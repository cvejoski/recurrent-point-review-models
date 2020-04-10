import pickle
from functools import partial

import numpy as np
import spacy
import torch
import tyche.data as data
from dpp.data import datasets
from pymongo import MongoClient
from torch.utils.data.dataloader import DataLoader
from tyche.data.field import BPTTField
from tyche.data.loader import ADataLoader

spacy_en = spacy.load('en_core_web_sm')


def tokenizer(x, punct=True):
    """
    Create a tokenizer function
    """
    if punct:
        return [token.orth_ for token in spacy_en.tokenizer(x) if not token.is_space]
    else:
        return [token.orth_ for token in spacy_en.tokenizer(x) if not token.is_punct | token.is_space]


def unpack_bow(x):
    x_unpacked = pickle.loads(x).toarray()[1:]
    x_unpacked = np.stack([x_unpacked[:-1], x_unpacked[1:]], 1)

    return x_unpacked


def unpack_bow2seq(x):
    return list(map(lambda i: pickle.loads(i), x[1:-1]))


def unpack_text(x):
    return x[1:-1]


def delta(x: list, t_max) -> np.ndarray:
    x = np.asarray(x, dtype=np.float) / t_max
    dt = np.diff(x)
    return np.stack([x[1:-1], dt[:-1], dt[1:]], 1)


def events(x: list):
    x = list(map(int, x))
    return [[x1, y] for (x1, y) in zip(x[:-1], x[1:])]


def min_max_scale(x, min_value, max_value):
    x = np.asarray(x, dtype=np.float)
    x = (x - min_value) / (max_value - min_value)
    return x


class BasicPointLoader(ADataLoader):
    def __init__(self, device, **kwargs):
        batch_size = kwargs.pop('batch_size')
        fix_len = kwargs.pop('fix_len', None)
        bptt_length = kwargs.pop('bptt_len')
        self._t_max = kwargs.pop('t_max')
        bow_size = kwargs.pop('bow_size')
        server = kwargs.pop('server', 'localhost')
        data_collection_name = kwargs.pop('data_collection')
        db_name = kwargs.pop('db')
        train_col = f'{data_collection_name}_{bow_size}_train'
        val_col = f'{data_collection_name}_{bow_size}_validation'
        test_col = f'{data_collection_name}_{bow_size}_test'

        db = MongoClient(f'mongodb://{server}/')[db_name]
        col = db[train_col]
        if self._t_max is None:
            min_max_values = list(col.aggregate([{"$project": {"_id": 0, "time": 1}}, {"$unwind": "$time"},
                                                 {"$group": {"_id": None, "max": {"$max": "$time"},
                                                             "min": {"$min": "$time"}}},
                                                 {"$limit": 1}]))[0]
            self._t_min = min_max_values['min']
            self._t_max = min_max_values['max']

        FIELD_TIME = BPTTField(bptt_length=bptt_length, use_vocab=False,
                               include_lengths=True, pad_token=np.array([0., 0., -1.]),
                               preprocessing=partial(delta, t_max=self._t_max), dtype=torch.float32,
                               fix_length=fix_len)

        FIELD_BOW = data.BPTTField(bptt_length=bptt_length, use_vocab=False, fix_length=fix_len,
                                   include_lengths=False,
                                   pad_token=np.zeros((2, bow_size)),
                                   # pad_token=[csr_matrix((1, bow_size)), csr_matrix((1, bow_size))],
                                   preprocessing=unpack_bow,  # postprocessing=expand_bow_vector,
                                   dtype=torch.float32)

        train, valid, test = datasets.BasicPointDataSet.splits(server, db_name, time_field=FIELD_TIME,
                                                               bow_field=FIELD_BOW,
                                                               train=train_col, validation=val_col, test=test_col,
                                                               **kwargs)

        if fix_len == -1:
            max_len = max([train.max_len, valid.max_len, test.max_len])
            FIELD_TIME.fix_length = max_len
            FIELD_BOW.fix_length = max_len

        self._train_iter, self._valid_iter, self._test_iter = data.BPTTIterator.splits(
                (train, valid, test), batch_sizes=(batch_size, batch_size, len(test)),
                sort_key=lambda x: x.time.shape[0],
                sort_within_batch=True, repeat=False, bptt_len=bptt_length, device=device)
        self._bptt_length = bptt_length
        self._fix_length = fix_len
        self._bow_size = bow_size

    @property
    def train(self):
        return self._train_iter

    @property
    def test(self):
        return self._test_iter

    @property
    def validate(self):
        return self._valid_iter

    @property
    def fix_len(self):
        return self._fix_length

    @property
    def bptt_length(self):
        return self._bptt_length

    @property
    def bow_size(self):
        return self._bow_size

    @property
    def t_max(self):
        return self._t_max


class TextPointLoader(ADataLoader):
    def __init__(self, device, dtype=torch.float32, **kwargs):
        super().__init__(device, **kwargs)
        kwargs = self.dataset_kwargs
        time_fix_len = kwargs.pop('time_fix_len', None)
        text_fix_len = kwargs.pop('text_fix_len', None)
        bptt_length = kwargs.pop('bptt_len')
        self._t_max = kwargs.pop('t_max')

        server = kwargs.pop('server', 'localhost')
        data_collection_name = kwargs.pop('data_collection')
        db_name = kwargs.pop('db')
        train_col = f'{data_collection_name}_train'
        val_col = f'{data_collection_name}_validation'
        test_col = f'{data_collection_name}_test'

        db = MongoClient(f'mongodb://{server}/')[db_name]
        col = db[train_col]
        if self._t_max is None:
            min_max_values = list(col.aggregate([{"$project": {"_id": 0, "time": 1}}, {"$unwind": "$time"},
                                                 {"$group": {"_id": None, "max": {"$max": "$time"},
                                                             "min": {"$min": "$time"}}},
                                                 {"$limit": 1}]))[0]
            self._t_min = min_max_values['min']
            self._t_max = min_max_values['max']
        # part_scale = partial(min_max_scale, min_value=self.__t_min, max_value=self.__t_max)
        FIELD_TIME = BPTTField(bptt_length=bptt_length, use_vocab=False,
                               include_lengths=True, pad_token=np.array([0., 0., -1.]),
                               preprocessing=partial(delta, t_max=self._t_max), dtype=dtype,
                               fix_length=time_fix_len)

        FIELD_TEXT = data.ReversibleField(init_token='<sos>', eos_token='<eos>', unk_token='<unk>',
                                          tokenize=partial(tokenizer, punct=self.punctuation), batch_first=True,
                                          use_vocab=True, fix_length=text_fix_len)
        NESTED_TEXT_FIELD = data.NestedBPTTField(FIELD_TEXT, bptt_length=bptt_length, use_vocab=False,
                                                 fix_length=time_fix_len, preprocessing=unpack_text,
                                                 include_lengths=True)

        train, valid, test = datasets.TextPointDataSet.splits(server, db_name, time_field=FIELD_TIME,
                                                              text_field=NESTED_TEXT_FIELD, train=train_col,
                                                              validation=val_col, test=test_col,
                                                              **kwargs)

        if time_fix_len == -1:
            max_len = max([train.max_len, valid.max_len, test.max_len])
            FIELD_TIME.fix_length = max_len
            NESTED_TEXT_FIELD.fix_length = max_len
        self._train_iter, self._valid_iter, self._test_iter = data.BPTTIterator.splits(
                (train, valid, test), batch_sizes=(self.batch_size, self.batch_size, len(test)),
                sort_key=lambda x: x.time.shape[0],
                sort_within_batch=True, repeat=False, bptt_len=bptt_length, device=device)
        NESTED_TEXT_FIELD.build_vocab(train, vectors=self.emb_dim, vectors_cache=self.path_to_vectors,
                                      max_size=self.voc_size,
                                      min_freq=self.min_freq)
        self._bptt_length = bptt_length
        self.train_vocab = NESTED_TEXT_FIELD.vocab
        self._time_fix_length = NESTED_TEXT_FIELD.fix_length
        self._text_fix_length = FIELD_TEXT.fix_length

    @property
    def train(self):
        return self._train_iter

    @property
    def test(self):
        return self._test_iter

    @property
    def validate(self):
        return self._valid_iter

    @property
    def time_fix_length(self):
        return self._time_fix_length

    @property
    def text_fix_length(self):
        return self._text_fix_length

    @property
    def bptt_length(self):
        return self._bptt_length

    @property
    def t_max(self):
        return self._t_max

    @property
    def t_min(self):
        return self._t_min

    @property
    def vocab(self):
        return self.train_vocab


class BasicPointEventDataLoader(ADataLoader):
    def __init__(self, device, **kwargs):
        self._t_max = kwargs.pop('t_max')
        batch_size = kwargs.pop('batch_size')
        fix_len = kwargs.pop('fix_len', None)
        bptt_length = kwargs.pop('bptt_length')
        path = kwargs.pop('path')

        FIELD_TIME = BPTTField(bptt_length=bptt_length, use_vocab=False,
                               include_lengths=True, pad_token=np.array([0., 0., -1.]),
                               preprocessing=partial(delta, t_max=self._t_max), dtype=torch.float32)
        FIELD_MARK = BPTTField(bptt_length=bptt_length, use_vocab=False,
                               include_lengths=False, pad_token=np.array([0, 0]), dtype=torch.int64,
                               preprocessing=events)
        train, valid, test = datasets.BasicPointEventDataSet.splits(path, time_field=FIELD_TIME,
                                                                    mark_field=FIELD_MARK, **kwargs)

        if fix_len == -1:
            max_len = max([train.max_len, valid.max_len, test.max_len])
            FIELD_TIME.fix_length = max_len

        self._train_iter, self._valid_iter, self._test_iter = data.BPTTPointIterator.splits(
                (train, valid, test),
                batch_sizes=(batch_size, batch_size, len(test)),
                shuffle=False,
                sort_within_batch=False,
                sort=True,
                sort_key=lambda x: len(x.time),
                repeat=False, bptt_len=bptt_length,
                device=device
        )
        # self._valid_iter = BucketIterator(valid, batch_size, train=False, sort=False, shuffle=False)
        # self._test_iter = BucketIterator(test, len(test), train=False, sort=False, shuffle=False)
        self.bptt_length = bptt_length

    @property
    def train(self):
        return self._train_iter

    @property
    def test(self):
        return self._test_iter

    @property
    def validate(self):
        return self._valid_iter

    @property
    def fix_len(self):
        return self.fix_length

    @property
    def bptt_len(self):
        return self.bptt_length

    @property
    def t_max(self):
        return self._t_max


class InteractingPointsDataLoader(ADataLoader):
    def __init__(self, **kwargs):
        data_path = kwargs.pop('data_path')
        self.__bptt_size = kwargs.pop('bptt_size')

        train_data = datasets.InteractingPointDataSet(data_path, bptt_size=self.__bptt_size)
        test_data = datasets.InteractingPointDataSet(data_path, train=False)

        self.__train_data_loader = DataLoader(train_data, **kwargs)
        self.__test_data_loader = DataLoader(test_data, **kwargs)

    @property
    def train(self):
        return self.__train_data_loader

    @property
    def validate(self):
        return self.__test_data_loader

    @property
    def bptt(self):
        return self.__bptt_size


class ConditionalWasssersteingPointDataLoader(ADataLoader):
    def __init__(self, **kwargs):
        data_path = kwargs.pop('data_path')
        self.past_of_sequence = kwargs.pop('past_of_sequence')

        train_data = datasets.ConditionalWasssersteingPointDataSet(data_path, past_of_sequence=self.past_of_sequence)
        test_data = datasets.ConditionalWasssersteingPointDataSet(data_path, past_of_sequence=self.past_of_sequence,
                                                                  train=False)

        self.past_size = train_data.past_size
        self.future_size = train_data.future_size

        self.__train_data_loader = DataLoader(train_data, **kwargs)
        self.__test_data_loader = DataLoader(test_data, **kwargs)

    @property
    def train(self):
        return self.__train_data_loader

    @property
    def validate(self):
        return self.__test_data_loader

    @property
    def bptt(self):
        return self.__bptt_size


class DataLoaderRatebeer(ADataLoader):
    def __init__(self, device, **kwargs):
        batch_size = kwargs.pop('batch_size')
        path_to_vectors = kwargs.pop('path_to_vectors')
        emb_dim = kwargs.pop('emb_dim')
        voc_size = kwargs.pop('voc_size', None)
        min_freq = kwargs.pop('min_freq')
        fix_len = kwargs.pop('fix_len', None)
        bptt_length = kwargs.pop('bptt_len')
        bow_size = kwargs.get('bow_size')
        server = kwargs.pop('server', 'localhost')
        data_collection_name = kwargs.pop('data_collection')
        self.__t_max = kwargs.pop('t_max')

        train_col = f'{data_collection_name}_train'
        val_col = f'{data_collection_name}_validation'
        test_col = f'{data_collection_name}_test'

        db = MongoClient('mongodb://' + server)['hawkes_text']
        col = db[train_col]
        if self.__t_max is None:
            min_max_values = list(col.aggregate([{"$project": {"_id": 0, "time": 1}}, {"$unwind": "$time"},
                                                 {"$group": {"_id": None, "max": {"$max": "$time"},
                                                             "min": {"$min": "$time"}}},
                                                 {"$limit": 1}]))[0]
            self.__t_min = min_max_values['min']
            self.__t_max = min_max_values['max']
        # part_scale = partial(min_max_scale, min_value=self.__t_min, max_value=self.__t_max)
        FIELD_TIME = data.BPTTField(bptt_length=bptt_length, use_vocab=False, fix_length=fix_len,
                                    include_lengths=True, pad_token=[-1.0, -1.0, -1.0],
                                    preprocessing=partial(delta, t_max=self.__t_max), dtype=torch.float64)
        FIELD_BOW = data.BPTTField(bptt_length=bptt_length, use_vocab=False, fix_length=fix_len,
                                   include_lengths=False,
                                   pad_token=np.zeros(bow_size),
                                   preprocessing=unpack_bow2seq,
                                   dtype=torch.float64)

        FIELD_TEXT = data.ReversibleField(init_token='<sos>', eos_token='<eos>',
                                          tokenize=tokenizer, batch_first=True, use_vocab=True)
        NESTED_TEXT_FIELD = data.NestedBPTTField(FIELD_TEXT, bptt_length=bptt_length, use_vocab=False,
                                                 fix_length=fix_len, preprocessing=unpack_text, include_lengths=True)

        train, valid, test = datasets.TextPointDataSet.splits(server, time_field=FIELD_TIME,
                                                              text_field=NESTED_TEXT_FIELD, bow_field=FIELD_BOW,
                                                              train=train_col, validation=val_col, test=test_col,
                                                              **kwargs)

        if fix_len == -1:
            max_len = max([train.max_len, valid.max_len, test.max_len])
            FIELD_TIME.fix_length = max_len
            FIELD_BOW.fix_length = max_len
            NESTED_TEXT_FIELD.fix_length = max_len
        self._train_iter, self._valid_iter, self._test_iter = data.BPTTIterator.splits(
                (train, valid, test), batch_sizes=(batch_size, batch_size, len(test)), sort_key=lambda x: len(x.time),
                sort_within_batch=True, repeat=False, bptt_len=bptt_length, device=device)
        self._bptt_length = bptt_length
        NESTED_TEXT_FIELD.build_vocab(train, vectors=emb_dim, vectors_cache=path_to_vectors, max_size=voc_size,
                                      min_freq=min_freq)
        self.train_vocab = NESTED_TEXT_FIELD.vocab
        self._fix_length = NESTED_TEXT_FIELD.fix_length
        self._bow_size = bow_size

    @property
    def train(self):
        return self._train_iter

    @property
    def test(self):
        return self._test_iter

    @property
    def validate(self):
        return self._valid_iter

    @property
    def fix_length(self):
        return self._fix_length

    @property
    def bptt_length(self):
        return self._bptt_length

    @property
    def bow_size(self):
        return self._bow_size

    @property
    def t_max(self):
        return self.__t_max

    @property
    def t_min(self):
        return self.__t_min


class DataLoaderRatebeerBow(ADataLoader):
    def __init__(self, device, **kwargs):
        batch_size = kwargs.pop('batch_size')
        fix_len = kwargs.pop('fix_len', None)
        bptt_length = kwargs.pop('bptt_len')
        bow_size = kwargs.pop('bow_size')
        server = kwargs.pop('server', 'localhost')
        data_collection_name = kwargs.pop('data_collection')
        self.__t_max = kwargs.pop('t_max')
        train_col = f'{data_collection_name}_train_' + str(bow_size)
        val_col = f'{data_collection_name}_validation_' + str(bow_size)
        test_col = f'{data_collection_name}_test_' + str(bow_size)

        db = MongoClient('mongodb://' + server)['hawkes_text']
        col = db[train_col]
        if self.__t_max is None:
            min_max_values = list(col.aggregate([{"$project": {"_id": 0, "time": 1}}, {"$unwind": "$time"},
                                                 {"$group": {"_id": None, "max": {"$max": "$time"},
                                                             "min": {"$min": "$time"}}},
                                                 {"$limit": 1}]))[0]
            self.__t_min = min_max_values['min']
            self.__t_max = min_max_values['max']
        # part_scale = partial(min_max_scale, min_value=self.min_time, max_value=self.max_time)
        FIELD_TIME = data.BPTTField(bptt_length=bptt_length, use_vocab=False, fix_length=fix_len,
                                    include_lengths=True, pad_token=np.array([-1.0, -1.0, -1.0]),
                                    preprocessing=partial(delta, t_max=self.__t_max), dtype=torch.float32)
        FIELD_BOW = data.BPTTField(bptt_length=bptt_length, use_vocab=False, fix_length=fix_len,
                                   include_lengths=False,
                                   pad_token=np.zeros((2, bow_size)),
                                   # pad_token=[csr_matrix((1, bow_size)), csr_matrix((1, bow_size))],
                                   preprocessing=unpack_bow,  # postprocessing=expand_bow_vector,
                                   dtype=torch.float32)

        train, valid, test = datasets.RatebeerBow.splits(server, time_field=FIELD_TIME, bow_field=FIELD_BOW,
                                                         train=train_col, validation=val_col, test=test_col, **kwargs)

        if fix_len == -1:
            max_len = max([train.max_len, valid.max_len, test.max_len])
            FIELD_TIME.fix_length = max_len
            FIELD_BOW.fix_length = max_len

        self._train_iter, self._valid_iter, self._test_iter = data.BPTTIterator.splits(
                (train, valid, test), batch_sizes=(batch_size, batch_size, len(test)), sort_key=lambda x: len(x.time),
                sort_within_batch=True, repeat=False, bptt_len=bptt_length, device=device)
        self.bptt_length = bptt_length
        self.bow_s = bow_size

    @property
    def train(self):
        return self._train_iter

    @property
    def test(self):
        return self._test_iter

    @property
    def validate(self):
        return self._valid_iter

    @property
    def fix_len(self):
        return self.fix_length

    @property
    def bptt_len(self):
        return self.bptt_length

    @property
    def bow_size(self):
        return self.bow_s

    @property
    def t_max(self):
        return self.__t_max

    @property
    def t_min(self):
        return self.__t_min
