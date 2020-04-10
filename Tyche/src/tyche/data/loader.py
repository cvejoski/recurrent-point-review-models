import pickle as pickle
import spacy
import torch
import torch.nn.functional as F
from abc import ABC
from collections import Counter
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.iterator import BucketIterator
from torchtext.datasets import text_classification
from torchtext.utils import download_from_url, extract_archive
from torchtext.vocab import Vocab
from tqdm import tqdm
from tyche import data
from tyche.data import datasets
from unittest.mock import Mock

spacy_en = spacy.load('en_core_web_sm')

URLS = {
    'AG_NEWS':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUDNpeUdjb0wxRms',
    'SogouNews':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUkVqNEszd0pHaFE',
    'DBpedia':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k',
    'YelpReviewPolarity':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbNUpYQ2N3SGlFaDg',
    'YelpReviewFull':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZlU4dXhHTFhZQU0',
    'YahooAnswers':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9Qhbd2JNdDBsQUdocVU',
    'AmazonReviewPolarity':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbaW12WVVZS2drcnM',
    'AmazonReviewFull':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZVhsUnRWRDhETzA'
}


def tokenizer(x, punct=True):
    """
    Create a tokenizer function
    """
    if punct:
        return [token.orth_ for token in spacy_en.tokenizer(x) if not token.is_space]
    else:
        return [token.orth_ for token in spacy_en.tokenizer(x) if not token.is_punct | token.is_space]


def tokenizer_ptb(x, punct=True):
    """
    Create a tokenizer function and replaces unk tokens in source textfile
    """
    x = x.replace("<unk>", "unk")
    if punct:
        return [token.orth_ for token in spacy_en.tokenizer(x) if not token.is_space]
    else:
        return [token.orth_ for token in spacy_en.tokenizer(x) if not token.is_punct | token.is_space]


class ADataLoader(ABC):
    def __init__(self, device, **kwargs):
        self.device = device
        self.batch_size = kwargs.pop('batch_size')
        self.path_to_vectors = kwargs.pop('path_to_vectors')
        self.emb_dim = kwargs.pop('emb_dim')
        self.voc_size = kwargs.pop('voc_size', None)
        self.min_freq = kwargs.pop('min_freq', 1)
        self._fix_length = kwargs.pop('fix_len', None)
        self.min_len = kwargs.pop('min_len', None)
        self.max_len = kwargs.pop('max_len', None)
        self.lower = kwargs.pop('lower', False)
        self.punctuation = kwargs.pop('punctuation', True)
        self.dataset_kwargs = kwargs

    @property
    def train(self):
        pass

    @property
    def validate(self):
        pass

    @property
    def test(self):
        pass


class DataLoaderPTB(ADataLoader):
    def __init__(self, device, **kwargs):
        path_to_data = kwargs.pop('path_to_data')
        super().__init__(device, **kwargs)

        # Defining fields
        TEXT = data.ReversibleField(init_token='<sos>', eos_token='<eos>', lower=self.lower,
                                    tokenize=tokenizer_ptb,
                                    include_lengths=True, fix_length=self._fix_length, batch_first=True)
        train, valid, test = datasets.PennTreebank.splits(TEXT, root=path_to_data)

        if self.min_len is not None:
            train.examples = [x for x in train.examples if len(x.text) >= self.min_len]
            valid.examples = [x for x in valid.examples if len(x.text) >= self.min_len]
            test.examples = [x for x in test.examples if len(x.text) >= self.min_len]
        if self.max_len is not None:
            train.examples = [x for x in train.examples if len(x.text) <= self.max_len]
            valid.examples = [x for x in valid.examples if len(x.text) <= self.max_len]
            test.examples = [x for x in test.examples if len(x.text) <= self.max_len]

        if self._fix_length == -1:
            TEXT.fix_length = max([train.max_len, valid.max_len, test.max_len])

        self._train_iter, self._valid_iter, self._test_iter = BucketIterator.splits(
                (train, valid, test),
                batch_sizes=(self.batch_size, self.batch_size, self.batch_size),
                sort_key=lambda x: len(x.text),
                sort_within_batch=True,
                repeat=False,
                device=self.device
        )

        TEXT.build_vocab(train, vectors=self.emb_dim, vectors_cache=self.path_to_vectors,
                         max_size=self.voc_size, min_freq=self.min_freq)
        self.train_vocab = TEXT.vocab
        self._fix_length = TEXT.fix_length

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
    def vocab(self):
        return self.train_vocab

    @property
    def fix_len(self):
        return self._fix_length


def _preprocess_wiki(dataset, min_len, max_len):
    if min_len is not None:
        dataset.examples = [sent for sent in dataset.examples if len(sent.text) >= min_len]
    if max_len is not None:
        dataset.examples = [sent for sent in dataset.examples if len(sent.text) <= max_len]
    for sent in dataset.examples:
        words = sent.text
        words = [word for word in words if not word == '@-@']
        sent.text = words
    dataset.examples = [sent for sent in dataset.examples if sent.text.count('=') < 2]
    return dataset


class DataLoaderWiki2(ADataLoader):
    def __init__(self, device, **kwargs):
        path_to_data = kwargs.pop('path_to_data')
        super().__init__(device, **kwargs)

        # Defining fields
        TEXT = data.ReversibleField(init_token='<sos>', eos_token='<eos>', unk_token='<unk>',
                                    tokenize=None, lower=self.lower,
                                    include_lengths=True, fix_length=self._fix_length, batch_first=True)
        train, valid, test = datasets.WikiText2.splits(TEXT, root=path_to_data)

        for dataset in [train, valid, test]:
            dataset = _preprocess_wiki(dataset, self.min_len, self.max_len)

        if self._fix_length == -1:
            TEXT.fix_length = max([train.max_len, valid.max_len, test.max_len])

        self._train_iter, self._valid_iter, self._test_iter = BucketIterator.splits(
                (train, valid, test),
                batch_sizes=(self.batch_size, self.batch_size, len(test)),
                sort_key=lambda x: len(x.text),
                sort_within_batch=True,
                repeat=False,
                device=self.device

        )

        TEXT.build_vocab(train, vectors=self.emb_dim, vectors_cache=self.path_to_vectors, max_size=self.voc_size,
                         min_freq=self.min_freq)
        self.train_vocab = TEXT.vocab
        self._fix_length = TEXT.fix_length

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
    def vocab(self):
        return self.train_vocab

    @property
    def fix_len(self):
        return self._fix_length


class DataLoaderWiki103(ADataLoader):
    def __init__(self, device, **kwargs):
        path_to_data = kwargs.pop('path_to_data')
        super().__init__(device, **kwargs)

        # Defining fields
        TEXT = data.ReversibleField(init_token='<sos>', eos_token='<eos>',
                                    tokenize=None, lower=self.lower,
                                    include_lengths=True, fix_length=self._fix_length, batch_first=True)
        train, valid, test = datasets.WikiText103.splits(TEXT, root=path_to_data)

        for dataset in [train, valid, test]:
            dataset = _preprocess_wiki(dataset, self.min_len, self.max_len)

        if self._fix_length == -1:
            TEXT.fix_length = max([train.max_len, valid.max_len, test.max_len])

        self._train_iter, self._valid_iter, self._test_iter = BucketIterator.splits(
                (train, valid, test),
                batch_sizes=(self.batch_size, self.batch_size, len(test)),
                sort_key=lambda x: len(x.text),
                sort_within_batch=True,
                repeat=False,
                device=self.device
        )

        TEXT.build_vocab(train, vectors=self.emb_dim, vectors_cache=self.path_to_vectors, max_size=self.voc_size,
                         min_freq=self.min_freq)
        self.train_vocab = TEXT.vocab
        self._fix_length = TEXT.fix_length

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
    def vocab(self):
        return self.train_vocab

    @property
    def fix_len(self):
        return self._fix_length


class DataLoaderYelp2019(ADataLoader):
    def __init__(self, device, **kwargs):
        server = kwargs.pop('server', 'localhost')
        db_name = kwargs.pop('db')
        data_collection_name = kwargs.pop('data_collection')
        super().__init__(device, **kwargs)

        train_col = f'{data_collection_name}_train'
        val_col = f'{data_collection_name}_validation'
        test_col = f'{data_collection_name}_test'

        FIELD_TEXT = data.ReversibleField(init_token='<sos>', eos_token='<eos>', unk_token='<unk>',
                                          tokenize=partial(tokenizer, punct=self.punctuation), batch_first=True,
                                          use_vocab=True,
                                          fix_length=self._fix_length,
                                          include_lengths=True, lower=self.lower)

        train, valid, test = datasets.Yelp2019.splits(server, db_name, text_field=FIELD_TEXT, train=train_col,
                                                      validation=val_col, test=test_col, **self.dataset_kwargs)
        if self._fix_length == -1:
            FIELD_TEXT.fix_length = max([train.max_len, valid.max_len, test.max_len])

        self._train_iter, self._valid_iter, self._test_iter = BucketIterator.splits(
                (train, valid, test), batch_sizes=(self.batch_size, self.batch_size, len(test)),
                sort_key=lambda x: len(x.text), sort_within_batch=True, repeat=False, device=self.device)
        FIELD_TEXT.build_vocab(train, vectors=self.emb_dim, vectors_cache=self.path_to_vectors, max_size=self.voc_size,
                               min_freq=self.min_freq)
        self.train_vocab = FIELD_TEXT.vocab
        self._fix_length = FIELD_TEXT.fix_length

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
    def vocab(self):
        return self.train_vocab


class DataLoaderYelp(ADataLoader):
    def __init__(self, device, **kwargs):
        self.device = device
        self.batch_size = kwargs.get('batch_size')
        self.path_to_data = kwargs.pop('path_to_data')
        self.path_to_vectors = kwargs.pop('path_to_vectors')
        self.emb_dim = kwargs.pop('emb_dim')
        self.voc_size = kwargs.pop('voc_size')
        self.min_freq = kwargs.pop('min_freq', 1)
        self.fix_length = kwargs.pop('fix_len', None)
        self.ngrams = kwargs.pop('ngrams', 1)
        self.download = kwargs.pop('download', False)
        self.save_split = kwargs.pop('save_split', False)
        self.load_split = kwargs.pop('load_split', False)

        self.train_sample = 100000
        self.valid_sample = 10000

        if self.load_split:
            print("loading pickle sample split")
            with open('/opt/mlfta/ansmodels/yelp/yelp_split_sample.pkl', 'rb') as input:
                split = pickle.load(input)
                sub_train_ = split[0]
                sub_valid_ = split[1]

            print("loading pickle sample split done")

        else:
            self.train_dataset, self.test_dataset = self._setup_datasets("yelp_review_full_csv", self.path_to_data,
                                                                         self.ngrams, vocab=None, include_unk=True,
                                                                         download=self.download)

            print("sample {} train data and {} valid data".format(self.train_sample, self.valid_sample))
            # sample 100k train, 10k valid, 10k test
            sub_train_, remain_train = random_split(self.train_dataset,
                                                    [self.train_sample, len(self.train_dataset) - self.train_sample])
            sub_valid_, _ = random_split(remain_train,
                                         [self.valid_sample, len(remain_train) - self.valid_sample])

        if self.save_split:
            print("saving sample split in pickle")

            split = [sub_train_, sub_valid_]

            with open('/opt/mlfta/ansmodels/yelp/yelp_split_sample.pkl', 'wb') as output:
                pickle.dump(split, output, pickle.HIGHEST_PROTOCOL)

        self.train_vocab = sub_train_.dataset.get_vocab()

        if self.fix_length is not None:
            sub_train_ = [x for x in sub_train_ if len(x[1]) < self.fix_length - 2]
            sub_valid_ = [x for x in sub_valid_ if len(x[1]) < self.fix_length - 2]

        else:
            print("fix length will be longest part in dataset")
            longest_train = max(sub_train_, key=lambda x: len(x[1]))
            longest_valid = max(sub_valid_, key=lambda x: len(x[1]))
            self.fix_length = max(len(longest_train[1]), len(longest_valid[1]))

        print("fix length to {}".format(self.fix_length))

        self._train_iter = DataLoader(sub_train_, batch_size=self.batch_size, shuffle=True,
                                      collate_fn=self.generate_batch, )

        self._valid_iter = DataLoader(sub_valid_, batch_size=self.batch_size, shuffle=True,
                                      collate_fn=self.generate_batch)

    def _setup_datasets(self, dataset_name, root='./data', ngrams=1, vocab=None, include_unk=True, download=False):
        if download:
            dataset_tar = download_from_url(URLS[dataset_name], root=root)
            extracted_files = extract_archive(dataset_tar)

            for fname in extracted_files:
                if fname.endswith('train.csv'):
                    train_csv_path = fname
                if fname.endswith('test.csv'):
                    test_csv_path = fname

        else:
            dir_name = root + "/" + dataset_name + "/"
            train_csv_path = dir_name + "train.csv"
            test_csv_path = dir_name + "test.csv"

        if vocab is None:
            print('Building Vocab based on {}'.format(train_csv_path))
            vocab = self.build_vocab_from_iterator(text_classification._csv_iterator(train_csv_path, ngrams))
        else:
            if not isinstance(vocab, Vocab):
                raise TypeError("Passed vocabulary is not of type Vocab")
        print('Vocab has {} entries'.format(len(vocab)))
        print('Creating training data')
        train_data, train_labels = text_classification._create_data_from_iterator(
                vocab, text_classification._csv_iterator(train_csv_path, ngrams, yield_cls=True), include_unk)
        print('Creating testing data')
        test_data, test_labels = text_classification._create_data_from_iterator(
                vocab, text_classification._csv_iterator(test_csv_path, ngrams, yield_cls=True), include_unk)
        if len(train_labels ^ test_labels) > 0:
            raise ValueError("Training and test labels don't match")
        return (text_classification.TextClassificationDataset(vocab, train_data, train_labels),
                text_classification.TextClassificationDataset(vocab, test_data, test_labels))

    def build_vocab_from_iterator(self, iterator):
        """
        Build a Vocab from an iterator.

        Arguments:
            iterator: Iterator used to build Vocab. Must yield list or iterator of tokens.
        """

        counter = Counter()
        with tqdm(unit_scale=0, unit='lines') as t:
            for tokens in iterator:
                counter.update(tokens)
                t.update(1)
        word_vocab = Vocab(counter=counter, max_size=self.voc_size, min_freq=self.min_freq,
                           specials=['<unk>', '<pad>', '<sos>', '<eos>'],
                           vectors=self.emb_dim, vectors_cache=self.path_to_vectors, specials_first=True)
        return word_vocab

    def generate_batch(self, batch):

        sos = self.train_vocab.stoi["<sos>"]
        eos = self.train_vocab.stoi["<eos>"]
        pad = self.train_vocab.stoi["<pad>"]
        text = [self.add_padding(entry[1], sos, eos, pad) for entry in batch]
        label = torch.tensor([entry[0] for entry in batch])
        seq_len = [torch.tensor(len(entry)) for entry in text]
        text = torch.stack(text)
        seq_len = torch.stack(seq_len)
        text, seq_len, label = text.to(self.device), seq_len.to(self.device), label.to(self.device)
        minibatch = Mock()
        minibatch.text = (text, seq_len)
        minibatch.label = label

        return minibatch

    def add_padding(self, text_entry, sos, eos, pad):
        pad_start = (1, 0)
        pad_end = (0, 1)
        text_entry = F.pad(text_entry, pad_start, value=sos)
        text_entry = F.pad(text_entry, pad_end, value=eos)
        difference = self.fix_length - len(text_entry)
        if difference > 0:
            pad_length = (0, difference)
            text_entry = F.pad(text_entry, pad_length, value=pad)

        return text_entry

    def add_word_to_vocab(self, word):
        vocab = self.train_dataset.get_vocab()
        if word not in vocab.stoi:
            vocab.itos.append(word)
            vocab.stoi[word] = len(vocab.itos) - 1

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
    def vocab(self):
        return self.train_vocab

    @property
    def fix_len(self):
        return self.fix_length


class DataLoaderYahoo(ADataLoader):
    def __init__(self, device, **kwargs):
        self.device = device
        self.batch_size = kwargs.get('batch_size')
        self.path_to_data = kwargs.pop('path_to_data')
        self.path_to_vectors = kwargs.pop('path_to_vectors')
        self.emb_dim = kwargs.pop('emb_dim')
        self.voc_size = kwargs.pop('voc_size')
        self.min_freq = kwargs.pop('min_freq', 1)
        self.fix_length = kwargs.pop('fix_len', None)
        self.ngrams = kwargs.pop('ngrams', 1)
        self.download = kwargs.pop('download', False)
        self.save_split = kwargs.pop('save_split', False)
        self.load_split = kwargs.pop('load_split', False)

        self.train_sample = 100000
        self.valid_sample = 10000

        if self.load_split:
            print("loading pickle sample split")
            with open('/opt/mlfta/ansmodels/yahoo/yahoo_split_sample.pkl', 'rb') as input:
                split = pickle.load(input)
                sub_train_ = split[0]
                sub_valid_ = split[1]

            print("loading pickle sample split done")

        else:
            self.train_dataset, self.test_dataset = self._setup_datasets("yahoo_answers_csv", self.path_to_data,
                                                                         self.ngrams, vocab=None, include_unk=True,
                                                                         download=self.download)

            print("sample {} train data and {} valid data".format(self.train_sample, self.valid_sample))
            # sample 100k train, 10k valid, 10k test
            sub_train_, remain_train = random_split(self.train_dataset,
                                                    [self.train_sample, len(self.train_dataset) - self.train_sample])
            sub_valid_, _ = random_split(remain_train,
                                         [self.valid_sample, len(remain_train) - self.valid_sample])

        if self.save_split:
            print("saving sample split in pickle")

            split = [sub_train_, sub_valid_]

            with open('/opt/mlfta/ansmodels/yahoo/yahoo_split_sample.pkl', 'wb') as output:
                pickle.dump(split, output, pickle.HIGHEST_PROTOCOL)

        self.train_vocab = sub_train_.dataset.get_vocab()

        if self.fix_length is not None:
            sub_train_ = [x for x in sub_train_ if len(x[1]) < self.fix_length - 2]
            sub_valid_ = [x for x in sub_valid_ if len(x[1]) < self.fix_length - 2]

        else:
            print("fix length will be longest part in dataset")
            longest_train = max(sub_train_, key=lambda x: len(x[1]))
            longest_valid = max(sub_valid_, key=lambda x: len(x[1]))
            self.fix_length = max(len(longest_train[1]), len(longest_valid[1]))

        print("fix length to {}".format(self.fix_length))

        self._train_iter = DataLoader(sub_train_, batch_size=self.batch_size, shuffle=True,
                                      collate_fn=self.generate_batch, )

        self._valid_iter = DataLoader(sub_valid_, batch_size=self.batch_size, shuffle=True,
                                      collate_fn=self.generate_batch)

    def _setup_datasets(self, dataset_name, root='./data', ngrams=1, vocab=None, include_unk=True, download=False):
        if download:
            dataset_tar = download_from_url(URLS[dataset_name], root=root)
            extracted_files = extract_archive(dataset_tar)

            for fname in extracted_files:
                if fname.endswith('train.csv'):
                    train_csv_path = fname
                if fname.endswith('test.csv'):
                    test_csv_path = fname

        else:
            dir_name = root + "/" + dataset_name + "/"
            train_csv_path = dir_name + "train.csv"
            test_csv_path = dir_name + "test.csv"

        if vocab is None:
            print('Building Vocab based on {}'.format(train_csv_path))
            vocab = self.build_vocab_from_iterator(text_classification._csv_iterator(train_csv_path, ngrams))
        else:
            if not isinstance(vocab, Vocab):
                raise TypeError("Passed vocabulary is not of type Vocab")
        print('Vocab has {} entries'.format(len(vocab)))
        print('Creating training data')
        train_data, train_labels = text_classification._create_data_from_iterator(
                vocab, text_classification._csv_iterator(train_csv_path, ngrams, yield_cls=True), include_unk)
        print('Creating testing data')
        test_data, test_labels = text_classification._create_data_from_iterator(
                vocab, text_classification._csv_iterator(test_csv_path, ngrams, yield_cls=True), include_unk)
        if len(train_labels ^ test_labels) > 0:
            raise ValueError("Training and test labels don't match")
        return (text_classification.TextClassificationDataset(vocab, train_data, train_labels),
                text_classification.TextClassificationDataset(vocab, test_data, test_labels))

    def build_vocab_from_iterator(self, iterator):
        """
        Build a Vocab from an iterator.

        Arguments:
            iterator: Iterator used to build Vocab. Must yield list or iterator of tokens.
        """

        counter = Counter()
        with tqdm(unit_scale=0, unit='lines') as t:
            for tokens in iterator:
                counter.update(tokens)
                t.update(1)
        word_vocab = Vocab(counter=counter, max_size=self.voc_size, min_freq=self.min_freq,
                           specials=['<unk>', '<pad>', '<sos>', '<eos>'],
                           vectors=self.emb_dim, vectors_cache=self.path_to_vectors, specials_first=True)
        return word_vocab

    def generate_batch(self, batch):

        sos = self.train_vocab.stoi["<sos>"]
        eos = self.train_vocab.stoi["<eos>"]
        pad = self.train_vocab.stoi["<pad>"]
        text = [self.add_padding(entry[1], sos, eos, pad) for entry in batch]
        label = torch.tensor([entry[0] for entry in batch])
        seq_len = [torch.tensor(len(entry)) for entry in text]
        text = torch.stack(text)
        seq_len = torch.stack(seq_len)
        text, seq_len, label = text.to(self.device), seq_len.to(self.device), label.to(self.device)
        minibatch = Mock()
        minibatch.text = (text, seq_len)
        minibatch.label = label

        return minibatch

    def add_padding(self, text_entry, sos, eos, pad):
        pad_start = (1, 0)
        pad_end = (0, 1)
        text_entry = F.pad(text_entry, pad_start, value=sos)
        text_entry = F.pad(text_entry, pad_end, value=eos)
        difference = self.fix_length - len(text_entry)
        if difference > 0:
            pad_length = (0, difference)
            text_entry = F.pad(text_entry, pad_length, value=pad)

        return text_entry

    def add_word_to_vocab(self, word):
        vocab = self.train_dataset.get_vocab()
        if word not in vocab.stoi:
            vocab.itos.append(word)
            vocab.stoi[word] = len(vocab.itos) - 1

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
    def vocab(self):
        return self.train_vocab

    @property
    def fix_len(self):
        return self.fix_length


class DataLoaderYelp15(ADataLoader):
    """
    @article{yang2017improved,
    title={Improved Variational Autoencoders for Text Modeling using Dilated
    Convolutions},
    author={Yang, Zichao and Hu, Zhiting and Salakhutdinov, Ruslan and
    Berg-Kirkpatrick, Taylor},
    journal={arXiv preprint arXiv:1702.08139},
    year={2017}
    }
    """

    def __init__(self, device, **kwargs):
        self.device = device
        self.batch_size = kwargs.get('batch_size')
        self.path_to_data = kwargs.pop('path_to_data')
        self.path_to_vectors = kwargs.pop('path_to_vectors')
        self.emb_dim = kwargs.pop('emb_dim')
        self.voc_size = kwargs.pop('voc_size')
        self.min_freq = kwargs.pop('min_freq', 1)
        self.fix_length = kwargs.pop('fix_len', 203)

        self.path_train_data = self.path_to_data + '/yelp_15/yelp.train.txt'
        self.path_val_data = self.path_to_data + '/yelp_15/yelp.valid.txt'
        self.path_test_data = self.path_to_data + '/yelp_15/yelp.test.txt'

        print("build vocab")
        vocab = self.build_vocab_from_textfile(self.path_train_data)

        print("create train split")
        list_train_data, list_train_labels = self.create_data_from_textfile(vocab, self.path_train_data,
                                                                            include_unk=True)
        train = text_classification.TextClassificationDataset(vocab, list_train_data, list_train_labels)

        print("create val split")
        list_val_data, list_val_labels = self.create_data_from_textfile(vocab, self.path_val_data, include_unk=True)
        valid = text_classification.TextClassificationDataset(vocab, list_val_data, list_val_labels)

        print("create test split")
        list_test_data, list_test_labels = self.create_data_from_textfile(vocab, self.path_test_data, include_unk=True)
        test = text_classification.TextClassificationDataset(vocab, list_test_data, list_test_labels)

        print("create data loaders")
        self._train_iter = DataLoader(train, batch_size=self.batch_size, shuffle=True,
                                       collate_fn=self.generate_batch, )

        self._valid_iter = DataLoader(valid, batch_size=self.batch_size, shuffle=True,
                                       collate_fn=self.generate_batch)

        self._test_iter = DataLoader(test, batch_size=self.batch_size, shuffle=True,
                                      collate_fn=self.generate_batch)

        self.train_vocab = vocab

    def build_vocab_from_textfile(self, fname):
        """
        read in frequencies from text file and build new vocab
        """
        counter = Counter()
        with open(fname) as fin:
            for line in fin:
                split_line = line.split('\t')
                split_line = split_line[1].split()
                counter.update(split_line)

        word_vocab = Vocab(counter=counter, max_size=self.voc_size, min_freq=self.min_freq,
                           specials=['<unk>', '<pad>', '<sos>', '<eos>'],
                           vectors=self.emb_dim, vectors_cache=self.path_to_vectors, specials_first=True)
        print("vocab len: {}".format(len(word_vocab)))

        return word_vocab

    def create_data_from_textfile(self, vocab, filename, include_unk):
        """
        replaces tokens from file with ids from vocab
        return data, labels
        """
        data = []
        labels = []
        with open(filename) as fin:
            for line in fin:
                split_line = line.split('\t')
                cls = int(split_line[0])
                split_line = split_line[1].split()
                if include_unk:
                    tokens = torch.tensor([vocab[token] for token in split_line])
                else:
                    token_ids = list(filter(lambda x: x is not Vocab.UNK, [vocab[token]
                                                                           for token in tokens]))
                    tokens = torch.tensor(token_ids)
                if len(tokens) == 0:
                    print('Row contains no tokens.')
                data.append((cls, tokens))
                labels.append(cls)
        return data, set(labels)

    def generate_batch(self, batch):
        """
        help function for data loader
        adds special tokens to each sentence and stores data as tensors in minibatch
        """

        sos = self.train_vocab.stoi["<sos>"]
        eos = self.train_vocab.stoi["<eos>"]
        pad = self.train_vocab.stoi["<pad>"]
        text = [self.add_padding(entry[1], sos, eos, pad) for entry in batch]
        label = torch.tensor([entry[0] for entry in batch])
        seq_len = [torch.tensor(len(entry)) for entry in text]
        text = torch.stack(text)
        seq_len = torch.stack(seq_len)
        text, seq_len, label = text.to(self.device), seq_len.to(self.device), label.to(self.device)
        minibatch = Mock()
        minibatch.text = (text, seq_len)
        minibatch.label = label

        return minibatch

    def add_padding(self, text_entry, sos, eos, pad):
        """adds padding to create same length sequences"""

        pad_start = (1, 0)
        pad_end = (0, 1)
        text_entry = F.pad(text_entry, pad_start, value=sos)
        text_entry = F.pad(text_entry, pad_end, value=eos)
        difference = self.fix_length - len(text_entry)
        if difference > 0:
            pad_length = (0, difference)
            text_entry = F.pad(text_entry, pad_length, value=pad)

        return text_entry

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
    def vocab(self):
        return self.train_vocab

    @property
    def fix_len(self):
        return self.fix_length
