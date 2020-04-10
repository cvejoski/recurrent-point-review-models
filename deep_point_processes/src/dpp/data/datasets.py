import os
from typing import Any

import numpy as np
import torch
from pymongo import MongoClient
from torch.utils.data.dataset import Dataset
from torchtext import data

make_example = data.Example.fromdict


def merge_examples(a, b):
    for t, e in zip(a, b):
        setattr(t, "mark", e.mark)
    return a


class BasicPointDataSet(data.Dataset):
    def __init__(self, server: str, db: str, collection: str, time_field, bow_field, **kwargs):

        fields = {'time': ('time', time_field), 'bow': ('bow', bow_field)}

        col = MongoClient(f'mongodb://{server}/')[db][collection]
        c = col.find({})
        examples = [make_example(i, fields) for i in c]

        fields, field_dict = [], fields
        for field in field_dict.values():
            if isinstance(field, list):
                fields.extend(field)
            else:
                fields.append(field)

        super().__init__(examples, fields, **kwargs)
        self.max_len = max([len(f.time) for f in self.examples])

    @classmethod
    def splits(cls, server: str, db: str, train=None, validation=None, test=None, **kwargs):
        train_data = None if train is None else cls(server, db, train, **kwargs)
        val_data = None if validation is None else cls(server, db, validation, **kwargs)
        test_data = None if train is None else cls(server, db, test, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data) if d is not None)


class TextPointDataSet(data.Dataset):
    def __init__(self, server: str, db: str, collection: str, time_field, text_field, **kwargs):
        fields = {'time': ('time', time_field), 'text': ('text', text_field)}
        col = MongoClient(f'mongodb://{server}/')[db][collection]
        # cursor_text = col.find({}).limit(10)
        cursor_text = col.find({})
        examples = [make_example(i, fields) for i in cursor_text]

        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super(TextPointDataSet, self).__init__(examples, fields, **kwargs)
        self.max_len = max([len(f.time) for f in self.examples])

    @classmethod
    def splits(cls, server: str, db: str, train=None, validation=None, test=None, **kwargs):
        train_data = None if train is None else cls(server, db, train, **kwargs)
        val_data = None if validation is None else cls(server, db, validation, **kwargs)
        test_data = None if train is None else cls(server, db, test, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data) if d is not None)


class RatebeerBow(data.Dataset):
    def __init__(self, server: str, collection: str, time_field, bow_field, **kwargs):

        fields = {'time': ('time', time_field), 'bow': ('bow', bow_field)}

        col = MongoClient('mongodb://' + server)['hawkes_text'][collection]
        c = col.find({}).limit(50)
        examples = [make_example(i, fields) for i in c]

        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super().__init__(examples, fields, **kwargs)
        self.max_len = max([len(f.time) for f in self.examples])

    @classmethod
    def splits(cls, server: str, train='ratebeer_by_user_train_2000',
               validation='ratebeer_by_user_validation_2000', test='ratebeer_by_user_test_2000',
               **kwargs):

        train_data = None if train is None else cls(server, train, **kwargs)
        val_data = None if validation is None else cls(server, validation, **kwargs)
        test_data = None if train is None else cls(server, test, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)


class RatebeerBow2Seq(data.Dataset):
    def __init__(self, server: str, collection: str, time_field, text_field, bow_field, **kwargs):
        bow_size = kwargs.pop('bow_size')
        fields = {'time': ('time', time_field), 'text': (
            'text', text_field), 'bow': ('bow', bow_field)}
        collection_name_bow = f"{collection}_{bow_size}"
        db = MongoClient('mongodb://' + server)['hawkes_text']
        col_bow = db[collection_name_bow]
        col = db[collection]
        cursor_text = col.find({}).limit(100)
        cursor_bow = col_bow.find({}).limit(100)

        examples = []
        for bow, text in zip(cursor_bow, cursor_text):
            example = {**bow, **text}
            examples.append(make_example(example, fields))

        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super(RatebeerBow2Seq, self).__init__(examples, fields, **kwargs)
        self.max_len = max([len(f.time) for f in self.examples])

    @classmethod
    def splits(cls, server: str, train='ratebeer_by_user_train',
               validation='ratebeer_by_user_validation', test='ratebeer_by_user_test',
               **kwargs):

        train_data = None if train is None else cls(server, train, **kwargs)
        val_data = None if validation is None else cls(server, validation, **kwargs)
        test_data = None if train is None else cls(server, test, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)

    @classmethod
    def iters(cls, text_field, batch_size=32, device='cpu', root='.data',
              vectors=None, vectors_cache=None, max_size=None, min_freq=1, **kwargs):
        """Create iterator objects for splits of the Penn Treebank dataset.
        This is the simplest way to use the dataset, and assumes common
        defaults for field, vocabulary, and iterator parameters.
        Arguments:
            text_field: The field that will be used for text data.
            batch_size: Batch size.
            bptt_len: Length of sequences for backpropagation through time.
            device: Device to create batches on. Use -1 for CPU and None for
                the currently active GPU device.
            root: The root directory where the data files will be stored.
            wv_dir, wv_type, wv_dim: Passed to the Vocab constructor for the
                text field. The word vectors are accessible as
                train.dataset.fields['text'].vocab.vectors.
            Remaining keyword arguments: Passed to the splits method.
        """
        train, val, test = cls.splits(text_field, root=root, **kwargs)

        return data.BucketIterator.splits((train, val, test), batch_size=batch_size, device=device)


class BasicPointEventDataSet(data.Dataset):
    def __init__(self, path: str, file_names: tuple, time_field, mark_field, **kwargs):

        path_time = os.path.join(path, file_names[0])
        path_event = os.path.join(path, file_names[1])
        time_examples = data.TabularDataset(path_time, format='csv', fields=[('time', time_field)]).examples
        event_examples = data.TabularDataset(path_event, format='csv', fields=[('mark', mark_field)]).examples
        examples = merge_examples(time_examples, event_examples)
        fields = [('time', time_field), ('mark', mark_field)]
        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super(BasicPointEventDataSet, self).__init__(examples, fields, **kwargs)
        self.max_len = max([len(f.time) for f in self.examples])

    @classmethod
    def splits(cls, root: str, train=('time-train.txt', 'event-train.txt'),
               validation=('time-test.txt', 'event-test.txt'), test=('time-test.txt', 'event-test.txt'),
               **kwargs):

        train_data = None if train is None else cls(root, train, **kwargs)
        val_data = None if validation is None else cls(root, validation, **kwargs)
        test_data = None if test is None else cls(root, test, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)


class InteractingPointDataSet(Dataset):
    def __init__(self, data_path: str, train=True, bptt_size=50) -> Dataset:
        self.data_path = data_path
        suffix = "train" if train else "test"

        events = self.__prepare_event_types(data_path, suffix)
        self.arrivals = self.__prepare_arrivals(data_path, suffix)
        self.arrivals = self._prepare_events_and_arrivals(events)
        self.prepare_batches_interacting(suffix)
        self.edges = self.__prepare_edges(data_path, suffix)

    def _prepare_events_and_arrivals(self, events):
        """
        transform the input data as needed for the recurrent market point processes
        into the data as needed in neural relational for interacting systems

        :param events:
        :return:
        """
        self.number_of_nodes = max(list(set(events[0, :]))) + 1
        arrivals_variate_k = self.arrivals[0, np.where(events[0, :] == 0)[0]]
        biggest = len(arrivals_variate_k)
        for k, realization in enumerate(self.arrivals):
            for i in range(self.number_of_nodes):
                arrivals_variate_k = self.arrivals[k, np.where(events[k, :] == i)[0]]
                biggest = max(biggest, len(arrivals_variate_k))

        interacting_list = []
        for k, realization in enumerate(self.arrivals):
            realization_interaction = []
            for i in range(self.number_of_nodes):
                arrivals_variate_k = self.arrivals[k, np.where(events[k, :] == i)[0]]
                missing = biggest - len(arrivals_variate_k)
                arrivals_variate_k = np.pad(arrivals_variate_k, (0, missing), mode="constant")
                realization_interaction.append(arrivals_variate_k)
            realization_interaction = np.asarray(realization_interaction)
            interacting_list.append(realization_interaction)
        interacting_list = np.asarray(interacting_list)
        return interacting_list

    def __prepare_event_types(self, data_path, suffix):
        events = np.int64(np.loadtxt(os.path.join(data_path, "event-" + suffix + ".txt")))
        return events

    def __prepare_edges(self, data_path: str, suffix: str) -> torch.Tensor:
        edges = np.load(self.data_path + 'edges-' + suffix + '.npy')
        edges = np.reshape(edges, [-1, self.number_of_nodes ** 2])
        edges = np.array((edges + 1) / 2, dtype=np.int64)
        edges = torch.LongTensor(edges)
        # Exclude self edges
        off_diag_idx = np.ravel_multi_index(
                np.where(np.ones((self.number_of_nodes, self.number_of_nodes)) - np.eye(self.number_of_nodes)),
                [self.number_of_nodes, self.number_of_nodes])
        edges = edges[:, off_diag_idx]
        return edges

    def __prepare_arrivals(self, data_path: str, suffix: str) -> torch.Tensor:
        time = np.loadtxt(os.path.join(data_path, "time-" + suffix + ".txt"))
        return time

    def prepare_batches_interacting(self, suffix):
        dt = self.arrivals[:, :, 1:] - self.arrivals[:, :, :-1]
        self.arrivals = np.stack((self.arrivals[:, :, 1:-1], dt[:, :, :-1]), axis=-1)
        self.arrivals = np.transpose(self.arrivals, [0, 2, 3, 1])

        # [num_samples, num_timesteps, num_dims, num_atoms]
        num_atoms = self.arrivals.shape[3]

        self.arrivals_max = self.arrivals.max()
        self.arrivals_min = self.arrivals.min()

        # Normalize to [-1, 1]
        self.arrivals = (self.arrivals - self.arrivals_min) * 2 / (self.arrivals_max - self.arrivals_min) - 1

        # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
        self.arrivals = np.transpose(self.arrivals, [0, 3, 1, 2])

        # edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
        # edges_train = np.array((edges_train + 1) / 2, dtype=np.int64)

        self.arrivals = torch.FloatTensor(self.arrivals)
        # edges_train = torch.LongTensor(edges_train)

        # Exclude self edges
        self.off_diag_idx = np.ravel_multi_index(
                np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
                [num_atoms, num_atoms])
        # edges_train = edges_train[:, off_diag_idx]

    def __len__(self) -> int:
        return self.arrivals.size()[0]

    def __getitem__(self, ix: int) -> Any:
        return self.arrivals[ix], self.edges[ix]


class ConditionalWasssersteingPointDataSet(Dataset):
    def __init__(self, data_path: str, train=True, past_of_sequence=.7) -> Dataset:
        self.data_path = data_path
        self.past_of_sequence = past_of_sequence
        suffix = "train" if train else "test"

        events = self.__prepare_event_types(data_path, suffix)
        self.arrivals = self.__prepare_arrivals(data_path, suffix)
        self.arrivals = self._prepare_events_and_arrivals(events)
        self.prepare_batches_interacting(suffix)
        self.edges = self.__prepare_edges(data_path, suffix)

    def __prepare_event_types(self, data_path, suffix):
        events = np.int64(np.loadtxt(os.path.join(data_path, "event-" + suffix + ".txt")))
        return events

    def __prepare_arrivals(self, data_path: str, suffix: str) -> torch.Tensor:
        time = np.loadtxt(os.path.join(data_path, "time-" + suffix + ".txt"))
        return time

    def _prepare_events_and_arrivals(self, events):
        """
        transform the input data as needed for the recurrent market point processes
        into the data as needed in neural relational for interacting systems

        :param events:
        :return: [num_samples, num_timesteps, num_dims, num_atoms]
        """
        self.number_of_nodes = max(list(set(events[0, :]))) + 1
        arrivals_variate_k = self.arrivals[0, np.where(events[0, :] == 0)[0]]
        biggest = len(arrivals_variate_k)
        for k, realization in enumerate(self.arrivals):
            for i in range(self.number_of_nodes):
                arrivals_variate_k = self.arrivals[k, np.where(events[k, :] == i)[0]]
                biggest = max(biggest, len(arrivals_variate_k))

        interacting_list = []
        for k, realization in enumerate(self.arrivals):
            realization_interaction = []
            for i in range(self.number_of_nodes):
                arrivals_variate_k = self.arrivals[k, np.where(events[k, :] == i)[0]]
                missing = biggest - len(arrivals_variate_k)
                arrivals_variate_k = np.pad(arrivals_variate_k, (0, missing), mode="constant")
                realization_interaction.append(arrivals_variate_k)
            realization_interaction = np.asarray(realization_interaction)
            interacting_list.append(realization_interaction)
        interacting_list = np.asarray(interacting_list)
        return interacting_list

    def prepare_batches_interacting(self, suffix):
        """
        same transformation as in interactnig systems and then collapses all atoms
        such that there is only one process

        :param suffix:
        :return: [num_sims, num_atoms, num_timesteps, num_dims]
        """
        dt = self.arrivals[:, :, 1:] - self.arrivals[:, :, :-1]
        self.arrivals = np.stack((self.arrivals[:, :, 1:-1], dt[:, :, :-1]), axis=-1)
        self.arrivals = np.transpose(self.arrivals, [0, 2, 3, 1])

        # [num_samples*num_atoms, num_timesteps, num_dims]
        num_atoms = self.arrivals.shape[3]

        self.arrivals_max = self.arrivals.max()
        self.arrivals_min = self.arrivals.min()

        # Normalize to [-1, 1]
        self.arrivals = (self.arrivals - self.arrivals_min) * 2 / (self.arrivals_max - self.arrivals_min) - 1

        # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
        self.arrivals = np.transpose(self.arrivals, [0, 3, 1, 2])

        # edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
        # edges_train = np.array((edges_train + 1) / 2, dtype=np.int64)

        self.arrivals = torch.FloatTensor(self.arrivals)
        number_of_simulations, number_of_atoms, num_timesteps, num_dim = self.arrivals.shape
        self.arrivals = self.arrivals.view(-1, num_timesteps, num_dim)
        last_past_index = int(self.past_of_sequence * num_timesteps)
        self.past_arrivals = self.arrivals[:, :last_past_index, :]
        self.future_arrivals = self.arrivals[:, last_past_index:, :]

        self.past_size = last_past_index
        self.future_size = self.arrivals.shape[1] - self.past_size
        # edges_train = torch.LongTensor(edges_train)

    def __prepare_edges(self, data_path: str, suffix: str) -> torch.Tensor:
        edges = np.load(self.data_path + 'edges-' + suffix + '.npy')
        edges = np.reshape(edges, [-1, self.number_of_nodes ** 2])
        edges = np.array((edges + 1) / 2, dtype=np.int64)
        edges = torch.LongTensor(edges)
        # Exclude self edges
        off_diag_idx = np.ravel_multi_index(
                np.where(np.ones((self.number_of_nodes, self.number_of_nodes)) - np.eye(self.number_of_nodes)),
                [self.number_of_nodes, self.number_of_nodes])
        edges = edges[:, off_diag_idx]
        return edges

    def __len__(self) -> int:
        return self.arrivals.size()[0]

    def __getitem__(self, ix: int) -> Any:
        return self.past_arrivals[ix], self.future_arrivals[ix]
