# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals
import numpy as np
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader

def preprocess_data(x, batch_size, window_size):
    """
    Preprocess input sequences with variable length for training with BPTT.

    It is added the sequence length to each window and this information is used for the dynamic rnn.

    Args:
        x (list): N sequences, each sequence i has length L_i and each item j in the sequence i has dimension D.
        batch_size (int):
        window_size (int):

    Returns:
        ndarray: with shape [num_batches, batch_size, num_windows, window_size, D]
    """
    D = len(x[0][0])
    N = len(x)

    seqences_l = np.asarray(map(len, x))

    max_s_l = np.max(seqences_l)
    reminder = max_s_l % window_size
    if reminder == 0:
        max_l = max_s_l
    else:
        max_l = max_s_l + (window_size - reminder)
    data = []
    for seq, s_len in zip(x, seqences_l):
        s_padded = np.pad(seq, ((0, max_l - s_len), (0, 0)), 'constant')
        data.append(s_padded)
    data = np.asarray(data)
    num_windows = max_l / window_size
    f_w = seqences_l / window_size
    p_w = seqences_l % window_size
    z_w = num_windows - f_w - 1

    f_w_m = map(lambda x: np.ones(x) * window_size, f_w)
    if reminder != 0:
        f_w_m = map(lambda x, y: np.append(x, y), f_w_m, p_w)
        z_w_m = map(lambda x: np.zeros(x), z_w)
        f_w_m = map(lambda x, y: np.append(x, y), f_w_m, z_w_m)
    f_w_m = np.asarray(f_w_m)
    mask = np.zeros((N, max_l))
    for i in range(N):
        mask[i, :seqences_l[i]] = 1
    data = data.reshape(-1, batch_size, num_windows, window_size, D)
    w_l = f_w_m.reshape(-1, batch_size, num_windows)
    mask = mask.reshape(-1, batch_size, num_windows, window_size)
    return data, w_l, mask


#==========================================================================
# Interacting systems
#==========================================================================

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def interacting_systems_load_data(data_dir,batch_size=1, suffix=''):
    loc_train = np.load(data_dir+'loc_train' + suffix + '.npy')
    vel_train = np.load(data_dir+'vel_train' + suffix + '.npy')
    edges_train = np.load(data_dir+'edges_train' + suffix + '.npy')

    loc_valid = np.load(data_dir+'loc_valid' + suffix + '.npy')
    vel_valid = np.load(data_dir+'vel_valid' + suffix + '.npy')
    edges_valid = np.load(data_dir+'edges_valid' + suffix + '.npy')

    loc_test = np.load(data_dir+'loc_test' + suffix + '.npy')
    vel_test = np.load(data_dir+'vel_test' + suffix + '.npy')
    edges_test = np.load(data_dir+'edges_test' + suffix + '.npy')

    # [num_samples, num_timesteps, num_dims, num_atoms]
    num_atoms = loc_train.shape[3]

    loc_max = loc_train.max()
    loc_min = loc_train.min()
    vel_max = vel_train.max()
    vel_min = vel_train.min()

    # Normalize to [-1, 1]
    loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_valid = (loc_valid - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_valid = (vel_valid - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_test = (vel_test - vel_min) * 2 / (vel_max - vel_min) - 1

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    loc_train = np.transpose(loc_train, [0, 3, 1, 2])
    vel_train = np.transpose(vel_train, [0, 3, 1, 2])
    feat_train = np.concatenate([loc_train, vel_train], axis=3)
    edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
    edges_train = np.array((edges_train + 1) / 2, dtype=np.int64)

    loc_valid = np.transpose(loc_valid, [0, 3, 1, 2])
    vel_valid = np.transpose(vel_valid, [0, 3, 1, 2])
    feat_valid = np.concatenate([loc_valid, vel_valid], axis=3)
    edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])
    edges_valid = np.array((edges_valid + 1) / 2, dtype=np.int64)

    loc_test = np.transpose(loc_test, [0, 3, 1, 2])
    vel_test = np.transpose(vel_test, [0, 3, 1, 2])
    feat_test = np.concatenate([loc_test, vel_test], axis=3)
    edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])
    edges_test = np.array((edges_test + 1) / 2, dtype=np.int64)

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges
    off_diag_idx = np.ravel_multi_index(
        np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
        [num_atoms, num_atoms])
    edges_train = edges_train[:, off_diag_idx]
    edges_valid = edges_valid[:, off_diag_idx]
    edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader, loc_max, loc_min, vel_max, vel_min