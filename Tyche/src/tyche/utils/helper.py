# coding: utf-8

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import copy
import itertools
import numpy as np
import torch as to
import torch as torch
import yaml
from functools import reduce
from importlib import import_module
from scipy import linalg as la


def create_class_instance(module_name, class_name, kwargs, *args):
    """Create an instance of a given class.

    :param module_name: where the class is located
    :param class_name:
    :param kwargs: arguments needed for the class constructor
    :returns: instance of 'class_name'

    """
    module = import_module(module_name)
    clazz = getattr(module, class_name)
    if kwargs is None:
        instance = clazz(*args)
    else:
        instance = clazz(*args, **kwargs)

    return instance


def create_instance(name, params, *args):
    """Creates an instance of class given configuration.

    :param name: of the module we want to create
    :param params: dictionary containing information how to instanciate the class
    :returns: instance of a class
    :rtype:

    """
    i_params = params[name]
    if type(i_params) is list:
        instance = [create_class_instance(
                p['module'], p['name'], p['args'], *args) for p in i_params]
    else:
        instance = create_class_instance(
                i_params['module'], i_params['name'], i_params['args'], *args)
    return instance


def create_nonlinearity(name):
    """
    Returns instance of nonlinearity class (from torch.nn)
    """
    module = import_module("torch.nn")
    clazz = getattr(module, name)
    instance = clazz()

    return instance


def get_class_nonlinearity(name):
    """
    Returns nonlinearity class (from torch.nn)
    """
    module = import_module("torch.nn")
    clazz = getattr(module, name)

    return clazz


def create_cost_function(name, *args):
    """
    Returns instance of cost fuctions (from torch.nn)
    """
    module = import_module("torch.nn")
    clazz = getattr(module, name)
    instance = clazz(*args)

    return instance


def load_params(path):
    """Loads experiment parameters from json file.

    :param path: to the json file
    :returns: param needed for the experiment
    :rtype: dictionary

    """
    try:
        with open(path, "rb") as f:
            params = yaml.full_load(f)
        return params
    except Exception as e:
        print(e)
        with open(path, "r") as f:
            params = yaml.full_load(f, encoding='utf-8')
        return params


def to_one_hot(labels, num_classes):
    """
    Convert tensor of labels to one hot encoding of the labels.
    :param labels: to be encoded
    :param num_classes:
    :return:
    """
    shape = labels.size()
    shape = shape + (num_classes,)
    one_hot = torch.FloatTensor(shape)
    one_hot.zero_()
    dim = 1 if len(shape) == 2 else 2
    one_hot.scatter_(dim, labels.unsqueeze(-1), 1)
    return one_hot


def convert_tuples_2_list(arg):
    for key, value in arg.items():
        if isinstance(value, dict):
            convert_tuples_2_list(value)
        else:
            if isinstance(value, tuple):
                arg[key] = list(value)

    return arg


def unpack_cv_parameters(params, prefix=None):
    cv_params = []
    for key, value in params.items():
        if isinstance(value, dict):
            if prefix is None:
                prefix = key
            else:
                prefix = ".".join([prefix, key])
            param_pool = unpack_cv_parameters(value, prefix)
            if '.' in prefix:
                prefix = prefix.rsplit('.', 1)[0]
            else:
                prefix = None

            if len(param_pool) > 0:
                cv_params.extend(param_pool)
        elif isinstance(value, tuple) and len(value) != 0 and isinstance(value[0], dict):
            for ix, v in enumerate(value):
                if isinstance(v, dict):
                    if prefix is None:
                        prefix = key
                    else:
                        prefix = ".".join([prefix, key + f"#{ix}"])
                    param_pool = unpack_cv_parameters(v, prefix)
                    if '.' in prefix:
                        prefix = prefix.rsplit('.', 1)[0]
                    else:
                        prefix = None
                    if len(param_pool) > 0:
                        cv_params.extend(param_pool)
        elif isinstance(value, list):
            if prefix is None:
                prefix = key
            else:
                key = ".".join([prefix, key])
            cv_params.append([(key, v) for v in value])
    return cv_params


def dict_set_nested(d, keys, value):
    node = d
    key_count = len(keys)
    key_idx = 0

    for key in keys:
        key_idx += 1

        if key_idx == key_count:
            node[key] = value
            return d
        else:
            if "#" in key:
                key, _id = key.split("#")
                if not key in node:
                    node[key] = dict()
                    node = node[key][int(_id)]
                else:
                    node = node[key][int(_id)]
            else:
                if not key in node:
                    node[key] = dict()
                    node = node[key]
                else:
                    node = node[key]


def expand_params(params):
    """
    Expand the hyperparamers for grid search

    :param params:
    :return:
    """
    cv_params = []
    param_pool = unpack_cv_parameters(params)

    for i in list(itertools.product(*param_pool)):
        d = copy.deepcopy(params)
        name = d['name']
        for j in i:
            dict_set_nested(d, j[0].split("."), j[1])
            name += "_" + j[0] + "_" + str(j[1])
            d['name'] = name.replace('.args.', "_")
        d = convert_tuples_2_list(d)
        cv_params.append(d)
    if not cv_params:
        return [params] * params['num_runs']

    gs_params = []
    for p in cv_params:
        gs_params += [p] * p['num_runs']
    return gs_params


def get_cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def get_device(params, logger=None):
    gpus = params.get("gpus", [])
    if len(gpus) > 0:
        if not torch.cuda.is_available():
            if logger is not None:
                logger.warning("No GPU's available. Using CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:" + str(gpus[0]))
    else:
        device = torch.device("cpu")
    return device


def gauss_legender_points(N=30):
    """ Returns the quadratures_nodes anb weights of the Gaussian-Lenegendre quadrature """
    beta = np.array([(n + 1.0) / np.sqrt((2.0 * n + 1.0) * (2.0 * n + 3.0))
                     for n in range(N - 1)], dtype=np.float32)
    M = np.diag(beta, -1) + np.diag(beta, 1)
    nodes, V = la.eigh(M, overwrite_a=True, overwrite_b=True)
    weight = 2 * V[0, :] ** 2
    return nodes, weight


def quadratures(f, a=-1, b=1, n=30):
    """
    Performing Legendre-Gauss quadrature integral approximation.

    :param f:
    :param a:
    :param b:
    :param n:
    :return:
    """
    nodes, weights = gauss_legender_points(n)
    w = to.tensor(weights.reshape(1, 1, -1))
    nodes = to.tensor(nodes.reshape(1, 1, -1))

    scale = (b - a) / 2.

    x = scale * nodes + (b + a) / 2.
    y = w * f(x)
    y = to.sum(scale * y, dim=-1)
    return y.type(dtype=to.float)


def gumbel_sample(shape, device, epsilon=1e-20):
    """
    Sample Gumbel(0,1)
    """
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u + epsilon) + epsilon)


def gumbel_softmax_sample(pi, tau, device, epsilon=1e-12):
    """
    Sample Gumbel-softmax
    """
    y = torch.log(pi + epsilon) + gumbel_sample(pi.size(), device)
    return torch.nn.functional.softmax(y / tau, dim=-1)


def gumbel_softmax(pi, tau, device):
    """
    Gumbel-Softmax distribution.
    Implementation from https://github.com/ericjang/gumbel-softmax.
    pi: [B, ..., n_classes] class probs of categorical z
    tau: temperature
    Returns [B, ..., n_classes] as a one-hot vector
    """
    y = gumbel_softmax_sample(pi, tau, device)
    shape = y.size()
    _, ind = y.max(dim=-1)  # [B, ...]
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y


def is_primitive(v):
    """
    Checks if v is of primitive type.
    """
    return isinstance(v, (int, float, bool, str))


def free_params(module):
    for p in module.parameters():
        p.requires_grad = True


def frozen_params(module):
    for p in module.parameters():
        p.requires_grad = False


def sum_dictionares(dicts: dict):
    """
    Sums the values of the common keys in dictionary.

    Parameters
    ----------
    dicts (list) dictionaries containing numeric values

    Returns
    -------
    dictionary with summed values
    """

    def reducer(accumulator, element):
        for key, value in element.items():
            accumulator[key] = accumulator.get(key, 0) + value
        return accumulator

    return reduce(reducer, dicts, {})
