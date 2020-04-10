# coding: utf-8

import argparse
import logging
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import itertools
import torch
from tyche.utils.helper import create_instance, load_params, expand_params, get_device

logger = logging.getLogger(__file__)


def train(args):
    params = load_params(args.parameters)
    resume = args.resume
    debug = args.debug
    gs_params = expand_params(params)
    if debug:
        for search in gs_params:
            train_params(search, resume, debug)
    else:
        p = Pool(params['num_workers'])
        p.map(partial(train_params, resume=resume), gs_params)


def train_params(params, resume, debug=False):


    if debug:
        torch.manual_seed(params["seed"])
    dtype_ = params.get("dtype", "float32")
    dtype_ = getattr(torch, dtype_)
    torch.set_default_dtype(dtype_)
    logger.info("Name of the Experiment: " + params['name'])
    device = get_device(params)
    data_loader = create_instance('data_loader', params, device, dtype_)
    model = create_instance('model', params, data_loader)
    # Optimizers
    optimizers = dict()
    if params['model']['name'] in ['TextTPP', 'TextARPP']:
        optimizer = create_instance('optimizer', params, model.tpp_model.parameters())

        lm_parm = model.language_model.parameters()
        if model.attention:
            lm_parm = itertools.chain(*[model.language_model.parameters(),
                                        model.attention_layer.parameters()])

        language_optimizer = create_instance('language_optimizer', params, lm_parm)
        optimizers['loss_optimizer'] = optimizer
        optimizers['language_optimizer'] = language_optimizer
    else:
        optimizer = create_instance('optimizer', params, model.parameters())
        optimizers['loss_optimizer'] = optimizer

    trainer = create_instance('trainer', params, model, optimizers, resume, params, data_loader)
    best_model = trainer.train()
    with open(os.path.join(params['trainer']['logging']['logging_dir'], 'best_models.txt'), 'a+') as f:
        f.write(str(best_model) + "\n")



def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "parameters",
            type=Path,
            help="path to the file containing the parameters \
                        for the experiment")
    parser.add_argument(
            "-r",
            "--resume",
            type=str,
            help="path to the file to stored model to resume \
                            training.")
    parser.add_argument("-d",
                        "--debug",
                        type=bool,
                        default=False,
                        const=True,
                        nargs='?',
                        help="Run in debug moode")
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    train(args)


if __name__ == '__main__':
    import sys

    main(sys.argv[1:])
