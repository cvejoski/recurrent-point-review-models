# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import argparse
import itertools
import logging
import os
from functools import partial
from multiprocessing import Pool

import torch
from pathlib import Path
from gentext.models.languagemodels import WAE, DiscreteWAE, SupervisedWAE, SemiSupervisedWAE
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
    logger.info("Name of the Experiment: " + params['name'])
    print(("Name of the Experiment: " + params['name']))
    device = get_device(params)

    # Data loader
    data_loader = create_instance('data_loader', params, device)
    # Model
    model = create_instance('model', params, data_loader.vocab, data_loader.fix_len)
    # Optimizers
    optimizers = dict()
    if isinstance(model, WAE) or isinstance(model, DiscreteWAE) or isinstance(model, SupervisedWAE) or isinstance(model, SemiSupervisedWAE):
        model_params = itertools.chain(*[model.encoder.parameters(), model.decoder.parameters()])
        optimizer = create_instance('optimizer', params, model_params)
        critic_optimizer = create_instance('critic_optimizer', params, model.wasserstein_distance.parameters())
        optimizers['loss_optimizer'] = optimizer
        optimizers['critic_optimizer'] = critic_optimizer

        if isinstance(model, SemiSupervisedWAE):
            cat_critic_optimizer = create_instance('cat_critic_optimizer', params, model.cat_wasserstein_distance.parameters())
            optimizers['cat_critic_optimizer'] = cat_critic_optimizer
            classification_optimizer = create_instance('class_optimizer', params, model.encoder.parameters())
            optimizers['class_optimizer'] = classification_optimizer

    else:
        optimizer = create_instance('optimizer', params, model.parameters())
        optimizers['loss_optimizer'] = optimizer
    # Trainer
    trainer = create_instance('trainer', params, model, optimizers, resume, params, data_loader)
    best_model = trainer.train()
    with open(os.path.join(params['trainer']['logging']['logging_dir'], 'best_models.txt'), 'a+') as f:
        f.write(str(best_model) + "\n")
    trainer.__del__()


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
    parser.add_argument(
            "-d",
            "--debug",
            type=bool,
            default=False,
            const=True,
            nargs='?',
            help="Run in debug mode")
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    train(args)


if __name__ == '__main__':
    import sys

    main(sys.argv[1:])
