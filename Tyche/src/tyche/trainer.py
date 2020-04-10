import datetime
import json
import logging
import matplotlib
import os
import torch
import tqdm
import yaml
from abc import ABCMeta
from tensorboardX import SummaryWriter
from typing import Any
from typing import Dict

from .utils.helper import get_device, is_primitive, create_instance


class BaseTrainingProcedure(metaclass=ABCMeta):

    def __init__(self, model, optimizer, resume, params, data_loader, train_logger=None, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = model
        self.optimizer = optimizer
        self.params = params

        self._prepare_dirs()
        self._save_params()
        self.t_logger = self._setup_logging()
        self.summary = SummaryWriter(self.tensorboard_dir)
        self.device = get_device(params)
        self.model.to(self.device)
        self.start_epoch = 0
        self.n_epochs = self.params['trainer']['epochs']
        self.save_after_epoch = self.params['trainer']['args']['save_after_epoch']
        self.batch_size = self.params['data_loader']['args']['batch_size']
        self.bm_metric = self.params['trainer']['args']['bm_metric']

        if "schedulers" in self.params['trainer']['args']:
            self.schedulers = dict()
            schedulers_ = create_instance('schedulers', self.params['trainer']['args'])
            if type(schedulers_) is not list:
                schedulers_ = [schedulers_]
            for a, b in zip(self.params['trainer']['args']['schedulers'], schedulers_):
                self.schedulers[a["label"]] = b
        else:
            self.schedulers = None

        self.data_loader = data_loader
        self.n_train_batches = len(data_loader.train)
        self.n_val_batches = len(data_loader.validate)

        self.global_step = 0
        self.best_model = {'train_loss': float('inf'),
                           'val_loss': float('inf'),
                           'train_metric': float('inf'),
                           'val_metric': float('inf')}

        self.train_logger = train_logger
        if resume:
            self._resume_check_point(resume)

    def train(self):
        e_bar = tqdm.tqdm(
                desc='Epoch: ',
                total=self.n_epochs,
                unit='epoch',
                initial=self.start_epoch,
                postfix='train loss: -, validation loss: -')
        for epoch in range(self.start_epoch, self.n_epochs):
            train_log = self._train_epoch(epoch)
            validate_log = self._validate_epoch(epoch)

            self._update_p_bar(e_bar, train_log, validate_log)
            self._check_and_save_best_model(train_log, validate_log)
            if epoch % self.save_after_epoch == 0 and epoch != 0:
                self._save_check_point(epoch)
        e_bar.close()
        self.best_model['name'] = self.params['name']
        return self.best_model

    def _train_epoch(self, epoch: int) -> Dict:
        self.model.train()
        p_bar = tqdm.tqdm(
                desc='Training batch: ', total=self.n_train_batches, unit='batch')

        epoch_stats = None
        for batch_idx, data in enumerate(self.data_loader.train):
            batch_stat = self._train_step(data, batch_idx, epoch, p_bar)
            epoch_stats = self._update_stats(epoch_stats, batch_stat)
        p_bar.close()

        self._normalize_stats(self.n_train_batches, epoch_stats)
        self._log_epoch('train/epoch/', epoch_stats)

        return epoch_stats

    def _train_step(self, minibatch: Any, batch_idx: int, epoch: int, p_bar):
        stats = self.model.train_step(minibatch, self.optimizer, self.global_step, scheduler=self.schedulers)
        self.tensor_2_item(stats)
        self._log_train_step(epoch, batch_idx, stats)
        p_bar.set_postfix_str("loss: {:4.8g}".format(stats['loss']))
        p_bar.update()
        self.global_step += 1
        return stats

    def _validate_epoch(self, epoch: int) -> Dict:
        self.model.eval()
        with torch.no_grad():
            p_bar = tqdm.tqdm(
                    desc="Validation batch: ",
                    total=self.n_val_batches,
                    unit="batch")

            epoch_stats = None
            for batch_idx, data in enumerate(self.data_loader.validate):
                batch_stat = self._validate_step(data, batch_idx, epoch, p_bar)
                epoch_stats = self._update_stats(epoch_stats, batch_stat)
            p_bar.close()

            if epoch % self.save_after_epoch == 0:
                final_stats = self.model.validate_epoch(self.data_loader, epoch)
                epoch_stats.update(final_stats)

            self._normalize_stats(self.n_val_batches, epoch_stats)
            self._log_epoch('validate/epoch/', epoch_stats)

        return epoch_stats

    def _test_epoch(self, epoch: int) -> Dict:
        self.model.eval()
        with torch.no_grad():
            p_bar = tqdm.tqdm(
                    desc="Test batch: ",
                    total=len(self.data_loader.test),
                    unit="batch")

            epoch_stats = None
            for batch_idx, data in enumerate(self.data_loader.test):
                batch_stat = self._validate_step(data, batch_idx, epoch, p_bar)
                epoch_stats = self._update_stats(epoch_stats, batch_stat)
            p_bar.close()

            self._normalize_stats(self.n_test_batches, epoch_stats)
            self._log_epoch('validate/epoch/', epoch_stats)

        return epoch_stats

    def _validate_step(self, minibatch: Any, batch_idx: int, epoch: int, p_bar):
        if type(self.model).__name__ in ("WAE", "SupervisedWAE"):
            stats = self.model.validate_step(minibatch, self.global_step, scheduler=self.schedulers)
        else:
            stats = self.model.validate_step(minibatch)

        self.tensor_2_item(stats)
        self._log_validation_step(epoch, batch_idx, stats)
        p_bar.set_postfix_str("loss: {:4.8g}".format(stats['loss']))
        p_bar.update()
        return stats

    @staticmethod
    def _update_stats(epoch_stat, batch_stat):
        if epoch_stat is None:
            return batch_stat.copy()
        for k, v in batch_stat.items():
            epoch_stat[k] += v

        return epoch_stat

    @staticmethod
    def _normalize_stats(n_batches, statistics):
        for k, v in statistics.items():
            if is_primitive(v):
                statistics[k] /= n_batches
        return statistics

    def _log_epoch(self, log_label, statistics):
        for k, v in statistics.items():
            if is_primitive(v):
                self.summary.add_scalar(log_label + k, v, self.global_step)
            elif isinstance(v, list) and isinstance(v[0], int):
                self.summary.add_histogram(log_label + k, v, self.global_step)
            elif isinstance(v, matplotlib.figure.Figure):
                self.summary.add_figure(log_label + k, figure=v, global_step=self.global_step)

    def __del__(self):
        self.summary.close()

    def _prepare_dirs(self) -> None:
        trainer_par = self.params['trainer']
        start_time = datetime.datetime.now().strftime('%d%m_%H%M%S')
        name = self.params['name']
        if len(name) > 200:
            name = "_".join([i if i.isdigit() else i[0:3] for i in name.split('_')])
        self.checkpoint_dir = os.path.join(trainer_par['save_dir'],
                                           name, start_time)
        self.logging_dir = os.path.join(trainer_par['logging']['logging_dir'],
                                        name, start_time)
        self.tensorboard_dir = os.path.join(trainer_par['logging']['tensorboard_dir'],
                                            name, start_time)

        os.makedirs(self.logging_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)

    def _save_params(self):
        params_path = os.path.join(self.logging_dir, 'config.yaml')
        self.logger.info(f'saving config into {params_path}')
        yaml.dump(self.params, open(params_path, 'w'), default_flow_style=False)

    def _save_model(self, file_name, **kwargs) -> None:
        model_type = type(self.model).__name__
        state = {
            'model_type': model_type,
            'epoch': kwargs.get('epoch'),
            'model_state': self.model.state_dict(),
            'params': self.params
        }
        for key in self.optimizer:
            state[key] = self.optimizer[key].state_dict()

        torch.save(state, file_name)

    def _save_model_parameters(self, file_name):
        '''
        Args:
            file_name:
        '''
        with open(file_name, 'w') as f:
            json.dump(self.params, f, indent=4)

    def _save_check_point(self, epoch: int) -> None:
        """

        :param epoch:
        :returns:
        :rtype:^^

        """

        file_name = os.path.join(self.checkpoint_dir,
                                 'checkpoint-epoch{}.pth'.format(epoch))
        self.t_logger.info('Saving checkpoint: {} ...'.format(file_name))
        self._save_model(file_name, epoch=epoch)

    def _save_best_model(self) -> None:
        file_name = os.path.join(self.checkpoint_dir,
                                 'best_model.pth')
        self.t_logger.info('Saving best model ...')
        self._save_model(file_name)

    def _resume_check_point(self, path: str) -> None:
        """

        :param path:
        :returns:
        :rtype:

        """
        self.logger.info('Loading checkpoint: {} ...'.format(path))
        if torch.cuda.is_available() is False:
            state = torch.load(path, map_location='cpu')
        else:
            state = torch.load(path)
        self.params = state['params']
        if state['epoch'] is None:
            self.start_epoch = 1
        else:
            self.start_epoch = state['epoch'] + 1
        self.model.load_state_dict(state['model_state'])
        for key in self.optimizer:
            self.optimizer[key].load_state_dict(state[key])
        self.logger.info('Finished loading checkpoint: {} ...'.format(path))

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('train_logger')
        logger.propagate = False
        logger.setLevel(logging.INFO)

        file_name = os.path.join(self.logging_dir, 'train.log')
        fh = logging.FileHandler(file_name)
        formatter = logging.Formatter(
                self.params['trainer']['logging']['formatters']['simple'])
        fh.setLevel(logging.INFO)

        fh.setFormatter(formatter)
        logger.addHandler(fh)

        return logger

    def _log_train_step(self, epoch: int, batch_idx: int, stats: Dict) -> None:
        data_len = len(self.data_loader.train.dataset)
        log = self._build_raw_log_str('Train epoch', batch_idx, epoch, stats, data_len, self.batch_size)
        self.t_logger.info(log)
        for k, v in stats.items():
            if is_primitive(v):
                self.summary.add_scalar('train/batch/' + k, v, self.global_step)

    def _log_validation_step(self, epoch: int, batch_idx: int, logs: Dict) -> None:
        data_len = len(self.data_loader.validate.dataset)
        log = self._build_raw_log_str('Validation epoch', batch_idx, epoch, logs, data_len, self.batch_size)
        self.t_logger.info(log)
        for k, v in logs.items():
            if is_primitive(v):
                self.summary.add_scalar('validate/batch/' + k, v, self.global_step)

    @staticmethod
    def _build_raw_log_str(prefix: str, batch_idx: int, epoch: int, logs: Dict, data_len: int, batch_size: int):
        sb = prefix + ': {} [{}/{} ({:.0%})]'.format(
                epoch,
                batch_idx * batch_size,
                data_len,
                100.0 * batch_idx / data_len)
        for k, v in logs.items():
            if is_primitive(v):
                sb += ' {}: {:.6f}'.format(k, v)
        return sb

    def _check_and_save_best_model(self, train_log: Dict, validate_log: Dict) -> None:
        if validate_log[self.bm_metric] < self.best_model['val_metric']:
            self._save_best_model()
            self._update_best_model_flag(train_log, validate_log)

    def _update_p_bar(self, e_bar, train_log: Dict, validate_log: Dict) -> None:
        e_bar.update()
        e_bar.set_postfix_str(
                f"train loss: {train_log['loss']:6.6g} train {self.bm_metric}: {train_log[self.bm_metric]:6.6g}, "
                f"validation loss: {validate_log['loss']:6.6g}, validation {self.bm_metric}: "
                f"{validate_log[self.bm_metric]:6.4g}")

    def _update_best_model_flag(self, train_log: Dict, validate_log: Dict) -> None:
        self.best_model['train_loss'] = train_log['loss']
        self.best_model['val_loss'] = validate_log['loss']
        self.best_model['train_metric'] = train_log[self.bm_metric]
        self.best_model['val_metric'] = validate_log[self.bm_metric]

    @staticmethod
    def tensor_2_item(stats):
        for key, value in stats.items():
            if type(value) is torch.Tensor:
                stats[key] = value.item()
