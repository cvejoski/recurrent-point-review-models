from abc import ABC, abstractmethod
from typing import Any, List, Dict

import torch.nn as nn

from tyche.utils.helper import create_instance


class AModel(nn.Module, ABC):
    def __init__(self, **kwargs):
        super().__init__()
        self.reduce = kwargs.get('reduce')
        if 'metrics' in kwargs:
            metrics = create_instance('metrics', kwargs)
            if type(metrics) is not list:
                metrics = [metrics]
            self.metrics = metrics
        else:
            self.metrics = None

    @abstractmethod
    def new_stats(self) -> Dict:
        """
        Create dictionary where it will hold the results (_loss_ and _metrics_) after each training step.
        :return:
        """
        raise NotImplementedError("The new_stats method is not implemented in your class!")

    @abstractmethod
    def loss(self, y: Any, y_target: Any) -> Dict:
        raise NotImplementedError("The loss method is not implemented in your class!")

    @abstractmethod
    def metric(self, y: Any, y_target: Any) -> Dict:
        raise NotImplementedError("The metric method is not implemented in your class!")

    @abstractmethod
    def train_step(self, minibatch: Any, optimizer: Any, step: int, scheduler: Any = None) -> Dict:
        raise NotImplementedError("The train_step method is not implemented in your class!")

    @abstractmethod
    def validate_step(self, minibatch: Any) -> Dict:
        raise NotImplementedError("The validate_step method is not implemented in your class!")

    def validate_epoch(self,dataloader: Any = None, epoch: int = 1) -> Dict:
        """
        This is performed only once every
            save_after_epoch: 10
            and usually accounts for global validation metrics, like the evaluation of the full graph etc
        :return:
        """
        return {}

    @property
    def device(self):
        return next(self.parameters()).device
