import torch
import torch.nn

from tyche.utils import helper
from tyche.utils.param_scheduler import ConstantScheduler


def test_stats():
    loss = torch.nn.CrossEntropyLoss()
    metrics = [torch.nn.CrossEntropyLoss(), torch.nn.SmoothL1Loss(), torch.nn.MSELoss()]
    stats = helper.Stats(loss, metrics)
    assert len(stats.metrics_name) == 3


def test_create_instance():
    l = ConstantScheduler(beta=1)
    print(l(39))
