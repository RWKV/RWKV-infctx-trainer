import torch
import torch.nn as nn
from torch import Tensor

from .interface import IMetric, MetricArgs

class AvgMetric(nn.Module, IMetric):
    def __init__(self):
        super().__init__()
        self.running_total = None
        self.count = 0

    def compute(self):
        if self.running_total is None:
            return 0.0
        return self.running_total / self.count

    def clear(self):
        self.running_total = None
        self.count = 0

class Accuracy(AvgMetric):
    def __init__(self):
        super().__init__()

    def update(self, margs:MetricArgs):
        with torch.no_grad():
            value = margs.predictions.eq(margs.labels).sum().float() / (margs.labels.size(0)*margs.labels.size(1))
            if self.running_total is None:
                self.running_total = value
            else:
                self.running_total += value
            self.count += 1

class Loss(AvgMetric):
    def __init__(self):
        super().__init__()

    def update(self, margs:MetricArgs):
        with torch.no_grad():
            value = margs.loss.float().detach().clone()
            if self.running_total is None:
                self.running_total = value
            else:
                self.running_total += value
            self.count += 1
