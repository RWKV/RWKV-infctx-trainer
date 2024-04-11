import torch
import torch.nn as nn
from torch import Tensor

from .interface import IMetric, MetricArgs

class AvgMetric(nn.Module, IMetric):
    def __init__(self):
        super().__init__()
        self.values = []

    def compute(self):
        with torch.no_grad():
            return sum(self.values) / max(1, len(self.values))

    def clear(self):
        self.values = []

class Accuracy(AvgMetric):
    def __init__(self):
        super().__init__()

    def update(self, margs:MetricArgs):
        with torch.no_grad():
            self.values.append(margs.predictions.eq(margs.labels).sum() / (margs.labels.size(0)*margs.labels.size(1)))

class Loss(AvgMetric):
    def __init__(self):
        super().__init__()

    def update(self, margs:MetricArgs):
        with torch.no_grad():
            self.values.append(margs.loss)
