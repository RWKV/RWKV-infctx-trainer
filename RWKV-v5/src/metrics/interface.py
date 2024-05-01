import torch
import torch.nn as nn
from torch import Tensor

from abc import abstractmethod

class MetricArgs():
    def __init__(self, inputs, logits:Tensor, predictions:Tensor, labels:Tensor, loss:Tensor):
        with torch.no_grad():
            self.inputs = inputs
            self.logits = logits
            self.predictions = predictions
            self.labels = labels
            self.loss = loss

class IMetric():
    @abstractmethod
    def update(self, margs:MetricArgs):
        raise NotImplementedError()

    @abstractmethod
    def compute(self):
        raise NotImplementedError()

    @abstractmethod
    def clear(self):
        raise NotImplementedError()

