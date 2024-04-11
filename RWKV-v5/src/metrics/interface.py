import torch
import torch.nn as nn
from torch import Tensor

from abc import abstractmethod

class MetricArgs():
    def __init__(self, inputs, logits:Tensor, predictions:Tensor, labels:Tensor, loss:Tensor):
        with torch.no_grad():
            self.inputs = inputs.detach() if isinstance(inputs, Tensor) else inputs
            self.logits = logits.detach() if isinstance(logits, Tensor) else logits
            self.predictions = predictions.detach() if isinstance(predictions, Tensor) else predictions
            self.labels = labels.detach() if isinstance(labels, Tensor) else labels
            self.loss = loss.detach() if isinstance(loss, Tensor) else loss

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

