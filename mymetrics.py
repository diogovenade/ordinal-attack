from torcheval.metrics.metric import Metric
import torch

class OneOff(Metric):
    def __init__(self):
        self.num = self.den = 0
    def update(self, preds, labels):
        self.num += (torch.abs(preds - labels) <= 1).sum().item()
        self.den += labels.size(0)
    def compute(self):
        return self.num / self.den
    def merge_state(self, metrics):
        pass

class MeanAbsoluteError(Metric):
    def __init__(self):
        self.sum_absolute_error = 0
        self.total = 0
    def update(self, preds, labels):
        self.sum_absolute_error += torch.abs(preds - labels).sum().item()
        self.total += labels.size(0)
    def compute(self):
        return self.sum_absolute_error / self.total
    def merge_state(self, metrics):
        pass
