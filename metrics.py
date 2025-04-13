from torcheval.metrics.metric import Metric
from sklearn.metrics import cohen_kappa_score
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

class QuadraticWeightedKappa(Metric):
    def __init__(self):
        self.preds = []
        self.labels = []
    def update(self, preds, labels):
        self.preds.extend(preds.cpu().numpy())
        self.labels.extend(labels.cpu().numpy())
    def compute(self):
        return cohen_kappa_score(self.labels, self.preds, weights="quadratic")
    def merge_state(self, metrics):
        for metric in metrics:
            self.preds.extend(metric.preds)
            self.labels.extend(metric.labels)