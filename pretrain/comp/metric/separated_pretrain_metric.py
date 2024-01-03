from typing import Optional

import torch
from allennlp.training.metrics import Metric

class F1Measure:
    def __init__(self, name):
        self.name = name
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def update_metric(self,
                      preds: torch.Tensor,
                      labels: torch.Tensor):
        positive_idxes = (preds == 1).int().nonzero().squeeze(-1)
        negative_idxes = (preds == 0).int().nonzero().squeeze(-1)

        true_pos = labels[positive_idxes].sum().item()  # pred=1, label=1
        false_pos = len(positive_idxes) - true_pos      # pred=1, label!=1
        false_neg = labels[negative_idxes].sum().item() # pred=0, label=1
        true_neg = len(negative_idxes) - false_neg      # pred=0, label!=1

        self.tp += true_pos
        self.fp += false_pos
        self.tn += true_neg
        self.fn += false_neg

    def get_metric(self):
        try:
            precision = self.tp / (self.tp + self.fp)
            recall = self.tp / (self.tp + self.fn)
            f1 = 2*precision*recall / (precision + recall)
        except ZeroDivisionError:
            f1 = 0

        return {
            f'{self.name}_f1': f1
        }

    def reset(self):
        self.tp = self.tn = self.fp = self.fn = 0


@Metric.register('separated_single_mask_f1')
class SeparatedSingleMaskF1(Metric):
    def __init__(self, name):
        self.f1 = F1Measure(name)

    def _check_shape(self,
                     predictions: torch.Tensor,
                     gold_labels: torch.Tensor,
                     mask: Optional[torch.BoolTensor]):
        assert predictions.shape == gold_labels.shape == mask.shape

    def _mask_select(self, pred, label, mask):
        masked_pred = torch.masked_select(pred, mask).long()
        masked_label = torch.masked_select(label, mask).long()
        return masked_pred, masked_label

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.BoolTensor]):
        # First detach tensors to avoid gradient flow.
        predictions, gold_labels, mask = self.detach_tensors(predictions, gold_labels, mask)
        self._check_shape(predictions, gold_labels, mask)

        masked_pred, masked_label = self._mask_select(predictions, gold_labels, mask)

        self.f1.update_metric(masked_pred, masked_label)

    def get_metric(self, reset: bool):
        metric = self.f1.get_metric()
        if reset:
            self.reset()
        return metric

    def reset(self) -> None:
        self.f1.reset()

    def get_detailed_metrics(self):
        return {
            'tp': self.f1.tp,
            'fp': self.f1.fp,
            'tn': self.f1.tn,
            'fn': self.f1.fn,
        }