from typing import Union, List

import numpy
import torch
from allennlp.training.metrics import Metric

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

@Metric.register('multiclass_classification')
class MulticlassClassificationMetric(Metric):
    def __init__(self,
                 average: str = 'macro',
                 zero_division: Union[str, int] = 0):
        self.average = average
        self.zero_division = zero_division

        self.predictions = []
        self.labels = []

    def __call__(self, predictions, gold_labels, mask):
        # First detach tensors to avoid gradient flow.
        predictions, gold_labels, mask = self.detach_tensors(predictions, gold_labels, mask)
        if mask is None:
            mask = torch.ones_like(gold_labels)
        mask = mask.bool()

        masked_preds = torch.masked_select(predictions, mask).cpu().tolist()
        masked_labels = torch.masked_select(gold_labels, mask).cpu().tolist()
        self.predictions.extend(masked_preds)
        self.labels.extend(masked_labels)

    def reset(self) -> None:
        self.predictions.clear()
        self.labels.clear()

    def get_metric(self, reset: bool):
        accuracy = numpy.round(accuracy_score(self.labels, self.predictions), 4)
        precision = precision_score(self.labels, self.predictions, average=self.average, zero_division=self.zero_division).round(4)
        recall = recall_score(self.labels, self.predictions, average=self.average, zero_division=self.zero_division).round(4)
        f1 = f1_score(self.labels, self.predictions, average=self.average, zero_division=self.zero_division).round(4)
        mcc = numpy.round(matthews_corrcoef(self.labels, self.predictions), 4)

        if reset:
            self.reset()
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mcc': mcc
        }

@Metric.register('multi_task_multi_class')
class MultitaskMulticlassClassificationMetric(Metric):
    def __init__(self,
                 task_num: int,
                 task_names: List[str],
                 average: str = 'macro',
                 zero_division: Union[str, int] = 0,
                 f1_only: bool = True):
        self.task_num = task_num
        self.task_metrics = [MulticlassClassificationMetric(average, zero_division) for i in range(task_num)]
        self.task_names = task_names
        self.f1_only = f1_only

    def __call__(self, predictions: List[torch.Tensor], gold_labels: List[torch.Tensor], mask):
        if mask is None:
            for i in range(self.task_num):
                self.task_metrics[i](predictions[i], gold_labels[i], mask)
        else:
            for i in range(self.task_num):
                self.task_metrics[i](predictions[i], gold_labels[i], mask[i])

    def reset(self):
        for metric in self.task_metrics:
            metric.reset()

    def get_metric(self, reset: bool):
        multi_task_metrics = {}
        f1s = []
        for name, metric in zip(self.task_names, self.task_metrics):
            task_metrics = metric.get_metric(reset)
            if self.f1_only:
                renamed_metrics = {f'{name}_f1': task_metrics['f1']}
                f1s.append(task_metrics['f1'])
            else:
                renamed_metrics = {f'{name}_{k}': v for k,v in task_metrics.items()}
            multi_task_metrics.update(renamed_metrics)
        multi_task_metrics['macro_f1_mean'] = numpy.mean(f1s)
        return multi_task_metrics
