from typing import Optional, Union, List

import torch
from allennlp.training.metrics import Metric, F1Measure

use_predicted_idxes_metrics = ['BooleanAccuracy', 'MulticlassClassificationMetric', 'MultitaskMulticlassClassificationMetric']
use_probabilities_metrics = {
    'Auc': {'squeeze': True, 'expand': False},
    'CategoricalAccuracy': {'squeeze': False, 'expand': False},
    'F1Measure': {'squeeze': False, 'expand': True},
}


def update_metric(metric: Metric,
                  pred_idxes: Union[torch.Tensor, List[torch.Tensor]],
                  probs: Union[torch.Tensor, List[torch.Tensor]],
                  labels: torch.Tensor,
                  mask: Optional[torch.BoolTensor] = None,
                  flatten_labels: bool = True,
                  ignore_shape: bool = False):
    if metric is None:
        return
    metric_class_name = metric.__class__.__name__
    if flatten_labels:
        labels = labels.flatten()
    if metric_class_name in use_predicted_idxes_metrics:
        metric(pred_idxes, labels, mask=mask)
    elif metric_class_name in use_probabilities_metrics:
        if use_probabilities_metrics[metric_class_name]['squeeze'] and (len(probs.shape) > 1 or ignore_shape):
            probs = probs[:, 1]
        if use_probabilities_metrics[metric_class_name]['expand'] and (len(probs.shape) == 1 or ignore_shape):
            probs = expand_1d_prob_tensor_to_2d(probs)
        metric(probs, labels, mask=mask)
    else:
        raise ValueError(f'Unregistered metric: {metric_class_name}')

def expand_1d_prob_tensor_to_2d(prob_tensor):
    """
    Expand the single probability (1D, shape:[batch,])
    to full probability distribution (2D, shape: [batch, 2]), with prob summed to 1.

    Example:
    [0.6, 0.8, 0.1] -> [[0.6, 0.4], [0.8, 0.2], [0.1, 0.9]]
    """
    residual_prob_tensor = torch.ones_like(prob_tensor)
    residual_prob_tensor = residual_prob_tensor - prob_tensor
    return torch.stack((residual_prob_tensor, prob_tensor), dim=-1)