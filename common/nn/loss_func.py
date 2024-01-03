from typing import Optional

import torch
from torch.nn import functional
from functools import reduce

from allennlp.common import Registrable


# TODO: Modules and functions in this file should be validated...

class LossFunc(Registrable, torch.nn.Module):
    def __init__(self, ignore_index: int=-100):
        self.ignore_index = ignore_index
        super().__init__()

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def align_pred_label_batch_size(self, pred, label):
        '''
        Call this method to align the batch dimension of predictions and
        labels, dealing with unmatched batch size in tail batch caused by
        different padding implementations of different fields (such as
        'TensorField' type labels will not be padded).
        # [Note]: This method assumes predictions and labels are matched
        #         at corresponding dimension, which may not be true.
        '''
        pred_size, label_size = pred.size(0), label.size(0)
        if pred_size == label_size:
            return pred, label
        else:
            smaller_batch_size = min(pred_size, label_size)
            return pred[:smaller_batch_size], \
                   label[:smaller_batch_size]


@LossFunc.register('binary_cross_entropy')
class BinaryCrossEntropyLoss(LossFunc):
    def forward(self, pred: torch.Tensor, label: torch.Tensor, **kwargs) -> torch.Tensor:
        return functional.binary_cross_entropy(pred, label, **kwargs)


@LossFunc.register('cross_entropy')
class CrossEntropyLoss(LossFunc):
    def forward(self, pred: torch.Tensor, label: torch.Tensor, **kwargs) -> torch.Tensor:
        return functional.cross_entropy(pred, label, ignore_index=self.ignore_index, **kwargs)


@LossFunc.register('nll')
class NllLoss(LossFunc):
    def forward(self, pred: torch.Tensor, label: torch.Tensor, **kwargs) -> torch.Tensor:
        pred_shape = pred.shape
        label_shape = label.shape

        # Adapt multi-dimensional pred/label tensor.
        if len(pred.shape) > 2:
            dummy_pred_batch_size = reduce(lambda x,y: x*y, pred_shape[:-1], 1)
            dummy_label_batch_size = reduce(lambda x,y: x*y, label_shape, 1)
            assert dummy_pred_batch_size==dummy_pred_batch_size, f'{dummy_pred_batch_size}!={dummy_pred_batch_size}'
            pred = pred.reshape((dummy_pred_batch_size, pred_shape[-1]))
            label = label.reshape(dummy_label_batch_size,)

        loss = functional.nll_loss(pred, label, ignore_index=self.ignore_index, **kwargs)

        # Adapt dimension keeping.
        if 'reduction' in kwargs and kwargs['reduction'] == 'none':
            return loss.reshape(label_shape)
        else:
            return loss



@LossFunc.register('mean_square')
class MeanSquareLoss(LossFunc):
    def forward(self, pred: torch.Tensor, label: torch.Tensor, **kwargs) -> torch.Tensor:
        pred = pred.squeeze()
        assert pred.size() == label.size(), f'MSE assumes logit and label have the same size,' \
                                             f'but got {pred.size()} and {label.size()}'

        return (pred - label) ** 2


@LossFunc.register('bce_logits')
class BCEWithLogitsLoss(LossFunc):
    def __init__(self):
        super().__init__()
        self._loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, label: torch.Tensor, **kwargs) -> torch.Tensor:
        # align batch_size of predictions and labels in tail batch
        pred, label = self.align_pred_label_batch_size(pred, label)
        return self._loss(pred, label, **kwargs)


@LossFunc.register('bce')
class BCELoss(LossFunc):
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, label: torch.Tensor, **kwargs) -> torch.Tensor:
        # align batch_size of predictions and labels in tail batch
        # pred, label = self.align_pred_label_batch_size(pred, label)
        # pred = pred.view(pred.size(0),)
        # label = label.view(pred.size(0),)
        if len(pred.size()) > 1 and len(pred.size()) > len(label.size()):
            pred = pred.squeeze(-1)
        return torch.nn.functional.binary_cross_entropy(pred, label.float(), **kwargs)  # float type tensor is expected for 'label'


def binary_focal_loss(probs, labels, loss_tensor, gamma=2., class_weight=None):
    """
    Binary alpha-balanced focal loss.
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf.

    :param probs: Predicted probabilities ranged (0,1)
    :param labels: Real binary labels with either 0 or 1
    :param loss_tensor: Non-reduced loss tensor, such as output of F.binary_cross_entropy(probs, labels, reduction='none')
    :param gamma: Exponential factor of focal loss
    :param class_weight: Balanced class weights, must be in shape (2,).
                         Note if None is given, both classes will have weight=1 but not 0.5
    """
    # [BugFix] 11.11:
    # “torch.abs” should be called before the gamma works, to prevent nan of sqrt on nagatives
    modulating_factor = torch.abs(probs - labels) ** gamma
    non_balanced_focal_loss = modulating_factor * loss_tensor
    assert torch.all(non_balanced_focal_loss >= 0), 'Fatal error of focal loss, negative loss term found (maybe a bug)'

    if class_weight is not None:
        class_weight_alphas = torch.index_select(class_weight, 0, labels.long())
    else:
        class_weight_alphas = 1

    return torch.mean(non_balanced_focal_loss * class_weight_alphas)

@LossFunc.register('bce_focal')
class BCEFocalLoss(LossFunc):
    def __init__(self,
                 gamma: float = 2,
                 positive_loss_weight: Optional[float] = None):
        super().__init__()
        self.gamma = gamma
        self.loss_weight = torch.FloatTensor([1-positive_loss_weight, positive_loss_weight]) \
                            if positive_loss_weight is not None else positive_loss_weight

    def _to_cuda_device_as(self, tgt_tensor, src_tensor):
        if tgt_tensor is None:
            return None
        return tgt_tensor.to(src_tensor.device)

    def forward(self, pred: torch.Tensor, label: torch.Tensor, **kwargs) -> torch.Tensor:
        loss_tensor = torch.nn.functional.binary_cross_entropy(pred, label.float(), reduction='none', **kwargs)  # float type tensor is expected for 'label'
        return binary_focal_loss(pred, label, loss_tensor, self.gamma, self._to_cuda_device_as(self.loss_weight, loss_tensor))
