from typing import Tuple, Optional

import torch

from allennlp.common.registrable import Registrable

from common.nn.loss_func import LossFunc
from pretrain.comp.nn.utils import replace_int_value


class LossSampler(Registrable, torch.nn.Module):
    def __init__(self,
                 loss_func: LossFunc,
                 **kwargs):
        super().__init__()
        self.loss_func = loss_func

    def get_loss(self, edge_matrix: torch.Tensor,
                 predicted_matrix: torch.Tensor,
                 elem_mask: Optional[torch.Tensor] = None
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def get_elem_mask_matrix(self, edge_matrix: torch.Tensor, elem_mask: torch.Tensor):
        # elem_mask shape: [bsz, seq]
        if elem_mask is not None:
            assert elem_mask.size(1) == edge_matrix.size(1), \
                f'Unmatched shape between elem_mask({elem_mask.size(1)}) and edge_matrix({edge_matrix.size(1)})'
            elem_matrix_mask = torch.bmm(elem_mask.unsqueeze(-1), elem_mask.unsqueeze(1)).bool()
        else:
            elem_matrix_mask = torch.ones_like(edge_matrix).bool()

        return elem_matrix_mask

    def cal_matrix_masked_loss_mean(self,
                                    predicted_matrix: torch.Tensor,
                                    edge_matrix: torch.Tensor,
                                    loss_mask: torch.Tensor) -> torch.Tensor:
        selected_label_edge = torch.masked_select(edge_matrix, loss_mask)
        selected_pred_edge = torch.masked_select(predicted_matrix, loss_mask)
        loss = self.loss_func(selected_pred_edge, selected_label_edge)
        return loss
        # Replace -1 with an arbitrary label to prevent error.
        # edge_matrix = replace_int_value(edge_matrix, -1, 0)
        # loss_matrix = self.loss_func(predicted_matrix, edge_matrix, reduction='none')
        # loss_mask = loss_mask.int()
        # return (loss_matrix * loss_mask).sum() / loss_mask.sum()
