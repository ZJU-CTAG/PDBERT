from typing import Dict, Tuple, Optional

import torch

from common.nn.loss_func import LossFunc
from pretrain.comp.nn.utils import stat_true_count_in_batch_dim, sample_2D_mask_by_count_along_batch_dim
from pretrain.comp.nn.loss_sampler.loss_sampler import LossSampler


@LossSampler.register('separated_full_single')
class SeparatedFullSingleLossSampler(LossSampler):
    def __init__(self, loss_func: LossFunc, **kwargs):
        super().__init__(loss_func, **kwargs)

    def get_loss(self,
                 edge_matrix: torch.Tensor,
                 predicted_matrix: torch.Tensor,
                 elem_mask: Optional[torch.Tensor] = None
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param edge_matrix: [batch, max_vertice, max_vertice]
        :param predicted_matrix: [batch, max_vertice, max_vertice]
        :return:
        """
        # Minus one to make label range [0,1].
        edge_matrix -= 1
        assert edge_matrix.shape == predicted_matrix.shape[:3], \
            f"Unmatched shape between label edges ({edge_matrix.size()}) and predicted edges({predicted_matrix.size()})"

        # Manually operating "masked_mean"
        loss_mask = (edge_matrix != -1).bool()
        # BugFix: Only do element masking at training time, make it full at testing time.
        if self.training:
            elem_mask_matrix = self.get_elem_mask_matrix(edge_matrix, elem_mask)
            loss_mask = loss_mask & elem_mask_matrix
        return self.cal_matrix_masked_loss_mean(predicted_matrix, edge_matrix, loss_mask), \
               loss_mask


@LossSampler.register('separated_balanced_single')
class SeparatedBalancedSingleLossSampler(LossSampler):
    """
    This sampler balances edged pairs and non-edged pairs by sampling partial non-edged pairs
    from all the empty positions to make the have the same size.
    """
    def __init__(self,
                 loss_func: LossFunc,
                 be_full_when_test: bool = False,
                 balanced_ratio: float = 1,
                 **kwargs):
        super().__init__(loss_func, **kwargs)
        self.be_full_when_test = be_full_when_test
        self.balanced_ratio = balanced_ratio

    def get_loss(self,
                 edge_matrix: torch.Tensor,
                 predicted_matrix: torch.Tensor,
                 elem_mask: Optional[torch.Tensor] = None
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Minus one to make label range [0,1].
        edge_matrix -= 1
        assert edge_matrix.shape == predicted_matrix.shape, "Unmatched shape between label edges and predicted edges"
        # assert predicted_matrix.size(1) == 2, "The second dimension should be 2 for data and control predictions"

        elem_mask_matrix = self.get_elem_mask_matrix(edge_matrix, elem_mask).flatten(start_dim=1)

        if self.be_full_when_test and not self.training:
            loss_mask = (edge_matrix != -1).bool()
            return self.cal_matrix_masked_loss_mean(predicted_matrix, edge_matrix, loss_mask), \
                   loss_mask
        else:
            bsz, v_num = edge_matrix.shape[:2]
            # Here we must use .reshape instead of .view
            edge_matrix = edge_matrix.reshape(bsz, v_num*v_num)
            predicted_matrix = predicted_matrix.reshape(bsz, v_num*v_num)

            # Get indexes for edge/non-edge positions.
            # Note non-edged positions have index 0.
            # Here we use element mask to exclude positions regarding masked elements.
            non_edge_mask = (edge_matrix == 0) & elem_mask_matrix
            edge_mask = (edge_matrix == 1) & elem_mask_matrix       # This seems to be trivial, but important to ensure consistency

            edge_count = stat_true_count_in_batch_dim(edge_mask)
            non_edge_count = stat_true_count_in_batch_dim(non_edge_mask)
            # Sampled non-edge count can not exceed the count of all non-edges
            non_edge_sampled_count = torch.min((edge_count*self.balanced_ratio).int(), non_edge_count)
            sampled_non_edge_mask = sample_2D_mask_by_count_along_batch_dim(non_edge_mask, non_edge_sampled_count)

            # Include both non-edges and edges
            sampled_mask = sampled_non_edge_mask.bool() | edge_mask
            return self.cal_matrix_masked_loss_mean(predicted_matrix, edge_matrix, sampled_mask), \
                   sampled_mask.view(bsz, v_num, v_num)