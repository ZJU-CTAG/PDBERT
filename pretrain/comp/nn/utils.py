from typing import Optional

import torch


def stat_true_count_in_batch_dim(mask: torch.Tensor):
    """
    Count the number of true elements among each row of the mask.
    Input can be arbitrary shape.
    It wil return a 1D int tensor to indicate the count of 'True' in the batch.
    """
    bsz = mask.size(0)
    idx_list = mask.nonzero()

    # Count number of edges per instance in the batch.
    zeros_to_be_filled = torch.zeros((bsz), dtype=torch.int, device=mask.device)
    ones_to_fill = torch.ones_like(idx_list[:, 0], dtype=torch.int, device=mask.device)
    true_count = torch.scatter_add(zeros_to_be_filled, 0, idx_list[:, 0], ones_to_fill)

    return true_count


def sample_2D_mask_by_count_along_batch_dim(
        source_mask: torch.Tensor,
        sample_num_list: torch.Tensor
) -> torch.Tensor:
    """
    Sample 'True' value from from the 2D input mask based on the given
    count of each row.
    Source mask must be in 2D shape and count tensor should be derived from
    torch.nonzero() method.
    Return a new mask that only sampled 'True' positions are of True value.
    """

    sampled_mask = torch.zeros_like(source_mask)
    bsz = source_mask.size(0)
    batch_candidate_idx_list = source_mask.nonzero()

    # Sample from each item in the batch by given count.
    # This operation can not be easily implemented in a batched manner, thus
    # sampling is done iteratively in the batch.
    sampled_idxes = []
    for i in range(bsz):
        sampled_num_i = int(sample_num_list[i].item())
        # Skip null sampling.
        if sampled_num_i == 0:
            continue
        # Selected indexes are in 1D form, reshape to 2D.
        i_idxes_to_be_sampled = torch.masked_select(batch_candidate_idx_list, batch_candidate_idx_list[:, 0:1] == i).view(-1, 2)
        i_sampled_idxes = i_idxes_to_be_sampled[torch.randperm(i_idxes_to_be_sampled.size(0))[:sampled_num_i]]
        sampled_idxes.append(i_sampled_idxes)

    # Set sampled non-edged positions to be True.
    # [Note]
    # Sampled items may be less than given count, since the number of total candidates implied
    # by the source mask may be less than required count.
    if len(sampled_idxes) > 0:
        sampled_idxes = torch.cat(sampled_idxes, dim=0)
        # Set sampled positions to be 1.
        sampled_mask[sampled_idxes[:, 0], sampled_idxes[:, 1]] = 1

    return sampled_mask


def multinomial_sample_2D_mask_by_count_along_batch_dim(
        source_mask: torch.Tensor,
        sample_num_list: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Sample 'True' value from from the 2D input mask based on the given
    count of each row, supporting sampling weights of multinomial distribution.
    Source mask must be in 2D shape and count tensor should be derived from
    torch.nonzero() method.
    Return a new mask that only sampled 'True' positions are of True value.
    """
    if weight is None:
        weight = torch.ones_like(source_mask)

    sampled_mask = torch.zeros_like(source_mask)
    bsz = source_mask.size(0)
    batch_candidate_idx_list = source_mask.nonzero()
    batch_candidate_weight = weight[batch_candidate_idx_list[:,0], batch_candidate_idx_list[:,1]]

    # Sample from each item in the batch by given count.
    # This operation can not be easily implemented in a batched manner, thus
    # sampling is done iteratively in the batch.
    sampled_idxes = []
    for i in range(bsz):
        sampled_num_i = int(sample_num_list[i].item())
        # Skip null sampling.
        if sampled_num_i == 0:
            continue
        # Selected indexes are in 1D form, reshape to 2D.
        i_idxes_to_be_sampled = torch.masked_select(batch_candidate_idx_list, batch_candidate_idx_list[:, 0:1] == i).view(-1, 2)
        # Select weights
        i_idxes_sample_weight = torch.masked_select(batch_candidate_weight, batch_candidate_idx_list[:, 0] == i)
        # Sample from indexes according to weights
        i_sampled_inner_idxes = torch.multinomial(i_idxes_sample_weight, sampled_num_i)
        i_real_sampled_idxes = i_idxes_to_be_sampled[i_sampled_inner_idxes]
        # i_sampled_idxes = i_idxes_to_be_sampled[torch.randperm(i_idxes_to_be_sampled.size(0))[:sampled_num_i]]
        sampled_idxes.append(i_real_sampled_idxes)

    # Set sampled non-edged positions to be True.
    # [Note]
    # Sampled items may be less than given count, since the number of total candidates implied
    # by the source mask may be less than required count.
    if len(sampled_idxes) > 0:
        sampled_idxes = torch.cat(sampled_idxes, dim=0)
        # Set sampled positions to be 1.
        sampled_mask[sampled_idxes[:, 0], sampled_idxes[:, 1]] = 1

    return sampled_mask


def replace_int_value(tensor: torch.Tensor,
                      replaced_value: int,
                      new_value: int) -> torch.Tensor:
    replace_mask = tensor == replaced_value
    return torch.masked_fill(tensor, replace_mask, new_value)

def construct_matrix_from_opt_edge_idxes(opt_edge_idxes: torch.Tensor, token_mask: torch.Tensor,
                                         edge_value: int = 2, non_edge_value: int = 1, pad_value: int = 0) -> torch.Tensor:
    """
        Construct edge matrix from optimized edge indices list.

        e.g.:

        Optimized edge indices list, Shape: [2, 3, 2]
        [[[0,1], [0,2], [2,2]],
         [[0,0], [1,1], [0,0]]]

        Output matrix, Shape: [2, 4, 4] (max_item=4, valid_token_count=[3,2])
        [[[1,2,2,0],   [[2,1,0,0],
          [1,1,1,0],    [1,2,0,0],
          [1,1,2,0],    [0,0,0,0],
          [0,0,0,0]],   [0,0,0,0]]]

        Note: Since the number of valid elements of each item of the batch can be different,
              we use "token_mask" to collect the real valid item count and set the padded rows and columns to 0s.

    """
    bsz, max_token_num = token_mask.shape
    # opt_edge_idxes shape: [bsz, max_edge, 2]
    # meta_idxes shape: [num_of_edges_in_batch, 2]

    # First we want to filter padded edges in the input "opt_edge_idxes"
    # For padded edges, they must be (0,0) and sum to 0, thus we can use "nonzero" to filter them
    # (Side Effect: real (0,0) edge may also be filtered)
    non_pad_opt_edge_meta_idxes = opt_edge_idxes.sum(2).nonzero()
    # unshaped_items shape: [num_of_edges_in_batch, 2]
    unshaped_non_pad_opt_edge_items = opt_edge_idxes[non_pad_opt_edge_meta_idxes[:,0], non_pad_opt_edge_meta_idxes[:,1]]
    # item_inbatch_idxes shape: [num_of_edges_in_batch]
    items_inbatch_idxes = non_pad_opt_edge_meta_idxes[:,0]

    # Then, we concat the batch_index as the first dimension of the "start-end" edge pair,
    # to make convenience of parallel processing the whole batch.
    # matrix_edge_idxes shape: [num_of_edges_in_batch, 3]
    matrix_edge_idxes = torch.cat((items_inbatch_idxes.unsqueeze(-1), unshaped_non_pad_opt_edge_items), dim=-1)
    matrix_edge_idxes = matrix_edge_idxes.long()
    # token_mask shape: [bsz, max_token, max_token]
    matrix: torch.Tensor = torch.zeros((bsz,max_token_num,max_token_num), device=token_mask.device) + non_edge_value
    # Set edges
    matrix[matrix_edge_idxes[:,0],matrix_edge_idxes[:,1],matrix_edge_idxes[:,2]] = edge_value

    # Set padded positions in matrix to zero
    # This process is hard to parallelize, thus we sequentially do it
    for i, matrix_i in enumerate(matrix):
        non_pad_token_num = token_mask[i].sum()
        matrix_i[non_pad_token_num:,non_pad_token_num:] = pad_value

    return matrix
