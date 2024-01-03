from typing import Tuple

import torch

from allennlp.common.registrable import Registrable


class LineExtractor(torch.nn.Module, Registrable):
    def __init__(self, max_lines: int, **kwargs):
        super().__init__()
        self.max_lines = max_lines

    def forward(self,
                token_features: torch.Tensor,
                token_mask: torch.Tensor,
                line_idxes: torch.Tensor,
                vertice_num: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


@LineExtractor.register('avg')
class AvgLineExtractor(LineExtractor):
    def __init__(self, max_lines: int, **kwargs):
        super().__init__(max_lines)

    def forward(self,
                token_features: torch.Tensor,   # [batch, seq, dim]
                token_mask: torch.Tensor,
                line_idxes: torch.Tensor,        # [batch, seq, 2]
                vertice_num: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, dim = token_features.shape
        max_lines = int(torch.max(vertice_num).item())
        line_features = torch.zeros((bsz, max_lines+1, dim),
                                    dtype=token_features.dtype,
                                    device=token_features.device)
        # Use row indexes, namely line number to scatter
        line_row_idxes = line_idxes[:, :, 0:1].repeat(1,1,dim)
        line_features = torch.scatter_add(line_features, 1, line_row_idxes, token_features)

        # Make each line's data as one, scatter_add to accumulate count of each line index.
        # [Note] here we fill an "ones" matrix to prevent zero-division.
        line_item_count = (torch.scatter_add(input=torch.ones((bsz, max_lines+1), device=token_features.device),
                                             dim=1,
                                             index=line_idxes[:, :, 0],
                                             src=torch.ones((bsz, seq_len), device=token_features.device))
                                .unsqueeze(-1))
                                # .repeat(1,1,dim))
        line_features /= line_item_count

        # Drop first line of padded values
        line_features = line_features[:, 1:, :]
        # [Note] Default count value is 1, instead of 0.
        line_mask = (line_item_count != 1).squeeze(-1)[:, 1:]
        return line_features, line_mask


