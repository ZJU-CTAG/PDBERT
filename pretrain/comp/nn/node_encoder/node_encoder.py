from typing import Dict, Optional

import torch

from allennlp.common.registrable import Registrable


class NodeEncoder(torch.nn.Module, Registrable):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self,
                node_features: torch.Tensor,
                node_mask: Optional[torch.Tensor] = None,
                node_extra_features: Dict[str, torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def get_output_dim(self):
        return self.output_dim


