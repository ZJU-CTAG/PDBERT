from typing import List, Dict, Optional

import torch

from pretrain.comp.nn.node_encoder.node_encoder import NodeEncoder
from common.nn.mlp import unified_mlp_module


@NodeEncoder.register('mlp')
class MLPNodeEncoder(NodeEncoder):
    """
    Use a MLP to simply project node features into a latent feature space.
    Mostly for dimension reduction purpose.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: List[int],
                 activation: str = 'relu',
                 dropout: float = 0.,
                 **kwargs):
        super().__init__(input_dim, hidden_dims[-1], **kwargs)
        self.mlp = unified_mlp_module([input_dim] + hidden_dims + [output_dim], activation, dropout)

    def forward(self,
                node_features: torch.Tensor,
                node_mask: Optional[torch.Tensor] = None,
                node_extra_features: Dict[str, torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        return self.mlp(node_features)
