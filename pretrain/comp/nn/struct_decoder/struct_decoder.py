from typing import Dict, Tuple

import torch
from allennlp.common.registrable import Registrable


class StructDecoder(torch.nn.Module, Registrable):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self,
                node_features: torch.Tensor,
                extra_features: Dict[str, torch.Tensor] = None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
