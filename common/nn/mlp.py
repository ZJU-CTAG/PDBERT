import torch
from torch import nn

from common.nn.activation_builder import build_activation

def unified_mlp_module(dims, activation, dropout):
    assert len(dims) >= 2
    blocks = [mlp_block(dims[i], dims[i+1], activation, dropout) for i in range(len(dims)-1)]
    return torch.nn.Sequential(*blocks)


def mlp_block(in_dim, out_dim, activation, dropout):
    blocks = [nn.Linear(in_dim, out_dim)]

    activation_block = build_activation(activation)
    blocks.append(activation_block)

    if dropout is not None:
        blocks.append(nn.Dropout(dropout))

    return nn.Sequential(*blocks)