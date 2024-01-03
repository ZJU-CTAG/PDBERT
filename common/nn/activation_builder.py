import torch

def _identity():
    return torch.nn.Identity()

def _relu(**kwargs):
    return torch.nn.ReLU()

def _gelu(**kwargs):
    return torch.nn.GELU()

def _tanh(**kwargs):
    return torch.nn.Tanh()

def _sigmoid(**kwargs):
    return torch.nn.Sigmoid()

_activation_map = {
    'relu': _relu,
    'gelu': _gelu,
    'tanh': _tanh,
    'sigmoid': _sigmoid,

    None: _identity,
    'none': _identity,
    'identity': _identity
}

def build_activation(name, **kwargs):
    return _activation_map.get(name)(**kwargs)
