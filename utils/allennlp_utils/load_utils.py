from typing import Dict, Union
import torch
from allennlp.models import Model

from utils import GlobalLogger as mylogger

def partial_load_state_dict(model: Union[Model, torch.nn.Module],
                            state_dict: Dict[str, torch.Tensor],
                            prefix_remap: Dict[str, str]):
    """
    Load parameters from a state dict according to the given mapping.
    Note we try to remap the name of parameters of this model to match the
    keys of the given state dict in a prefix-matching manner.

    Also to note, unmapped parameters will be ignored, thus even the key is identical,
    it is also necessary to place this item in the map.
    """
    partial_state_dict = {}
    for name, par in model.named_parameters():
        load_name = None
        for prefix in prefix_remap:
            if name.startswith(prefix):
                load_prefix = prefix_remap[prefix]
                load_name = load_prefix + name[len(prefix):]
                # Always match first
                break
        # Only load mapped parameters
        if load_name is None:
            continue
        if name in state_dict:
            partial_state_dict[name] = state_dict[load_name]

    load_res = model.load_state_dict(partial_state_dict, strict=False)
    mylogger.info('partial_load_state_dict',
                  f'State dict loading result: {load_res}')