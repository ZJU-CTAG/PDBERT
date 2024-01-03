from typing import List
import torch
from allennlp.data import Token

from utils.pretrain_utils.lexer_based_token_analyse_utils import lexer_match_tokens_and_intersect_allennlp_tokens

def _none_token_mask(raw_code: str, tokens: List[Token]) -> torch.Tensor:
    mask = torch.ones(len(tokens),)
    return mask

_token_name_types = ['Token.Name']

def _token_name_token_mask(raw_code: str, tokens: List[Token]) -> torch.Tensor:
    mask = torch.zeros(len(tokens),)
    unmasked_indices = lexer_match_tokens_and_intersect_allennlp_tokens(raw_code, tokens, _token_name_types, is_filtered_list=False)
    mask[unmasked_indices] = 1
    return mask

_token_mask_method_dispatcher = {
    'token_name': _token_name_token_mask,

    'none': _none_token_mask,
    None: _none_token_mask,
}

from utils import GlobalLogger as mylogger

def dispatch_token_mask_method(method_name: str):
    if method_name not in _token_mask_method_dispatcher:
        mylogger.warning('dispatch_token_mask_method',
                         f'method {method_name} not in accepted list: {_token_mask_method_dispatcher.keys()}')
    return _token_mask_method_dispatcher.get(method_name, _none_token_mask)
