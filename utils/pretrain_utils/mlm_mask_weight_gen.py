from typing import List, Tuple

import torch
from allennlp.data.tokenizers import Token
from pygments.lexers.c_cpp import CppLexer

from utils.pretrain_utils.lexer_based_token_analyse_utils import lexer_match_tokens_and_intersect_allennlp_tokens

cpp_lexer = CppLexer()

def uniform_mlm_gen_mask_weights(raw_code: str, tokens: List[Token]) -> Tuple[torch.Tensor, List]:
    """
    Equal weights for all tokens.
    """
    return torch.ones(len(tokens),), []


def basic_lexer_filter_mlm_gen_mask_weights(raw_code: str, tokens: List[Token]) -> Tuple[torch.Tensor, List]:
    """
    Filter whitespaces, operators, punctuations and literals.
    """
    filtered_types = ['Token.Punctuation', 'Token.Text.Whitespace',
                      'Token.Literal.String', 'Token.Operator', 'Token.Literal.Number.Integer']
    unfiltered_indices = lexer_match_tokens_and_intersect_allennlp_tokens(raw_code, tokens, filtered_types, is_filtered_list=True)
    weights = torch.zeros(len(tokens),)
    weights[unfiltered_indices] += 1
    return weights, unfiltered_indices


def enhanced_lexer_filter_mlm_gen_mask_weights(raw_code: str, tokens: List[Token]) -> Tuple[torch.Tensor, List]:
    """
    Enhanced filter of all subtypes of operators, punctuations, literals, texts, comments and errors.
    Leave only Token.Name and its subtypes (not checked yet).
    """
    filtered_types = ['Token.Punctuation', 'Token.Text', 'Token.Literal', 'Token.Operator', 'Token.Comment', 'Token.Error']
    unfiltered_indices = lexer_match_tokens_and_intersect_allennlp_tokens(raw_code, tokens, filtered_types, is_filtered_list=True)
    weights = torch.zeros(len(tokens),)
    weights[unfiltered_indices] += 1
    return weights, unfiltered_indices


from utils import GlobalLogger as mylogger

mlm_weight_gen_dispatcher = {
    'uniform': uniform_mlm_gen_mask_weights,
    'basic_lexer_filter': basic_lexer_filter_mlm_gen_mask_weights,
    'enhanced_lexer_filter': enhanced_lexer_filter_mlm_gen_mask_weights,

    'none': uniform_mlm_gen_mask_weights,
    None: uniform_mlm_gen_mask_weights,
}

def dispatch_mlm_weight_gen_method(method: str = 'uniform'):
    if method not in mlm_weight_gen_dispatcher:
        mylogger.warning('mlm_weight_gen_dispatch',
                         f'No such method when dispatching mlm weight gen: {method}')
    return mlm_weight_gen_dispatcher.get(method, uniform_mlm_gen_mask_weights)