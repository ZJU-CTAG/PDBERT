from typing import List, Optional

import torch
from allennlp.data import TextFieldTensors, Vocabulary, Token

from utils.allennlp_utils.tokenizer_vocab_sensitive_utils import codebert_families, microsoft_pretrain_families


def check_identifier_matching(edges: torch.Tensor,
                              code: TextFieldTensors,
                              vocab: Vocabulary,
                              namespace: str = 'code_tokens',
                              token_id_key: str = 'token_ids',
                              batch_i: int = 0):
    for edge in edges[batch_i]:
        if edge.sum().item() == 0:
            continue
        s_id = code[namespace][token_id_key][batch_i][edge[0].int().item()].item()
        e_id = code[namespace][token_id_key][batch_i][edge[1].int().item()].item()
        print(vocab.get_token_from_index(s_id, namespace), end=' ')
        print(vocab.get_token_from_index(e_id, namespace))

def check_identifier_matching_one_edge_group(edges: torch.Tensor,
                              code: TextFieldTensors,
                              vocab: Vocabulary,
                              namespace: str = 'code_tokens',
                              token_id_key: str = 'token_ids',
                              batch_i: int = 0):
    for edge in edges:
        s_id = code[namespace][token_id_key][batch_i][edge[0].int().item()].item()
        e_id = code[namespace][token_id_key][batch_i][edge[1].int().item()].item()
        print(vocab.get_token_from_index(s_id, namespace), end=' ')
        print(vocab.get_token_from_index(e_id, namespace))

from utils import GlobalLogger as mylogger

def check_pretrain_code_field_correctness(special_tokenizer_token_handler_type: str,
                                          original_code: str,
                                          tokenized_code: List[Token],
                                          line_indexes: torch.Tensor,
                                          edge_matrix: Optional[torch.Tensor] = None,
                                          input_mode_count: int = 1):
    # 1. Check tokenized code
    if special_tokenizer_token_handler_type in codebert_families:
        if len(tokenized_code) <= 2:
            mylogger.error('check_pretrain_code_field_correctness',
                           f'Found empty tokenized code, original code: {original_code}')
    else:
        mylogger.warning('check_pretrain_code_field_correctness',
                         f'Unhandled tokenized type: {special_tokenizer_token_handler_type}')

    # 2. Check consistency between code and line index
    if special_tokenizer_token_handler_type in codebert_families:
        special_token_count = 1 + input_mode_count
    else:
        raise NotImplementedError
    if len(tokenized_code) - special_token_count != len(line_indexes):
        mylogger.error('check_pretrain_code_field_correctness',
                       f'Inconsistency found between line index and tokenized code: ' +
                       f'code_len({len(tokenized_code)}) - {special_token_count} != index_len({len(line_indexes)}). '
                       f'\noriginal code: {original_code}')