from functools import reduce
from typing import List, Tuple
import re

from allennlp.data import Token
from pygments.lexers.c_cpp import CppLexer

cpp_lexer = CppLexer()

def cpp_lexer_parse(raw_code: str):
    return list(cpp_lexer.get_tokens_unprocessed(raw_code))

def get_token_span_by_cpplexer(raw_code: str,
                               token_types: List[str],
                               is_filtered_list: bool = True):
    """
    Given raw c/cpp code, analyze with cpp lexer and return
    char spans that are not filtered by given token types.
    """
    lexer_tokens = list(cpp_lexer.get_tokens_unprocessed(raw_code))
    not_masked_spans = []
    lexer_token_char_span = [t[0] for t in lexer_tokens] + [len(raw_code)]
    for i,token in enumerate(lexer_tokens):
        token_type_str = str(token[1])
        token_type_matched = reduce(lambda v,e: v or re.match(e, token_type_str) is not None, token_types, False)
        # Check matched and filtered relationship
        if token_type_matched ^ is_filtered_list:
            # Left-close right-open span
            not_masked_spans.append((lexer_token_char_span[i], lexer_token_char_span[i+1]))
    return not_masked_spans

def lexer_match_tokens_and_intersect_allennlp_tokens(raw_code: str,
                                                     allennlp_tokens: List[Token],
                                                     token_types: List[str],
                                                     is_filtered_list: bool = True):
    """
    Given raw code and allennlp tokenized tokens, this function first analyze
    raw code using lexer to get token types, and filter them based on given
    filtered_types.

    Then unfiltered tokens will be intersected with allennlp tokens to determine
    exactly which allennlp tokens are unfiltered based on resulted spans of lexer
    analysis.

    Finally, indices of these intersected allennlp tokens will be returned.
    """
    target_token_char_spans = get_token_span_by_cpplexer(raw_code, token_types, is_filtered_list=is_filtered_list)
    span_i = 0
    allennlp_target_token_indices = []
    for token_i, token in enumerate(allennlp_tokens):
        idx, idx_end = token.idx, token.idx_end

        # Skip speicial tokens
        if idx is None or idx_end is None:
            continue
        # No more spans to check, break
        if span_i >= len(target_token_char_spans):
            break

        cur_span = target_token_char_spans[span_i]
        # Check span intersection
        if (idx - cur_span[1]) * (idx_end - cur_span[0]) < 0:
            allennlp_target_token_indices.append(token_i)
        # Check if current token span ends and move to next span
        if idx_end >= cur_span[1]:
            span_i += 1

    return allennlp_target_token_indices


def get_token_type_char_spans(raw_code: str, token_types: List[str]) -> List[Tuple[int,int]]:
    lexer_tokens = cpp_lexer_parse(raw_code)
    token_char_spans = [t[0] for t in lexer_tokens] + [len(raw_code)]
    identifier_char_spans = []
    for i, token in enumerate(lexer_tokens):
        if str(token[1]) in token_types:
            identifier_char_spans.append((token_char_spans[i], token_char_spans[i+1]))
    return identifier_char_spans
