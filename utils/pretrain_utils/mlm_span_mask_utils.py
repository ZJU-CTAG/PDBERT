import torch
from typing import List, Tuple

from allennlp.data import Token

from utils.pretrain_utils.lexer_based_token_analyse_utils import get_token_type_char_spans



def get_span_mask_from_token_mask(span_tags: torch.Tensor, token_masks: torch.Tensor) -> torch.Tensor:
    """
    Given the span tags, generate the span-level masks based on token-level masks.
    This function should be used in pairs with "generate_span_tags".
    """
    span_masks = []
    for span_tag, token_mask in zip(span_tags, token_masks):
        selected_span_tags = torch.masked_select(input=span_tag, mask=token_mask)
        # Filtering tag=1 (independent tokens) before collecting span tags
        selected_span_tags = torch.masked_select(input=selected_span_tags, mask=selected_span_tags.gt(1))
        # Make the masks of tokens with span_tag greater than 1 as True
        span_mask = (span_tag == selected_span_tags.unsqueeze(-1)).sum(0).bool()
        # Include independent tokens here
        span_mask = (span_mask | token_mask)
        span_masks.append(span_mask)

    return torch.stack(span_masks, dim=0)

from utils.pretrain_utils.const import identifier_token_types, identifier_keyword_token_types


def generate_span_tags(char_spans: List[Tuple[int, int]],
                       allennlp_tokens: List[Token]):
    """
    Given target spans of the sequence, generate the span tags of the whole sequence.
    """
    span_i = 0
    allennlp_token_span_tags = []
    for token_i, token in enumerate(allennlp_tokens):
        idx, idx_end = token.idx, token.idx_end

        # Skip speicial tokens, or no more spans to check, give tag 1 as default
        # Default span tag is 1, means independent token
        if idx is None or idx_end is None or span_i >= len(char_spans):
            allennlp_token_span_tags.append(1)
            continue

        cur_span = char_spans[span_i]
        # Check span intersection
        if (idx - cur_span[1]) * (idx_end - cur_span[0]) < 0:
            allennlp_token_span_tags.append(span_i + 2)    # ! Span tag starts from 2
        else:
            allennlp_token_span_tags.append(1)

        # Check if current token span ends and move to next span
        if idx_end >= cur_span[1]:
            span_i += 1

    return torch.LongTensor(allennlp_token_span_tags)


def generate_null_span_tags(raw_code: str, allennlp_tokens: List[Token]):
    return None


def generate_identifier_span_tags(raw_code: str, allennlp_tokens: List[Token]):
    target_char_spans = get_token_type_char_spans(raw_code, token_types=identifier_token_types)
    return generate_span_tags(target_char_spans, allennlp_tokens)


def generate_identifier_keyword_span_tags(raw_code: str, allennlp_tokens: List[Token]):
    target_char_spans = get_token_type_char_spans(raw_code, token_types=identifier_keyword_token_types)
    return generate_span_tags(target_char_spans, allennlp_tokens)

mlm_span_tag_strategy_dispatcher = {
    'identifier': generate_identifier_span_tags,
    'identifier_keyword': generate_identifier_keyword_span_tags,

    'none': generate_null_span_tags,
    None: generate_null_span_tags,
}

from utils import GlobalLogger as mylogger

def dispatch_mlm_span_mask_tag_method(name: str):
    if name not in mlm_span_tag_strategy_dispatcher:
        mylogger.warning('dispatch_mlm_span_mask_tag_method',
                         f'No such method when dispatching mlm span mask gen strategy: {name}')
    return mlm_span_tag_strategy_dispatcher.get(name, generate_null_span_tags)
