from typing import Optional, List

from allennlp.data import Tokenizer, Token

from utils.allennlp_utils.tokenizer_vocab_sensitive_utils import pre_handle_special_tokenizer_tokens, post_handle_special_tokenizer_tokens

def downstream_tokenize(tokenizer: Tokenizer,
                        raw_code: str,
                        tokenizer_type: str,
                        mode: Optional[str] = None) -> List[Token]:
    '''
        Reuse the utils of pre-train to tokenize code for downstream.
        The core action lies on "post_handle_special_tokenizer_tokens", such as adding special prefix.
        This function is designed to simulate "model.tokenize" of CodeBERT code.
    '''
    # NOTE: The max tokens are not stritly constrained here,
    #       thus we need to leave a few tokens from max_tokens of pre-traiend model to avoid overflow.
    tokens = tokenizer.tokenize(raw_code)
    pre_handled_tokens = pre_handle_special_tokenizer_tokens(tokenizer_type, tokens)
    post_handled_tokens, _ = post_handle_special_tokenizer_tokens(tokenizer_type, (pre_handled_tokens,), None, mode)
    return post_handled_tokens