from typing import Tuple, Iterable, Dict, List, Optional
import os
import re
from tqdm import tqdm

import torch
from allennlp.data import Tokenizer, TokenIndexer, Instance, Token
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, TensorField

from common.modules.code_cleaner import CodeCleaner, TrivialCodeCleaner
from utils.allennlp_utils.tokenizer_vocab_sensitive_utils import pre_handle_special_tokenizer_tokens, \
    post_handle_special_tokenizer_tokens
from utils.file import read_dumped
from utils.pretrain_utils.check import check_pretrain_code_field_correctness
from utils.pretrain_utils.mlm_mask_weight_gen import dispatch_mlm_weight_gen_method
from utils.pretrain_utils.mlm_span_mask_utils import dispatch_mlm_span_mask_tag_method
from utils.pretrain_utils.token_pdg_matrix_mask_utils import dispatch_token_mask_method


@DatasetReader.register('packed_hybrid_line_token_pdg')
class PackedHybridLineTokenPDGDatasetReader(DatasetReader):
    def __init__(self,
                 code_tokenizer: Tokenizer,
                 code_indexer: TokenIndexer,
                 # volume_range: Tuple[int,int],  # closed interval: [a,b]
                 pdg_max_vertice: int,  # For line-level approach, this should be equal to "max_lines"
                 max_lines: int,
                 code_max_tokens: int,
                 code_namespace: str = "code_tokens",
                 code_cleaner: CodeCleaner = TrivialCodeCleaner(),  # Do not set this to keep consistent with char span of token level joern-parse
                 tokenized_newline_char: str = 'Ċ',  # \n after tokenization by CodeBERT
                 special_tokenizer_token_handler_type: str = 'codebert',
                 only_keep_complete_lines: bool = True,
                 mlm_sampling_weight_strategy: str = 'uniform',
                 mlm_span_mask_strategy: str = 'none',
                 multi_vs_multi_strategy: str = 'first',
                 hybrid_data_is_processed: bool = False,
                 processed_tokenizer_name: str = 'microsoft/codebert-base',
                 optimize_data_edge_input_memory: bool = True,
                 ctrl_edge_version: str = 'v1',                     # To adapt new version of ctrl edges input, only line-level ctrl edges but not data edges
                 token_data_edge_mask_strategy: str = 'none',       # To exclude some token-pairs when calculating loss of token-data prediction, set this param
                 token_data_edge_mask_kwargs: Dict = {},              # Param of token_mask_method
                 model_mode: Optional[str] = None,
                 debug: bool = False,
                 is_train: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.code_tokenizer = code_tokenizer
        self.code_token_indexers = {code_namespace: code_indexer}
        self.pdg_max_vertice = pdg_max_vertice
        self.max_lines = max_lines
        self.code_max_tokens = code_max_tokens
        self.tokenized_newline_char = tokenized_newline_char
        self.code_cleaner = code_cleaner
        self.special_tokenizer_token_handler_type = special_tokenizer_token_handler_type
        self.only_keep_complete_lines = only_keep_complete_lines
        self.hybrid_data_is_processed = hybrid_data_is_processed
        self.processed_tokenizer_name = processed_tokenizer_name
        self.optimize_data_edge_input_memory = optimize_data_edge_input_memory
        self.mlm_sampling_weight_method = dispatch_mlm_weight_gen_method(mlm_sampling_weight_strategy)
        self.mlm_span_mask_tag_gen_method = dispatch_mlm_span_mask_tag_method(mlm_span_mask_strategy)
        self.multi_vs_multi_strategy = multi_vs_multi_strategy
        self.ctrl_edge_matrix_func = {
            'v1': self.make_ctrl_edge_matrix_v1,
            'v2': self.make_ctrl_edge_matrix_v2,
        }[ctrl_edge_version]
        self.token_data_edge_mask_func = dispatch_token_mask_method(token_data_edge_mask_strategy)
        self.token_data_edge_mask_kwargs = token_data_edge_mask_kwargs
        self.model_mode = model_mode

        self.is_train = is_train
        self.actual_read_samples = 0
        self.debug = debug


    def make_ctrl_edge_matrix_v1(self,
                                 line_edges: List[str],
                                 line_count: int) -> torch.LongTensor:
        """
        Make line-level ctrl dependency matrix from edge data.
        V1: Results from latest version of joern, including line-level data edges.

        """
        # To cover the last line (line_count-th), we have to allocate one more line here.
        # Set 1 as default to distinguish from padded positions
        matrix = torch.ones((line_count+1, line_count+1))

        for edge in line_edges:
            tail, head, etype = re.split(',| ', edge)   # tail/head vertice index start from 1 instead of 0
            tail, head, etype = int(tail), int(head), int(etype)
            # Ignore uncovered vertices (lines)
            if tail > line_count or head > line_count:
                continue
            if etype == 3 or etype == 2:
                matrix[tail, head] = 2

        # Drop 0-th row and column, since line index starts from 1.
        return matrix[1:, 1:]

    def make_ctrl_edge_matrix_v2(self,
                                 line_edges: List[str],
                                 line_count: int) -> torch.LongTensor:
        """
        Make line-level ctrl dependency matrix from edge data.
        V2: Results from 0.3.1 version of joern, only line-level ctrl edges.

        """
        # To cover the last line (line_count-th), we have to allocate one more line here.
        # Set 1 as default to distinguish from padded positions
        matrix = torch.ones((line_count+1, line_count+1))

        for edge in line_edges:
            tail, head = edge
            # Ignore uncovered vertices (lines)
            if tail > line_count or head > line_count:
                continue
            matrix[tail, head] = 2

        # Drop 0-th row and column, since line index starts from 1.
        # Now line index start from 0.
        return matrix[1:, 1:]

    def make_data_edge_matrix_from_processed(self,
                                             tokens: List[Token],
                                             processed_data_edges: List[Tuple[int,int]]) -> torch.Tensor:
        token_len = len(tokens)
        # Set 1 as default to distinguish from padded positions
        matrix = torch.ones((token_len, token_len))

        for edge in processed_data_edges:
            s_token_idx, e_token_idx = edge
            if s_token_idx >= token_len or e_token_idx >= token_len:
                continue
            if s_token_idx == e_token_idx:
                continue
            # Set edge value to 2
            matrix[s_token_idx, e_token_idx] = 2

        return matrix

    def make_data_edge_matrix_from_processed_optimized(self,
                                                       tokens: List[Token],
                                                       processed_data_edges: List[Tuple[int,int]]) -> torch.Tensor:
        """
        Compared to making matrix at loading time, here we only give the edges idxes,
        to allow construct token matrix at run-time to avoid unaffordable memory consumption.
        """
        token_len = len(tokens)
        idxes = []
        for edge in processed_data_edges:
            s_token_idx, e_token_idx = edge
            if s_token_idx >= token_len or e_token_idx >= token_len:
                continue
            if s_token_idx == e_token_idx:
                continue
            idxes.append([s_token_idx, e_token_idx])

        # Append a placeholder idx, to avoid key missing error when calling "batch_tensor"
        if len(idxes) == 0:
            idxes.append([0,0])

        return torch.Tensor(idxes)


    def truncate_and_make_line_index(self, tokens: List[Token]) -> Tuple[List[Token],torch.Tensor,int]:
        """
        Truncate code tokens based on max_lines and max_tokens and determine line index for each token after tokenization.
        Line indexes (2D) will be used to aggregate line representation from token representations.
        Indexes and tokens are matched one-by-one.
        """
        line_idxes = []
        line_tokens = []
        current_line = 1        # line_index start from 1, to distinguish from padded zeros
        current_column = 0
        tokens = pre_handle_special_tokenizer_tokens(self.special_tokenizer_token_handler_type, tokens)

        for i, token in enumerate(tokens):
            line_idxes.append([current_line, current_column])   # 2D line-column index
            line_tokens.append(token)
            current_column += 1
            if token.text == self.tokenized_newline_char:
                current_line += 1
                current_column = 0
            # truncate code tokens if exceeding max_lines or max_tokens
            # NOTE: Since post-handle may not be the invert operation of pre-handle, the number of
            #       max tokens here may be slightly different from the given number.
            if current_line > self.max_lines or len(line_tokens) == self.code_max_tokens:
                break

        if self.only_keep_complete_lines:
            # FixBug: Empty tokens when 'current_column' is 0 or 'current_line' is 1.
            if current_column > 0 and current_line > 1:
                line_tokens = line_tokens[:-current_column]
                line_idxes = line_idxes[:-current_column]

        line_tokens, line_idxes = post_handle_special_tokenizer_tokens(self.special_tokenizer_token_handler_type, (line_tokens,), line_idxes,
                                                                       mode=self.model_mode)
        return line_tokens, torch.LongTensor(line_idxes), current_line-1

    def _test_text_to_instance(self, packed_data: Dict) -> Tuple[bool, Instance]:
        raw_code = packed_data['raw_code']
        raw_code = self.code_cleaner.clean_code(raw_code)
        tokenized_code = self.code_tokenizer.tokenize(raw_code)
        tokenized_code, token_line_idxes, line_count = self.truncate_and_make_line_index(tokenized_code)
        check_pretrain_code_field_correctness(self.special_tokenizer_token_handler_type, raw_code, tokenized_code, token_line_idxes, None, input_mode_count=1)

        # Ignore single-line code samples.
        if line_count <= 1:
            return False, Instance({})

        fields = {
            'code': TextField(tokenized_code, self.code_token_indexers),
            'line_idxes': TensorField(token_line_idxes),
            'vertice_num': TensorField(torch.Tensor([line_count])), # num. of line is vertice num.
        }
        return True, Instance(fields)

    def process_test_labels(self, raw_code, ctrl_edges, data_edges):
        raw_code = self.code_cleaner.clean_code(raw_code)
        tokenized_code = self.code_tokenizer.tokenize(raw_code)
        tokenized_code, token_line_idxes, line_count = self.truncate_and_make_line_index(tokenized_code)
        ctrl_matrix = self.ctrl_edge_matrix_func(ctrl_edges, line_count)
        data_matrix = self.make_data_edge_matrix_from_processed(tokenized_code, data_edges)
        return ctrl_matrix, data_matrix, line_count

    def text_to_instance(self, packed_pdg: Dict) -> Tuple[bool, Instance]:
        if not self.is_train:
            return self._test_text_to_instance(packed_pdg)

        raw_code = packed_pdg['raw_code']
        line_edges = packed_pdg['line_edges']
        # original_total_line = packed_pdg['total_line']

        raw_code = self.code_cleaner.clean_code(raw_code)
        tokenized_code = self.code_tokenizer.tokenize(raw_code)
        tokenized_code, token_line_idxes, line_count = self.truncate_and_make_line_index(tokenized_code)
        edge_matrix = self.ctrl_edge_matrix_func(line_edges, line_count)
        if self.hybrid_data_is_processed:
            if self.optimize_data_edge_input_memory:
                data_matrix = self.make_data_edge_matrix_from_processed_optimized(tokenized_code,
                                                                                  packed_pdg['processed_token_data_edges'][self.processed_tokenizer_name])
            else:
                data_matrix = self.make_data_edge_matrix_from_processed(tokenized_code,
                                                                        packed_pdg['processed_token_data_edges'][self.processed_tokenizer_name])
        else:
            raise ValueError("Data must be processed")

        # Ignore single-line code samples.
        if line_count == 1:
            return False, Instance({})

        mlm_sampling_weights, _ = self.mlm_sampling_weight_method(raw_code, tokenized_code)
        token_data_token_mask = self.token_data_edge_mask_func(raw_code, tokenized_code, **self.token_data_edge_mask_kwargs)

        check_pretrain_code_field_correctness(self.special_tokenizer_token_handler_type, raw_code, tokenized_code, token_line_idxes, line_edges)
        fields = {
            'code': TextField(tokenized_code, self.code_token_indexers),
            'line_idxes': TensorField(token_line_idxes),
            'line_ctrl_edges': TensorField(edge_matrix),
            'token_data_edges': TensorField(data_matrix),
            'vertice_num': TensorField(torch.Tensor([line_count])), # num. of line is vertice num.
            'mlm_sampling_weights': TensorField(mlm_sampling_weights),
            'token_data_token_mask': TensorField(token_data_token_mask),
        }

        span_tags = self.mlm_span_mask_tag_gen_method(raw_code, tokenized_code)
        if span_tags is not None:
            fields['mlm_span_tags'] = TensorField(span_tags)

        return True, Instance(fields)


    def _read(self, dataset_config: Dict) -> Iterable[Instance]:
        from utils import GlobalLogger as logger
        data_base_path = dataset_config['data_base_path']
        volume_range = dataset_config['volume_range']   # close interval

        for vol in range(volume_range[0], volume_range[1]+1):
            logger.info('PackedHybridTokenLineReader.read', f'Reading Vol. {vol}')

            vol_path = os.path.join(data_base_path, f'packed_hybrid_vol_{vol}.pkl')
            packed_vol_data_items = read_dumped(vol_path)
            packed_vol_data_items = packed_vol_data_items[:100] if self.debug else packed_vol_data_items
            for pdg_data_item in packed_vol_data_items:
                try:
                    ok, instance = self.text_to_instance(pdg_data_item)
                    if ok:
                        self.actual_read_samples += 1
                        yield instance
                # TODO: revert
                except FileNotFoundError as e:
                    logger.error('read', f'error: {e}. \npdg-item content: {pdg_data_item}')

        logger.info('reader', f'Total samples loaded: {self.actual_read_samples}')