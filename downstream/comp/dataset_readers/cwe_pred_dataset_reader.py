import torch
from typing import Iterable, Dict, Optional, List, Tuple
from tqdm import tqdm

from allennlp.data import Instance, Tokenizer, TokenIndexer
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, TensorField, LabelField

from common.modules.code_cleaner import CodeCleaner, PreLineTruncateCodeCleaner
from utils.downstream_utils.tokenize_utils import downstream_tokenize
from utils.file import read_dumped


@DatasetReader.register('cwe_pred_base')
class CwePredBaseDatasetReader(DatasetReader):
    def __init__(self,
                 cwe_label_space: List[str],
                 code_tokenizer: Tokenizer,
                 code_indexer: TokenIndexer,
                 code_max_tokens: int,
                 code_namespace: str = "code_tokens",
                 code_cleaner: CodeCleaner = PreLineTruncateCodeCleaner(200),  # Pre-truncate lines to prevent long time waited
                 tokenizer_type: str = 'codebert',
                 model_mode: Optional[str] = None,
                 debug: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.cwe_label_space = cwe_label_space
        self.cwe_label_map = {cwe:i for i,cwe in enumerate(cwe_label_space)}

        self.code_tokenizer = code_tokenizer
        self.code_indexers = {code_namespace: code_indexer}
        self.code_max_tokens = code_max_tokens
        self.code_cleaner = code_cleaner
        self.tokenizer_type = tokenizer_type
        self.model_mode = model_mode
        self.debug = debug

    def text_to_instance(self, data_item: Dict) -> Tuple[bool,Optional[Instance]]:
        code = data_item['code']
        cwe_id = data_item['cwe_id']
        if cwe_id in self.cwe_label_map:
            label = self.cwe_label_map[cwe_id]
        else:
            return False, None

        code = self.code_cleaner.clean_code(code)
        tokenized_code = downstream_tokenize(self.code_tokenizer, code, self.tokenizer_type, self.model_mode)
        fields = {
            'code': TextField(tokenized_code, self.code_indexers),
            'label': TensorField(torch.LongTensor([label]))
        }
        return True, Instance(fields)


    def _read(self, file_path) -> Iterable[Instance]:
        data = read_dumped(file_path)
        if self.debug:
            data = data[:150]
        for item in tqdm(data):
            ok, instance = self.text_to_instance(item)
            if ok:
                yield instance