import torch
from typing import Iterable, Dict, Optional, List, Tuple
from tqdm import tqdm

from allennlp.data import Instance, Tokenizer, TokenIndexer
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, TensorField, LabelField, ListField

from common.modules.code_cleaner import CodeCleaner, PreLineTruncateCodeCleaner
from utils.downstream_utils.tokenize_utils import downstream_tokenize
from utils.file import read_dumped


@DatasetReader.register('cvss_metric_pred_base')
class CvssMetricPredBaseDatasetReader(DatasetReader):
    def __init__(self,
                 metrics_to_predict: List[str],
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
        self.metrics_to_predict = metrics_to_predict

        self.code_tokenizer = code_tokenizer
        self.code_indexers = {code_namespace: code_indexer}
        self.code_max_tokens = code_max_tokens
        self.code_cleaner = code_cleaner
        self.tokenizer_type = tokenizer_type
        self.model_mode = model_mode
        self.debug = debug

    def text_to_instance(self, data_item: Dict) -> Instance:
        code = data_item['code']
        code = self.code_cleaner.clean_code(code)
        tokenized_code = downstream_tokenize(self.code_tokenizer, code, self.tokenizer_type, self.model_mode)
        fields = {
            'code': TextField(tokenized_code, self.code_indexers),
        }

        metric_labels = []
        for metric in self.metrics_to_predict:
            metric_val = data_item[metric]
            metric_label = LabelField(metric_val, label_namespace=f'{metric}_labels')
            metric_labels.append(metric_label)
        label_field = ListField(metric_labels)
        fields['label'] = label_field

        return Instance(fields)


    def _read(self, file_path) -> Iterable[Instance]:
        data = read_dumped(file_path)
        if self.debug:
            data = data[:150]
        for item in tqdm(data):
            yield self.text_to_instance(item)