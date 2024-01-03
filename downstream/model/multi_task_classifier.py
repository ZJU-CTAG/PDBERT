from typing import Dict, Optional, List, Tuple

import torch
from allennlp.data import Vocabulary, TextFieldTensors

from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Seq2VecEncoder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import Metric

from common.nn.classifier import Classifier
from common.nn.loss_func import LossFunc
from utils import GlobalLogger as mylogger
from utils.allennlp_utils.metric_update import update_metric


@Model.register('multi_task_classifier')
class MultiTaskClassifier(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        code_embedder: TextFieldEmbedder,
        code_encoder: Seq2SeqEncoder,
        code_feature_squeezer: Seq2VecEncoder,
        loss_func: LossFunc,
        classifiers: List[Classifier],
        metric: Optional[Metric] = None,
        wrapping_dim_for_code: int = 0,
        **kwargs
    ):
        super().__init__(vocab, **kwargs)
        self.code_embedder = code_embedder
        self.code_encoder = code_encoder
        self.code_feature_squeezer = code_feature_squeezer
        self.loss_func = loss_func
        self.classifiers = torch.nn.ModuleList(classifiers)
        self.metric = metric

        self.wrapping_dim_for_code = wrapping_dim_for_code


    def embed_encode_code(self, code: TextFieldTensors):
        num_wrapping_dim = self.wrapping_dim_for_code

        # shape: (batch_size, max_input_sequence_length)
        mask = get_text_field_mask(code, num_wrapping_dims=num_wrapping_dim)
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_features = self.code_embedder(code, num_wrapping_dims=num_wrapping_dim)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self.code_encoder(embedded_features, mask)
        code_feature = self.code_feature_squeezer(encoder_outputs, mask)
        return {
            "outputs": code_feature
        }

    def _forward_multi_classifiers(self, features: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        multi_task_logits, multi_task_labels = [], []
        for classifier in self.classifiers:
            pred_logits, pred_labels = classifier(features)
            multi_task_logits.append(pred_logits)
            multi_task_labels.append(pred_labels)
        # First dim is task dim.
        # Since output_num for each task can differ, we can not stack them.
        # But the predicted indices is consistent, we can stack them as output.
        return multi_task_logits, multi_task_labels

    def _get_multi_task_loss(self, logits_list: List[torch.Tensor], labels_list: List[torch.Tensor]):
        total_loss = torch.zeros((1,), device=logits_list[0].device)
        for logits, labels in zip(logits_list, labels_list):
            # labels = labels.squeeze(-1)
            loss = self.loss_func(logits, labels)
            total_loss += loss
        return total_loss

    def forward(self,
                code: TextFieldTensors,
                label: torch.Tensor,
                **kwargs) -> Dict[str, torch.Tensor]:
        # Shape: [batch, seq, dim]
        encoded_code_outputs = self.embed_encode_code(code)
        code_features = encoded_code_outputs['outputs']

        multi_task_logits, multi_task_labels = self._forward_multi_classifiers(code_features)
        # multi_task_labels = multi_task_labels.squeeze(-1)
        # loss = self.loss_func(multi_task_logits.flatten(0,1), label.flatten(0,1))
        label = label.permute((1, 0))
        loss = self._get_multi_task_loss(multi_task_logits, label)

        if self.metric is not None:
            update_metric(self.metric, multi_task_labels, multi_task_logits, label, flatten_labels=False)

        return {
            'logits': multi_task_logits,
            'pred': multi_task_labels,
            'loss': loss
        }

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if self.metric is not None:
            metric = self.metric.get_metric(reset)
            # no metric name returned, use its class name instead
            if type(metric) != dict:
                metric_name = self.metric.__class__.__name__
                metric = {metric_name: metric}
            metrics.update(metric)
        return metrics
