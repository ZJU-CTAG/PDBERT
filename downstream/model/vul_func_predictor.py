from typing import Dict, Optional

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


@Model.register('vul_func_predictor')
@Model.register('downstream_classifier')
class VulFuncPredictor(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        code_embedder: TextFieldEmbedder,
        code_encoder: Seq2SeqEncoder,
        code_feature_squeezer: Seq2VecEncoder,
        loss_func: LossFunc,
        classifier: Classifier,
        metric: Optional[Metric] = None,
        wrapping_dim_for_code: int = 0,
        **kwargs
    ):
        super().__init__(vocab, **kwargs)
        self.code_embedder = code_embedder
        self.code_encoder = code_encoder
        self.code_feature_squeezer = code_feature_squeezer
        self.loss_func = loss_func
        self.classifier = classifier
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

    def forward(self,
                code: TextFieldTensors,
                label: torch.Tensor,
                **kwargs) -> Dict[str, torch.Tensor]:
        # Shape: [batch, seq, dim]
        encoded_code_outputs = self.embed_encode_code(code)
        code_features = encoded_code_outputs['outputs']

        pred_logits, pred_labels = self.classifier(code_features)
        label = label.squeeze(-1)
        loss = self.loss_func(pred_logits, label)

        if self.metric is not None:
            update_metric(self.metric, pred_labels, pred_logits, label)

        return {
            'logits': pred_logits,
            'pred': pred_labels,
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
