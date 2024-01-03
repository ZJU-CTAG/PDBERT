from typing import List, Tuple

import torch
from torch import nn

from allennlp.common import Registrable
from transformers import RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

from common.nn.mlp import mlp_block

class Classifier(Registrable, nn.Module):
    def __init__(self, num_class: int):
        super().__init__()
        self._num_class = num_class

    def forward(self, feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        @return: logits and predicted label indexes
        """
        raise NotImplementedError

    def get_output_dim(self) -> int:
        return self._num_class

    def get_exp_input_dim(self) -> int:
        raise NotImplementedError


@Classifier.register('linear_softmax')
class LinearSoftmaxClassifier(Classifier):
    def __init__(self,
                 in_feature_dim: int,
                 out_feature_dim: int,
                 hidden_dims: List[int],
                 activations: List[str],
                 dropouts: List[float],
                 ahead_feature_dropout: float = 0.,
                 log_softmax: bool = False,
                 return_logits: bool = True):   # actual layer_num = len(hidden_dims) + 1
        super().__init__(out_feature_dim)

        assert len(hidden_dims) == len(activations) == len(dropouts)

        in_dims = [in_feature_dim, *hidden_dims]
        out_dims = [*hidden_dims, out_feature_dim]
        activations = [*activations, None]      # no activation at last last output layer
        dropouts = [*dropouts, None]            # no dropout at last output layer

        layers = [
            mlp_block(in_dim, out_dim, activation, dropout)
            for in_dim, out_dim, activation, dropout in
            zip(in_dims, out_dims, activations, dropouts)
        ]
        self._layers = nn.Sequential(*layers)
        self._in_feature_dim = in_feature_dim
        self._ahead_feature_dropout = torch.nn.Dropout(ahead_feature_dropout)
        self.softmax = torch.nn.LogSoftmax(dim=-1) if log_softmax else torch.nn.Softmax(dim=-1)
        self.return_logits = return_logits

    def forward(self, feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feature = self._ahead_feature_dropout(feature)
        logits = self._layers(feature)
        pred_idxes = torch.max(logits, dim=-1).indices

        if self.return_logits:
            return logits, pred_idxes
        else:
            probs = self.softmax(logits)
            return probs, pred_idxes

    def get_exp_input_dim(self) -> int:
        return self._in_feature_dim


@Classifier.register('linear_sigmoid')
class LinearSigmoidClassifier(LinearSoftmaxClassifier):
    def __init__(self,
                 in_feature_dim: int,
                 hidden_dims: List[int],
                 activations: List[str],
                 dropouts: List[float],
                 ahead_feature_dropout: float = 0.,
                 out_dim: int = 1):   # actaul layer_num = len(hidden_dims) + 1
        super().__init__(in_feature_dim,
                         out_dim,
                         hidden_dims,
                         activations,
                         dropouts,
                         ahead_feature_dropout)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feature = self._ahead_feature_dropout(feature)
        logits = self._layers(feature)
        logits = self.sigmoid(logits).squeeze(-1)
        pred_idxes = (logits > 0.5).long()
        return logits, pred_idxes

# MODEL_CLASSES = {
#     'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
#     'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
#     'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
#     'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
#     'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
# }

@Classifier.register('roberta_binary_header')
class RobertaBinaryHeader(Classifier):
    """
        Use RobertaHeader as classifier, as CodeXGLUE benchmark.
    """
    def __init__(self,
                 model_name_or_path: str,
                 return_logits: bool = True):
        config = RobertaConfig.from_pretrained(model_name_or_path)
        config.num_labels = 1
        super().__init__(1)

        encoder = RobertaClassificationHead(config)
        self.encoder = encoder
        self.config = config
        self.return_logits = return_logits

    def forward(self, feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Since RobertaHeader expect input shape [bsz, len, dim],
        # we have to expand the `seq` dimension of squeezed features.
        feature = feature.unsqueeze(1)

        outputs = self.encoder(feature)
        logits = outputs
        probs = torch.sigmoid(logits)

        if logits.size(-1) > 1:
            pred_idxes = torch.max(logits, dim=-1).indices
        else:
            pred_idxes = (probs > 0.5).long().squeeze(-1)

        if self.return_logits:
            return logits, pred_idxes
        else:
            return probs, pred_idxes

    def get_exp_input_dim(self) -> int:
        return self.config.hidden_size