import torch

class DecoderOutputActivationAdapter(torch.nn.Module):
    def __init__(self, activation: str):
        super().__init__()
        if activation == 'sigmoid':
            self.forward_func = self._sigmoid_forward
        elif activation == 'softmax':
            self.forward_func = self._softmax_forward
        else:
            raise NotImplementedError(f'Activation: {activation}')

    def _sigmoid_forward(self, output_features: torch.Tensor) -> torch.Tensor:
        pred_scores = torch.sigmoid(output_features).squeeze(-1)
        return pred_scores

    def _softmax_forward(self, output_features: torch.Tensor) -> torch.Tensor:
        pred_scores = torch.softmax(output_features, dim=1)[:, :, :, 1]
        return pred_scores

    def forward(self, output_features: torch.Tensor) -> torch.Tensor:
        return self.forward_func(output_features)