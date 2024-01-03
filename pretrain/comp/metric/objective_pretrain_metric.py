import torch
from allennlp.training.metrics import Metric


@Metric.register('objective_loss')
class ObjectiveLoss(Metric):
    def __init__(self, name: str):
        self.name = name
        self.total_loss = 0.
        self.total_count = 0

    def __call__(self, loss_val):
        self.total_count += 1
        if isinstance(loss_val, torch.Tensor):
            self.total_loss += loss_val.item()
        else:
            self.total_loss += loss_val

    def reset(self) -> None:
        self.total_loss = 0.
        self.total_count = 0

    def get_metric(self, reset: bool):
        avg_loss = self.total_loss / self.total_count
        if reset:
            self.reset()
        return {f'{self.name}_loss': avg_loss}
