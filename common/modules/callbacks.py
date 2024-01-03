from typing import Dict, Any, List, Optional
import shutil
import os
import math

import torch
from allennlp.data import TensorDict
from allennlp.models import Model
from allennlp.training import TrainerCallback
from allennlp.models.archival import archive_model
from utils.stat import stat_model_param_number
from utils.file import read_dumped

from allennlp.data.vocabulary import Vocabulary

from utils import GlobalLogger as mylogger

@TrainerCallback.register('epoch_print')
class EpochPrintCallback(TrainerCallback):
    def __init__(self, serialization_dir=None):
        super().__init__(serialization_dir)

    def on_epoch(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any],
        epoch: int,
        is_primary: bool = True,
        **kwargs,
    ) -> None:
        print(f'* Epoch {epoch} ended')
        print('-'*100)


@TrainerCallback.register('model_param_stat')
class ModelParamStatCallback(TrainerCallback):
    def __init__(self, serialization_dir=None):
        super().__init__(serialization_dir)

    def on_start(
        self, trainer: "GradientDescentTrainer", is_primary: bool = True, **kwargs
    ) -> None:
        print('-' * 100)
        # print('Test for on_start callback...')
        stat_model_param_number(trainer.model)
        print('-' * 100)

@TrainerCallback.register('save_jsonnet_config')
class SaveJsonnetConfigCallback(TrainerCallback):
    def __init__(self,
                 file_src: str,
                 serialization_dir=None):
        super().__init__(serialization_dir)
        self._file_src = file_src
        self._serial_dir = serialization_dir

    def on_start(
        self, trainer: "GradientDescentTrainer", is_primary: bool = True, **kwargs
    ) -> None:
        # print(f'* serial_dir given to callback constructor: {self._serial_dir}')
        dst = os.path.join(self._serial_dir, '_train_config.jsonnet')
        print(f'[SaveJsonnetConfigCallback] Saving jsonnet from {self._file_src} to {dst}')
        shutil.copy(self._file_src, dst)

@TrainerCallback.register('save_epoch_model')
class SaveEpochModelCallback(TrainerCallback):
    def __init__(self,
                 serialization_dir,
                 save_epoch_points: List[int] = []):
        super().__init__(serialization_dir)
        self._serial_dir = serialization_dir
        self.save_epoch_points = save_epoch_points

    def on_epoch(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any],
        epoch: int,
        is_primary: bool = True,
        **kwargs,
    ) -> None:
        if epoch in self.save_epoch_points:
            weight_file_name = f'epoch_{epoch}.th'
            weight_file_path = os.path.join(self.serialization_dir, weight_file_name)
            trainer._save_model_state(weight_file_path)
            archive_model(self.serialization_dir,
                          weight_file_name,
                          os.path.join(self.serialization_dir, f'model_epoch_{epoch}.tar.gz'))
            # remove the weight file after archiving
            os.remove(weight_file_path)

@TrainerCallback.register('save_epoch_state')
class SaveEpochStateCallback(TrainerCallback):
    def __init__(self,
                 serialization_dir,
                 save_epoch_points: List[int] = []):
        super().__init__(serialization_dir)
        self._serial_dir = serialization_dir
        self.save_epoch_points = save_epoch_points

    def on_epoch(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any],
        epoch: int,
        is_primary: bool = True,
        **kwargs,
    ) -> None:
        if epoch in self.save_epoch_points:
            weight_file_name = f'state_epoch_{epoch}.th'
            weight_file_path = os.path.join(self.serialization_dir, weight_file_name)
            trainer._save_model_state(weight_file_path)
            print(f'[SaveEpochStateCallback] Save weight of {epoch}-th to {weight_file_path}')


@TrainerCallback.register('log_grad_norm')
class LogGradNormCallback(TrainerCallback):
    def __init__(self,
                 serialization_dir,
                 grad_norm_stat_period: int = 50):
        super().__init__(serialization_dir)
        self._serial_dir = serialization_dir
        self.grad_norm_stat_period = grad_norm_stat_period
        self.norms = []


    def on_batch(
        self,
        trainer: "GradientDescentTrainer",
        batch_inputs: List[TensorDict],
        batch_outputs: List[Dict[str, Any]],
        batch_metrics: Dict[str, Any],
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_primary: bool = True,
        batch_grad_norm: Optional[float] = None,
        **kwargs,
    ) -> None:
        assert batch_grad_norm is not None
        self.norms.append(batch_grad_norm)
        if len(self.norms) == self.grad_norm_stat_period:
            parameters_to_clip = [p for p in trainer.model.parameters() if p.grad is not None]
            grads = torch.stack([torch.norm(p.grad.detach()) for p in parameters_to_clip])
            grad_max = max(grads) / self.grad_norm_stat_period
            grad_median = torch.median(grads) / self.grad_norm_stat_period
            mylogger.debug('LogGradNormCallback',
                           f'avg grad_norm = {"%.4f" % (sum(self.norms) / len(self.norms) /len(parameters_to_clip))}, ' +
                           f'max = {"%.4f" % grad_max.item()}, ' +
                           f'median = {"%.4f" % grad_median.item()}')
            self.norms.clear()

from utils import GlobalLogger as mylogger

@TrainerCallback.register('decay_sampling')
class DecaySamplingCallback(TrainerCallback):
    def __init__(self,
                 serialization_dir = None,
                 miu: float = 12.):
        super().__init__(serialization_dir)
        self._serial_dir = serialization_dir
        self.miu = miu

    def on_start(
        self, trainer: "GradientDescentTrainer", is_primary: bool = True, **kwargs
    ) -> None:
        epoch = 0
        m = self.miu
        s_epoch = float(epoch // 4)
        p = m / (m + math.exp(s_epoch / m))
        trainer.model.set_decay_sampling_p(p)

        mylogger.debug('DecaySamplingCallback', f'Epoch={epoch}, p = {p}')


    def on_epoch(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any],
        epoch: int,
        is_primary: bool = True,
        **kwargs,
    ) -> None:
        epoch += 1  # Fot next batch
        m = self.miu
        s_epoch = float(epoch // 4)
        p = m / (m + math.exp(s_epoch / m))
        trainer.model.set_decay_sampling_p(p)

        mylogger.debug('DecaySamplingCallback', f'Epoch={epoch}, p = {p}')

from utils.allennlp_utils.load_utils import partial_load_state_dict

@TrainerCallback.register('partial_load_state_dict')
class PartialLoadStateDictCallback(TrainerCallback):
    def __init__(self,
                 load_state_dict_path: str,
                 load_prefix_remap: Dict = {},  # Note this map is "mapping name of model parameter to match state dict"
                 serialization_dir=None):
        super().__init__(serialization_dir)
        self.load_state_dict_path = load_state_dict_path
        self.load_prefix_remap = load_prefix_remap

    def on_start(self,
                 trainer: "GradientDescentTrainer",
                 is_primary: bool = True,
                 **kwargs) -> None:
        model = trainer.model
        state_dict = torch.load(self.load_state_dict_path, map_location=trainer.cuda_device)
        partial_load_state_dict(model, state_dict, self.load_prefix_remap)


@TrainerCallback.register('model_epoch_increment')
class ModelEpochIncrementCallback(TrainerCallback):
    def on_epoch(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any],
        epoch: int,
        is_primary: bool = True,
        **kwargs,
    ) -> None:
        trainer.model.cur_epoch += 1

@TrainerCallback.register('add_pretraiend_vocab_tokens')
class AddPretrainedVocabTokensCallback(TrainerCallback):
    def __init__(self,
                 pretrained_config_path: str,
                 namespace: str,
                 serialization_dir=None):
        super().__init__(serialization_dir)
        self.namespace = namespace
        self.pretrained_config_path = pretrained_config_path

    def on_start(self,
                 trainer: "GradientDescentTrainer",
                 is_primary: bool = True,
                 **kwargs) -> None:
        pretrained_config = read_dumped(self.pretrained_config_path)
        vocab_to_load = pretrained_config['model']['vocab']
        tokens_to_add = [t for t in vocab_to_load]
        print(f"[AddPretrainedVocabTokensCallback] Loading {len(tokens_to_add)} tokens from pretrained config: {self.pretrained_config_path}")
        trainer.model.vocab.add_tokens_to_namespace(tokens_to_add, self.namespace)
        print(f'[AddPretrainedVocabTokensCallback] Current vocab: {trainer.model.vocab}')


