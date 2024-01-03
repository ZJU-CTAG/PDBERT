import sys
from pprint import pprint
from typing import Tuple, List
from tqdm import tqdm
import json
import platform

import torch
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.models.model import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

sys.path.extend([f'../'])

from downstream import *
from utils import GlobalLogger as mylogger
from utils.allennlp_utils.build_utils import build_dataset_reader_from_config
from utils.file import save_evaluate_results, dump_pred_results
from utils.cmd_args import read_classification_eval_args

args = read_classification_eval_args()
mylogger.info('eval_classification', f"Args: {args}")

data_file_name = args.data_file_name
model_name = args.model_name
cuda_device = args.cuda

data_base_path = args.data_base_path
data_file_path = data_base_path + data_file_name
serialization_dir = args.serial_dir
model_path = serialization_dir + model_name
batch_size = args.batch_size

def predict_on_dataloader(_model, _data_loader) -> Tuple[List, List, List]:
    all_pred = []
    all_ref = []
    all_score = []
    with torch.no_grad():
        _model.eval()
        for i, batch in enumerate(tqdm(_data_loader)):
            outputs = _model(**batch)
            all_pred.extend(outputs['pred'].cpu().detach().tolist())
            all_score.extend(outputs['logits'].cpu().detach().tolist())
            all_ref.extend(batch['label'].cpu().detach().squeeze().tolist())
    return all_ref, all_pred, all_score

dataset_reader = build_dataset_reader_from_config(
    config_path=serialization_dir + 'config.json',
    serialization_dir=serialization_dir
)
model = Model.from_archive(model_path)

if cuda_device != -1:
    model = model.cuda(cuda_device)
    torch.cuda.set_device(cuda_device)

data_loader = MultiProcessDataLoader(dataset_reader,
                                     data_file_path,
                                     shuffle=False,
                                     batch_size=batch_size,
                                     cuda_device=cuda_device)
data_loader.index_with(model.vocab)

# if cuda_device != -1:
#     model = model.cuda(cuda_device)
#     torch.cuda.set_device(cuda_device)

all_ref, all_pred, all_score = predict_on_dataloader(model, data_loader)
result_dict = {
    'Accuracy': accuracy_score(all_ref, all_pred),
    'Precision': precision_score(all_ref, all_pred, average=args.average),
    'Recall': recall_score(all_ref, all_pred, average=args.average),
    'F1-Score': f1_score(all_ref, all_pred, average=args.average),
    'MCC': matthews_corrcoef(all_ref, all_pred),
}

if args.extra_averages is not None:
    extra_average_methods = args.extra_averages.split(',')
    for average in extra_average_methods:
        result_dict[f'{average}_F1'] = f1_score(all_ref, all_pred, average=average)

print('*'*80)
pprint(result_dict)

sys.exit(0)