import os
import re
from typing import List
import torch
from tqdm import tqdm
import numpy
import sys
import pprint

sys.path.append("../")

from allennlp.common import JsonDict
from allennlp.data import Instance, Token
from allennlp.predictors import Predictor
from allennlp.models import Model
from allennlp.data.tokenizers import PretrainedTransformerTokenizer

from common.modules.code_cleaner import TrivialCodeCleaner
from pretrain import *
from common import *
from utils.file import read_dumped, dump_json
from utils.allennlp_utils.build_utils import build_dataset_reader_from_config
from utils.pretrain_utils.format import convert_func_signature_to_one_line
from utils.pretrain_utils.mat import remove_consecutive_lines, shift_edges_in_matrix


class PDGPredictor(Predictor):
    def predict_pdg(self, code: str):
        instance = self._json_to_instance({
            'raw_code': code
        })
        return self.predict_instance(instance)

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        ok, instance = self._dataset_reader.text_to_instance(json_dict)
        if not ok:
            raise ValueError
        else:
            return instance


def set_reader(_reader, _max_lines):
    _reader.is_train = False
    _reader.code_cleaner = TrivialCodeCleaner()     # To avoid multiple nl elimination, may affect performance
    return _reader


def get_line_count_from_tokens(raw_code: str, tokens: List[Token]):
    """
        Compute the line count from tokens, where raw_code is complete but tokens are partial.
    """
    # Find the char indices of new-lines
    new_line_indices = []
    for m in re.finditer('\n', raw_code):
        new_line_indices.append(m.start())
    # Add a dummy nl at last to avoid out-of-bound
    new_line_indices.append(1e10)

    cur_line = 0
    cur_nl_idx = 0
    for i, t in enumerate(tokens):
        if t.idx is None:
            continue
        while t.idx <= new_line_indices[cur_nl_idx] <= t.idx_end:
            cur_line += 1
            cur_nl_idx += 1
    return cur_line


def split_partial_code(code: str, n_line: int):
    code_lines = code.split("\n")
    return '\n'.join(code_lines[:n_line])


def predict_one_file(code, ctrl_edges, data_edges, n):
    code_snippet = split_partial_code(code, n)
    pdg_output = predictor.predict_pdg(code_snippet)
    ctrl_pred, data_pred = pdg_output['ctrl_edge_labels'], pdg_output['data_edge_labels']
    ctrl_pred = torch.IntTensor(ctrl_pred).flatten().tolist()
    data_pred = torch.IntTensor(data_pred).flatten().tolist()

    ctrl_label, data_label, line_count = dataset_reader.process_test_labels(code_snippet, ctrl_edges, data_edges)
    # Minus one to revert the real matrix.
    ctrl_label = (ctrl_label-1).flatten().tolist()
    data_label = (data_label-1).flatten().tolist()

    if line_count <= n:
        return ctrl_pred, data_pred, ctrl_label, data_label
    else:
        return [], [], [], []


def process_my_code(code: str):
    """
        Since in the original tokenization system, multiple new lines were not properly
        handled causing line count error.
        Here we insert a space into them to prevent from this special case.
    """
    if code[-1] != '\n':
        return code + '\n'
    else:
        return code


def process_result_as_conf_matrix(predicts, labels):
    """
    Return: TN, FN, FP, TP
    """
    p_tensor = numpy.array(predicts, dtype=int)
    l_tensor = numpy.array(labels, dtype=int)
    indices = numpy.arange(len(predicts))
    m = numpy.zeros((len(predicts), 2, 2), dtype=int)
    m[indices, p_tensor, l_tensor] = 1
    return m.sum(0)


def cal_f1_from_conf_matrix(conf_m):
    TN, FN, FP, TP = conf_m.flatten()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return 2*precision*recall / (precision + recall)

#---------------------------------------------------------------------------

cuda_device = 0
model_path = f'../data/models/intrinsic/pdbert_base_with_header.tar.gz'
config_path = f'../data/models/intrinsic/config.json'
tokenizer_name = '../data/models/pdbert-base/'

max_lines = 50
full_Ns = [10, 20, 30, 10000]


#---------------------------------------------------------------------------

torch.cuda.set_device(cuda_device)
print('\n\n')
print(f'[main] Building model from: {model_path}\n')
model = Model.from_archive(model_path)
model = model.cuda(cuda_device)
print(f'[main] Building reader from: {config_path}\n')
dataset_reader = build_dataset_reader_from_config(config_path)
dataset_reader = set_reader(dataset_reader, max_lines)
predictor = PDGPredictor(model, dataset_reader, frozen=True)

#---------------------------------------------------------------------------

full_ctrl_results = {k:numpy.zeros((2, 2), dtype=int) for k in full_Ns}
full_data_results = {k:numpy.zeros((2, 2), dtype=int) for k in full_Ns}
full_counts = {n:0 for n in full_Ns}

#---------------------------------------------------------------------------

def eval_full_func_for_my_data(data_base_path,
                               file_name_temp,
                               vol_range,
                               max_tokens=512,
                               rm_consecutive_nls=False):
    print("Building components...")
    svol, evol = vol_range
    vols = list(range(svol, evol+1))
    data_edges_pretokenized_name = 'pdbert-base'

    # No inner truncation to identify full function
    tokenizer = PretrainedTransformerTokenizer(tokenizer_name)
    results = {}
    total_instance = 0

    for vol in vols:
        test_file_path = data_base_path + file_name_temp.format(vol)
        print(f'Eval on Vol.{vol} ...')
        data_items = read_dumped(test_file_path)
        total_instance += len(data_items)
        for i, data_item in tqdm(enumerate(data_items), total=len(data_items)):
            raw_code = data_item['raw_code']
            raw_code = process_my_code(convert_func_signature_to_one_line(code=raw_code))
            if rm_consecutive_nls:
                raw_code, del_line_indices = remove_consecutive_lines(raw_code)
            else:
                del_line_indices = None
            tokens = tokenizer.tokenize(raw_code)
            # Not full within max_tokens, skip
            if len(tokens) > max_tokens:
                continue
            max_lines = get_line_count_from_tokens(raw_code, tokens)

            cdg_edges = data_item['line_edges']
            if rm_consecutive_nls:
                cdg_edges = shift_edges_in_matrix(cdg_edges, del_line_indices)
            ddg_edges = data_item['processed_token_data_edges'][data_edges_pretokenized_name]

            for n in full_Ns:
                # Only work on the first n larger than max_lines, to locale range
                if n >= max_lines:
                    try:
                        # Set n=max_lines to reuse utils
                        cdg_preds, ddg_preds, cdg_labels, ddg_labels = predict_one_file(raw_code, cdg_edges, ddg_edges, max_lines)
                    except Exception as e:
                        print(f"Error when predicting #{i}, n={n}, err_type: {type(e)}, err: {e}, skipped")
                        continue

                    assert len(cdg_preds) == len(cdg_labels), \
                           f"CDG: pred ({len(cdg_preds)}) != label ({len(cdg_labels)}). \n- Code: {raw_code}"
                    assert len(ddg_preds) == len(ddg_labels), \
                           f"DDG: pred ({len(ddg_preds)}) != label ({len(ddg_labels)}). \n- Code: {raw_code}"

                    ctrl_res_m = process_result_as_conf_matrix(cdg_preds, cdg_labels)
                    data_res_m = process_result_as_conf_matrix(ddg_preds, ddg_labels)
                    full_ctrl_results[n] += ctrl_res_m
                    full_data_results[n] += data_res_m
                    full_counts[n] += 1
                    break

    for n in full_Ns:
        c_pairs = int(full_ctrl_results[n].sum())
        d_pairs = int(full_data_results[n].sum())
        c_f1 = cal_f1_from_conf_matrix(full_ctrl_results[n]) if c_pairs > 0 else None
        d_f1 = cal_f1_from_conf_matrix(full_data_results[n]) if d_pairs > 0 else None
        overall_f1 = cal_f1_from_conf_matrix(full_ctrl_results[n] + full_data_results[n]) if c_pairs + d_pairs > 0 else None
        n_result = {
            'total_instance': total_instance,
            'total_valid_instance': full_counts[n],
            'c_pairs': c_pairs,
            'd_pairs': d_pairs,
            'ctrl_f1': c_f1 ,
            'data_f1': d_f1,
            'overall_f1': overall_f1
        }
        results[n] = n_result
        print("\n" + '*' * 50)
        print(f"N = {n}")
        print(n_result)

    print("\n\n" + "*" * 75)
    pprint.pp('Intrinsic - Full function PDG prediction')
    print(results)
    print("*" * 75)

if __name__ == '__main__':
    eval_full_func_for_my_data(data_base_path="../data/datasets/intrinsic/",
                               file_name_temp="packed_hybrid_vol_{}.pkl",
                               vol_range=(221228, 221228),
                               max_tokens=512,
                               rm_consecutive_nls=False)
