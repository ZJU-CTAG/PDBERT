
# README
This is the replication package of paper "Pre-training by Predicting Program Dependencies for Vulnerability Analysis Tasks".

Our datasets and online appendix can be found [here](https://zenodo.org/records/10140638).

## Requirements
### Softwares
- CUDA: 11.1
- torch==1.10.2+cu111
- allennlp==2.8.0
- allennlp_models==2.8.0
- transformers==4.12.5
- numpy==1.22.4
- scipy==1.8.1
- torchtext==0.11.2
- torchvision==0.11.3+cu111

## Hardwares (opt.)
- GPU: RTX 3090 24GB

## Organization of the Replication Package
- `common`: Common modules for both pre-training and downstream tasks.
- `data`: Datasets and saved models.
  - `datasets`: Datasets for experiments.
  - `models`: Saved models for intrinsic evaluation and extrinsic evaluation (fine-tuned).
- `dist_importing`: For importing modules when dist_training is enabled for allennlp.
- `downstreams`: Modules, scripts and configs for downstreams tasks (extrinsic evaluation).
- `pretrain`: Modules, scripts and configs for pre-training task (intrinsic evaluation).
- `utils`: Utility functions.

## How to Run
### Intrinsicn Evaluation
To obtain the results of Table 1 & Table 2 in our paper.
1. Go to the `pretrain` folder (**This is important for relative path retrieving**).
2. For partial code intrinsic evaluation results in Table 1, run: ```python eval_partial_func_pdg.py```
3. For full function only intrinsic evaluation results in Table 2, run: ```python eval_full_func_pdg.py```

#### Note
- we convert the test set into one file named `packed_hybrid_vol_221228.pkl`, and the ground truth of control dependency prediction (CDG) and data dependency prediction (DDG) has been constructed based on the outputs of Joern and provided in this file.
- A pre-trained model with CDP and DDP headers has been provided in `models/intrinsic`, but this is only for intrinsic evaluation.

### Extrinsic Evaluation
We use three vulnerability analysis tasks for extrinsic evaluation: vulnerability detection, vulnerability classification and vulnerability assessment.

#### Preparation
To make training and testing as a unified pipeline, you should open `downstream/global_vars.json` to make some configurations. 
In detail, the key of the object in `downstream/global_vars.json` should be the name of your machine (run Python command ```import platform; print(platform.node())``` to check), and the `python_bin` should be the path your Python binary located.

#### Vulnerability Detection
1. Go to `downstream` folder (**This is important for relative path retrieving**).
2. For three datasets, run:
   - ReVeal: ```python train_eval_from_config.py -config configs/vul_detect/pdbert_reveal.jsonnet -task_name vul_detect/reveal -average binary```
   - Devign: ```python train_eval_from_config.py -config configs/vul_detect/pdbert_devign.jsonnet -task_name vul_detect/devign -average binary```
   - BigVul: ```python train_eval_from_config.py -config configs/vul_detect/pdbert_bigvul.jsonnet -task_name vul_detect/bigvul -average binary```

#### CWE Classification
1. Go to `downstream` folder (**This is important for relative path retrieving**).
2. Run ```python train_eval_from_config.py -config configs/cwe_class/pdbert.jsonnet -task_name cwe_class -average macro -extra_averages weighted```

#### Vulnerability Assessment
1. Go to `downstream` folder (**This is important for relative path retrieving**).
2. Run ```python train_eval_multi_task_from_config.py -config configs/vul_assess/pdbert.jsonnet -task_name vul_assess -extra_eval_configs "{\"task_names\":\"CPL,AVL,CFD,ITG\"}" -eval_script eval_multi_task_classification -average macro -extra_averages weighted```
   
#### Note: 
- If you want to change the configuration of the running task, check `downstream/configs` accordingly.
- GPU running is enabled by default. If you run these experiments on GPU with small memory and encounter "CUDA out of memory" error, try to decrease the `data_loader/batch_size` in the config. But to keep consistent with our configuration, you should correspondingly increase the `trainer/num_gradient_accumulation_steps`, since the real batch size is `batch_size * num_gradient_accumulation_steps`.
- Due to unavaliable network or other connection problem, process will fail sometimes and report errors like "The TLS connection was non-properly terminated" or "Make sure that 'microsoft/codebert-base' is a correct model identifier listed on 'https://huggingface.co/models'". This is because our model was trained based on CodeBERT and transformers need to fetch meta info of CodeBERT from remote. As a solution, you can download the archived CodeBERT model and put it in the right path for local retrieving. Take these steps:
  - Download the CodeBERT model by:
  ```shell
    git lfs install
    git clone https://huggingface.co/microsoft/codebert-base
  ```
  - For intrinsic evaluation, move downloaded CodeBERT directory to "pretrain/microsoft/codebert-base"
  - For extrinsic evaluation, move downloaded CodeBERT directory to "downstreams/microsoft/codebert-base"
