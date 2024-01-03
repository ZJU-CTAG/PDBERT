local data_base_path = '../data/datasets/extrinsic/vul_assess/';
local pretrained_model = '../data/models/pdbert-base';
local code_embed_dim = 768;
local code_encode_dim = 768;
local code_out_dim = 768;

local code_max_tokens = 512;
local code_namespace = "code_tokens";
local tokenizer_type = "codebert";

local mlm_mask_token = "<MLM>";
local additional_special_tokens = [mlm_mask_token];     # Add this special token to avoid embedding size mismatch
local cuda_device = 4;

local debug = false;

local metrics_to_predict = ['Complexity', 'Availability', 'Confidentiality', 'Integrity'];     # Here are keys to fetch metric value from data item
local task_names = ["mCPL", "mAVL", "mCFD", "mITG"];         # Here are nicknames for each metric to print
local metric_label_space_sizes = [3, 3, 3, 3];
local task_num = 4;

# -config configs/vul_assess/pdbert.jsonnet -task_name vul_assess -extra_eval_configs "{\"task_names\":\"CPL,AVL,CFD,ITG\"}" -eval_script eval_multi_task_classification -average macro -extra_averages weighted
{
    dataset_reader: {
        type: "cvss_metric_pred_base",
        metrics_to_predict: metrics_to_predict,
        code_tokenizer: {
            type: "pretrained_transformer",
            model_name: pretrained_model,
            max_length: code_max_tokens,
            tokenizer_kwargs: {
              additional_special_tokens: additional_special_tokens
            }
        },
        code_indexer: {
            type: "pretrained_transformer",
            model_name: pretrained_model,
            namespace: code_namespace,
            tokenizer_kwargs: {
              additional_special_tokens: additional_special_tokens
            }
        },
        code_max_tokens: code_max_tokens,
        code_namespace: code_namespace,
        code_cleaner: { type: "space_sub"},
        tokenizer_type: tokenizer_type,
        model_mode: null,

        debug: debug,
    },

    train_data_path: data_base_path + "train.json",
    validation_data_path: data_base_path + "validate.json",

    model: {
        type: "multi_task_classifier",
        wrapping_dim_for_code: 0,
        code_embedder: {
          token_embedders: {
            code_tokens: {
              type: "pretrained_transformer",
              model_name: pretrained_model,
              train_parameters: true,
              tokenizer_kwargs: {
                additional_special_tokens: additional_special_tokens
             }
            }
          }
        },
        code_encoder: {
            type: "pass_through",
            input_dim: code_embed_dim,
        },
        code_feature_squeezer: {
            type: "cls_pooler",
            embedding_dim: code_embed_dim,
        },
        loss_func: {
            type: "cross_entropy"
        },
        classifiers: [
            {
                type: "linear_softmax",
                out_feature_dim: metric_label_space_sizes[0],
                in_feature_dim: code_out_dim,
                hidden_dims: [128],
                activations: ["relu"],
                dropouts: [0.3],
                ahead_feature_dropout: 0.3,
                return_logits: true,
            },
            {
                type: "linear_softmax",
                out_feature_dim: metric_label_space_sizes[1],
                in_feature_dim: code_out_dim,
                hidden_dims: [128],
                activations: ["relu"],
                dropouts: [0.3],
                ahead_feature_dropout: 0.3,
                return_logits: true,
            },
            {
                type: "linear_softmax",
                out_feature_dim: metric_label_space_sizes[2],
                in_feature_dim: code_out_dim,
                hidden_dims: [128],
                activations: ["relu"],
                dropouts: [0.3],
                ahead_feature_dropout: 0.3,
                return_logits: true,
            },
            {
                type: "linear_softmax",
                out_feature_dim: metric_label_space_sizes[3],
                in_feature_dim: code_out_dim,
                hidden_dims: [128],
                activations: ["relu"],
                dropouts: [0.3],
                ahead_feature_dropout: 0.3,
                return_logits: true,
            },
        ],
        metric: {
            type: "multi_task_multi_class",
            task_num: task_num,
            task_names: task_names,
            average: "macro",
            zero_division: 0,
            f1_only: true
        },
    },

  data_loader: {
    batch_size: 8,
    shuffle: true,
  },
  validation_data_loader: {
    batch_size: 32,
    shuffle: true,
  },

  trainer: {
    num_epochs: 20,
    patience: null,
    cuda_device: cuda_device,
    validation_metric: "+macro_f1_mean",
    optimizer: {
      type: "adam",
      lr: 1e-5
    },
    num_gradient_accumulation_steps: 4,
    callbacks: [
      { type: "epoch_print" },
    ],
    checkpointer: null,     // checkpointer is set to null to avoid saving model state at each episode
  },
}