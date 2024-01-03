local data_base_path = '../data/datasets/extrinsic/cwe_class/';
local pretrained_model = '../data/models/pdbert-base';
local code_embed_dim = 768;
local code_encode_dim = 768;
local code_out_dim = 768;

local code_max_tokens = 512;
local code_namespace = "code_tokens";
local tokenizer_type = "codebert";

local mlm_mask_token = "<MLM>";
local hunk_separator = "<HUNK>";
local additional_special_tokens = [mlm_mask_token, hunk_separator];     # Add this special token to avoid embedding size mismatch
local cuda_device = 3;

local debug = false;

local cwe_label_space = ['CWE-264', 'CWE-119', 'CWE-189', 'CWE-200', 'CWE-20', 'CWE-416', 'CWE-399', 'CWE-125', 'CWE-190', 'CWE-362', 'CWE-476', 'CWE-787', 'CWE-254', 'CWE-284'];  # cwe_id with more than 100 times for Fan's data
local label_space_size = 14;

# -config configs/cwe_class/pdbert.jsonnet -task_name cwe_class -average macro -extra_averages weighted

{
    vocabulary: {
        type: "from_pretrained_transformer",
        model_name: pretrained_model,
        namespace: code_namespace
    },

    dataset_reader: {
        type: "cwe_pred_base",
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
        cwe_label_space: cwe_label_space,
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
        type: "downstream_classifier",
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
        classifier: {
            type: "linear_softmax",
            out_feature_dim: label_space_size,
            in_feature_dim: code_out_dim,
            hidden_dims: [512, 256],
            activations: ["relu", "relu"],
            dropouts: [0.3, 0.3],
            ahead_feature_dropout: 0.3,
            return_logits: true,
        },
        metric: {
            type: "multiclass_classification",
            average: "macro",
            zero_division: 0,
        },
    },

  data_loader: {
    batch_size: 16,
    shuffle: true,
  },
  validation_data_loader: {
    batch_size: 32,
    shuffle: true,
  },

  trainer: {
    num_epochs: 30,
    patience: null,
    cuda_device: cuda_device,
    validation_metric: "+mcc",
    optimizer: {
      type: "adam",
      lr: 1e-5
    },
    num_gradient_accumulation_steps: 2,
    callbacks: [
      { type: "epoch_print" },
      { type: "model_param_stat" },
    ],
    checkpointer: null,     // checkpointer is set to null to avoid saving model state at each episode
  },
}