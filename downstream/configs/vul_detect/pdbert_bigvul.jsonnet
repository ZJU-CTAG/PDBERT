local data_base_path = '../data/datasets/extrinsic/vul_detect/bigvul/';
local pretrained_model = '../data/models/pdbert-base';
local code_embed_dim = 768;
local code_encode_dim = 768;
local code_out_dim = 768;

local code_max_tokens = 512;
local code_namespace = "code_tokens";
local tokenizer_type = "codebert";

local mlm_mask_token = "<MLM>";
local additional_special_tokens = [mlm_mask_token];     # Add this special token to avoid embedding size mismatch
local cuda_device = 2;

# -config configs/vul_detect/pdbert_bigvul.jsonnet -task_name vul_detect/bigvul -average binary

local debug = false;

{
    dataset_reader: {
        type: "func_vul_detect_base",
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
        func_code_field_key: "code",
        vul_label_field_key: "vul",
        code_max_tokens: code_max_tokens,
        code_namespace: code_namespace,
        code_cleaner: { type: "space_sub"},
        tokenizer_type: tokenizer_type,

        debug: debug
    },

    train_data_path: data_base_path + "train.json",
    validation_data_path: data_base_path + "validate.json",

    model: {
        type: "vul_func_predictor",
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
            type: "bce"
        },
        classifier: {
            type: "linear_sigmoid",
            in_feature_dim: code_out_dim,
            hidden_dims: [256, 128],
            activations: ["relu", "relu"],
            dropouts: [0.3, 0.3],
            ahead_feature_dropout: 0.3,
        },
        metric: {
            type: "f1",
            positive_label: 1,
        },
    },

  data_loader: {
    batch_size: 16,
    shuffle: true,
  },
  validation_data_loader: {
    batch_size: 64,
    shuffle: true,
  },

  trainer: {
    num_epochs: 10,
    patience: null,
    cuda_device: cuda_device,
    validation_metric: "+f1",
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