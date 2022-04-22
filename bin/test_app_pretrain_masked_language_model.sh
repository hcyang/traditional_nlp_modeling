#!/usr/bin/sh

export APP_MODULE_PATH=model.language_understanding.unsupervised.neural_network.learner_pretrain
export APP_MODULE_FILE_NAME=app_pretrain_masked_language_model

export DATA_ROOT=/home/hcyang

export TRAINER_ARGUMENT_USE_CUDA=False

export INPUT_PATH=$DATA_ROOT/data/text/us_presidential_speeches/text_files
export OUTPUT_MODEL_CHECKPOINT_PATH=$DATA_ROOT/data/text/us_presidential_speeches/model_checkpoints
export INPUT_PATH=$DATA_ROOT/data_wikipedia/_dumps.wikimedia.org_zhwiki_20220401_output
export OUTPUT_MODEL_CHECKPOINT_PATH=$DATA_ROOT/data_wikipedia/_dumps.wikimedia.org_zhwiki_20220401_model_checkpoints

python -m $APP_MODULE_PATH.$APP_MODULE_FILE_NAME --trainer_argument_use_cuda=$TRAINER_ARGUMENT_USE_CUDA --input_path=$INPUT_PATH --output_model_checkpoint_path=$OUTPUT_MODEL_CHECKPOINT_PATH


# ---- curl https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip
# ---- curl https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
# ---- curl https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip