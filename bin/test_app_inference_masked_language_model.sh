#!/usr/bin/sh

export APP_MODULE_PATH=model.language_understanding.unsupervised.neural_network.learner_pretrain
export APP_MODULE_FILE_NAME=app_inference_pretrained_masked_language_model

export DATA_ROOT=/home/hcyang

export TRAINER_ARGUMENT_USE_CUDA=False

export INPUT_FILE=$DATA_ROOT/data_wikipedia/dumps.wikimedia.org_zhwiki_20211020_model_checkpoints/input.txt

export OUTPUT_MODEL_CHECKPOINT_PATH=$DATA_ROOT/data_wikipedia/dumps.wikimedia.org_zhwiki_20211020_model_checkpoints/checkpoint-60
export OUTPUT_FILE=$DATA_ROOT/data_wikipedia/dumps.wikimedia.org_zhwiki_20211020_model_checkpoints/checkpoint-60-output.txt

python -m $APP_MODULE_PATH.$APP_MODULE_FILE_NAME --trainer_argument_use_cuda=$TRAINER_ARGUMENT_USE_CUDA --transformers_model_folder=$OUTPUT_MODEL_CHECKPOINT_PATH --output_file=$OUTPUT_FILE --input_file=$INPUT_FILE

export OUTPUT_MODEL_CHECKPOINT_PATH=$DATA_ROOT/pyludispatch/_models/pytorch_transformers/bert-base-multilingual-uncased/model
export OUTPUT_FILE=$DATA_ROOT/data_wikipedia/dumps.wikimedia.org_zhwiki_20211020_model_checkpoints/original-output.txt

python -m $APP_MODULE_PATH.$APP_MODULE_FILE_NAME --trainer_argument_use_cuda=$TRAINER_ARGUMENT_USE_CUDA --transformers_model_folder=$OUTPUT_MODEL_CHECKPOINT_PATH --output_file=$OUTPUT_FILE --input_file=$INPUT_FILE
