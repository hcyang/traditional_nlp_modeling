# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
This module uses a HuggingfaceDatasetWrapperInputTextFiles
and train a masked language model.
"""

# from typing import Any
from typing import List
from typing import Union

import argparse

import os

# import torch

import datasets

from transformers \
    import BertTokenizer

from transformers \
    import DataCollatorForLanguageModeling

from transformers \
    import TrainingArguments
from transformers \
    import Trainer

# from transformers \
#     import BertModel
from transformers \
    import BertForMaskedLM
# from transformers \
#     import BertForPreTraining

from data.pytorch.datasets_huggingface.dataset_wrapper_input_text_files \
    import HuggingfaceDatasetWrapperInputTextFiles

from data.pytorch.datasets_huggingface.huggingface_datasets_utility \
    import HuggingfaceDatasetsUtility

from model.language_understanding.helper.pytorch_language_understanding_transformers_helper \
    import PytorchLanguageUnderstandingTransformersPretainedModelHelper

from utility.pytorch_utility.pytorch_utility \
    import PytorchUtility

from utility.datatype_helper.datatype_helper \
    import DatatypeHelper
from utility.debugging_helper.debugging_helper \
    import DebuggingHelper
# from utility.configuration_helper.configuration_helper \
#     import ConfigurationHelper

def process_app_pretrain_masked_language_model_arguments(parser):
    """
    To process data manager related arguments.
    """
    # DebuggingHelper.write_line_to_system_console_out(
    #     f'Calling process_app_pretrain_masked_language_model_arguments() in {__name__}')
    if parser is None:
        DebuggingHelper.throw_exception(
            'input argument, parser, is None')
    # ------------------------------------------------------------------------
    parser.add_argument(
        '--transformers_model_name',
        type=str,
        required=False,
        default='bert-base-multilingual-uncased',
        help='transformers model name.')
    parser.add_argument(
        '--transformers_model_folder',
        type=str,
        required=False,
        default='pyludispatch/_models/pytorch_transformers',
        help='transformers model folder.')
    parser.add_argument(
        '--transformers_tokenizer_folder',
        type=str,
        required=False,
        default='pyludispatch/_tokenizers/pytorch_transformers',
        help='transformers tokenizer folder.')
    # ------------------------------------------------------------------------
    # ---- NOTE-NO-NEED ---- parser.add_argument(
    # ---- NOTE-NO-NEED ----     '--name',
    # ---- NOTE-NO-NEED ----     type=str,
    # ---- NOTE-NO-NEED ----     required=True,
    # ---- NOTE-NO-NEED ----     help='dataset name.')
    parser.add_argument(
        '--input_path',
        type=str,
        required=True,
        help='input path.')
    # ---- NOTE-USE-input_path-INSTEAD ---- parser.add_argument(
    # ---- NOTE-USE-input_path-INSTEAD ----     '--input_text_files',
    # ---- NOTE-USE-input_path-INSTEAD ----     type=str,
    # ---- NOTE-USE-input_path-INSTEAD ----     required=True,
    # ---- NOTE-USE-input_path-INSTEAD ----     help='input text files.')
    # ------------------------------------------------------------------------
    parser.add_argument(
        '--output_model_checkpoint_path',
        type=str,
        required=True,
        help='output model checkpoint path.')
    # ------------------------------------------------------------------------
    parser.add_argument(
        '--trainer_argument_use_cuda',
        type=DatatypeHelper.to_bool,
        required=False,
        default=False,
        help='use cuda or not, default to True')
    parser.add_argument(
        '--trainer_argument_num_train_epochs',
        type=int,
        required=False,
        default=4,
        help='number of training epochs')
    parser.add_argument(
        '--trainer_argument_per_device_train_batch_size',
        type=int,
        required=False,
        default=8,
        help='per device train batch size')
    parser.add_argument(
        '--trainer_argument_gradient_accumulation_steps',
        type=int,
        required=False,
        default=4,
        help='gradient accumulation steps')
    parser.add_argument(
        '--trainer_argument_per_device_eval_batch_size',
        type=int,
        required=False,
        default=8,
        help='per device eval batch size')
    parser.add_argument(
        '--trainer_argument_logging_steps',
        type=int,
        required=False,
        default=4,
        help='logging steps')
    parser.add_argument(
        '--trainer_argument_save_steps',
        type=int,
        required=False,
        default=8,
        help='save size')
    parser.add_argument(
        '--trainer_argument_save_total_limit',
        type=int,
        required=False,
        default=4,
        help='save total limit')
    return parser

# ---- NOTE-PYLINT ---- C0103: Function name "" doesn't conform to snake_case naming style
# pylint: disable=C0103
def example_function_app_pretrain_masked_language_model():
    """
    The main function to quickly test HuggingfaceDatasetWrapperInputTextFiles.

    REFERENCE:
        https://www.thepythoncode.com/article/pretraining-bert-huggingface-transformers-in-python
    """
    # ------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    # ------------------------------------------------------------------------
    process_app_pretrain_masked_language_model_arguments(parser)
    # ------------------------------------------------------------------------
    args: argparse.Namespace = parser.parse_args()
    DebuggingHelper.write_line_to_system_console_out(
        f'args={str(args)}')
    # ------------------------------------------------------------------------
    transformers_model_folder: str = \
        os.path.normpath(args.transformers_model_folder)
    transformers_tokenizer_folder: str = \
        os.path.normpath(args.transformers_tokenizer_folder)
    # ------------------------------------------------------------------------
    input_path: str = \
        os.path.normpath(args.input_path)
    output_model_checkpoint_path: str = \
        os.path.normpath(args.output_model_checkpoint_path)
    # ------------------------------------------------------------------------
    trainer_argument_use_cuda: bool = \
        args.trainer_argument_use_cuda
    trainer_argument_num_train_epochs: int = \
        args.trainer_argument_num_train_epochs
    trainer_argument_per_device_train_batch_size: int = \
        args.trainer_argument_per_device_train_batch_size
    trainer_argument_gradient_accumulation_steps: int = \
        args.trainer_argument_gradient_accumulation_steps
    trainer_argument_per_device_eval_batch_size: int = \
        args.trainer_argument_per_device_eval_batch_size
    trainer_argument_logging_steps: int = \
        args.trainer_argument_logging_steps
    trainer_argument_save_steps: int = \
        args.trainer_argument_save_steps
    trainer_argument_save_total_limit: int = \
        args.trainer_argument_save_total_limit
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    os.environ["WANDB_DISABLED"] = "true"
    # ---- NOTE-PYLINT ---- C0301: Line too long
    # pylint: disable=C0301
    # ---- NOTE-FOR-REFERENCE ---- ***** Running training *****
    # ---- NOTE-FOR-REFERENCE ----   Num examples = 45
    # ---- NOTE-FOR-REFERENCE ----   Num Epochs = 10
    # ---- NOTE-FOR-REFERENCE ----   Instantaneous batch size per device = 10
    # ---- NOTE-FOR-REFERENCE ----   Total train batch size (w. parallel, distributed & accumulation) = 80
    # ---- NOTE-FOR-REFERENCE ----   Gradient Accumulation steps = 8
    # ---- NOTE-FOR-REFERENCE ----   Total optimization steps = 10
    # ---- NOTE-FOR-REFERENCE ---- Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"
    # ---- NOTE-FOR-REFERENCE ---- wandb: (1) Create a W&B account
    # ---- NOTE-FOR-REFERENCE ---- wandb: (2) Use an existing W&B account
    # ---- NOTE-FOR-REFERENCE ---- wandb: (3) Don't visualize my results
    # ---- NOTE-FOR-REFERENCE ---- wandb: Enter your choice: 3
    # ---- NOTE-FOR-REFERENCE ---- wandb: You chose 'Don't visualize my results'
    # ---- NOTE-FOR-REFERENCE ---- wandb: W&B syncing is set to `offline` in this directory.  Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.
    # ------------------------------------------------------------------------
    PytorchUtility.set_all_random_number_generator_seeds(47)
    # ------------------------------------------------------------------------
    pytorch_transformers_model_name: str = \
        args.transformers_model_name
    # ------------------------------------------------------------------------
    # ---- NOTE-PYLINT ---- C0301: Line too long
    # pylint: disable=C0301
    pytorch_transformers_model_cache_dir_model: str = \
        os.path.join(os.path.os.sep, transformers_model_folder, pytorch_transformers_model_name, 'model')
    DebuggingHelper.write_line_to_system_console_out(
        '==== pytorch_transformers_model_cache_dir_model:"{}"'.format(pytorch_transformers_model_cache_dir_model))
    pytorch_transformers_model: BertForMaskedLM = \
        BertForMaskedLM.from_pretrained(
            pretrained_model_name_or_path=pytorch_transformers_model_cache_dir_model,
            cache_dir=pytorch_transformers_model_cache_dir_model)
    # ------------------------------------------------------------------------
    # ---- NOTE-PYLINT ---- C0301: Line too long
    # pylint: disable=C0301
    pytorch_transformers_model_cache_dir_tokenizer: str = \
        os.path.join(os.path.os.sep, transformers_tokenizer_folder, pytorch_transformers_model_name)
    pytorch_transformers_model_tokenizer_do_lower_case: bool = True
    DebuggingHelper.write_line_to_system_console_out(
        '==== pytorch_transformers_model_cache_dir_tokenizer:"{}"'.format(pytorch_transformers_model_cache_dir_tokenizer))
    pytorch_transformers_tokenizer: BertTokenizer = \
        BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=pytorch_transformers_model_cache_dir_tokenizer,
            cache_dir=pytorch_transformers_model_cache_dir_tokenizer,
            do_lower_case=pytorch_transformers_model_tokenizer_do_lower_case)
    if pytorch_transformers_tokenizer is None:
        DebuggingHelper.throw_exception(
            f'Could not deserialize a '
            f'BertTokenizer object from '
            f'pretrained_model_name_or_path={pytorch_transformers_model_cache_dir_tokenizer}'
            f',cache_dir={pytorch_transformers_model_cache_dir_tokenizer}'
            f',do_lower_case={pytorch_transformers_model_tokenizer_do_lower_case}')
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    input_text_files: List[str] = \
        HuggingfaceDatasetWrapperInputTextFiles.scan_input_text_files_folder( \
            input_text_files_folder=input_path)
    DebuggingHelper.write_line_to_system_console_out(
        '==== input_text_files:"{}"'.format(input_text_files))
    # ------------------------------------------------------------------------
    dataset_wrapper: HuggingfaceDatasetWrapperInputTextFiles = HuggingfaceDatasetWrapperInputTextFiles(
        huggingface_datasets_argument_input_text_files=input_text_files)
    dataset: Union[datasets.DatasetDict, datasets.Dataset, datasets.IterableDatasetDict, datasets.IterableDataset] = \
        dataset_wrapper.get_dataset()
    DebuggingHelper.write_line_to_system_console_out(
        f'dataset={dataset}')
    # ------------------------------------------------------------------------
    train_test_datasets: datasets.DatasetDict = \
        HuggingfaceDatasetsUtility.split_to_train_test_sets(dataset)
    DebuggingHelper.write_line_to_system_console_out(
        f'train_test_datasets={train_test_datasets}')
    train_dataset: datasets.Dataset = \
        train_test_datasets["train"]
    test_dataset: datasets.Dataset = \
        train_test_datasets["test"]
    DebuggingHelper.write_line_to_system_console_out(
        f'train_dataset={train_dataset}')
    DebuggingHelper.write_line_to_system_console_out(
        f'test_dataset={test_dataset}')
    # ---- train_dataset_text_feature: List[str] = \
    # ----     train_dataset['text']
    # ---- test_dataset_text_feature: List[str] = \
    # ----     test_dataset['text']
    # ---- DebuggingHelper.write_line_to_system_console_out(
    # ----     f'train_dataset_text_feature={train_dataset_text_feature}')
    # ---- DebuggingHelper.write_line_to_system_console_out(
    # ----     f'test_dataset_text_feature={test_dataset_text_feature}')
    # ---- NOTE-FOR-DEBUGGING ---- train_dataset_text_feature_tokenized = \
    # ---- NOTE-FOR-DEBUGGING ----     HuggingfaceDatasetsUtility.tokenize_dataset_text_feature( \
    # ---- NOTE-FOR-DEBUGGING ----         tokenizer=pytorch_transformers_tokenizer, \
    # ---- NOTE-FOR-DEBUGGING ----         dataset=train_dataset)
    # ---- NOTE-FOR-DEBUGGING ---- test_dataset_text_feature_tokenized = \
    # ---- NOTE-FOR-DEBUGGING ----     HuggingfaceDatasetsUtility.tokenize_dataset_text_feature( \
    # ---- NOTE-FOR-DEBUGGING ----         tokenizer=pytorch_transformers_tokenizer, \
    # ---- NOTE-FOR-DEBUGGING ----         dataset=test_dataset)
    # ---- NOTE-FOR-DEBUGGING ---- # ---- DebuggingHelper.write_line_to_system_console_out(
    # ---- NOTE-FOR-DEBUGGING ---- # ----     f'train_dataset_text_feature_tokenized={train_dataset_text_feature_tokenized}')
    # ---- NOTE-FOR-DEBUGGING ---- # ---- DebuggingHelper.write_line_to_system_console_out(
    # ---- NOTE-FOR-DEBUGGING ---- # ----     f'test_dataset_text_feature_tokenized={test_dataset_text_feature_tokenized}')
    train_dataset_tokenized: datasets.Dataset = \
        HuggingfaceDatasetsUtility.tokenize_dataset( \
            tokenizer=pytorch_transformers_tokenizer, \
            dataset=train_dataset)
    test_dataset_tokenized: datasets.Dataset = \
        HuggingfaceDatasetsUtility.tokenize_dataset( \
            tokenizer=pytorch_transformers_tokenizer, \
            dataset=test_dataset)
    DebuggingHelper.write_line_to_system_console_out(
        f'train_dataset_tokenized={train_dataset_tokenized}')
    DebuggingHelper.write_line_to_system_console_out(
        f'test_dataset_tokenized={test_dataset_tokenized}')
    # ------------------------------------------------------------------------
    data_collator: DataCollatorForLanguageModeling = \
        PytorchLanguageUnderstandingTransformersPretainedModelHelper.create_data_collator_for_language_modeling( \
            tokenizer=pytorch_transformers_tokenizer)
    DebuggingHelper.write_line_to_system_console_out(
        f'data_collator={data_collator}')
    # ------------------------------------------------------------------------
    training_args = TrainingArguments(
        no_cuda=not trainer_argument_use_cuda,
        output_dir=output_model_checkpoint_path,                                    # ---- output directory for saving model checkpoint
        evaluation_strategy="steps",                                                # ---- "steps" for evaluate each `logging_steps`
        overwrite_output_dir=True,                                                  # ---- whether to overrite output directory
        num_train_epochs=trainer_argument_num_train_epochs,                         # ---- number of training epochs, feel free to tweak (10)
        per_device_train_batch_size=trainer_argument_per_device_train_batch_size,   # ---- the training batch size, put it as high as your GPU memory fits (10)
        gradient_accumulation_steps=trainer_argument_gradient_accumulation_steps,   # ---- accumulating the gradients before updating the weights (8)
        per_device_eval_batch_size=trainer_argument_per_device_eval_batch_size,     # ---- evaluation batch size (64)
        logging_steps=trainer_argument_logging_steps,                               # ---- evaluate, log and save model checkpoints every (500) step
        save_steps=trainer_argument_save_steps,                                     # ---- save the checkpoint for number of steps (500)
        load_best_model_at_end=True,                                                # ---- whether to load the best model (in terms of loss) at the end of training
        save_total_limit=trainer_argument_save_total_limit,                         # ---- whether you don't have much space so you let only 3 model weights saved in the disk
    )
    trainer = Trainer(
        model=pytorch_transformers_model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset_tokenized,
        eval_dataset=test_dataset_tokenized)
    trainer.train()
    # ------------------------------------------------------------------------

def main():
    """
    The main() function can quickly test a HuggingfaceDatasetWrapperInputTextFiles object
    and train a masked language model.
    """
    example_function_app_pretrain_masked_language_model()

if __name__ == '__main__':
    main()
