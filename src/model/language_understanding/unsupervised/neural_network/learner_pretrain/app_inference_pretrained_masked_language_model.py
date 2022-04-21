# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
This module uses a masked language model and do some evaluation.
"""

# from typing import Any
# from typing import List
# from typing import Union

import argparse

import codecs
import os
from sklearn.pipeline import Pipeline

# import torch

# ---- NOTE-NO-NEED-FROM-PRETRAIN ---- import datasets

from transformers \
    import BertTokenizer

from transformers \
    import pipeline

# ---- NOTE-NO-NEED-FROM-PRETRAIN ---- from transformers \
# ---- NOTE-NO-NEED-FROM-PRETRAIN ----     import DataCollatorForLanguageModeling

# ---- NOTE-NO-NEED-FROM-PRETRAIN ---- from transformers \
# ---- NOTE-NO-NEED-FROM-PRETRAIN ----     import TrainingArguments
# ---- NOTE-NO-NEED-FROM-PRETRAIN ---- from transformers \
# ---- NOTE-NO-NEED-FROM-PRETRAIN ----     import Trainer

# from transformers \
#     import BertModel
from transformers \
    import BertForMaskedLM
# from transformers \
#     import BertForPreTraining

# ---- NOTE-PYLINT ---- C0301: Line too long
# pylint: disable=C0301

# ---- NOTE-NO-NEED-FROM-PRETRAIN ---- from data.pytorch.datasets_huggingface.dataset_wrapper_input_text_files \
# ---- NOTE-NO-NEED-FROM-PRETRAIN ----     import HuggingfaceDatasetWrapperInputTextFiles

# ---- NOTE-NO-NEED-FROM-PRETRAIN ---- from data.pytorch.datasets_huggingface.huggingface_datasets_utility \
# ---- NOTE-NO-NEED-FROM-PRETRAIN ----     import HuggingfaceDatasetsUtility

# ---- NOTE-NO-NEED-FROM-PRETRAIN ---- from model.language_understanding.helper.pytorch_language_understanding_transformers_helper \
# ---- NOTE-NO-NEED-FROM-PRETRAIN ----     import PytorchLanguageUnderstandingTransformersPretainedModelHelper

from utility.pytorch_utility.pytorch_utility \
    import PytorchUtility

from utility.datatype_helper.datatype_helper \
    import DatatypeHelper
from utility.debugging_helper.debugging_helper \
    import DebuggingHelper
from utility.string_helper.string_helper \
    import StringHelper
# from utility.configuration_helper.configuration_helper \
#     import ConfigurationHelper

def process_app_inference_pretrained_masked_language_model_arguments(parser):
    """
    To process data manager related arguments.
    """
    # DebuggingHelper.write_line_to_system_console_out(
    #     f'Calling process_app_inference_pretrained_masked_language_model_arguments() in {__name__}')
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
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='input file.')
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='output file.')
    # ---- NOTE-NO-NEED ---- parser.add_argument(
    # ---- NOTE-NO-NEED ----     '--name',
    # ---- NOTE-NO-NEED ----     type=str,
    # ---- NOTE-NO-NEED ----     required=True,
    # ---- NOTE-NO-NEED ----     help='dataset name.')
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- parser.add_argument(
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     '--input_path',
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     type=str,
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     required=True,
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     help='input path.')
    # ---- NOTE-USE-input_path-INSTEAD ---- parser.add_argument(
    # ---- NOTE-USE-input_path-INSTEAD ----     '--input_text_files',
    # ---- NOTE-USE-input_path-INSTEAD ----     type=str,
    # ---- NOTE-USE-input_path-INSTEAD ----     required=True,
    # ---- NOTE-USE-input_path-INSTEAD ----     help='input text files.')
    # ------------------------------------------------------------------------
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- parser.add_argument(
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     '--output_model_checkpoint_path',
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     type=str,
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     required=True,
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     help='output model checkpoint path.')
    # ------------------------------------------------------------------------
    parser.add_argument(
        '--trainer_argument_use_cuda',
        type=DatatypeHelper.to_bool,
        required=False,
        default=False,
        help='use cuda or not, default to True')
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- parser.add_argument(
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     '--trainer_argument_num_train_epochs',
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     type=int,
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     required=False,
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     default=4,
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     help='number of training epochs')
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- parser.add_argument(
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     '--trainer_argument_per_device_train_batch_size',
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     type=int,
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     required=False,
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     default=8,
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     help='per device train batch size')
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- parser.add_argument(
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     '--trainer_argument_gradient_accumulation_steps',
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     type=int,
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     required=False,
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     default=4,
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     help='gradient accumulation steps')
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- parser.add_argument(
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     '--trainer_argument_per_device_eval_batch_size',
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     type=int,
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     required=False,
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     default=8,
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     help='per device eval batch size')
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- parser.add_argument(
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     '--trainer_argument_logging_steps',
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     type=int,
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     required=False,
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     default=4,
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     help='logging steps')
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- parser.add_argument(
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     '--trainer_argument_save_steps',
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     type=int,
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     required=False,
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     default=8,
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     help='save size')
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- parser.add_argument(
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     '--trainer_argument_save_total_limit',
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     type=int,
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     required=False,
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     default=4,
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     help='save total limit')
    return parser

# ---- NOTE-PYLINT ---- C0103: Function name "" doesn't conform to snake_case naming style
# pylint: disable=C0103
def example_function_app_inference_pretrained_masked_language_model():
    """
    The main function to quickly test HuggingfaceDatasetWrapperInputTextFiles.

    REFERENCE:
        https://www.thepythoncode.com/article/pretraining-bert-huggingface-transformers-in-python
    """
    # ------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    # ------------------------------------------------------------------------
    process_app_inference_pretrained_masked_language_model_arguments(parser)
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
    input_file: str = \
        os.path.normpath(args.input_file)
    output_file: str = \
        os.path.normpath(args.output_file)
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- input_path: str = \
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     os.path.normpath(args.input_path)
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- output_model_checkpoint_path: str = \
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     os.path.normpath(args.output_model_checkpoint_path)
    # ------------------------------------------------------------------------
    trainer_argument_use_cuda: bool = \
        args.trainer_argument_use_cuda
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- trainer_argument_num_train_epochs: int = \
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     args.trainer_argument_num_train_epochs
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- trainer_argument_per_device_train_batch_size: int = \
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     args.trainer_argument_per_device_train_batch_size
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- trainer_argument_gradient_accumulation_steps: int = \
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     args.trainer_argument_gradient_accumulation_steps
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- trainer_argument_per_device_eval_batch_size: int = \
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     args.trainer_argument_per_device_eval_batch_size
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- trainer_argument_logging_steps: int = \
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     args.trainer_argument_logging_steps
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- trainer_argument_save_steps: int = \
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     args.trainer_argument_save_steps
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- trainer_argument_save_total_limit: int = \
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     args.trainer_argument_save_total_limit
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- os.environ["WANDB_DISABLED"] = "true"
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
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- pytorch_transformers_model_cache_dir_model: str = \
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     os.path.join(os.path.os.sep, transformers_model_folder, pytorch_transformers_model_name, 'model')
    pytorch_transformers_model_cache_dir_model: str = \
        transformers_model_folder
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
    fill_mask_pipeline: Pipeline = pipeline(
        "fill-mask",
        model=pytorch_transformers_model,
        tokenizer=pytorch_transformers_tokenizer)
    # ---- NOTE-FOR-DEBUGGING ---- # ---- NOTE perform predictions
    # ---- NOTE-FOR-DEBUGGING ---- examples = [ \
    # ---- NOTE-FOR-DEBUGGING ----     "名可名非常名，名不正則言不順，言不順則事不成。", \
    # ---- NOTE-FOR-DEBUGGING ----     "中國的文學成就最大的是詩歌，從《離騷》到唐代律詩，詩歌一直對中國文壇有着巨大的影響", \
    # ---- NOTE-FOR-DEBUGGING ----     "在很長一段時間", \
    # ---- NOTE-FOR-DEBUGGING ----     "材料科學", \
    # ---- NOTE-FOR-DEBUGGING ----     "中國文學", \
    # ---- NOTE-FOR-DEBUGGING ---- ]
    # ---- NOTE-FOR-DEBUGGING ---- # ---- NOTE-FOR-REFERENCE ---- transformers.pipelines.base.PipelineException: More than one mask_token ([MASK]) is not supported
    # ---- NOTE-FOR-DEBUGGING ---- # ---- examples = [ \
    # ---- NOTE-FOR-DEBUGGING ---- # ----     "名可名非常名，[MASK]則言不順，言不順則事不成。", \
    # ---- NOTE-FOR-DEBUGGING ---- # ----     "中國的文學成就最大的是[MASK]，從《離騷》到唐代律詩，詩歌一直對中國文壇有着巨大的影響", \
    # ---- NOTE-FOR-DEBUGGING ---- # ----     "在很長一段[MASK]間", \
    # ---- NOTE-FOR-DEBUGGING ---- # ----     "材料[MASK]學", \
    # ---- NOTE-FOR-DEBUGGING ---- # ----     "中國[MASK]學", \
    # ---- NOTE-FOR-DEBUGGING ---- # ---- ]
    # ---- NOTE-FOR-DEBUGGING ---- examples = [ \
    # ---- NOTE-FOR-DEBUGGING ----     "名可名非常名，[MASK]不正則言不順，言不順則事不成。", \
    # ---- NOTE-FOR-DEBUGGING ----     "中國的文學成就最大的是[MASK]歌，從《離騷》到唐代律詩，詩歌一直對中國文壇有着巨大的影響", \
    # ---- NOTE-FOR-DEBUGGING ----     "在很長一段[MASK]間", \
    # ---- NOTE-FOR-DEBUGGING ----     "材料[MASK]學", \
    # ---- NOTE-FOR-DEBUGGING ----     "中國[MASK]學", \
    # ---- NOTE-FOR-DEBUGGING ---- ]
    # ---- NOTE-FOR-DEBUGGING ---- with ccodecs.open( \
    # ---- NOTE-FOR-DEBUGGING ----     filename=output_file, \
    # ---- NOTE-FOR-DEBUGGING ----     mode="w", \
    # ---- NOTE-FOR-DEBUGGING ----     encoding="utf8") as output_file_writer:
    # ---- NOTE-FOR-DEBUGGING ----     for example in examples:
    # ---- NOTE-FOR-DEBUGGING ----         for prediction in fill_mask_pipeline(example):
    # ---- NOTE-FOR-DEBUGGING ----             prediction_sequence: str = prediction['sequence']
    # ---- NOTE-FOR-DEBUGGING ----             prediction_confidence_score: float = prediction['score']
    # ---- NOTE-FOR-DEBUGGING ----             # ---- print(f"{prediction['sequence']}, confidence: {prediction['score']}")
    # ---- NOTE-FOR-DEBUGGING ----             output_file_writer.write(f"{example}\t{prediction_sequence}\t{prediction_confidence_score}\n")
    # ------------------------------------------------------------------------
    with codecs.open( \
        filename=input_file, \
        mode="r", \
        encoding="utf8") as input_file_reader, \
        codecs.open( \
        filename=output_file, \
        mode="w", \
        encoding="utf8") as output_file_writer:
        while True:
            example: str = input_file_reader.readline()
            example = example.strip()
            if StringHelper.is_none_empty_or_whitespaces(example):
                break
            for prediction in fill_mask_pipeline(example):
                prediction_sequence: str = prediction['sequence']
                prediction_confidence_score: float = prediction['score']
                # ---- print(f"{prediction['sequence']}, confidence: {prediction['score']}")
                output_file_writer.write(f"{example}\t{prediction_sequence}\t{prediction_confidence_score}\n")
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    # ---- NOTE-PYLINT ---- C0301: Line too long
    # pylint: disable=C0301
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- input_text_files: List[str] = \
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     HuggingfaceDatasetWrapperInputTextFiles.scan_input_text_files_folder( \
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----         input_text_files_folder=input_path)
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- DebuggingHelper.write_line_to_system_console_out(
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     '==== input_text_files:"{}"'.format(input_text_files))
    # ------------------------------------------------------------------------
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- dataset_wrapper: HuggingfaceDatasetWrapperInputTextFiles = HuggingfaceDatasetWrapperInputTextFiles(
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     huggingface_datasets_argument_input_text_files=input_text_files)
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- dataset: Union[datasets.DatasetDict, datasets.Dataset, datasets.IterableDatasetDict, datasets.IterableDataset] = \
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     dataset_wrapper.get_dataset()
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- DebuggingHelper.write_line_to_system_console_out(
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     f'dataset={dataset}')
    # ------------------------------------------------------------------------
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- train_test_datasets: datasets.DatasetDict = \
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     HuggingfaceDatasetsUtility.split_to_train_test_sets(dataset)
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- DebuggingHelper.write_line_to_system_console_out(
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     f'train_test_datasets={train_test_datasets}')
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- train_dataset: datasets.Dataset = \
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     train_test_datasets["train"]
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- test_dataset: datasets.Dataset = \
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     train_test_datasets["test"]
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- DebuggingHelper.write_line_to_system_console_out(
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     f'train_dataset={train_dataset}')
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- DebuggingHelper.write_line_to_system_console_out(
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     f'test_dataset={test_dataset}')
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- # ---- train_dataset_text_feature: List[str] = \
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- # ----     train_dataset['text']
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- # ---- test_dataset_text_feature: List[str] = \
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- # ----     test_dataset['text']
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- # ---- DebuggingHelper.write_line_to_system_console_out(
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- # ----     f'train_dataset_text_feature={train_dataset_text_feature}')
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- # ---- DebuggingHelper.write_line_to_system_console_out(
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- # ----     f'test_dataset_text_feature={test_dataset_text_feature}')
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- # ---- NOTE-FOR-DEBUGGING ---- train_dataset_text_feature_tokenized = \
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- # ---- NOTE-FOR-DEBUGGING ----     HuggingfaceDatasetsUtility.tokenize_dataset_text_feature( \
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- # ---- NOTE-FOR-DEBUGGING ----         tokenizer=pytorch_transformers_tokenizer, \
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- # ---- NOTE-FOR-DEBUGGING ----         dataset=train_dataset)
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- # ---- NOTE-FOR-DEBUGGING ---- test_dataset_text_feature_tokenized = \
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- # ---- NOTE-FOR-DEBUGGING ----     HuggingfaceDatasetsUtility.tokenize_dataset_text_feature( \
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- # ---- NOTE-FOR-DEBUGGING ----         tokenizer=pytorch_transformers_tokenizer, \
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- # ---- NOTE-FOR-DEBUGGING ----         dataset=test_dataset)
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- # ---- NOTE-FOR-DEBUGGING ---- # ---- DebuggingHelper.write_line_to_system_console_out(
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- # ---- NOTE-FOR-DEBUGGING ---- # ----     f'train_dataset_text_feature_tokenized={train_dataset_text_feature_tokenized}')
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- # ---- NOTE-FOR-DEBUGGING ---- # ---- DebuggingHelper.write_line_to_system_console_out(
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- # ---- NOTE-FOR-DEBUGGING ---- # ----     f'test_dataset_text_feature_tokenized={test_dataset_text_feature_tokenized}')
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- train_dataset_tokenized: datasets.Dataset = \
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     HuggingfaceDatasetsUtility.tokenize_dataset( \
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----         tokenizer=pytorch_transformers_tokenizer, \
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----         dataset=train_dataset)
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- test_dataset_tokenized: datasets.Dataset = \
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     HuggingfaceDatasetsUtility.tokenize_dataset( \
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----         tokenizer=pytorch_transformers_tokenizer, \
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----         dataset=test_dataset)
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- DebuggingHelper.write_line_to_system_console_out(
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     f'train_dataset_tokenized={train_dataset_tokenized}')
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- DebuggingHelper.write_line_to_system_console_out(
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     f'test_dataset_tokenized={test_dataset_tokenized}')
    # ------------------------------------------------------------------------
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- data_collator: DataCollatorForLanguageModeling = \
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     PytorchLanguageUnderstandingTransformersPretainedModelHelper.create_data_collator_for_language_modeling( \
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----         tokenizer=pytorch_transformers_tokenizer)
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- DebuggingHelper.write_line_to_system_console_out(
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     f'data_collator={data_collator}')
    # ------------------------------------------------------------------------
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- training_args = TrainingArguments(
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     no_cuda=not trainer_argument_use_cuda,
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     output_dir=output_model_checkpoint_path,                                    # ---- output directory for saving model checkpoint
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     evaluation_strategy="steps",                                                # ---- "steps" for evaluate each `logging_steps`
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     overwrite_output_dir=True,                                                  # ---- whether to overrite output directory
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     num_train_epochs=trainer_argument_num_train_epochs,                         # ---- number of training epochs, feel free to tweak (10)
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     per_device_train_batch_size=trainer_argument_per_device_train_batch_size,   # ---- the training batch size, put it as high as your GPU memory fits (10)
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     gradient_accumulation_steps=trainer_argument_gradient_accumulation_steps,   # ---- accumulating the gradients before updating the weights (8)
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     per_device_eval_batch_size=trainer_argument_per_device_eval_batch_size,     # ---- evaluation batch size (64)
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     logging_steps=trainer_argument_logging_steps,                               # ---- evaluate, log and save model checkpoints every (500) step
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     save_steps=trainer_argument_save_steps,                                     # ---- save the checkpoint for number of steps (500)
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     load_best_model_at_end=True,                                                # ---- whether to load the best model (in terms of loss) at the end of training
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     save_total_limit=trainer_argument_save_total_limit,                         # ---- whether you don't have much space so you let only 3 model weights saved in the disk
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- )
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- trainer = Trainer(
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     model=pytorch_transformers_model,
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     args=training_args,
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     data_collator=data_collator,
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     train_dataset=train_dataset_tokenized,
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ----     eval_dataset=test_dataset_tokenized)
    # ---- NOTE-NO-NEED-FROM-PRETRAIN ---- trainer.train()
    # ------------------------------------------------------------------------

def main():
    """
    The main() function can quickly test a masked language model and do some evaluation.
    """
    example_function_app_inference_pretrained_masked_language_model()

if __name__ == '__main__':
    main()
