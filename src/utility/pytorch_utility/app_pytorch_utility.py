# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
This module tests Pytorch utility functions defined in pytorch_utility.
"""
# ---- NOTE-PYLINT ---- C0302: Too many lines in module (*/1000) (too-many-lines)
# pylint: disable=C0302

# from typing import Any
from typing import List
# from typing import NoReturn
from typing import Tuple

import argparse

import os

import  numpy

import torch
from transformers.optimization import AdamW
# from torch.utils.data import DataLoader
# from torch.utils.data import SequentialSampler
from torch.utils.data._utils.collate import default_collate

from pandas import DataFrame

# import sklearn
# from sklearn.metrics import classification_report

from simpletransformers.classification \
    import ClassificationModel
from simpletransformers.classification \
    import ClassificationArgs

from data.manager.base_pytorch_dataset_nop_collator \
    import BasePytorchDatasetNopCollator

# from data.manager.pytorch_transformers_dataset_kaggle_entity_annotated_corpus \
#     import KaggleEntityAnnotatedCorpusPytorchTransformersDataset

# from data.manager.pytorch_dataset_kaggle_entity_annotated_corpus \
#     import KaggleEntityAnnotatedCorpusPytorchDataset
from data.manager.pytorch_dataset_text_spam_int_id_feature \
    import TextSpamIntIdFeaturePytorchDataset
from data.manager.pytorch_dataset_text_spam_raw_str_feature \
    import TextSpamRawStrFeaturePytorchDataset
from data.manager.pytorch_dataset_iris \
    import IrisPytorchDataset

from data.manager.base_pytorch_transformers_dataset \
    import BasePytorchTransformersDatasetStrStr
# from data.manager.base_pytorch_transformers_dataset \
#     import BasePytorchTransformersDataset

# from data.manager.base_pytorch_dataset \
#     import BasePytorchDatasetIntFloats
# from data.manager.base_pytorch_dataset \
#     import BasePytorchDatasetIntInts
# from data.manager.base_pytorch_dataset \
#     import BasePytorchDatasetIntsInts
# from data.manager.base_pytorch_dataset \
#     import BasePytorchDatasetIntStr
# from data.manager.base_pytorch_dataset \
#     import BasePytorchDataset

from data.manager.dataset_manager_kaggle_entity_annotated_corpus \
    import KaggleEntityAnnotatedCorpusDatasetManager
from data.manager.dataset_manager_text_spam \
    import TextSpamDatasetManager
from data.manager.dataset_manager_iris \
    import IrisDatasetManager
# from data.manager.base_dataset_manager \
#     import BaseDatasetManager

# from data.manager.base_dataframe_text_labels_manager \
#     import BasePandasDataframeTextLabelsManagerIntIntStrStr
# from data.manager.base_dataframe_text_labels_manager \
#     import BasePandasDataframeTextLabelsManagerIntFloatStrStr
# from data.manager.base_dataframe_text_labels_manager \
#     import BasePandasDataframeTextLabelsManagerIntStrStrStr
# from data.manager.base_dataframe_text_labels_manager \
#     import BasePandasDataframeTextLabelsManager

from data_proprietary.manager.pytorch_dataset_dunmore_testing \
    import DunmoreTestingIdIntFeaturePytorchDataset
from data_proprietary.manager.pytorch_dataset_dunmore_training \
    import DunmoreTrainingIdIntFeaturePytorchDataset
# from data_proprietary.manager.pytorch_dataset_dunmore_validating_p0 \
#     import DunmoreValidatingP0IdIntFeaturePytorchDataset
# from data_proprietary.manager.pytorch_dataset_dunmore_validating_p1 \
#     import DunmoreValidatingP1IdIntFeaturePytorchDataset
# from data_proprietary.manager.pytorch_dataset_dunmore_validating_p2 \
#     import DunmoreValidatingP2IdIntFeaturePytorchDataset

from data_proprietary.manager.dataset_manager_dunmore_testing \
    import DunmoreTestingDatasetManager
from data_proprietary.manager.dataset_manager_dunmore_training \
    import DunmoreTrainingDatasetManager
# from data_proprietary.manager.dataset_manager_dunmore_validating_p0 \
#     import DunmoreValidatingP0DatasetManager
# from data_proprietary.manager.dataset_manager_dunmore_validating_p1 \
#     import DunmoreValidatingP1DatasetManager
# from data_proprietary.manager.dataset_manager_dunmore_validating_p2 \
#     import DunmoreValidatingP2DatasetManager

# from tensor.pytorch.manager.base_tensor_label_feature_dataset \
#     import TensorLabelFeatureDatasetLongFloat

# from tensor.pytorch.manager.tensor_label_feature_dataset_manager_text_spam \
#     import TensorLabelFeatureDatasetManagerTextSpam
# from tensor.pytorch.manager.tensor_label_feature_dataset_manager_iris \
#     import TensorLabelFeatureDatasetManagerIris
# from tensor.pytorch.manager.base_tensor_label_feature_dataset_manager \
#     import TensorLabelFeatureDatasetManagerLongFloat
# from tensor.pytorch.manager.base_tensor_label_feature_dataset_manager \
#     import BaseTensorLabelFeatureDatasetManager

from model.language_understanding.featurizer.\
    pytorch_transformers_feature_manager_bert_tokenization \
    import PytorchTransformersFeatureManagerBertTokenization

from model.language_understanding.helper.pytorch_language_understanding_transformers_helper \
    import PytorchLanguageUnderstandingTransformersPretainedModelHelper

from utility.classification_report_helepr.classification_report_helepr \
    import ClassificationReportHelper

from utility.pytorch_utility.pytorch_utility \
    import PytorchUtility

from utility.debugging_helper.debugging_helper \
    import DebuggingHelper

def process_pytorch_arguments(parser):
    """
    To process data manager related arguments.
    """
    if parser is None:
        DebuggingHelper.throw_exception(
            'input argument, parser, is None')
    parser.add_argument(
        '--input_pytorch_model_filename',
        type=str,
        required=True,
        help='Input Pytorch model filename.')
    parser.add_argument(
        '--output_pytorch_fine_tuned_model_filename',
        type=str,
        required=True,
        help='Output Pytorch fine-tuned model filename.')
    parser.add_argument(
        '--output_pytorch_model_json_filename',
        type=str,
        required=True,
        help='Output Pytorch model in JSON filename.')
    return parser

def main_kaggle_entity_annotated_corpus_entity_per_token_classification():
    """
    The main_dunmore_simple_transformers() function can quickly test PytorchUtility functions
    """
    # ---- NOTE-PYLINT ---- C0301: Line too long
    # pylint: disable=C0301
    # ---- NOTE-PYLINT ---- W0612 Unused Variable
    # pylint: disable=W0612
    # ---- NOTE-PYLINT ---- R0914: Too many local variables (*/15) (too-many-locals)
    # pylint: disable=R0914
    # ------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    # ------------------------------------------------------------------------
    # process_pytorch_arguments(parser)
    # ------------------------------------------------------------------------
    args: argparse.Namespace = parser.parse_args()
    DebuggingHelper.write_line_to_system_console_out(
        f'args={str(args)}')
    # ------------------------------------------------------------------------
    PytorchUtility.set_all_random_number_generator_seeds(47)
    # ------------------------------------------------------------------------
    use_gpu: bool = True
    device: torch.device = None
    device_cpu: torch.device = torch.device('cpu')
    # device = torch.device('cpu')
    if use_gpu:
        device = torch.device('cuda')
    # ------------------------------------------------------------------------
    PytorchUtility.set_torch_cuda_per_process_memory_fraction(fraction=1.0, device=None)
    # ------------------------------------------------------------------------
    pytorch_transformers_model_tokenizer: str = \
        'bert-base-uncased'
    pytorch_transformers_model_cache_dir_tokenizer: str = \
        'pytorch_transformers'
    # ---- NOTE ---- use a fixed transformer tokenizer cache
    pytorch_transformers_model_cache_dir_tokenizer: str = \
        os.path.join(os.path.os.sep, 'pyludispatch', '_tokenizers_cache', 'pytorch_transformers', 'bert-base-uncased')
        # ---- '/pyludispatch/_tokenizers_cache/pytorch_transformers/bert-base-uncased')
    pytorch_transformers_model_tokenizer_do_lower_case: bool = True
    pytorch_transformers_model_tokenizer_kind: int = \
        PytorchTransformersFeatureManagerBertTokenization.PYTORCH_TRANSFORMERS_TOKENIZER_KIND
    pytorch_transformers_feature_manager_bert_tokenization: \
        PytorchTransformersFeatureManagerBertTokenization = \
        PytorchTransformersFeatureManagerBertTokenization( \
            pytorch_transformers_model_tokenizer, \
            pytorch_transformers_model_cache_dir_tokenizer, \
            pytorch_transformers_model_tokenizer_do_lower_case, \
            pytorch_transformers_model_tokenizer_kind)
    # ------------------------------------------------------------------------
    dataset_manager: KaggleEntityAnnotatedCorpusDatasetManager = \
        KaggleEntityAnnotatedCorpusDatasetManager( \
        pytorch_transformers_feature_manager_bert_tokenization)
    DebuggingHelper.write_line_to_system_console_out(
        f'dataset_manager.get_shape()='
        f'{dataset_manager.get_shape()}')
    number_labels = dataset_manager.get_number_labels()
    DebuggingHelper.write_line_to_system_console_out(
        f'number_labels='
        f'{number_labels}')
    # ------------------------------------------------------------------------
    max_sequence_length: int = 128
    # ------------------------------------------------------------------------
    number_partitions: int = 100
    partition_index: int = 0
    to_except: bool = False
    # ------------------------------------------------------------------------
    training_record_generator = \
        dataset_manager.get_record_generator(
            generator_configuration_int_0=number_partitions,
            generator_configuration_int_1=partition_index,
            generator_configuration_bool_0=to_except)
    DebuggingHelper.write_line_to_system_console_out(
        f'number of training partition records='
        f'{training_record_generator.number_partition_records}')
    DebuggingHelper.write_line_to_system_console_out(
        f'number of training generator records='
        f'{training_record_generator.number_generator_records}')
    training_records: List[str] = \
        list(training_record_generator.generate())
    training_pytorch_transformers_dataset: BasePytorchTransformersDatasetStrStr = \
        BasePytorchTransformersDatasetStrStr( \
            records=training_records, \
            max_sequence_length=max_sequence_length)
    # ------------------------------------------------------------------------
    partition_index: int = 1
    to_except = False
    # ------------------------------------------------------------------------
    testing_record_generator = \
        dataset_manager.get_record_generator(
            generator_configuration_int_0=number_partitions,
            generator_configuration_int_1=partition_index,
            generator_configuration_bool_0=to_except)
    DebuggingHelper.write_line_to_system_console_out(
        f'number of testing partition records='
        f'{testing_record_generator.number_partition_records}')
    DebuggingHelper.write_line_to_system_console_out(
        f'number of testing generator records='
        f'{testing_record_generator.number_generator_records}')
    testing_records: List[str] = \
        list(testing_record_generator.generate())
    testing_pytorch_transformers_dataset: BasePytorchTransformersDatasetStrStr = \
        BasePytorchTransformersDatasetStrStr( \
            records=testing_records, \
            max_sequence_length=max_sequence_length)
    # ------------------------------------------------------------------------
    training_data_loader_batch_size: int = 3
    # ------------------------------------------------------------------------
    training_sequential_sampler: torch.utils.data.SequentialSampler = \
        torch.utils.data.SequentialSampler(training_pytorch_transformers_dataset)
    training_data_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        dataset=training_pytorch_transformers_dataset,
        sampler=training_sequential_sampler,
        batch_size=training_data_loader_batch_size)
    # ------------------------------------------------------------------------
    testing_data_loader_batch_size: int = training_data_loader_batch_size
    # ------------------------------------------------------------------------
    testing_sequential_sampler: torch.utils.data.SequentialSampler = \
        torch.utils.data.SequentialSampler(testing_pytorch_transformers_dataset)
    testing_data_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        dataset=testing_pytorch_transformers_dataset,
        sampler=testing_sequential_sampler,
        batch_size=testing_data_loader_batch_size)
    # ------------------------------------------------------------------------
    pytorch_transformers_model_dir_learner: str = \
        os.path.join(os.path.os.sep, 'pyludispatch', '_models', 'pytorch_transformers', 'bert-base-uncased', 'model')
        # ---- '/pyludispatch/_models/pytorch_transformers/bert-base-uncased/model'
    base_transformers_model = PytorchLanguageUnderstandingTransformersPretainedModelHelper.model_from_pretrained( \
        pretrained_model_name_or_path=pytorch_transformers_model_dir_learner, \
        model_type=PytorchLanguageUnderstandingTransformersPretainedModelHelper.MODEL_TYPE_BERT)
    model = PytorchUtility.TorchNeuralNetworkModuleEntityBertPerTokenClassificationModel( \
        number_entity_per_token_labels=number_labels, \
        component_module_base_transformers_model_bert=base_transformers_model, \
        component_module_base_transformers_parameter_embedding_dimension=768, \
        component_module_dropout_parameter_rate=0.3)
    # ------------------------------------------------------------------------
    number_epochs: int = 2
    # ------------------------------------------------------------------------
    optimizer_learning_rate: float = 3e-5
    no_decay_parameters = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = PytorchUtility.get_model_optimizer_parameters(
        model=model,
        no_decay_parameters=no_decay_parameters)
    if device is not None:
        model.to(device)
    optimizer = AdamW(optimizer_parameters, lr=optimizer_learning_rate)
    # ------------------------------------------------------------------------
    best_loss = numpy.inf
    for epoch in range(number_epochs):
        train_loss = PytorchUtility.train_model_per_epoch(
            data_loader=training_data_loader,
            model=model,
            optimizer=optimizer,
            device=device,
            device_resettled_after_operation=device_cpu,
            scheduler=None)
        DebuggingHelper.write_line_to_system_console_out(
            f'epoch={epoch}, train_loss={train_loss}')
        # ---- NOTE-MAY-CONSUME-GPU-RAM-AND-POSSIBLY-EXHAUST-IT ---- test_loss = PytorchUtility.evaluate_model_per_epoch(
        # ---- NOTE-MAY-CONSUME-GPU-RAM-AND-POSSIBLY-EXHAUST-IT ----     data_loader=testing_data_loader,
        # ---- NOTE-MAY-CONSUME-GPU-RAM-AND-POSSIBLY-EXHAUST-IT ----     model=model,
        # ---- NOTE-MAY-CONSUME-GPU-RAM-AND-POSSIBLY-EXHAUST-IT ----     device=device, # ---- NOTE-MAY-CONSUME-GPU-RAM-AND-POSSIBLY-EXHAUST-IT ----
        # ---- NOTE-MAY-CONSUME-GPU-RAM-AND-POSSIBLY-EXHAUST-IT ----     device_resettled_after_operation=device_cpu)
        # ---- NOTE-FOR-REFERENCE ---- Exception has occurred: RuntimeError       (note: full exception trace is shown but execution is paused at: _run_module_as_main)
        # ---- NOTE-FOR-REFERENCE ---- Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking arugment for argument index in method wrapper_index_select)
        # ---- NOTE-MAY-CONSUME-GPU-RAM-AND-POSSIBLY-EXHAUST-IT ---- DebuggingHelper.write_line_to_system_console_out(
        # ---- NOTE-MAY-CONSUME-GPU-RAM-AND-POSSIBLY-EXHAUST-IT ----     f'epoch={epoch}, train_loss={train_loss}, test_loss={test_loss}')
        # ---- NOTE-MAY-CONSUME-GPU-RAM-AND-POSSIBLY-EXHAUST-IT ---- if test_loss < best_loss:
        # ---- NOTE-MAY-CONSUME-GPU-RAM-AND-POSSIBLY-EXHAUST-IT ----     # ---- torch.save(model.state_dict(), config.MODEL_PATH)
        # ---- NOTE-MAY-CONSUME-GPU-RAM-AND-POSSIBLY-EXHAUST-IT ----     best_loss = test_loss
        # ---- NOTE-MAY-CONSUME-GPU-RAM-AND-POSSIBLY-EXHAUST-IT ---- DebuggingHelper.write_line_to_system_console_out(
        # ---- NOTE-MAY-CONSUME-GPU-RAM-AND-POSSIBLY-EXHAUST-IT ----     f'epoch={epoch}, train_loss={train_loss}, best_loss={best_loss}, test_loss={test_loss}')
    if device_cpu is not None:
        model.to(device_cpu)
    test_loss = PytorchUtility.evaluate_model_per_epoch(
        data_loader=testing_data_loader,
        model=model,
        device=device_cpu, # ---- NOTE-MAY-CONSUME-GPU-RAM-AND-POSSIBLY-EXHAUST-IT ---- if using GPU device
        device_resettled_after_operation=None)
    # ------------------------------------------------------------------------
    pytorch_transformers_model_learner_finetuned: str = \
        os.path.join(os.path.os.sep, 'pyludispatch', '_models_finetuned', 'pytorch_transformers', 'bert-base-uncased', 'model', 'pytorch_model.bin')
        # ---- '/pyludispatch/_models_finetuned/pytorch_transformers/bert-base-uncased/model/pytorch_model.bin'
    PytorchUtility.save_model(
        model=model,
        output_torch_model_filename=pytorch_transformers_model_learner_finetuned)
    # ------------------------------------------------------------------------
    # model_loaded: Any = \
    #     PytorchUtility.load_model(input_pytorch_model_filename=pytorch_transformers_model_learner_finetuned)
    # ---- NOTE-TODO ---- PytorchUtility.predict_model( \
    # ---- NOTE-TODO ----     model=model_loaded, \
    # ---- NOTE-TODO ----     data_loader=testing_data_loader)
    # ---- NOTE-TODO ---- test_classification_report: classification_report = \
    # ---- NOTE-TODO ----     ClassificationReportHelper.create_bio_classification_report( \
    # ---- NOTE-TODO ----     y_ground_trues=y_test,
    # ---- NOTE-TODO ----     y_predictions=y_predictions)
    # ---- NOTE-TODO ---- DebuggingHelper.write_line_to_system_console_out(
    # ---- NOTE-TODO ----     f'test_classification_report=\n{test_classification_report}')
    # ------------------------------------------------------------------------
    # ---- NOTE-PYLINT ---- C0301: Line too long
    # pylint: disable=C0301
    # ====    number_partitions: int = 100
    # ====    partition_index: int = 0 # ---- 1 for testing
    # ====    to_except: bool = True # ---- for training and True for testing
    # ====    number_epochs: int = 5
    # ====    use_gpu: bool = True
    # ====    training_data_loader_batch_size: int = 6
    # ====    testing_data_loader_batch_size: int = 6
    # ---- STDOUT-INFO: [2021-08-04T09:08:56.207484][main_kaggle_entity_annotated_corpus_entity_per_token_classification @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:160][DEBUG:10]: args=Namespace()
    # ---- STDOUT-INFO: [2021-08-04T09:12:23.823178][main_kaggle_entity_annotated_corpus_entity_per_token_classification @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:184][DEBUG:10]: dataset_manager.get_shape()=(47959, 17, 30522)
    # ---- STDOUT-INFO: [2021-08-04T09:12:23.825178][main_kaggle_entity_annotated_corpus_entity_per_token_classification @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:188][DEBUG:10]: number_labels=17
    # ---- STDOUT-INFO: [2021-08-04T09:12:23.827181][main_kaggle_entity_annotated_corpus_entity_per_token_classification @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:203][DEBUG:10]: number of training partition records=480
    # ---- STDOUT-INFO: [2021-08-04T09:12:23.828182][main_kaggle_entity_annotated_corpus_entity_per_token_classification @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:206][DEBUG:10]: number of training generator records=480
    # ---- STDOUT-INFO: [2021-08-04T09:12:23.832212][main_kaggle_entity_annotated_corpus_entity_per_token_classification @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:222][DEBUG:10]: number of testing partition records=480
    # ---- STDOUT-INFO: [2021-08-04T09:12:23.833180][main_kaggle_entity_annotated_corpus_entity_per_token_classification @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:225][DEBUG:10]: number of testing generator records=480
    # ---- 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 80/80 [00:44<00:00,  1.81it/s]
    # ---- 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 80/80 [00:12<00:00,  6.47it/s]
    # ---- STDOUT-INFO: [2021-08-04T09:13:24.146962][main_kaggle_entity_annotated_corpus_entity_per_token_classification @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:301][DEBUG:10]: epoch=0, train_loss=0.7808341555297375, test_loss=0.40981995570473373
    # ---- 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 80/80 [00:43<00:00,  1.82it/s]
    # ---- 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 80/80 [00:12<00:00,  6.46it/s]
    # ---- STDOUT-INFO: [2021-08-04T09:14:20.454637][main_kaggle_entity_annotated_corpus_entity_per_token_classification @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:301][DEBUG:10]: epoch=1, train_loss=0.2677281618118286, test_loss=0.3103304022923112
    # ---- 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 80/80 [00:43<00:00,  1.82it/s]
    # ---- 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 80/80 [00:12<00:00,  6.47it/s]
    # ---- STDOUT-INFO: [2021-08-04T09:15:16.696960][main_kaggle_entity_annotated_corpus_entity_per_token_classification @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:301][DEBUG:10]: epoch=2, train_loss=0.14232921125367284, test_loss=0.2907575017074123
    # ---- 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 80/80 [00:43<00:00,  1.83it/s]
    # ---- 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 80/80 [00:12<00:00,  6.48it/s]
    # ---- STDOUT-INFO: [2021-08-04T09:16:12.867902][main_kaggle_entity_annotated_corpus_entity_per_token_classification @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:301][DEBUG:10]: epoch=3, train_loss=0.09091263832524418, test_loss=0.3079919566982426
    # ---- 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 80/80 [00:43<00:00,  1.83it/s]
    # ---- 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 80/80 [00:12<00:00,  6.49it/s]
    # ---- STDOUT-INFO: [2021-08-04T09:17:09.027636][main_kaggle_entity_annotated_corpus_entity_per_token_classification @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:301][DEBUG:10]: epoch=4, train_loss=0.053777554113185036, test_loss=0.3130848284898093
    # ====    number_partitions: int = 5
    # ====    partition_index: int = 0
    # ====    to_except: bool = True # ---- for training and False for testing
    # ====    number_epochs: int = 5
    # ====    use_gpu: bool = True
    # ====    training_data_loader_batch_size: int = 3
    # ====    testing_data_loader_batch_size: int = 3
    # ---- STDOUT-INFO: [2021-08-03T18:16:09.638644][main_kaggle_entity_annotated_corpus_entity_per_tokenification @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\y\pytorch_utility\app_pytorch_utility.py:160][DEBUG:10]: args=Namespace()
    # ---- STDOUT-INFO: [2021-08-03T18:19:37.713254][main_kaggle_entity_annotated_corpus_entity_per_tokenification @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\y\pytorch_utility\app_pytorch_utility.py:184][DEBUG:10]: dataset_manager.get_shape()=(47959, 122)
    # ---- STDOUT-INFO: [2021-08-03T18:19:37.720253][main_kaggle_entity_annotated_corpus_entity_per_tokenclassification @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\pyhon\utility\pytorch_utility\app_pytorch_utility.py:205][DEBUG:10]: number of training generator records=38367
    # ---- STDOUT-INFO: [2021-08-03T18:19:37.909283][main_kaggle_entity_annotated_corpus_entity_per_tokenclassification @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\pyhon\utility\pytorch_utility\app_pytorch_utility.py:218][DEBUG:10]: number of testing partition records=9592
    # ---- STDOUT-INFO: [2021-08-03T18:19:37.911246][main_kaggle_entity_annotated_corpus_entity_per_tokenclassification @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\pyhon\utility\pytorch_utility\app_pytorch_utility.py:221][DEBUG:10]: number of testing generator records=9592
    # ---- 100%|██████████████████████████████████████████████████████| 12789/12789 [1:16:38<00:00,  2.78it/s]
    # ---- 100%|████████████████████████████████████████████████████████████████| 3198/3198 [04:31<00:00, 11.77it/s]
    # ---- STDOUT-INFO: [2021-08-03T19:40:51.794820][main_kaggle_entity_annotated_corpus_entity_per_token_classification @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:296][DEBUG:10]: epoch=0, train_loss=0.159513104451654, test_loss=0.13886238159329792
    # ---- 100%|████████████████████████████████████████████████████████████| 12789/12789 [1:17:12<00:00,  2.76it/s]
    # ---- 100%|████████████████████████████████████████████████████████████████| 3198/3198 [04:32<00:00, 11.72it/s]
    # ---- STDOUT-INFO: [2021-08-03T21:02:37.172340][main_kaggle_entity_annotated_corpus_entity_per_token_classification @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:296][DEBUG:10]: epoch=1, train_loss=0.10214252754420913, test_loss=0.14447486992360975
    # ---- 100%|████████████████████████████████████████████████████████████| 12789/12789 [1:16:41<00:00,  2.78it/s]
    # ---- 100%|████████████████████████████████████████████████████████████████| 3198/3198 [04:31<00:00, 11.79it/s]
    # ---- STDOUT-INFO: [2021-08-03T22:23:49.935228][main_kaggle_entity_annotated_corpus_entity_per_token_classification @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:296][DEBUG:10]: epoch=2, train_loss=0.07288529425896227, test_loss=0.160199985277989
    # ---- 100%|████████████████████████████████████████████████████████████| 12789/12789 [1:16:40<00:00,  2.78it/s]
    # ---- 100%|████████████████████████████████████████████████████████████████| 3198/3198 [04:30<00:00, 11.82it/s]
    # ---- STDOUT-INFO: [2021-08-03T23:45:00.742802][main_kaggle_entity_annotated_corpus_entity_per_token_classification @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:296][DEBUG:10]: epoch=3, train_loss=0.05474682348685943, test_loss=0.1776494393791195
    # ---- 100%|████████████████████████████████████████████████████████████| 12789/12789 [1:16:40<00:00,  2.78it/s]
    # ---- 100%|████████████████████████████████████████████████████████████████| 3198/3198 [04:31<00:00, 11.79it/s]
    # ---- STDOUT-INFO: [2021-08-04T01:06:12.809247][main_kaggle_entity_annotated_corpus_entity_per_token_classification @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:296][DEBUG:10]: epoch=4, train_loss=0.04296390854169653, test_loss=0.19683858065119406


def main_dunmore_simple_transformers():
    """
    The main_dunmore_simple_transformers() function can quickly test PytorchUtility functions
    """
    # ---- NOTE-PYLINT ---- C0301: Line too long
    # pylint: disable=C0301
    # ---- NOTE-PYLINT ---- W0612 Unused Variable
    # pylint: disable=W0612
    # ---- NOTE-PYLINT ---- R0914: Too many local variables (*/15) (too-many-locals)
    # pylint: disable=R0914
    # ------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    # ------------------------------------------------------------------------
    # process_pytorch_arguments(parser)
    # ------------------------------------------------------------------------
    args: argparse.Namespace = parser.parse_args()
    DebuggingHelper.write_line_to_system_console_out(
        f'args={str(args)}')
    # ------------------------------------------------------------------------
    PytorchUtility.set_all_random_number_generator_seeds(47)
    # ------------------------------------------------------------------------
    # ---- NOTE-PYLINT ---- E1133: Non-iterable value * is used in an iterating context (not-an-iterable)
    # pylint: disable=E1133
    feature_raw_array_for_training: List[str] = \
        [x[0] for x in DunmoreTrainingDatasetManager.static_get_feature_raw_arrays()]
    label_index_array_for_training: List[int] = \
        DunmoreTrainingDatasetManager.static_get_label_single_indexes()
    text_labels_for_training: List[List[str, int]] = \
        [[x[0], x[1]] for x in zip(feature_raw_array_for_training, label_index_array_for_training)]
    text_labels_dataframe_for_training: DataFrame = \
        DataFrame(text_labels_for_training)
    # ==== text_labels_dataframe_for_training.columns = ["text", "labels"]
    # ------------------------------------------------------------------------
    simple_transformers_model_type: str = 'xlnet'
    simple_transformers_tokenizer_type: str = 'xlnet'
    simple_transformers_use_cuda: bool = True
    # ---- NOTE-bert-large-uncased ---- simple_transformers_use_cuda: bool = False
    pytorch_transformers_model: str = \
        'xlnet-base-cased'
    # ---- NOTE-bert-base-uncased ---- pytorch_transformers_model: str = \
    # ---- NOTE-bert-base-uncased ----     'bert-base-uncased'
    # ---- NOTE-bert-large-uncased ---- pytorch_transformers_model: str = \
    # ---- NOTE-bert-large-uncased ----     'bert-large-uncased'
    pytorch_transformers_model_tokenizer: str = \
        'xlnet-base-cased'
    # ---- NOTE-bert-base-uncased ---- pytorch_transformers_model_tokenizer: str = \
    # ---- NOTE-bert-base-uncased ----     'bert-base-uncased'
    # ---- NOTE-bert-large-uncased ---- pytorch_transformers_model_tokenizer: str = \
    # ---- NOTE-bert-large-uncased ----     'bert-large-uncased'
    # ---- NOTE-FOR-REFERENCE ---- 'xlnet-large-uncased' cannot load on a machine with only 4GB GPU
    # ---- NOTE-FOR-REFERENCE ---- Exception has occurred: RuntimeError       (note: full exception trace is shown but execution is paused at: _run_module_as_main)
    # ---- NOTE-FOR-REFERENCE ---- CUDA out of memory. Tried to allocate 16.00 MiB (GPU 0; 4.00 GiB total capacity; 3.01 GiB already allocated; 0 bytes free; 3.02 GiB reserved in total by PyTorch)
    # ---- NOTE-FOR-REFERENCE ---- 'bert-large-uncased' cannot load on a machine with only 4GB GPU
    # ---- NOTE-FOR-REFERENCE ---- Exception has occurred: RuntimeError       (note: full exception trace is shown but execution is paused at: _run_module_as_main)
    # ---- NOTE-FOR-REFERENCE ---- CUDA out of memory. Tried to allocate 20.00 MiB (GPU 0; 4.00 GiB total capacity; 3.00 GiB already allocated; 0 bytes free; 3.01 GiB reserved in total by PyTorch)
    pytorch_transformers_model_cache_dir_learner: str = \
        'pytorch_transformers'
    # ---- NOTE ---- use a fixed transformer tokenizer cache
    pytorch_transformers_model_cache_dir_learner: str = \
        os.path.join(os.path.os.sep, 'pyludispatch', '_model_and_tokenizers', 'pytorch_transformers', 'xlnet-base-cased')
        # ---- '/pyludispatch/_model_and_tokenizers/pytorch_transformers/xlnet-base-cased'
    # ---- NOTE-bert-base-uncased ---- pytorch_transformers_model_cache_dir_learner: str = \
    # ---- NOTE-bert-base-uncased ----     os.path.join(os.path.os.sep, 'pyludispatch', '_model_and_tokenizers', 'pytorch_transformers', 'bert-base-uncased')
    # ---- NOTE-bert-base-uncased ----     # ---- '/pyludispatch/_model_and_tokenizers/pytorch_transformers/bert-base-uncased'
    # ---- NOTE-bert-large-uncased ---- pytorch_transformers_model_cache_dir_learner: str = \
    # ---- NOTE-bert-large-uncased ----     os.path.join(os.path.os.sep, 'pyludispatch', '_model_and_tokenizers', 'pytorch_transformers', 'bert-large-uncased')
    # ---- NOTE-bert-large-uncased ----     # ---- '/pyludispatch/_model_and_tokenizers/pytorch_transformers/bert-large-uncased'
    # ---- NOTE-USE_SAME-AS-MODEL ---- pytorch_transformers_model_cache_dir_tokenizer: str = \
    # ---- NOTE-USE_SAME-AS-MODEL ----     'pytorch_transformers'
    # ---- NOTE-USE_SAME-AS-MODEL ---- # ---- NOTE ---- use a fixed transformer tokenizer cache
    # ---- NOTE-USE_SAME-AS-MODEL ---- pytorch_transformers_model_cache_dir_tokenizer: str = \
    # ---- NOTE-USE_SAME-AS-MODEL ----     os.path.join(os.path.os.sep, 'pyludispatch', '_model_and_tokenizers', 'pytorch_transformers', 'xlnet-base-cased')
    # ---- NOTE-USE_SAME-AS-MODEL ----     # ---- '/pyludispatch/_model_and_tokenizers/pytorch_transformers/xlnet-base-cased'
    # ---- NOTE-USE_SAME-AS-MODEL ---- # ---- NOTE-bert-base-uncased ---- pytorch_transformers_model_cache_dir_tokenizer: str = \
    # ---- NOTE-USE_SAME-AS-MODEL ---- # ---- NOTE-bert-base-uncased ----     os.path.join(os.path.os.sep, 'pyludispatch', '_model_and_tokenizers', 'pytorch_transformers', 'bert-base-uncased')
    # ---- NOTE-USE_SAME-AS-MODEL ---- # ---- NOTE-bert-base-uncased ----     # ---- '/pyludispatch/_model_and_tokenizers/pytorch_transformers/bert-base-uncased'
    # ---- NOTE-USE_SAME-AS-MODEL ---- # ---- NOTE-bert-large-uncased ---- pytorch_transformers_model_cache_dir_tokenizer: str = \
    # ---- NOTE-USE_SAME-AS-MODEL ---- # ---- NOTE-bert-large-uncased ----     os.path.join(os.path.os.sep, 'pyludispatch', '_model_and_tokenizers', 'pytorch_transformers', 'bert-large-uncased')
    # ---- NOTE-USE_SAME-AS-MODEL ---- # ---- NOTE-bert-large-uncased ----     # ---- '/pyludispatch/_model_and_tokenizers/pytorch_transformers/bert-large-uncased'
    # pytorch_transformers_model_tokenizer_do_lower_case: bool = True
    simple_transformers_classification_model_arguments = \
        ClassificationArgs( \
            num_train_epochs=6)
    # ---- simple_transformers_classification_model_arguments = \
    # ----     ClassificationArgs( \
    # ----         num_train_epochs=7,
    # ----         train_batch_size=4,
    # ----         eval_batch_size=4)
    simple_transformers_classification_model: \
        ClassificationModel = \
            PytorchUtility.create_simple_transformers_classification_model( \
            simple_transformers_model_type=simple_transformers_model_type, \
            simple_transformers_model_name=pytorch_transformers_model, \
            simple_transformers_tokenizer_type=simple_transformers_tokenizer_type, \
            simple_transformers_tokenizer_name=pytorch_transformers_model_tokenizer, \
            simple_transformers_use_cuda=simple_transformers_use_cuda, \
            simple_transformers_arguments=simple_transformers_classification_model_arguments, \
            pytorch_transformers_cache_dir=pytorch_transformers_model_cache_dir_learner)
    # ------------------------------------------------------------------------
    # results_before_training, model_outputs_before_training, wrong_predictions_before_training = \
    #     simple_transformers_classification_model.eval_model(\
    #         text_labels_dataframe_for_training)
    # ------------------------------------------------------------------------
    simple_transformers_classification_model.train_model(text_labels_dataframe_for_training)
    # ------------------------------------------------------------------------
    # results_after_training, model_outputs_after_training, wrong_predictions_after_training = \
    #     simple_transformers_classification_model.eval_model(\
    #         text_labels_dataframe_for_training)
    # ------------------------------------------------------------------------
    # ---- NOTE-PYLINT ---- E1133: Non-iterable value * is used in an iterating context (not-an-iterable)
    # pylint: disable=E1133
    feature_raw_array_for_testing: List[str] = \
        [x[0] for x in DunmoreTestingDatasetManager.static_get_feature_raw_arrays()]
    label_index_array_for_testing: List[int] = \
        DunmoreTestingDatasetManager.static_get_label_single_indexes()
    text_labels_for_testing: List[List[str, int]] = \
        [[x[0], x[1]] for x in zip(feature_raw_array_for_testing, label_index_array_for_testing)]
    text_labels_dataframe_for_testing: DataFrame = \
        DataFrame(text_labels_for_testing)
    # ==== text_labels_dataframe_for_testing.columns = ["text", "labels"]
    # ------------------------------------------------------------------------
    results_for_testing, model_outputs_for_testing, wrong_predictions_for_testing = \
        simple_transformers_classification_model.eval_model(\
            text_labels_dataframe_for_testing)
    # ------------------------------------------------------------------------
    # DebuggingHelper.write_line_to_system_console_out(
    #     f'results_before_training:'
    #     f'\n{results_before_training}')
    # DebuggingHelper.write_line_to_system_console_out(
    #     f'model_outputs_before_training:'
    #     f'\n{model_outputs_before_training}')
    # DebuggingHelper.write_line_to_system_console_out(
    #     f'wrong_predictions_before_training:'
    #     f'\n{wrong_predictions_before_training}')
    # ------------------------------------------------------------------------
    # DebuggingHelper.write_line_to_system_console_out(
    #     f'results_after_training:'
    #     f'\n{results_after_training}')
    # DebuggingHelper.write_line_to_system_console_out(
    #     f'model_outputs_after_training:'
    #     f'\n{model_outputs_after_training}')
    # DebuggingHelper.write_line_to_system_console_out(
    #     f'wrong_predictions_after_training:'
    #     f'\n{wrong_predictions_after_training}')
    # ------------------------------------------------------------------------
    DebuggingHelper.write_line_to_system_console_out(
        f'results_for_testing:'
        f'\n{results_for_testing}')
    DebuggingHelper.write_line_to_system_console_out(
        f'model_outputs_for_testing:'
        f'\n{model_outputs_for_testing}')
    DebuggingHelper.write_line_to_system_console_out(
        f'wrong_predictions_for_testing:'
        f'\n{wrong_predictions_for_testing}')
    # ------------------------------------------------------------------------
    # ---- NOTE-PYLINT ---- C0301: Line too long
    # pylint: disable=C0301
    # ------------------------------------------------------------------------
    # ---- NOTE-FOR-REFERENCE ----     model_name='bert-base-uncased'
    # ---- NOTE-FOR-REFERENCE ----     simple_transformers_use_cuda=True
    # ---- NOTE-FOR-REFERENCE ----     num_train_epochs=5
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ---- auprc = 0.8997687504685543
    # ---- NOTE-FOR-REFERENCE ---- auroc = 0.9515064249569539
    # ---- NOTE-FOR-REFERENCE ---- eval_loss = 0.7407259472685468
    # ---- NOTE-FOR-REFERENCE ---- fn = 40
    # ---- NOTE-FOR-REFERENCE ---- fp = 104
    # ---- NOTE-FOR-REFERENCE ---- mcc = 0.72097083041143
    # ---- NOTE-FOR-REFERENCE ---- tn = 639
    # ---- NOTE-FOR-REFERENCE ---- tp = 318
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----   > (318+639)/(318+639+40+104)
    # ---- NOTE-FOR-REFERENCE ----   [1] 0.8692098
    # ------------------------------------------------------------------------
    # ---- NOTE-FOR-REFERENCE ----     model_name='bert-base-uncased'
    # ---- NOTE-FOR-REFERENCE ----     simple_transformers_use_cuda=True
    # ---- NOTE-FOR-REFERENCE ----     num_train_epochs=6
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ---- auprc = 0.8784354560933525
    # ---- NOTE-FOR-REFERENCE ---- auroc = 0.9447938675308466
    # ---- NOTE-FOR-REFERENCE ---- eval_loss = 0.8098325022752313
    # ---- NOTE-FOR-REFERENCE ---- fn = 35
    # ---- NOTE-FOR-REFERENCE ---- fp = 102
    # ---- NOTE-FOR-REFERENCE ---- mcc = 0.7360427769300535
    # ---- NOTE-FOR-REFERENCE ---- tn = 641
    # ---- NOTE-FOR-REFERENCE ---- tp = 323
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----   > (323+641)/(323+641+25+102)
    # ---- NOTE-FOR-REFERENCE ----   [1] 0.883593
    # ------------------------------------------------------------------------
    # ---- NOTE-FOR-REFERENCE ----     model_name='bert-base-uncased'
    # ---- NOTE-FOR-REFERENCE ----     simple_transformers_use_cuda=True
    # ---- NOTE-FOR-REFERENCE ----     num_train_epochs=6
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ---- auprc = 0.9005559606288014
    # ---- NOTE-FOR-REFERENCE ---- auroc = 0.9499969924133628
    # ---- NOTE-FOR-REFERENCE ---- eval_loss = 0.7718211210292107
    # ---- NOTE-FOR-REFERENCE ---- fn = 36
    # ---- NOTE-FOR-REFERENCE ---- fp = 91
    # ---- NOTE-FOR-REFERENCE ---- mcc = 0.7517407129772101
    # ---- NOTE-FOR-REFERENCE ---- tn = 652
    # ---- NOTE-FOR-REFERENCE ---- tp = 322
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----   > (322+652)/(322+652+36+91)
    # ---- NOTE-FOR-REFERENCE ----   [1] 0.8846503
    # ------------------------------------------------------------------------
    # ---- NOTE-FOR-REFERENCE ----     model_name='bert-large-uncased'
    # ---- NOTE-FOR-REFERENCE ----     simple_transformers_use_cuda=False
    # ---- NOTE-FOR-REFERENCE ----     num_train_epochs=5
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ---- auprc = 0.8879423540062982
    # ---- NOTE-FOR-REFERENCE ---- auroc = 0.9595366812785251
    # ---- NOTE-FOR-REFERENCE ---- eval_loss = 0.6178433844109521
    # ---- NOTE-FOR-REFERENCE ---- fn = 34
    # ---- NOTE-FOR-REFERENCE ---- fp = 87
    # ---- NOTE-FOR-REFERENCE ---- mcc = 0.7630990143720023
    # ---- NOTE-FOR-REFERENCE ---- tn = 656
    # ---- NOTE-FOR-REFERENCE ---- tp = 324
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----   > (324+656)/(324+656+34+87)
    # ---- NOTE-FOR-REFERENCE ----   [1] 0.8900999
    # ------------------------------------------------------------------------
    # ---- NOTE-FOR-REFERENCE ----     model_name='xlnet-base-cased'
    # ---- NOTE-FOR-REFERENCE ----     simple_transformers_use_cuda=True
    # ---- NOTE-FOR-REFERENCE ----     num_train_epochs=5
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ---- auprc = 0.808566453288734
    # ---- NOTE-FOR-REFERENCE ---- auroc = 0.9362805176056603
    # ---- NOTE-FOR-REFERENCE ---- eval_loss = 0.3930719734938896
    # ---- NOTE-FOR-REFERENCE ---- fn = 27
    # ---- NOTE-FOR-REFERENCE ---- fp = 111
    # ---- NOTE-FOR-REFERENCE ---- mcc = 0.7407784976803038
    # ---- NOTE-FOR-REFERENCE ---- tn = 632
    # ---- NOTE-FOR-REFERENCE ---- tp = 331
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----   > (331+632)/(331+632+27+111)
    # ---- NOTE-FOR-REFERENCE ----   [1] 0.8746594
    # ------------------------------------------------------------------------
    # ---- NOTE-FOR-REFERENCE ----     model_name='xlnet-base-cased'
    # ---- NOTE-FOR-REFERENCE ----     simple_transformers_use_cuda=True
    # ---- NOTE-FOR-REFERENCE ----     num_train_epochs=6
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ---- auprc = auprc = 0.334883787692963
    # ---- NOTE-FOR-REFERENCE ---- auroc = 0.5220136544433334
    # ---- NOTE-FOR-REFERENCE ---- eval_loss = 0.6468220145806022
    # ---- NOTE-FOR-REFERENCE ---- fn = 358
    # ---- NOTE-FOR-REFERENCE ---- fp = 0
    # ---- NOTE-FOR-REFERENCE ---- mcc = 0.0
    # ---- NOTE-FOR-REFERENCE ---- tn = 743
    # ---- NOTE-FOR-REFERENCE ---- tp = 0
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----   > (0+743)/(0+743+0+358)
    # ---- NOTE-FOR-REFERENCE ----   [1] 0.6748411
    # ------------------------------------------------------------------------

def main_dunmore():
    """
    The main_dunmore() function can quickly test PytorchUtility functions
    """
    # ---- NOTE-PYLINT ---- C0301: Line too long
    # pylint: disable=C0301
    # ---- NOTE-PYLINT ---- W0612 Unused Variable
    # pylint: disable=W0612
    # ---- NOTE-PYLINT ---- R0914: Too many local variables (*/15) (too-many-locals)
    # pylint: disable=R0914
    # ------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    # ------------------------------------------------------------------------
    # process_pytorch_arguments(parser)
    # ------------------------------------------------------------------------
    args: argparse.Namespace = parser.parse_args()
    DebuggingHelper.write_line_to_system_console_out(
        f'args={str(args)}')
    # ------------------------------------------------------------------------
    PytorchUtility.set_all_random_number_generator_seeds(47)
    # ------------------------------------------------------------------------
    pytorch_transformers_model_tokenizer: str = \
        'bert-base-uncased'
    pytorch_transformers_model_cache_dir_tokenizer: str = \
        'pytorch_transformers'
    # ---- NOTE ---- use a fixed transformer tokenizer cache
    pytorch_transformers_model_cache_dir_tokenizer: str = \
        os.path.join(os.path.os.sep, 'pyludispatch', '_tokenizers_cache', 'pytorch_transformers', 'bert-base-uncased')
        # ---- '/pyludispatch/_tokenizers_cache/pytorch_transformers/bert-base-uncased'
    pytorch_transformers_model_tokenizer_do_lower_case: bool = True
    pytorch_transformers_model_tokenizer_kind: int = \
        PytorchTransformersFeatureManagerBertTokenization.PYTORCH_TRANSFORMERS_TOKENIZER_KIND
    pytorch_transformers_feature_manager_bert_tokenization: \
        PytorchTransformersFeatureManagerBertTokenization = \
        PytorchTransformersFeatureManagerBertTokenization( \
            pytorch_transformers_model_tokenizer, \
            pytorch_transformers_model_cache_dir_tokenizer, \
            pytorch_transformers_model_tokenizer_do_lower_case, \
            pytorch_transformers_model_tokenizer_kind)
    # ------------------------------------------------------------------------
    dunmore_training_dataset_manager: DunmoreTrainingDatasetManager = \
        DunmoreTrainingDatasetManager( \
        pytorch_transformers_feature_manager_bert_tokenization)
    dataset_shape: Tuple[int, int, int] = dunmore_training_dataset_manager.get_shape()
    linear_layer_number_input_linear_features: int = dataset_shape[1]
    linear_layer_number_output_linear_features: int = dataset_shape[0]
    # ---- DebuggingHelper.write_line_to_system_console_out(
    # ----     f'dataset_shape: {dataset_shape}')
    # ---- DebuggingHelper.write_line_to_system_console_out(
    # ----     f'dunmore_training_dataset_manager.get_records()[0]: {dunmore_training_dataset_manager.get_records()[0]}')
    # ------------------------------------------------------------------------
    dunmore_training_feature_id_pytorch_dataset: DunmoreTrainingIdIntFeaturePytorchDataset = \
        DunmoreTrainingIdIntFeaturePytorchDataset(dunmore_training_dataset_manager)
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ---- dunmore_training_raw_str_feature_pytorch_dataset: DunmoreTrainingRawStrFeaturePytorchDataset = \
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ----     DunmoreTrainingRawStrFeaturePytorchDataset(dunmore_training_dataset_manager)
    sequential_sampler: torch.utils.data.SequentialSampler = \
        torch.utils.data.SequentialSampler(dunmore_training_feature_id_pytorch_dataset)
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ---- sequential_sampler: torch.utils.data.SequentialSampler = \
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ----     torch.utils.data.SequentialSampler(dunmore_training_raw_str_feature_pytorch_dataset)
    base_pytorch_dataset_nop_collator: BasePytorchDatasetNopCollator = \
        BasePytorchDatasetNopCollator()
    data_loader_batch_size: int = 3
    data_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        dataset=dunmore_training_feature_id_pytorch_dataset,
        sampler=sequential_sampler,
        batch_size=data_loader_batch_size,
        collate_fn=base_pytorch_dataset_nop_collator)
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ---- data_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ----     dataset=dunmore_training_raw_str_feature_pytorch_dataset,
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ----     sampler=sequential_sampler,
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ----     batch_size=data_loader_batch_size)
    # ------------------------------------------------------------------------
    sequence_max_length: int = 64
    sequence_unknown_element_id: int = 0
    embedding_layer_number_embeddings: int = dataset_shape[1]
    embedding_layer_dimension: int = 256
    epoches: int = 30
    learning_rate: float = 0.01
    model_name_prefix: str = "model"
    batch_logging_interval: int = 4
    epoch_checkpoint_interval: int = 4
    torch_neural_network_module_linear_embedding: PytorchUtility.TorchNeuralNetworkModuleLinearEmbedding = \
        PytorchUtility.TorchNeuralNetworkModuleLinearEmbedding( \
            sequence_max_length, \
            sequence_unknown_element_id, \
            embedding_layer_number_embeddings, \
            embedding_layer_dimension, \
            linear_layer_number_input_linear_features, \
            linear_layer_number_output_linear_features)
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ---- torch_neural_network_module_linear_embedding: PytorchUtility.TorchNeuralNetworkModuleLinearEmbeddingRaw = \
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ----     PytorchUtility.TorchNeuralNetworkModuleLinearEmbeddingRaw( \
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ----         pytorch_transformers_feature_manager_bert_tokenization, \
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ----         sequence_max_length, \
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ----         sequence_unknown_element_id, \
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ----         embedding_layer_number_embeddings, \
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ----         embedding_layer_dimension, \
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ----         linear_layer_number_input_linear_features, \
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ----         linear_layer_number_output_linear_features)
    # ------------------------------------------------------------------------
    label_ids_true_after_training, label_ids_predicted_after_training, classification_report_object_before_training = \
        PytorchUtility.predict_model( \
            torch_neural_network_module_linear_embedding, \
            data_loader)
    # ------------------------------------------------------------------------
    PytorchUtility.train_model( \
        torch_neural_network_module_linear_embedding, \
        data_loader, \
        epoches, \
        learning_rate, \
        model_name_prefix, \
        batch_logging_interval, \
        epoch_checkpoint_interval)
    # ------------------------------------------------------------------------
    label_ids_true_after_training, label_ids_predicted_after_training, classification_report_object_after_training = \
        PytorchUtility.predict_model( \
            torch_neural_network_module_linear_embedding, \
            data_loader)
    # ------------------------------------------------------------------------
    DebuggingHelper.write_line_to_system_console_out(
        f'classification_report_object_before_training:'
        f'\n{classification_report_object_before_training}')
    DebuggingHelper.write_line_to_system_console_out(
        f'classification_report_object_after_training:'
        f'\n{classification_report_object_after_training}')
    # ------------------------------------------------------------------------
    dunmore_testing_dataset_manager: DunmoreTestingDatasetManager = DunmoreTestingDatasetManager( \
        pytorch_transformers_feature_manager_bert_tokenization)
    dataset_shape: Tuple[int, int, int] = dunmore_testing_dataset_manager.get_shape()
    linear_layer_number_input_linear_features: int = dataset_shape[1]
    linear_layer_number_output_linear_features: int = dataset_shape[0]
    # ---- DebuggingHelper.write_line_to_system_console_out(
    # ----     f'dataset_shape: {dataset_shape}')
    # ---- DebuggingHelper.write_line_to_system_console_out(
    # ----     f'dunmore_testing_dataset_manager.get_records()[0]: {dunmore_testing_dataset_manager.get_records()[0]}')
    # ------------------------------------------------------------------------
    dunmore_testing_feature_id_pytorch_dataset: DunmoreTestingIdIntFeaturePytorchDataset = \
        DunmoreTestingIdIntFeaturePytorchDataset(dunmore_testing_dataset_manager)
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ---- dunmore_testing_raw_str_feature_pytorch_dataset: DunmoreTestingRawStrFeaturePytorchDataset = \
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ----     DunmoreTestingRawStrFeaturePytorchDataset(dunmore_testing_dataset_manager)
    data_loader_for_testing_batch_size: int = 3
    data_loader_for_testing: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        dataset=dunmore_testing_feature_id_pytorch_dataset,
        batch_size=data_loader_for_testing_batch_size,
        collate_fn=base_pytorch_dataset_nop_collator)
    # ------------------------------------------------------------------------
    label_ids_true_testing, label_ids_predicted_testing, classification_report_object_testing = \
        PytorchUtility.predict_model( \
            torch_neural_network_module_linear_embedding, \
            data_loader_for_testing)
    # ------------------------------------------------------------------------
    DebuggingHelper.write_line_to_system_console_out(
        f'classification_report_object_before_training:'
        f'\n{classification_report_object_before_training}')
    DebuggingHelper.write_line_to_system_console_out(
        f'classification_report_object_after_training:\n'
        f'{classification_report_object_after_training}')
    # ------------------------------------------------------------------------
    DebuggingHelper.write_line_to_system_console_out(
        f'classification_report_object_testing:\n'
        f'{classification_report_object_testing}')
    # ------------------------------------------------------------------------
    # ---- NOTE-PYLINT ---- C0301: Line too long
    # pylint: disable=C0301
    # ------------------------------------------------------------------------
    # ---- NOTE-FOR-REFERENCE ----     embedding_layer_dimension: int = 32
    # ---- NOTE-FOR-REFERENCE ----     epoches: int = 10
    # ---- NOTE-FOR-REFERENCE ----     learning_rate: float = 0.01
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ---- STDOUT-INFO: [2021-07-23T14:19:36.330707][main_dunmore @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:198][DEBUG:10]: classification_report_object_before_training:
    # ---- NOTE-FOR-REFERENCE ----               precision    recall  f1-score   support
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----            0       0.55      0.01      0.01      1067
    # ---- NOTE-FOR-REFERENCE ----            1       0.44      0.99      0.61       827
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----     accuracy                           0.44      1894
    # ---- NOTE-FOR-REFERENCE ----    macro avg       0.49      0.50      0.31      1894
    # ---- NOTE-FOR-REFERENCE ---- weighted avg       0.50      0.44      0.27      1894
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ---- STDOUT-INFO: [2021-07-23T14:19:36.334706][main_dunmore @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:200][DEBUG:10]: classification_report_object_after_training:
    # ---- NOTE-FOR-REFERENCE ----               precision    recall  f1-score   support
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----            0       0.73      0.75      0.74      1067
    # ---- NOTE-FOR-REFERENCE ----            1       0.66      0.64      0.65       827
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----     accuracy                           0.70      1894
    # ---- NOTE-FOR-REFERENCE ----    macro avg       0.69      0.69      0.69      1894
    # ---- NOTE-FOR-REFERENCE ---- weighted avg       0.70      0.70      0.70      1894
    # ------------------------------------------------------------------------
    # ---- NOTE-FOR-REFERENCE ----     embedding_layer_dimension: int = 64
    # ---- NOTE-FOR-REFERENCE ----     epoches: int = 10
    # ---- NOTE-FOR-REFERENCE ----     learning_rate: float = 0.01
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ---- STDOUT-INFO: [2021-07-23T14:25:03.739163][main_dunmore @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:198][DEBUG:10]: classification_report_object_before_training:
    # ---- NOTE-FOR-REFERENCE ----               precision    recall  f1-score   support
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----            0       0.00      0.00      0.00      1067
    # ---- NOTE-FOR-REFERENCE ----            1       0.44      1.00      0.61       827
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----     accuracy                           0.44      1894
    # ---- NOTE-FOR-REFERENCE ----    macro avg       0.22      0.50      0.30      1894
    # ---- NOTE-FOR-REFERENCE ---- weighted avg       0.19      0.44      0.27      1894
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ---- STDOUT-INFO: [2021-07-23T14:25:03.742121][main_dunmore @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:200][DEBUG:10]: classification_report_object_after_training:
    # ---- NOTE-FOR-REFERENCE ----               precision    recall  f1-score   support
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----            0       0.81      0.81      0.81      1067
    # ---- NOTE-FOR-REFERENCE ----            1       0.75      0.76      0.76       827
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----     accuracy                           0.79      1894
    # ---- NOTE-FOR-REFERENCE ----    macro avg       0.78      0.78      0.78      1894
    # ---- NOTE-FOR-REFERENCE ---- weighted avg       0.79      0.79      0.79      1894
    # ------------------------------------------------------------------------
    # ---- NOTE-FOR-REFERENCE ----     embedding_layer_dimension: int = 64
    # ---- NOTE-FOR-REFERENCE ----     epoches: int = 20
    # ---- NOTE-FOR-REFERENCE ----     learning_rate: float = 0.01
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ---- STDOUT-INFO: [2021-07-23T15:12:44.527032][main_dunmore @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:198][DEBUG:10]: classification_report_object_before_training:
    # ---- NOTE-FOR-REFERENCE ----               precision    recall  f1-score   support
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----            0       0.00      0.00      0.00      1067
    # ---- NOTE-FOR-REFERENCE ----            1       0.44      1.00      0.61       827
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----     accuracy                           0.44      1894
    # ---- NOTE-FOR-REFERENCE ----    macro avg       0.22      0.50      0.30      1894
    # ---- NOTE-FOR-REFERENCE ---- weighted avg       0.19      0.44      0.27      1894
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ---- STDOUT-INFO: [2021-07-23T15:12:44.530039][main_dunmore @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:200][DEBUG:10]: classification_report_object_after_training:
    # ---- NOTE-FOR-REFERENCE ----               precision    recall  f1-score   support
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----            0       0.83      0.85      0.84      1067
    # ---- NOTE-FOR-REFERENCE ----            1       0.80      0.77      0.79       827
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----     accuracy                           0.82      1894
    # ---- NOTE-FOR-REFERENCE ----    macro avg       0.82      0.81      0.82      1894
    # ---- NOTE-FOR-REFERENCE ---- weighted avg       0.82      0.82      0.82      1894
    # ------------------------------------------------------------------------
    # ---- NOTE-FOR-REFERENCE ----     embedding_layer_dimension: int = 128
    # ---- NOTE-FOR-REFERENCE ----     epoches: int = 20
    # ---- NOTE-FOR-REFERENCE ----     learning_rate: float = 0.01
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ---- STDOUT-INFO: [2021-07-23T15:22:33.785657][main_dunmore @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:198][DEBUG:10]: classification_report_object_before_training:
    # ---- NOTE-FOR-REFERENCE ----               precision    recall  f1-score   support
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----            0       0.70      0.01      0.01      1067
    # ---- NOTE-FOR-REFERENCE ----            1       0.44      1.00      0.61       827
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----     accuracy                           0.44      1894
    # ---- NOTE-FOR-REFERENCE ----    macro avg       0.57      0.50      0.31      1894
    # ---- NOTE-FOR-REFERENCE ---- weighted avg       0.59      0.44      0.27      1894
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ---- STDOUT-INFO: [2021-07-23T15:22:33.788657][main_dunmore @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:200][DEBUG:10]: classification_report_object_after_training:
    # ---- NOTE-FOR-REFERENCE ----               precision    recall  f1-score   support
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----            0       0.90      0.96      0.93      1067
    # ---- NOTE-FOR-REFERENCE ----            1       0.94      0.86      0.90       827
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----     accuracy                           0.92      1894
    # ---- NOTE-FOR-REFERENCE ----    macro avg       0.92      0.91      0.91      1894
    # ---- NOTE-FOR-REFERENCE ---- weighted avg       0.92      0.92      0.92      1894
    # ------------------------------------------------------------------------
    # ---- NOTE-FOR-REFERENCE ----     embedding_layer_dimension: int = 128
    # ---- NOTE-FOR-REFERENCE ----     epoches: int = 20
    # ---- NOTE-FOR-REFERENCE ----     learning_rate: float = 0.01
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ---- STDOUT-INFO: [2021-07-23T15:38:48.347948][main_dunmore @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:198][DEBUG:10]: classification_report_object_before_training:
    # ---- NOTE-FOR-REFERENCE ----               precision    recall  f1-score   support
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----            0       0.55      0.71      0.62      1067
    # ---- NOTE-FOR-REFERENCE ----            1       0.40      0.24      0.30       827
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----     accuracy                           0.51      1894
    # ---- NOTE-FOR-REFERENCE ----    macro avg       0.47      0.48      0.46      1894
    # ---- NOTE-FOR-REFERENCE ---- weighted avg       0.48      0.51      0.48      1894
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ---- STDOUT-INFO: [2021-07-23T15:38:48.350757][main_dunmore @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:200][DEBUG:10]: classification_report_object_after_training:
    # ---- NOTE-FOR-REFERENCE ----               precision    recall  f1-score   support
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----            0       0.92      1.00      0.96      1067
    # ---- NOTE-FOR-REFERENCE ----            1       1.00      0.88      0.94       827
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----     accuracy                           0.95      1894
    # ---- NOTE-FOR-REFERENCE ----    macro avg       0.96      0.94      0.95      1894
    # ---- NOTE-FOR-REFERENCE ---- weighted avg       0.95      0.95      0.95      1894
    # ------------------------------------------------------------------------
    # ---- NOTE-FOR-REFERENCE ----     embedding_layer_dimension: int = 256
    # ---- NOTE-FOR-REFERENCE ----     epoches: int = 30
    # ---- NOTE-FOR-REFERENCE ----     learning_rate: float = 0.01
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ---- STDOUT-INFO: [2021-07-23T16:35:53.040549][main_dunmore @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:198][DEBUG:10]: classification_report_object_before_training:
    # ---- NOTE-FOR-REFERENCE ----               precision    recall  f1-score   support
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----            0       0.55      0.71      0.62      1067
    # ---- NOTE-FOR-REFERENCE ----            1       0.40      0.24      0.30       827
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----     accuracy                           0.51      1894
    # ---- NOTE-FOR-REFERENCE ----    macro avg       0.47      0.48      0.46      1894
    # ---- NOTE-FOR-REFERENCE ---- weighted avg       0.48      0.51      0.48      1894
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ---- STDOUT-INFO: [2021-07-23T16:35:53.043550][main_dunmore @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:200][DEBUG:10]: classification_report_object_after_training:
    # ---- NOTE-FOR-REFERENCE ----               precision    recall  f1-score   support
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----            0       1.00      1.00      1.00      1067
    # ---- NOTE-FOR-REFERENCE ----            1       1.00      1.00      1.00       827
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----     accuracy                           1.00      1894
    # ---- NOTE-FOR-REFERENCE ----    macro avg       1.00      1.00      1.00      1894
    # ---- NOTE-FOR-REFERENCE ---- weighted avg       1.00      1.00      1.00      1894
    # ---- NOTE-FOR-REFERENCE ---- STDOUT-INFO: [2021-07-23T19:26:56.368333][main_dunmore @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:232][DEBUG:10]: classification_report_object_testing:
    # ---- NOTE-FOR-REFERENCE ----               precision    recall  f1-score   support
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----            0       0.78      0.76      0.77       743
    # ---- NOTE-FOR-REFERENCE ----            1       0.53      0.56      0.55       358
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----     accuracy                           0.70      1101
    # ---- NOTE-FOR-REFERENCE ----    macro avg       0.66      0.66      0.66      1101
    # ---- NOTE-FOR-REFERENCE ---- weighted avg       0.70      0.70      0.70      1101
    # ------------------------------------------------------------------------
    # input_pytorch_model_filename: str = args.input_pytorch_model_filename
    # output_pytorch_model_json_filename: str = args.output_pytorch_model_json_filename
    # ---- TODO ---- PytorchUtility.dump_pytorch_model(
    # ---- TODO ----     input_pytorch_model_filename=input_pytorch_model_filename,
    # ---- TODO ----     output_pytorch_model_json_filename=output_pytorch_model_json_filename)
    # ------------------------------------------------------------------------
    # ---- TODO ---- PytorchUtility.check_pytorch_model(input_pytorch_model_filename=input_pytorch_model_filename)
    # ---- TODO ---- # ---- NOTE-TODO-NOT-WORKING-YET ----
    # ------------------------------------------------------------------------

def main_text_spam():
    """
    The main_text_spam() function can quickly test PytorchUtility functions
    """
    # ---- NOTE-PYLINT ---- W0612 Unused Variable
    # pylint: disable=W0612
    # ---- NOTE-PYLINT ---- R0914: Too many local variables (*/15) (too-many-locals)
    # pylint: disable=R0914
    # ------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    # ------------------------------------------------------------------------
    # process_pytorch_arguments(parser)
    # ------------------------------------------------------------------------
    args: argparse.Namespace = parser.parse_args()
    DebuggingHelper.write_line_to_system_console_out(
        f'args={str(args)}')
    # ------------------------------------------------------------------------
    PytorchUtility.set_all_random_number_generator_seeds(47)
    # ------------------------------------------------------------------------
    pytorch_transformers_model_tokenizer: str = \
        'bert-base-uncased'
    pytorch_transformers_model_cache_dir_tokenizer: str = \
        'pytorch_transformers'
    # ---- NOTE ---- use a fixed transformer tokenizer cache
    pytorch_transformers_model_cache_dir_tokenizer: str = \
        os.path.join(os.path.os.sep, 'pyludispatch', '_tokenizers_cache', 'pytorch_transformers', 'bert-base-uncased')
        # ---- '/pyludispatch/_tokenizers_cache/pytorch_transformers/bert-base-uncased'
    pytorch_transformers_model_tokenizer_do_lower_case: bool = True
    pytorch_transformers_model_tokenizer_kind: int = \
        PytorchTransformersFeatureManagerBertTokenization.PYTORCH_TRANSFORMERS_TOKENIZER_KIND
    pytorch_transformers_feature_manager_bert_tokenization: \
        PytorchTransformersFeatureManagerBertTokenization = \
        PytorchTransformersFeatureManagerBertTokenization( \
            pytorch_transformers_model_tokenizer, \
            pytorch_transformers_model_cache_dir_tokenizer, \
            pytorch_transformers_model_tokenizer_do_lower_case, \
            pytorch_transformers_model_tokenizer_kind)
    # ------------------------------------------------------------------------
    text_spam_dataset_manager: TextSpamDatasetManager = TextSpamDatasetManager( \
        pytorch_transformers_feature_manager_bert_tokenization)
    dataset_shape: Tuple[int, int, int] = text_spam_dataset_manager.get_shape()
    linear_layer_number_input_linear_features: int = dataset_shape[1]
    linear_layer_number_output_linear_features: int = dataset_shape[0]
    # ---- DebuggingHelper.write_line_to_system_console_out(
    # ----     f'dataset_shape: {dataset_shape}')
    # ---- DebuggingHelper.write_line_to_system_console_out(
    # ----     f'text_spam_dataset_manager.get_records()[0]: {text_spam_dataset_manager.get_records()[0]}')
    # ------------------------------------------------------------------------
    text_spam_feature_id_pytorch_dataset: TextSpamIntIdFeaturePytorchDataset = \
        TextSpamIntIdFeaturePytorchDataset(text_spam_dataset_manager)
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ---- text_spam_raw_feature_pytorch_dataset: TextSpamRawStrFeaturePytorchDataset = \
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ----     TextSpamRawStrFeaturePytorchDataset(text_spam_dataset_manager)
    sequential_sampler: torch.utils.data.SequentialSampler = \
        torch.utils.data.SequentialSampler(text_spam_feature_id_pytorch_dataset)
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ---- sequential_sampler: torch.utils.data.SequentialSampler = \
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ----     torch.utils.data.SequentialSampler(text_spam_raw_feature_pytorch_dataset)
    base_pytorch_dataset_nop_collator: BasePytorchDatasetNopCollator = \
        BasePytorchDatasetNopCollator()
    data_loader_batch_size: int = 3
    data_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        dataset=text_spam_feature_id_pytorch_dataset,
        sampler=sequential_sampler,
        batch_size=data_loader_batch_size,
        collate_fn=base_pytorch_dataset_nop_collator)
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ---- data_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ----     dataset=text_spam_raw_feature_pytorch_dataset,
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ----     sampler=sequential_sampler,
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ----     batch_size=data_loader_batch_size)
    # ------------------------------------------------------------------------
    sequence_max_length: int = 64
    sequence_unknown_element_id: int = 0
    embedding_layer_number_embeddings: int = dataset_shape[1]
    embedding_layer_dimension: int = 6
    epoches: int = 2
    learning_rate: float = 0.01
    model_name_prefix: str = "model"
    batch_logging_interval: int = 4
    epoch_checkpoint_interval: int = 4
    torch_neural_network_module_linear_embedding: PytorchUtility.TorchNeuralNetworkModuleLinearEmbedding = \
        PytorchUtility.TorchNeuralNetworkModuleLinearEmbedding( \
            sequence_max_length, \
            sequence_unknown_element_id, \
            embedding_layer_number_embeddings, \
            embedding_layer_dimension, \
            linear_layer_number_input_linear_features, \
            linear_layer_number_output_linear_features)
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ---- torch_neural_network_module_linear_embedding: PytorchUtility.TorchNeuralNetworkModuleLinearEmbeddingRaw = \
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ----     PytorchUtility.TorchNeuralNetworkModuleLinearEmbeddingRaw( \
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ----         pytorch_transformers_feature_manager_bert_tokenization, \
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ----         sequence_max_length, \
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ----         sequence_unknown_element_id, \
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ----         embedding_layer_number_embeddings, \
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ----         embedding_layer_dimension, \
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ----         linear_layer_number_input_linear_features, \
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ----         linear_layer_number_output_linear_features)
    # ------------------------------------------------------------------------
    label_ids_true_after_training, label_ids_predicted_after_training, classification_report_object_before_training = \
        PytorchUtility.predict_model( \
            torch_neural_network_module_linear_embedding, \
            data_loader)
    # ------------------------------------------------------------------------
    PytorchUtility.train_model( \
        torch_neural_network_module_linear_embedding, \
        data_loader, \
        epoches, \
        learning_rate, \
        model_name_prefix, \
        batch_logging_interval, \
        epoch_checkpoint_interval)
    # ------------------------------------------------------------------------
    label_ids_true_after_training, label_ids_predicted_after_training, classification_report_object_after_training = \
        PytorchUtility.predict_model( \
            torch_neural_network_module_linear_embedding, \
            data_loader)
    # ------------------------------------------------------------------------
    DebuggingHelper.write_line_to_system_console_out(
        f'classification_report_object_before_training:\n{classification_report_object_before_training}')
    DebuggingHelper.write_line_to_system_console_out(
        f'classification_report_object_after_training:\n{classification_report_object_after_training}')
    # ------------------------------------------------------------------------
    # ---- NOTE-FOR-REFERENCE ---- STDOUT-INFO: [2021-07-20T23:02:00.370132][main_text_spam @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:178][DEBUG:10]: classification_report_object_before_training:
    # ---- NOTE-FOR-REFERENCE ----               precision    recall  f1-score   support
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----            0       0.91      0.73      0.81      4826
    # ---- NOTE-FOR-REFERENCE ----            1       0.24      0.53      0.33       747
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----     accuracy                           0.71      5573
    # ---- NOTE-FOR-REFERENCE ----    macro avg       0.57      0.63      0.57      5573
    # ---- NOTE-FOR-REFERENCE ---- weighted avg       0.82      0.71      0.75      5573
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ---- STDOUT-INFO: [2021-07-20T23:02:00.374131][main_text_spam @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:180][DEBUG:10]: classification_report_object_after_training:
    # ---- NOTE-FOR-REFERENCE ----               precision    recall  f1-score   support
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----            0       0.96      0.97      0.97      4826
    # ---- NOTE-FOR-REFERENCE ----            1       0.81      0.72      0.77       747
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----     accuracy                           0.94      5573
    # ---- NOTE-FOR-REFERENCE ----    macro avg       0.89      0.85      0.87      5573
    # ---- NOTE-FOR-REFERENCE ---- weighted avg       0.94      0.94      0.94      5573
    # ------------------------------------------------------------------------
    # input_pytorch_model_filename: str = args.input_pytorch_model_filename
    # output_pytorch_model_json_filename: str = args.output_pytorch_model_json_filename
    # ---- TODO ---- PytorchUtility.dump_pytorch_model(
    # ---- TODO ----     input_pytorch_model_filename=input_pytorch_model_filename,
    # ---- TODO ----     output_pytorch_model_json_filename=output_pytorch_model_json_filename)
    # ------------------------------------------------------------------------
    # ---- TODO ---- PytorchUtility.check_pytorch_model(input_pytorch_model_filename=input_pytorch_model_filename)
    # ---- TODO ---- # ---- NOTE-TODO-NOT-WORKING-YET ----
    # ------------------------------------------------------------------------

def main_text_spam_raw():
    """
    The main_text_spam_raw() function can quickly test PytorchUtility functions
    """
    # ---- NOTE-PYLINT ---- W0612 Unused Variable
    # pylint: disable=W0612
    # ---- NOTE-PYLINT ---- R0914: Too many local variables (*/15) (too-many-locals)
    # pylint: disable=R0914
    # ------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    # ------------------------------------------------------------------------
    # process_pytorch_arguments(parser)
    # ------------------------------------------------------------------------
    args: argparse.Namespace = parser.parse_args()
    DebuggingHelper.write_line_to_system_console_out(
        f'args={str(args)}')
    # ------------------------------------------------------------------------
    PytorchUtility.set_all_random_number_generator_seeds(47)
    # ------------------------------------------------------------------------
    pytorch_transformers_model_tokenizer: str = \
        'bert-base-uncased'
    pytorch_transformers_model_cache_dir_tokenizer: str = \
        'pytorch_transformers'
    # ---- NOTE ---- use a fixed transformer tokenizer cache
    pytorch_transformers_model_cache_dir_tokenizer: str = \
        os.path.join(os.path.os.sep, 'pyludispatch', '_tokenizers_cache', 'pytorch_transformers', 'bert-base-uncased')
        # ---- '/pyludispatch/_tokenizers_cache/pytorch_transformers/bert-base-uncased'
    pytorch_transformers_model_tokenizer_do_lower_case: bool = True
    pytorch_transformers_model_tokenizer_kind: int = \
        PytorchTransformersFeatureManagerBertTokenization.PYTORCH_TRANSFORMERS_TOKENIZER_KIND
    pytorch_transformers_feature_manager_bert_tokenization: \
        PytorchTransformersFeatureManagerBertTokenization = \
        PytorchTransformersFeatureManagerBertTokenization( \
            pytorch_transformers_model_tokenizer, \
            pytorch_transformers_model_cache_dir_tokenizer, \
            pytorch_transformers_model_tokenizer_do_lower_case, \
            pytorch_transformers_model_tokenizer_kind)
    # ------------------------------------------------------------------------
    text_spam_dataset_manager: TextSpamDatasetManager = TextSpamDatasetManager( \
        pytorch_transformers_feature_manager_bert_tokenization)
    dataset_shape: Tuple[int, int, int] = text_spam_dataset_manager.get_shape()
    linear_layer_number_input_linear_features: int = dataset_shape[1]
    linear_layer_number_output_linear_features: int = dataset_shape[0]
    # ---- DebuggingHelper.write_line_to_system_console_out(
    # ----     f'dataset_shape: {dataset_shape}')
    # ---- DebuggingHelper.write_line_to_system_console_out(
    # ----     f'text_spam_dataset_manager.get_records()[0]: {text_spam_dataset_manager.get_records()[0]}')
    # ------------------------------------------------------------------------
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-3-works ---- text_spam_feature_id_pytorch_dataset: TextSpamIntIdFeaturePytorchDataset = \
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-3-works ----     TextSpamIntIdFeaturePytorchDataset(text_spam_dataset_manager)
    text_spam_raw_str_feature_pytorch_dataset: TextSpamRawStrFeaturePytorchDataset = \
        TextSpamRawStrFeaturePytorchDataset(text_spam_dataset_manager)
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-3-works ---- sequential_sampler: torch.utils.data.SequentialSampler = \
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-3-works ----     torch.utils.data.SequentialSampler(text_spam_feature_id_pytorch_dataset)
    sequential_sampler: torch.utils.data.SequentialSampler = \
        torch.utils.data.SequentialSampler(text_spam_raw_str_feature_pytorch_dataset)
    data_loader_batch_size: int = 3
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-3-works ---- base_pytorch_dataset_nop_collator: BasePytorchDatasetNopCollator = \
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-3-works ----     BasePytorchDatasetNopCollator()
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-3-works ---- data_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-3-works ----     dataset=text_spam_feature_id_pytorch_dataset,
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-3-works ----     sampler=sequential_sampler,
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-3-works ----     batch_size=data_loader_batch_size,
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-3-works ----     collate_fn=base_pytorch_dataset_nop_collator)
    data_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        dataset=text_spam_raw_str_feature_pytorch_dataset,
        sampler=sequential_sampler,
        batch_size=data_loader_batch_size,
        collate_fn=default_collate)
    # ------------------------------------------------------------------------
    sequence_max_length: int = 64
    sequence_unknown_element_id: int = 0
    embedding_layer_number_embeddings: int = dataset_shape[1]
    embedding_layer_dimension: int = 6
    epoches: int = 2
    learning_rate: float = 0.01
    model_name_prefix: str = "model"
    batch_logging_interval: int = 4
    epoch_checkpoint_interval: int = 4
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-3-works ---- torch_neural_network_module_linear_embedding: PytorchUtility.TorchNeuralNetworkModuleLinearEmbedding = \
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-3-works ----     PytorchUtility.TorchNeuralNetworkModuleLinearEmbedding( \
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-3-works ----         sequence_max_length, \
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-3-works ----         sequence_unknown_element_id, \
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-3-works ----         embedding_layer_number_embeddings, \
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-3-works ----         embedding_layer_dimension, \
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-3-works ----         linear_layer_number_input_linear_features, \
    # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-3-works ----         linear_layer_number_output_linear_features)
    torch_neural_network_module_linear_embedding: PytorchUtility.TorchNeuralNetworkModuleLinearEmbeddingRaw = \
        PytorchUtility.TorchNeuralNetworkModuleLinearEmbeddingRaw( \
            pytorch_transformers_feature_manager_bert_tokenization, \
            sequence_max_length, \
            sequence_unknown_element_id, \
            embedding_layer_number_embeddings, \
            embedding_layer_dimension, \
            linear_layer_number_input_linear_features, \
            linear_layer_number_output_linear_features)
    # ------------------------------------------------------------------------
    label_ids_true_after_training, label_ids_predicted_after_training, classification_report_object_before_training = \
        PytorchUtility.predict_model( \
            torch_neural_network_module_linear_embedding, \
            data_loader)
    # ------------------------------------------------------------------------
    PytorchUtility.train_model( \
        torch_neural_network_module_linear_embedding, \
        data_loader, \
        epoches, \
        learning_rate, \
        model_name_prefix, \
        batch_logging_interval, \
        epoch_checkpoint_interval)
    # ------------------------------------------------------------------------
    label_ids_true_after_training, label_ids_predicted_after_training, classification_report_object_after_training = \
        PytorchUtility.predict_model( \
            torch_neural_network_module_linear_embedding, \
            data_loader)
    # ------------------------------------------------------------------------
    DebuggingHelper.write_line_to_system_console_out(
        f'classification_report_object_before_training:\n{classification_report_object_before_training}')
    DebuggingHelper.write_line_to_system_console_out(
        f'classification_report_object_after_training:\n{classification_report_object_after_training}')
    # ------------------------------------------------------------------------
    # ---- NOTE-FOR-REFERENCE ---- STDOUT-INFO: [2021-07-20T23:06:05.653736][main_text_spam_raw @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:318][DEBUG:10]: classification_report_object_before_training:
    # ---- NOTE-FOR-REFERENCE ----               precision    recall  f1-score   support
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----            0       0.91      0.73      0.81      4826
    # ---- NOTE-FOR-REFERENCE ----            1       0.24      0.53      0.33       747
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----     accuracy                           0.71      5573
    # ---- NOTE-FOR-REFERENCE ----    macro avg       0.57      0.63      0.57      5573
    # ---- NOTE-FOR-REFERENCE ---- weighted avg       0.82      0.71      0.75      5573
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ---- STDOUT-INFO: [2021-07-20T23:06:05.656712][main_text_spam_raw @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:320][DEBUG:10]: classification_report_object_after_training:
    # ---- NOTE-FOR-REFERENCE ----               precision    recall  f1-score   support
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----            0       0.96      0.97      0.97      4826
    # ---- NOTE-FOR-REFERENCE ----            1       0.81      0.72      0.77       747
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----     accuracy                           0.94      5573
    # ---- NOTE-FOR-REFERENCE ----    macro avg       0.89      0.85      0.87      5573
    # ---- NOTE-FOR-REFERENCE ---- weighted avg       0.94      0.94      0.94      5573
    # ------------------------------------------------------------------------
    # input_pytorch_model_filename: str = args.input_pytorch_model_filename
    # output_pytorch_model_json_filename: str = args.output_pytorch_model_json_filename
    # ---- TODO ---- PytorchUtility.dump_pytorch_model(
    # ---- TODO ----     input_pytorch_model_filename=input_pytorch_model_filename,
    # ---- TODO ----     output_pytorch_model_json_filename=output_pytorch_model_json_filename)
    # ------------------------------------------------------------------------
    # ---- TODO ---- PytorchUtility.check_pytorch_model(input_pytorch_model_filename=input_pytorch_model_filename)
    # ---- TODO ---- # ---- NOTE-TODO-NOT-WORKING-YET ----
    # ------------------------------------------------------------------------

def main_iris():
    """
    The main_iris() function can quickly test PytorchUtility functions
    """
    # ---- NOTE-PYLINT ---- W0612 Unused Variable
    # pylint: disable=W0612
    # ---- NOTE-PYLINT ---- R0914: Too many local variables (*/15) (too-many-locals)
    # pylint: disable=R0914
    # ------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    # ------------------------------------------------------------------------
    # process_pytorch_arguments(parser)
    # ------------------------------------------------------------------------
    args: argparse.Namespace = parser.parse_args()
    DebuggingHelper.write_line_to_system_console_out(
        f'args={str(args)}')
    # ------------------------------------------------------------------------
    PytorchUtility.set_all_random_number_generator_seeds(47)
    # ------------------------------------------------------------------------
    iris_dataset_manager: IrisDatasetManager = \
        IrisDatasetManager()
    dataset_shape: Tuple[int, int, int] = iris_dataset_manager.get_shape()
    linear_layer_number_input_linear_features: int = dataset_shape[1]
    linear_layer_number_output_linear_features: int = dataset_shape[0]
    DebuggingHelper.write_line_to_system_console_out(
        f'dataset_shape: {dataset_shape}')
    # DebuggingHelper.write_line_to_system_console_out(
    #     f'iris_dataset_manager.get_records(): {iris_dataset_manager.get_records()}')
    # ------------------------------------------------------------------------
    iris_pytorch_dataset: IrisPytorchDataset = \
        IrisPytorchDataset(iris_dataset_manager)
    sequential_sampler: torch.utils.data.SequentialSampler = \
        torch.utils.data.SequentialSampler(iris_pytorch_dataset)
    base_pytorch_dataset_nop_collator: BasePytorchDatasetNopCollator = \
        BasePytorchDatasetNopCollator()
    data_loader_batch_size: int = 3
    data_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        dataset=iris_pytorch_dataset,
        sampler=sequential_sampler,
        batch_size=data_loader_batch_size,
        collate_fn=base_pytorch_dataset_nop_collator)
    # ------------------------------------------------------------------------
    epoches: int = 128
    learning_rate: float = 0.01
    model_name_prefix: str = "model"
    batch_logging_interval: int = 4
    epoch_checkpoint_interval: int = 4
    torch_neural_network_module_linear: PytorchUtility.TorchNeuralNetworkModuleLinear = \
        PytorchUtility.TorchNeuralNetworkModuleLinear( \
            linear_layer_number_input_linear_features, \
            linear_layer_number_output_linear_features)
    # ------------------------------------------------------------------------
    label_ids_true_after_training, label_ids_predicted_after_training, classification_report_object_before_training = \
        PytorchUtility.predict_model( \
            torch_neural_network_module_linear, \
            data_loader)
    # ------------------------------------------------------------------------
    PytorchUtility.train_model( \
        torch_neural_network_module_linear, \
        data_loader, \
        epoches, \
        learning_rate, \
        model_name_prefix, \
        batch_logging_interval, \
        epoch_checkpoint_interval)
    # ------------------------------------------------------------------------
    label_ids_true_after_training, label_ids_predicted_after_training, classification_report_object_after_training = \
        PytorchUtility.predict_model( \
            torch_neural_network_module_linear, \
            data_loader)
    # ------------------------------------------------------------------------
    DebuggingHelper.write_line_to_system_console_out(
        f'classification_report_object_before_training:\n{classification_report_object_before_training}')
    DebuggingHelper.write_line_to_system_console_out(
        f'classification_report_object_after_training:\n{classification_report_object_after_training}')
    # ------------------------------------------------------------------------
    # ---- NOTE-FOR-REFERENCE ---- epoches = 32
    # ---- NOTE-FOR-REFERENCE ---- STDOUT-INFO: [2021-07-20T23:07:59.290968][main_iris @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:422][DEBUG:10]: classification_report_object_before_training:
    # ---- NOTE-FOR-REFERENCE ----               precision    recall  f1-score   support
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----            0       0.00      0.00      0.00        50
    # ---- NOTE-FOR-REFERENCE ----            1       0.33      1.00      0.50        50
    # ---- NOTE-FOR-REFERENCE ----            2       0.00      0.00      0.00        50
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----     accuracy                           0.33       150
    # ---- NOTE-FOR-REFERENCE ----    macro avg       0.11      0.33      0.17       150
    # ---- NOTE-FOR-REFERENCE ---- weighted avg       0.11      0.33      0.17       150
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ---- STDOUT-INFO: [2021-07-20T23:07:59.293968][main_iris @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:424][DEBUG:10]: classification_report_object_after_training:
    # ---- NOTE-FOR-REFERENCE ----               precision    recall  f1-score   support
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----            0       1.00      1.00      1.00        50
    # ---- NOTE-FOR-REFERENCE ----            1       0.00      0.00      0.00        50
    # ---- NOTE-FOR-REFERENCE ----            2       0.50      1.00      0.67        50
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----     accuracy                           0.67       150
    # ---- NOTE-FOR-REFERENCE ----    macro avg       0.50      0.67      0.56       150
    # ---- NOTE-FOR-REFERENCE ---- weighted avg       0.50      0.67      0.56       150
    # ------------------------------------------------------------------------
    # ---- NOTE-FOR-REFERENCE ---- epoches = 64
    # ---- NOTE-FOR-REFERENCE ---- STDOUT-INFO: [2021-07-20T23:09:30.759652][main_iris @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:422][DEBUG:10]: classification_report_object_before_training:
    # ---- NOTE-FOR-REFERENCE ----               precision    recall  f1-score   support
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----            0       0.00      0.00      0.00        50
    # ---- NOTE-FOR-REFERENCE ----            1       0.33      1.00      0.50        50
    # ---- NOTE-FOR-REFERENCE ----            2       0.00      0.00      0.00        50
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----     accuracy                           0.33       150
    # ---- NOTE-FOR-REFERENCE ----    macro avg       0.11      0.33      0.17       150
    # ---- NOTE-FOR-REFERENCE ---- weighted avg       0.11      0.33      0.17       150
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ---- STDOUT-INFO: [2021-07-20T23:09:30.761639][main_iris @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:424][DEBUG:10]: classification_report_object_after_training:
    # ---- NOTE-FOR-REFERENCE ----               precision    recall  f1-score   support
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----            0       1.00      1.00      1.00        50
    # ---- NOTE-FOR-REFERENCE ----            1       1.00      0.14      0.25        50
    # ---- NOTE-FOR-REFERENCE ----            2       0.54      1.00      0.70        50
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----     accuracy                           0.71       150
    # ---- NOTE-FOR-REFERENCE ----    macro avg       0.85      0.71      0.65       150
    # ---- NOTE-FOR-REFERENCE ---- weighted avg       0.85      0.71      0.65       150
    # ------------------------------------------------------------------------
    # ---- NOTE-FOR-REFERENCE ---- epoches = 128
    # ---- NOTE-FOR-REFERENCE ---- STDOUT-INFO: [2021-07-20T23:11:49.351859][main_iris @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:422][DEBUG:10]: classification_report_object_before_training:
    # ---- NOTE-FOR-REFERENCE ----               precision    recall  f1-score   support
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----            0       0.00      0.00      0.00        50
    # ---- NOTE-FOR-REFERENCE ----            1       0.33      1.00      0.50        50
    # ---- NOTE-FOR-REFERENCE ----            2       0.00      0.00      0.00        50
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----     accuracy                           0.33       150
    # ---- NOTE-FOR-REFERENCE ----    macro avg       0.11      0.33      0.17       150
    # ---- NOTE-FOR-REFERENCE ---- weighted avg       0.11      0.33      0.17       150
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ---- STDOUT-INFO: [2021-07-20T23:11:49.353860][main_iris @ d:\git\EmbedML\private\hunyang\project\LanguageUnderstandingOpenSource\src\python\utility\pytorch_utility\app_pytorch_utility.py:424][DEBUG:10]: classification_report_object_after_training:
    # ---- NOTE-FOR-REFERENCE ----               precision    recall  f1-score   support
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----            0       1.00      1.00      1.00        50
    # ---- NOTE-FOR-REFERENCE ----            1       1.00      0.60      0.75        50
    # ---- NOTE-FOR-REFERENCE ----            2       0.71      1.00      0.83        50
    # ---- NOTE-FOR-REFERENCE ----
    # ---- NOTE-FOR-REFERENCE ----     accuracy                           0.87       150
    # ---- NOTE-FOR-REFERENCE ----    macro avg       0.90      0.87      0.86       150
    # ---- NOTE-FOR-REFERENCE ---- weighted avg       0.90      0.87      0.86       150
    # ------------------------------------------------------------------------
    # input_pytorch_model_filename: str = args.input_pytorch_model_filename
    # output_pytorch_model_json_filename: str = args.output_pytorch_model_json_filename
    # ---- TODO ---- PytorchUtility.dump_pytorch_model(
    # ---- TODO ----     input_pytorch_model_filename=input_pytorch_model_filename,
    # ---- TODO ----     output_pytorch_model_json_filename=output_pytorch_model_json_filename)
    # ------------------------------------------------------------------------
    # ---- TODO ---- PytorchUtility.check_pytorch_model(input_pytorch_model_filename=input_pytorch_model_filename)
    # ---- TODO ---- # ---- NOTE-TODO-NOT-WORKING-YET ----
    # ------------------------------------------------------------------------

def main_functional_tests():
    """
    The main_functional_tests() function can quickly test PytorchUtility function functionalities.
    """
    # main_iris()
    # main_text_spam_raw()
    # main_text_spam()
    # main_dunmore()
    # main_dunmore_simple_transformers()
    main_kaggle_entity_annotated_corpus_entity_per_token_classification()

def main():
    """
    The main() function can quickly test PytorchUtility functions.
    """
    main_functional_tests()

if __name__ == '__main__':
    main()
