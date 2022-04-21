# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
This module implements feature manager objects using PytorchTransformers tokenizer.
"""

import os

from typing import List

from transformers import BertTokenizer

from model.language_understanding.featurizer.\
    base_data_feature_manager_single_label_pytorch_transformers \
    import PytorchTransformersSingleLabelDataFeatureManager

from model.language_understanding.helper.\
    pytorch_language_understanding_transformers_helper \
    import PytorchLanguageUnderstandingTransformersPretainedModelHelper

from data.manager.base_record_generator \
    import BaseRecordGenerator
from data.manager.base_data_manager \
    import BaseDataManager

from utility.io_helper.io_helper \
    import IoHelper
from utility.debugging_helper.debugging_helper \
    import DebuggingHelper
from utility.configuration_helper.configuration_helper \
    import ConfigurationHelper

class TokenizationPytorchTransformersSingleLabelDataFeatureManager(\
    PytorchTransformersSingleLabelDataFeatureManager):
    """
    This class can create features on an input text through a tokenizer.
    """
    PYTORCH_TRANSFORMERS_TOKENIZER_KIND: int = 0
    PYTORCH_TRANSFORMERS_WORDPIECE_TOKENIZER_KIND: int = 1
    PYTORCH_TRANSFORMERS_BASIC_TOKENIZER_KIND: int = 2

    def __init__(self, \
        data_manager: BaseDataManager, \
        featurization_max_sequence_length: int, \
        featurization_dataloader_batch_size: int, \
        pytorch_transformers_model_tokenizer: str = \
            'bert-base-uncased', \
        pytorch_transformers_model_cache_dir_tokenizer: str = \
            os.path.join( \
                ConfigurationHelper.MODEL_TOKENIZER_CACHE_DIRECTORY_STAMPED, \
                'pytorch_transformers'), \
        pytorch_transformers_model_tokenizer_do_lower_case: bool = True, \
        pytorch_transformers_model_tokenizer_kind: int = \
            PYTORCH_TRANSFORMERS_TOKENIZER_KIND, \
        column_index_label: int = 0, \
        column_index_feature_text: int = 1, \
        column_index_weight: int = -1, \
        column_index_feature_text_auxiliary: int = -1, \
        column_index_identity: int = -1, \
        record_generator: BaseRecordGenerator = None, \
        string_to_replace_null_label: str = '', \
        include_out_of_value_label: bool = True, \
        default_non_existent_label_id: int = 0, \
        include_out_of_value_feature: bool = False, \
        include_only_unique_feature: bool = True, \
        default_non_existent_feature_id: int = -1):
        """
        Init with a data manager object and a tokenizer object.
        """
        # ---- NOTE-PYLINT ---- R0913: Too many arguments
        # pylint: disable=R0913
        # ---- NOTE-PYLINT ---- R0914: Too many local variables
        # pylint: disable=R0914
        super(TokenizationPytorchTransformersSingleLabelDataFeatureManager, self).__init__( \
            data_manager=data_manager,
            featurization_max_sequence_length=featurization_max_sequence_length,
            featurization_dataloader_batch_size=featurization_dataloader_batch_size,
            column_index_label=column_index_label,
            column_index_feature_text=column_index_feature_text,
            column_index_weight=column_index_weight,
            column_index_feature_text_auxiliary=column_index_feature_text_auxiliary,
            column_index_identity=column_index_identity,
            record_generator=record_generator,
            string_to_replace_null_label=string_to_replace_null_label,
            include_out_of_value_label=include_out_of_value_label,
            default_non_existent_label_id=default_non_existent_label_id,
            include_out_of_value_feature=include_out_of_value_feature,
            include_only_unique_feature=include_only_unique_feature,
            default_non_existent_feature_id=default_non_existent_feature_id)
        self.pytorch_transformers_model_tokenizer = \
            pytorch_transformers_model_tokenizer
        self.pytorch_transformers_model_cache_dir_tokenizer = \
            pytorch_transformers_model_cache_dir_tokenizer
        self.pytorch_transformers_model_tokenizer_do_lower_case = \
            pytorch_transformers_model_tokenizer_do_lower_case
        self.pytorch_transformers_model_tokenizer_kind = \
            pytorch_transformers_model_tokenizer_kind
        self.tokenizer = None
        self.deserialize(self.pytorch_transformers_model_tokenizer)

    def copy_core_featurization_metadata(self, other_data_feature_manager):
        # ---- NOTE-CONCRETE-OVERRIDE-FUNCTION ----
        """
        Reference the documentation in the parent class.
        For this class, the core featurization metadata are the members being copied.
        """
        super().copy_core_featurization_metadata(
            other_data_feature_manager=other_data_feature_manager)
        other_data_feature_manager.pytorch_transformers_model_tokenizer = \
            self.pytorch_transformers_model_tokenizer
        other_data_feature_manager.pytorch_transformers_model_tokenizer_do_lower_case = \
            self.pytorch_transformers_model_tokenizer_do_lower_case
        other_data_feature_manager.pytorch_transformers_model_tokenizer_kind = \
            self.pytorch_transformers_model_tokenizer_kind

    def get_featurizer(self):
        # ---- NOTE-CONCRETE-OVERRIDE-FUNCTION ----
        """
        Return a featurizer.
        """
        return self.tokenizer
    def set_featurizer(self, tokenizer):
        # ---- NOTE-CONCRETE-OVERRIDE-FUNCTION ----
        """
        Set a featurizer.
        """
        if tokenizer is None:
            DebuggingHelper.throw_exception(
                f'input argument, tokenizer, is None')
        self.tokenizer = tokenizer
    def serialize_featurizer(self, serialization_destination: str, dump: bool = True):
        # ---- NOTE-CONCRETE-OVERRIDE-FUNCTION ----
        """
        Serialize a featurizer.
        """
        if self.tokenizer is None:
            DebuggingHelper.throw_exception(
                f'input argument, self.tokenizer, is None')
        if not os.path.exists(serialization_destination):
            os.makedirs(serialization_destination)
        self.tokenizer.save_vocabulary(save_directory=serialization_destination)
        featurization_metadata = super().serialize_featurizer(
            serialization_destination=serialization_destination,
            dump=True)
        return featurization_metadata
    def deserialize_featurizer(self, serialization_destination: str):
        # ---- NOTE-CONCRETE-OVERRIDE-FUNCTION ----
        """
        Deserialize a featurizer.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        if serialization_destination is None:
            DebuggingHelper.throw_exception(
                f'input argument, serialization_destination, is None')
        # ---- NOTE-PYLINT ---- diable pylint E1135 error message as it is spurious
        # ---- NOTE-PYLINT ---- E1135: Value 'PytorchLanguageUnderstandingTransformersPretainedModelHelper.
        # ---- NOTE-PYLINT ---- E1135: get_pretrained_vocabulary_file_key_set()' doesn't
        # ---- NOTE-PYLINT ---- E1135: support membership test
        # ---- NOTE-PYLINT ----        The instance is actually a set, so it can do
        # ---- NOTE-PYLINT ----        membership test.
        # pylint: disable=E1135
        featurization_metadata = None
        if serialization_destination not in \
            PytorchLanguageUnderstandingTransformersPretainedModelHelper.get_pretrained_vocabulary_file_key_set():
            if not IoHelper.isdir(serialization_destination):
                DebuggingHelper.throw_exception(
                    f'input argument, {serialization_destination}'
                    f', is not a directory for deserializing a tokenizer')
            featurization_metadata = super().deserialize_featurizer(
                serialization_destination=serialization_destination)
        if self.pytorch_transformers_model_cache_dir_tokenizer is None:
            DebuggingHelper.throw_exception(
                f'input argument, self.pytorch_transformers_model_cache_dir_tokenizer, is None')
        pytorch_transformers_tokenizer = \
            BertTokenizer.from_pretrained(
                pretrained_model_name_or_path=serialization_destination,
                cache_dir=self.pytorch_transformers_model_cache_dir_tokenizer,
                do_lower_case=self.pytorch_transformers_model_tokenizer_do_lower_case)
        if pytorch_transformers_tokenizer is None:
            DebuggingHelper.throw_exception(
                f'Could not deserialize a '
                f'TokenizationPytorchTransformersSingleLabelDataFeatureManager object from '
                f'pretrained_model_name_or_path={serialization_destination}'
                f',cache_dir={self.pytorch_transformers_model_cache_dir_tokenizer}'
                f',do_lower_case={self.pytorch_transformers_model_tokenizer_do_lower_case}')
        # ---- NOTE-PYLINT ---- diable pylint E1101 error message as it is spurious
        # ---- NOTE-PYLINT ---- E1101: Instance of 'PreTrainedTokenizer'
        # ---- NOTE-PYLINT ---- E1101: has no 'wordpiece_tokenizer' member (no-member)
        # ---- NOTE-PYLINT ----        The instance is actually a BertTokenizer, so it does
        # ---- NOTE-PYLINT ----        have the members.
        # pylint: disable=E1101
        wordpiece_tokenizer = \
            pytorch_transformers_tokenizer.wordpiece_tokenizer
        basic_tokenizer = \
            pytorch_transformers_tokenizer.basic_tokenizer
        if self.pytorch_transformers_model_tokenizer_kind == \
            TokenizationPytorchTransformersSingleLabelDataFeatureManager.PYTORCH_TRANSFORMERS_TOKENIZER_KIND:
            self.set_featurizer(pytorch_transformers_tokenizer)
        if self.pytorch_transformers_model_tokenizer_kind == \
            TokenizationPytorchTransformersSingleLabelDataFeatureManager.PYTORCH_TRANSFORMERS_WORDPIECE_TOKENIZER_KIND:
            self.set_featurizer(wordpiece_tokenizer)
        if self.pytorch_transformers_model_tokenizer_kind == \
            TokenizationPytorchTransformersSingleLabelDataFeatureManager.PYTORCH_TRANSFORMERS_BASIC_TOKENIZER_KIND:
            self.set_featurizer(basic_tokenizer)
        return featurization_metadata

    def create_features(self, text: str) -> List[str]:
        # ---- NOTE-CONCRETE-OVERRIDE-FUNCTION ----
        """
        Create a list of features, each is a list of strings.
        An abstract function that child classes must override.
        """
        return self.tokenizer.tokenize(text)
