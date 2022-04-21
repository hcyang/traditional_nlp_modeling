# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
This module implements feature manager objects using a NGram + subword tokenizer.
"""

from typing import List

import json
import os

from model.language_understanding.featurizer.base_data_feature_manager_single_label \
    import SingleLabelDataFeatureManager
from data.manager.base_record_generator \
    import BaseRecordGenerator
from data.manager.base_data_manager \
    import BaseDataManager

from utility.io_helper.io_helper \
    import IoHelper
# from utility.debugging_helper.debugging_helper \
#     import DebuggingHelper

class NGramSubwordSingleLabelDataFeatureManager(\
    SingleLabelDataFeatureManager):
    """
    This class can create features on an input text through a tokenizer.
    """

    # ---- NOTE-PYLINT ---- R0903: Too few public methods
    # pylint: disable=R0903
    def __init__(self, \
        data_manager: BaseDataManager, \
        column_index_label: int = 0, \
        column_index_weight: int = -1, \
        column_index_feature_text: int = 1, \
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
        super(NGramSubwordSingleLabelDataFeatureManager, self).__init__( \
            data_manager=data_manager,
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

    def copy_core_featurization_metadata(self, other_data_feature_manager):
        # ---- NOTE-CONCRETE-OVERRIDE-FUNCTION ----
        """
        Reference the documentation in the parent class.
        For this class, the core featurization metadata are the members being copied.
        """
        super().copy_core_featurization_metadata(
            other_data_feature_manager=other_data_feature_manager)

    def get_featurizer(self):
        # ---- NOTE-CONCRETE-OVERRIDE-FUNCTION ----
        """
        Return a featurizer.
        """
        # ---- NOTE-PYLINT ---- W0235: Useless super delegation in method
        # pylint: disable=W0235
        return super().get_featurizer()
        # ---- NOTE: Self is the featurizer.
    def set_featurizer(self, tokenizer):
        # ---- NOTE-CONCRETE-OVERRIDE-FUNCTION ----
        """
        Set a featurizer.
        """
        # ---- NOTE-PYLINT ---- W0235: Useless super delegation in method
        # pylint: disable=W0235
        super().set_featurizer(tokenizer)
        # ---- NOTE: Self is the featurizer.
    def serialize_featurizer(self, serialization_destination: str, dump: bool = True):
        # ---- NOTE-CONCRETE-OVERRIDE-FUNCTION ----
        """
        Serialize a featurizer.
        """
        featurization_metadata = super().serialize_featurizer(
            serialization_destination=serialization_destination,
            dump=False)
        featurization_metadata_self = {}
        featurization_metadata.update(featurization_metadata_self)
        if dump:
            if IoHelper.isdir(serialization_destination):
                serialization_destination = os.path.join(
                    serialization_destination,
                    f'{__name__}_featurization_metadata')
            serialization_destination += '.json'
            json.dump(
                featurization_metadata,
                IoHelper.codecs_open_file(
                    filename=serialization_destination,
                    mode='w',
                    encoding='utf-8'),
                separators=(',', ':'),
                sort_keys=True,
                indent=4)
        return featurization_metadata

    def deserialize_featurizer(self, serialization_destination: str):
        # ---- NOTE-CONCRETE-OVERRIDE-FUNCTION ----
        """
        Deserialize a featurizer.
        """
        if IoHelper.isdir(serialization_destination):
            serialization_destination = os.path.join(
                serialization_destination,
                f'{__name__}_featurization_metadata')
        featurization_metadata = super().deserialize_featurizer(
            serialization_destination=serialization_destination)
        return featurization_metadata

    def create_features(self, text: str) -> List[str]:
        # ---- NOTE-CONCRETE-OVERRIDE-FUNCTION ----
        """
        Create a list of features, each is a list of strings.
        An abstract function that child classes must override.
        """
        return text.split()
