# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
This module implements feature manager objects using Bert tokenizer.
"""

from typing import List
from typing import Dict

import json
import os
from tqdm import tqdm

from model.language_understanding.featurizer.base_data_feature_manager \
    import InstanceFeatures
from model.language_understanding.featurizer.base_data_feature_manager \
    import BaseDataFeatureManager
from data.manager.base_record_generator \
    import BaseRecordGenerator
from data.manager.base_data_manager \
    import BaseDataManager

from utility.io_helper.io_helper \
    import IoHelper
from utility.datatype_helper.datatype_helper \
    import DatatypeHelper
from utility.debugging_helper.debugging_helper \
    import DebuggingHelper

class SingleLabelInstance(InstanceFeatures):
    """
    SingleLabelInstance data structure.
    """
    # ---- NOTE-PYLINT ---- R0902: Too many instance attributes
    # pylint: disable=R0902
    # ---- NOTE-PYLINT ---- R0903: Too few public methods
    # pylint: disable=R0903
    def __init__(self, \
        label: str, \
        text: str, \
        weight: float, \
        text_auxiliary: str, \
        features: List[str], \
        features_auxiliary: List[str], \
        identity: str = None):
        """
        Init() for SingleLabelInstance.
        """
        # ---- NOTE-PYLINT ---- R0913: Too many arguments
        # pylint: disable=R0913
        super(SingleLabelInstance, self).__init__( \
            text=text,
            weight=weight,
            text_auxiliary=text_auxiliary,
            features=features,
            features_auxiliary=features_auxiliary,
            identity=identity)
        self.label: str = label

class SingleLabelDataFeatureManager(BaseDataFeatureManager):
    """
    This class can create features on an input text through a tokenizer.
    """
    def __init__(self, \
        data_manager: BaseDataManager, \
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
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # ---- NOTE-PYLINT ---- R0913: Too many arguments
        # pylint: disable=R0913
        super(SingleLabelDataFeatureManager, self).__init__(
            data_manager=data_manager,
            column_index_feature_text=column_index_feature_text,
            column_index_weight=column_index_weight,
            column_index_feature_text_auxiliary=column_index_feature_text_auxiliary,
            column_index_identity=column_index_identity,
            record_generator=record_generator,
            include_out_of_value_feature=include_out_of_value_feature,
            include_only_unique_feature=include_only_unique_feature,
            default_non_existent_feature_id=default_non_existent_feature_id)
        # ---- NOTE: for prediction-only tasks, labels are not needed ---- if column_index_label < 0:
        # ---- NOTE: for prediction-only tasks, labels are not needed ----     DebuggingHelper.throw_exception(
        # ---- NOTE: for prediction-only tasks, labels are not needed ----         f'input argument, column_index_label, is invalid')
        self.column_index_label = column_index_label
        self.string_to_replace_null_label = string_to_replace_null_label
        self.include_out_of_value_label = include_out_of_value_label
        self.default_non_existent_label_id = default_non_existent_label_id
        self.labels = None
        self.label_map: Dict[str, int] = None

    def copy_core_featurization_metadata(self, other_data_feature_manager):
        # ---- NOTE-CONCRETE-OVERRIDE-FUNCTION ----
        """
        Reference the documentation in the parent class.
        For this class, the core featurization metadata are the members being copied.
        self.column_index_label is indexing and not copied, it should be initialized by
        the contstructor.
        """
        super().copy_core_featurization_metadata(
            other_data_feature_manager=other_data_feature_manager)
        other_data_feature_manager.string_to_replace_null_label = \
            self.string_to_replace_null_label
        other_data_feature_manager.include_out_of_value_label = \
            self.include_out_of_value_label
        other_data_feature_manager.default_non_existent_label_id = \
            self.default_non_existent_label_id
        other_data_feature_manager.labels = \
            self.labels
        other_data_feature_manager.label_map = \
            self.label_map

    def get_instance(self, record: List[str]) -> SingleLabelInstance:
        """
        Create an SingleLabelInstance object from a list of columns.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        label: str = None
        weight: float = None
        text: str = None
        features: List[str] = None
        text_auxiliary: str = None
        features_auxiliary: List[str] = None
        try:
            label = self.get_label(record=record)
            weight = self.get_weight(record=record)
            text = record[self.column_index_feature_text]
            features = self.create_features(text=text)
            text_auxiliary = None
            features_auxiliary = None
            if self.has_feature_text_auxiliary:
                text_auxiliary = record[self.column_index_feature_text_auxiliary]
                features_auxiliary = self.create_features(text=text_auxiliary)
            identity = self.get_identity(record=record)
            return SingleLabelInstance(
                label=label,
                text=text,
                weight=weight,
                text_auxiliary=text_auxiliary,
                features=features,
                features_auxiliary=features_auxiliary,
                identity=identity)
        except:
            DebuggingHelper.write_line_to_system_console_err(
                f'ERROR in get_instance()'
                f', self.column_index_feature_text={self.column_index_feature_text}'
                f', self.column_index_feature_text_auxiliary={self.column_index_feature_text_auxiliary}'
                f', self.column_index_label={self.column_index_label}'
                f', self.column_index_weight={self.column_index_weight}'
                f', self.column_index_identity={self.column_index_identity}'
                f', text={text}'
                f', text_auxiliary={text_auxiliary}'
                f', label={label}'
                f', weight={weight}'
                f', record={record}')
            raise

    def get_label(self, record: List[str]) -> str:
        """
        Compile a list of labels use the records in the data_manager member.
        """
        if self.column_index_label < 0:
            return self.string_to_replace_null_label
        if self.column_index_label >= len(record):
            DebuggingHelper.throw_exception(
                f'column_index_label|{self.column_index_label}|'
                f'>=len(record)|{len(record)}|')
        label = record[self.column_index_label]
        if DatatypeHelper.is_none_empty_whitespaces_or_nan(label):
            # DebuggingHelper.write_line_to_system_console_err(
            #    f'DatatypeHelper.is_none_empty_whitespaces_or_nan({label}) is true')
            return self.string_to_replace_null_label
        return str(label)
        # ---- NOTE: Force label to be 'str' type, otherwise classification_report
        #            may not be happy with a 'float' typed label.

    def get_label_id(self, label: str) -> int:
        """
        Convert a label into ID.
        """
        if label in self.label_map:
            return self.label_map[label]
        if self.include_out_of_value_label:
            return self.default_non_existent_feature_id
        DebuggingHelper.throw_exception(
            f'ERROR get_label_id(), input label={label} does not exist!')
        return None

    def populate_labels_features(self):
        """
        Populate the labels and features members.
        """
        # ---- NOTE: commented-for-using-generator-instead ---- records = self.get_records()
        self.populate_labels(
            # ---- NOTE: commented-for-using-generator-instead ---- records=records,
            to_use_generator=False) # ---- NOTE: use all data_feature_manager records
                                    #            to find labels from all records!
        self.populate_features(
            # ---- NOTE: commented-for-using-generator-instead ---- records=records,
            to_use_generator=True) # ---- NOTE: use data_feature_manager generator
                                   #            to find features for training records only!

    def populate_labels(self, \
        records: List[List[str]] = None, \
        to_use_generator: bool = False):
        """
        Compile a list of labels use the records in the data_manager member.
        """
        self.labels = {}
        records, count_records = \
            self.prepare_records(records=records, to_use_generator=to_use_generator)
        # DebuggingHelper.write_line_to_system_console_out(
        #     f'populate_labels(): number of records, len(records) = {len(records)}')
        # DebuggingHelper.write_line_to_system_console_out(
        #     f'populate_labels(): number of records, count_records = {count_records}')
        for record in tqdm(records, total=count_records, desc='populate_labels()'):
            weight: float = self.get_weight(record=record)
            label: str = self.get_label(record=record)
            if label in self.labels:
                self.labels[label] = self.labels[label] + weight
            else:
                self.labels[label] = weight
        self.label_map = {label : i for i, label in enumerate(self.labels)}
        # DebuggingHelper.write_line_to_system_console_out(
        #     f'populate_labels(): self.label_map = {str(self.label_map)}')

    def produce_labels(self, \
        records: List[List[str]] = None, \
        to_use_generator: bool = True) -> List[str]:
        """
        Compile a list of labels use the records in the data_manager member.
        """
        labels: List[str] = []
        records, count_records = \
            self.prepare_records(records=records, to_use_generator=to_use_generator)
        # DebuggingHelper.write_line_to_system_console_out(
        #     f'produce_labels(): number of records, len(records) = {len(records)}')
        # DebuggingHelper.write_line_to_system_console_out(
        #     f'produce_labels(): number of records, count_records = {count_records}')
        for record in tqdm(records, total=count_records, desc='produce_labels()'):
            label: str = self.get_label(record=record)
            labels.append(label)
        return labels

    def produce_label_ids(self, \
        records: List[List[str]] = None, \
        to_use_generator: bool = True) -> List[int]:
        """
        Compile a list of labels use the records in the data_manager member.
        """
        labels: List[str] = \
            self.produce_labels(records=records, to_use_generator=to_use_generator)
        return [self.get_label_id(label=label) for label in labels]

    def get_featurizer(self):
        # ---- NOTE-CONCRETE-OVERRIDE-FUNCTION ----
        """
        Return a featurizer.
        """
        # ---- NOTE-PYLINT ---- W0235: Useless super delegation in method
        # pylint: disable=W0235
        return super().get_featurizer()
    def set_featurizer(self, tokenizer):
        # ---- NOTE-CONCRETE-OVERRIDE-FUNCTION ----
        """
        Set a featurizer.
        """
        # ---- NOTE-PYLINT ---- W0235: Useless super delegation in method
        # pylint: disable=W0235
        super().set_featurizer(tokenizer)
    def serialize_featurizer(self, serialization_destination: str, dump: bool = True):
        # ---- NOTE-CONCRETE-OVERRIDE-FUNCTION ----
        """
        Serialize a featurizer.
        """
        featurization_metadata = {
            'string_to_replace_null_label': self.string_to_replace_null_label,
            'include_out_of_value_label': self.include_out_of_value_label,
            'default_non_existent_label_id': self.default_non_existent_label_id,
            'labels': self.labels,
            'label_map': self.label_map}
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
        serialization_destination += '.json'
        featurization_metadata = \
            json.load(
                IoHelper.codecs_open_file(
                    filename=serialization_destination,
                    mode='r',
                    encoding='utf-8'))
        self.string_to_replace_null_label = \
            featurization_metadata['string_to_replace_null_label']
        self.include_out_of_value_label = \
            featurization_metadata['include_out_of_value_label']
        self.default_non_existent_label_id = \
            featurization_metadata['default_non_existent_label_id']
        self.labels = \
            featurization_metadata['labels']
        self.label_map = \
            featurization_metadata['label_map']
        return featurization_metadata

    def create_features(self, text: str) -> List[str]:
        # ---- NOTE-CONCRETE-OVERRIDE-FUNCTION ----
        """
        Create a list of features, each is a list of strings.
        An abstract function that child classes must override.
        """
        # ---- NOTE-PYLINT ---- W0235: Useless super delegation in method
        # pylint: disable=W0235
        return super().create_features(text)
