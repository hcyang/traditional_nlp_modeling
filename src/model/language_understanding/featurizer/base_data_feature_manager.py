# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
This module implements base feature manager objects
"""

from typing import Dict
from typing import List
from typing import Tuple

from joblib import Parallel, delayed

# from tqdm import tqdm

from data.manager.base_record_generator \
    import BaseRecordGenerator
from data.manager.base_data_manager \
    import BaseDataManager

from utility.list_array_helper.list_array_helper \
    import ListArrayHelper
from utility.parallel_helper.parallel_helper \
    import ParallelHelper
from utility.datatype_helper.datatype_helper \
    import DatatypeHelper
from utility.debugging_helper.debugging_helper \
    import DebuggingHelper

class InstanceFeatures:
    """
    Instance data structure with features only.
    """
    # ---- NOTE-PYLINT ---- R0902: Too many instance attributes
    # pylint: disable=R0902
    # ---- NOTE-PYLINT ---- R0903: Too few public methods
    # pylint: disable=R0903
    def __init__(self, \
        text: str, \
        weight: float, \
        text_auxiliary: str, \
        features: List[str], \
        features_auxiliary: List[str], \
        identity: str = None):
        """
        Init() for InstanceFeatures.
        """
        # ---- NOTE-PYLINT ---- R0913: Too many arguments
        # pylint: disable=R0913
        self.text: str = text
        self.weight: float = weight
        self.text_auxiliary: str = text_auxiliary
        self.features: List[str] = features
        self.features_auxiliary: List[str] = features_auxiliary
        self.identity: str = identity

    def __repr__(self):
        return str(self.__dict__)

class BaseDataFeatureManager:
    """
    This class can extract and manage feature.
    """
    # ---- NOTE-PYLINT ---- R0902: Too many instance attributes
    # pylint: disable=R0902
    # ---- NOTE-PYLINT ---- R0904: Too many public methods
    # pylint: disable=R0904
    def __init__(self, \
        data_manager: BaseDataManager, \
        column_index_feature_text: int = 1, \
        column_index_weight: int = -1, \
        column_index_feature_text_auxiliary: int = -1, \
        column_index_identity: int = -1, \
        record_generator: BaseRecordGenerator = None, \
        include_out_of_value_feature: bool = False, \
        include_only_unique_feature: bool = True, \
        default_non_existent_feature_id: int = -1):
        """
        Init with a data manager object.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # ---- NOTE-PYLINT ---- R0913: Too many arguments
        # pylint: disable=R0913
        # ---- NOTE-BaseDataFeatureManager-MAY-NOT-NEED-A-DATA-SOURCE ---- if data_manager is None:
        # ---- NOTE-BaseDataFeatureManager-MAY-NOT-NEED-A-DATA-SOURCE ----     DebuggingHelper.throw_exception(
        # ---- NOTE-BaseDataFeatureManager-MAY-NOT-NEED-A-DATA-SOURCE ----         f'input argument, data_manager, is None')
        self.data_manager = data_manager
        if column_index_feature_text < 0:
            DebuggingHelper.throw_exception(
                f'input argument, column_index_feature_text, is invalid')
        self.column_index_feature_text: int = column_index_feature_text
        self.column_index_weight: int = column_index_weight
        self.column_index_feature_text_auxiliary: int = column_index_feature_text_auxiliary
        self.column_index_identity: int = column_index_identity
        self.record_generator: BaseRecordGenerator = record_generator
        self.include_out_of_value_feature: bool = include_out_of_value_feature
        self.include_only_unique_feature: bool = include_only_unique_feature
        self.default_non_existent_feature_id: int = default_non_existent_feature_id
        self.features: Dict[str, float] = None
        self.feature_map: Dict[str, int] = None
        self.features_auxiliary: Dict[str, float] = None
        self.feature_auxiliary_map: Dict[str, int] = None
        self.features_combined: Dict[str, float] = None
        self.feature_combined_map: Dict[str, int] = None
        self.has_weight = (self.column_index_weight >= 0)
        self.has_feature_text_auxiliary: bool = (self.column_index_feature_text_auxiliary >= 0)
        self.has_identity: bool = (self.column_index_identity >= 0)

    def copy_core_featurization_metadata(self, other_data_feature_manager):
        # ---- NOTE-CONCRETE-BASE-FUNCTION ----
        """
        Reference the documentation in the parent class.
        For this class, the core featurization metadata are the members being copied.
        The 4 column indexes for feature_text, weight, feature_text_auxiliary, and identity are
        indexing and may change from one feature manager to another, so they are not copied.
        self.record_generator is not copied as record generator logic may be different among
        processing training dataset, testing dataset, and predicting dataset.
        """
        if other_data_feature_manager is None:
            DebuggingHelper.throw_exception(
                f'other_data_feature_manager is None')
        other_data_feature_manager.include_out_of_value_feature = \
            self.include_out_of_value_feature
        other_data_feature_manager.include_only_unique_feature = \
            self.include_only_unique_feature
        other_data_feature_manager.default_non_existent_feature_id = \
            self.default_non_existent_feature_id
        other_data_feature_manager.features = \
            self.features
        other_data_feature_manager.feature_map = \
            self.feature_map
        other_data_feature_manager.features_auxiliary = \
            self.features_auxiliary
        other_data_feature_manager.feature_auxiliary_map = \
            self.feature_auxiliary_map
        other_data_feature_manager.features_combined = \
            self.features_combined
        other_data_feature_manager.feature_combined_map = \
            self.feature_combined_map

    def get_number_features(self) -> int:
        """
        Return the number of features in this BaseDataFeatureManager object.
        """
        if self.features is None:
            DebuggingHelper.throw_exception(
                f'input argument, self.features, is None')
        return len(self.features)
    def get_number_features_auxiliary(self) -> int:
        """
        Return the number of auxiliary features in this BaseDataFeatureManager object.
        """
        if self.features_auxiliary is None:
            DebuggingHelper.throw_exception(
                f'input argument, self.features_auxiliary, is None')
        return len(self.features_auxiliary)
    def get_number_features_combined(self) -> int:
        """
        Return the number of combined features in this BaseDataFeatureManager object.
        """
        if self.features_combined is None:
            DebuggingHelper.throw_exception(
                f'input argument, self.features_combined, is None')
        return len(self.features_combined)

    def get_data_manager(self) -> BaseDataManager:
        """
        Return the data_manager used in this BaseDataFeatureManager object.
        """
        return self.data_manager

    def get_records(self) -> List[List[str]]:
        """
        Return the records used in this BaseDataFeatureManager object.
        """
        if self.get_data_manager() is None:
            DebuggingHelper.throw_exception(
                f'input argument, self.get_data_manager(), is None')
        return self.get_data_manager().get_records()

    def get_features(self, record: List[str], column_index_feature_text: int = -1) -> List[str]:
        """
        Get a list of features use the records in data_manager.
        """
        if column_index_feature_text < 0:
            column_index_feature_text = self.column_index_feature_text
        record_length = len(record)
        if column_index_feature_text >= record_length:
            DebuggingHelper.throw_exception(
                f'column_index_feature_text|{column_index_feature_text}|'
                f'>=record_length|{record_length}|')
        text = record[column_index_feature_text]
        individual_features = self.create_features(text)
        return individual_features

    def get_weight(self, record: List[str], column_index_weight: int = -1) -> float:
        """
        Get a list of weights using the records in data_manager.
        """
        weight = 1
        if column_index_weight < 0:
            column_index_weight = self.column_index_weight
        if column_index_weight < 0:
            return weight
        record_length = len(record)
        if self.column_index_weight >= record_length:
            DebuggingHelper.throw_exception(
                f'column_index_weight|{column_index_weight}|'
                f'>=record_length|{record_length}|')
        weight = float(record[column_index_weight])
        return weight

    def get_identity(self, record: List[str], column_index_identity: int = -1) -> str:
        """
        Get a list of identities using the records in data_manager.
        """
        identity = None
        if column_index_identity < 0:
            column_index_identity = self.column_index_identity
        if column_index_identity < 0:
            return identity
        record_length = len(record)
        if self.column_index_identity >= record_length:
            DebuggingHelper.throw_exception(
                f'column_index_identity|{column_index_identity}|'
                f'>=record_length|{record_length}|')
        identity = record[column_index_identity]
        return identity

    def get_feature_id(self, \
        input_features: List[str]) -> List[int]:
        """
        Convert a list of features into feature IDs.
        """
        feature_ids = []
        if self.include_out_of_value_feature:
            for input_feature in input_features:
                if input_feature in self.feature_map:
                    feature_ids.append(self.feature_map[input_feature])
                else:
                    feature_ids.append(self.default_non_existent_feature_id)
        else:
            for input_feature in input_features:
                if input_feature in self.feature_map:
                    feature_ids.append(self.feature_map[input_feature])
        if self.include_only_unique_feature:
            feature_ids = list(set(feature_ids))
        return feature_ids
    def get_feature_auxiliary_id(self, \
        input_features: List[str]) -> List[int]:
        """
        Convert a list of features into feature IDs.
        """
        feature_ids = []
        if self.include_out_of_value_feature:
            for input_feature in input_features:
                if input_feature in self.feature_auxiliary_map:
                    feature_ids.append(self.feature_auxiliary_map[input_feature])
                else:
                    feature_ids.append(self.default_non_existent_feature_id)
        else:
            for input_feature in input_features:
                if input_feature in self.feature_auxiliary_map:
                    feature_ids.append(self.feature_auxiliary_map[input_feature])
        if self.include_only_unique_feature:
            feature_ids = list(set(feature_ids))
        return feature_ids
    def get_feature_combined_ids(self, \
        input_features: List[str]) -> List[int]:
        """
        Convert a list of features into feature IDs.
        """
        feature_ids = []
        if self.include_out_of_value_feature:
            for input_feature in input_features:
                if input_feature in self.feature_combined_map:
                    feature_ids.append(self.feature_combined_map[input_feature])
                else:
                    feature_ids.append(self.default_non_existent_feature_id)
        else:
            for input_feature in input_features:
                if input_feature in self.feature_combined_map:
                    feature_ids.append(self.feature_combined_map[input_feature])
        if self.include_only_unique_feature:
            feature_ids = list(set(feature_ids))
        return feature_ids

    def populate_features(self, \
        records: List[List[str]] = None, \
        to_use_generator: bool = True):
        """
        Compile a list of features use the records in the data_manager member.
        """
        self.features: Dict[str, float] = {}
        self.features_auxiliary: Dict[str, float] = {}
        self.features_combined: Dict[str, float] = {}
        # records, count_records = \
        records, _ = \
            self.prepare_records(records=records, to_use_generator=to_use_generator)
        # # ---- DebuggingHelper.write_line_to_system_console_out(
        # # ----     f'populate_features(): number of records, len(records) = {len(records)}')
        # DebuggingHelper.write_line_to_system_console_out(
        #     f'populate_features(): number of records, count_records = {count_records}')
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # ---- NOTE: parallel with locking implementation below
        parallelism = ParallelHelper.get_parallelism()
        # DebuggingHelper.write_line_to_system_console_out(
        #     f'populate_features(): parallelism={parallelism}')
        # DebuggingHelper.write_line_to_system_console_out(
        #     f'populate_features(): len(records)={len(records)}')
        # DebuggingHelper.write_line_to_system_console_out(
        #     f'populate_features(): Parallel(): CHECKPOINT-0')
        features_tuples_collected_list = \
            Parallel(n_jobs=parallelism)\
                (delayed(self.populate_features_for_records_direct)(
                    records=records,
                    segment_id=segment_id,
                    number_segments=parallelism) for segment_id in range(parallelism))
        # DebuggingHelper.write_line_to_system_console_out(
        #     f'populate_features(): Parallel(): CHECKPOINT-1')
        features_tuples_collected: Dict[int, Tuple[Dict[str, float], Dict[str, float]]] = {}
        for features_tuples_collected_element in features_tuples_collected_list:
            features_tuples_collected.update(features_tuples_collected_element)
        # DebuggingHelper.write_line_to_system_console_out(
        #     f'populate_features(): Parallel(): CHECKPOINT-2')
        for segment_id in range(parallelism): # ---- NOTE: merge parallel output in order. # ---- NOTE: may not be necessary if Parallel already returns results in order.
            DatatypeHelper.update_count_dictionary_direct(
                self.features,
                features_tuples_collected[segment_id][0])
            DatatypeHelper.update_count_dictionary_direct(
                self.features_auxiliary,
                features_tuples_collected[segment_id][1])
        # DebuggingHelper.write_line_to_system_console_out(
        #     f'populate_features(): Parallel(): CHECKPOINT-3')
        DatatypeHelper.update_count_dictionary(
            self.features_combined,
            self.features)
        DatatypeHelper.update_count_dictionary(
            self.features_combined,
            self.features_auxiliary)
        DebuggingHelper.write_line_to_system_console_out(
            f'populate_features(): Parallel(): CHECKPOINT-4')
        self.feature_map = \
            {feature : i for i, feature in enumerate(self.features)}
        self.feature_auxiliary_map = \
            {feature : i for i, feature in enumerate(self.features_auxiliary)}
        self.feature_combined_map = \
            {feature : i for i, feature in enumerate(self.features_combined)}
        # DebuggingHelper.write_line_to_system_console_out(
        #     f'populate_features(): Parallel(): CHECKPOINT-5')
        # DebuggingHelper.write_line_to_system_console_out(
        #     f'populate_features(): self.feature_map = {str(self.feature_map)}')

    def produce_features(self, \
        records: List[List[str]] = None, \
        to_use_generator: bool = True) -> \
            Tuple[List[List[str]], List[List[str]], List[float], List[str], int]:
        """
        Compile a list of features use the records in the data_manager member.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        individual_features: List[List[str]] = []
        individual_features_auxiliary: List[List[str]] = []
        weights: List[float] = []
        identities: List[str] = []
        records, count_records = \
            self.prepare_records(records=records, to_use_generator=to_use_generator)
        # # ---- DebuggingHelper.write_line_to_system_console_out(
        # # ----     f'produce_features(): number of records, len(records) = {len(records)}')
        # DebuggingHelper.write_line_to_system_console_out(
        #     f'produce_features(): number of records, count_records = {count_records}')
        # ---- NOTE: parallel with locking implementation below
        parallelism = ParallelHelper.get_parallelism()
        # DebuggingHelper.write_line_to_system_console_out(
        #     f'produce_features(): parallelism={parallelism}')
        # DebuggingHelper.write_line_to_system_console_out(
        #     f'produce_features(): len(records)={len(records)}')
        # DebuggingHelper.write_line_to_system_console_out(
        #     f'produce_features(): Parallel(): CHECKPOINT-0')
        individual_features_tuples_collected_list = \
            Parallel(n_jobs=parallelism)\
                (delayed(self.produce_features_for_records_direct)(
                    records=records,
                    segment_id=segment_id,
                    number_segments=parallelism) for segment_id in range(parallelism))
        # DebuggingHelper.write_line_to_system_console_out(
        #     f'produce_features(): Parallel(): CHECKPOINT-1')
        individual_features_tuples_collected: Dict[int, Tuple[List[List[str], List[List[str]], List[float], List[str]]]] = {}
        for individual_features_tuples_collected_element in individual_features_tuples_collected_list:
            individual_features_tuples_collected.update(individual_features_tuples_collected_element)
        # DebuggingHelper.write_line_to_system_console_out(
        #     f'produce_features(): Parallel(): CHECKPOINT-2')
        for segment_id in range(parallelism): # ---- NOTE: merge parallel output in order. # ---- NOTE: may not be necessary if Parallel already returns results in order.
            individual_features_tuple = individual_features_tuples_collected[segment_id]
            individual_features.extend(individual_features_tuple[0])
            individual_features_auxiliary.extend(individual_features_tuple[1])
            weights.extend(individual_features_tuple[2])
            identities.extend(individual_features_tuple[3])
        # DebuggingHelper.write_line_to_system_console_out(
        #     f'produce_features(): Parallel(): CHECKPOINT-3')
        return (individual_features,
                individual_features_auxiliary,
                weights,
                identities,
                count_records)

    def produce_feature_ids(self, \
        records: List[List[str]] = None, \
        to_use_generator: bool = True) -> \
            Tuple[List[List[int]], List[List[int]], List[float], List[str], int]:
        """
        Compile a list of feature ids use the records in the data_manager member.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        individual_feature_ids: List[List[int]] = []
        individual_feature_ids_auxiliary: List[List[int]] = []
        weights: List[float] = []
        identities: List[str] = []
        records, count_records = \
            self.prepare_records(records=records, to_use_generator=to_use_generator)
        # # ---- DebuggingHelper.write_line_to_system_console_out(
        # # ----     f'produce_feature_ids(): number of records, len(records) = {len(records)}')
        # DebuggingHelper.write_line_to_system_console_out(
        #     f'produce_feature_ids(): number of records, count_records = {count_records}')
        # ---- NOTE: parallel with locking implementation below
        parallelism = ParallelHelper.get_parallelism()
        # DebuggingHelper.write_line_to_system_console_out(
        #     f'produce_feature_ids(): parallelism={parallelism}')
        # DebuggingHelper.write_line_to_system_console_out(
        #     f'produce_feature_ids(): len(records)={len(records)}')
        # DebuggingHelper.write_line_to_system_console_out(
        #     f'produce_feature_ids(): Parallel(): CHECKPOINT-0')
        individual_feature_ids_tuples_collected_list = \
            Parallel(n_jobs=parallelism)\
                (delayed(self.produce_feature_ids_for_records_direct)(
                    records=records,
                    segment_id=segment_id,
                    number_segments=parallelism) for segment_id in range(parallelism))
        # DebuggingHelper.write_line_to_system_console_out(
        #     f'produce_feature_ids(): Parallel(): CHECKPOINT-1')
        individual_feature_ids_tuples_collected: Dict[int, Tuple[List[List[int], List[List[int]], List[float], List[str]]]] = {}
        for individual_feature_ids_tuples_collected_element in individual_feature_ids_tuples_collected_list:
            individual_feature_ids_tuples_collected.update(individual_feature_ids_tuples_collected_element)
        # DebuggingHelper.write_line_to_system_console_out(
        #     f'produce_feature_ids(): Parallel(): CHECKPOINT-2')
        for segment_id in range(parallelism): # ---- NOTE: merge parallel output in order. # ---- NOTE: may not be necessary if Parallel already returns results in order.
            individual_feature_ids_tuple = individual_feature_ids_tuples_collected[segment_id]
            individual_feature_ids.extend(individual_feature_ids_tuple[0])
            individual_feature_ids_auxiliary.extend(individual_feature_ids_tuple[1])
            weights.extend(individual_feature_ids_tuple[2])
            identities.extend(individual_feature_ids_tuple[3])
        # DebuggingHelper.write_line_to_system_console_out(
        #     f'produce_feature_ids(): Parallel(): CHECKPOINT-3')
        return (individual_feature_ids,
                individual_feature_ids_auxiliary,
                weights,
                identities,
                count_records)

    # ---- NOTE-NOT-NEEDED-YET ----
    # def populate_features_for_record(self, record: List[str]):
    #     """
    #     Process one record and populate its processed features synchronized.
    #     """
    #     weight = self.get_weight(record=record)
    #     individual_features = \
    #         self.get_features(record, self.column_index_feature_text)
    #     DatatypeHelper.update_count_dictionary(
    #         self.features,
    #         {x: weight for x in individual_features})
    #     if self.has_feature_text_auxiliary:
    #         individual_features = \
    #             self.get_features(record, self.column_index_feature_text_auxiliary)
    #         DatatypeHelper.update_count_dictionary(
    #             self.features_auxiliary,
    #             {x: weight for x in individual_features})

    def populate_features_for_records_direct(self, \
        records: List[List[str]], \
        segment_id: int, \
        number_segments: int) -> \
            Dict[int, Tuple[Dict[str, float], Dict[str, float]]]:
        """
        Process one record and populate its processed features without synchronization.
        """
        features_per_segment: Dict[str, float] = {}
        features_auxiliary_per_segment: Dict[str, float] = {}
        number_records = len(records)
        begin_index_for_segment, end_index_for_segment, _, _, _ = \
            ListArrayHelper.get_segment_begin_end_index_tuple(
                number_records=number_records,
                number_segments=number_segments,
                segment_id=segment_id)
        for record_index in range(begin_index_for_segment, end_index_for_segment):
            self.populate_features_into_aggregate_tuple_for_record_direct(
                records[record_index],
                features_per_segment,
                features_auxiliary_per_segment)
        return {segment_id: (features_per_segment, features_auxiliary_per_segment)}

    def produce_features_for_records_direct(self, \
        records: List[List[str]], \
        segment_id: int, \
        number_segments: int) -> \
            Dict[int, Tuple[List[List[str]], List[List[str]], List[float], List[str]]]:
        """
        Process one record and populate its processed features without synchronization.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # ---- NOTE-PYLINT ---- R0914: Too many local variables
        # pylint: disable=R0914
        individual_features_per_segment: List[List[str]] = []
        individual_features_auxiliary_per_segment: List[List[str]] = []
        weights_per_segment: List[float] = []
        identities_per_segment: List[str] = []
        number_records = len(records)
        begin_index_for_segment, end_index_for_segment, _, _, _ = \
            ListArrayHelper.get_segment_begin_end_index_tuple(
                number_records=number_records,
                number_segments=number_segments,
                segment_id=segment_id)
        for record_index in range(begin_index_for_segment, end_index_for_segment):
            individual_features, individual_features_auxiliary, weight, identity = \
                self.produce_features_into_aggregate_tuple_for_record_direct(
                    records[record_index])
            individual_features_per_segment.append(individual_features)
            individual_features_auxiliary_per_segment.append(individual_features_auxiliary)
            weights_per_segment.append(weight)
            identities_per_segment.append(identity)
        return {segment_id: (individual_features_per_segment,
                             individual_features_auxiliary_per_segment,
                             weights_per_segment,
                             identities_per_segment)}

    def produce_feature_ids_for_records_direct(self, \
        records: List[List[str]], \
        segment_id: int, \
        number_segments: int) -> \
            Dict[int, Tuple[List[List[int]], List[List[int]], List[float], List[str]]]:
        """
        Process one record and populate its processed features without synchronization.
        """
        # ---- NOTE-PYLINT ---- R0914: Too many local variables
        # pylint: disable=R0914
        individual_feature_ids_per_segment: List[List[int]] = []
        individual_feature_ids_auxiliary_per_segment: List[List[int]] = []
        weights_per_segment: List[float] = []
        identities_per_segment: List[str] = []
        number_records = len(records)
        begin_index_for_segment, end_index_for_segment, _, _, _ = \
            ListArrayHelper.get_segment_begin_end_index_tuple(
                number_records=number_records,
                number_segments=number_segments,
                segment_id=segment_id)
        for record_index in range(begin_index_for_segment, end_index_for_segment):
            individual_feature_ids, individual_feature_ids_auxiliary, weight, identity = \
                self.produce_feature_ids_into_aggregate_tuple_for_record_direct(
                    records[record_index])
            individual_feature_ids_per_segment.append(individual_feature_ids)
            individual_feature_ids_auxiliary_per_segment.append(individual_feature_ids_auxiliary)
            weights_per_segment.append(weight)
            identities_per_segment.append(identity)
        return {segment_id: (individual_feature_ids_per_segment,
                             individual_feature_ids_auxiliary_per_segment,
                             weights_per_segment,
                             identities_per_segment)}

    # ---- NOTE-NOT-NEEDED-YET ----
    # def populate_features_for_record_direct(self, record: List[str]):
    #     """
    #     Process one record and populate its processed features without synchronization.
    #     """
    #     if self.features is None:
    #         self.features: Dict[str, float] = {}
    #     if self.features_auxiliary is None:
    #         self.features_auxiliary: Dict[str, float] = {}
    #     self.populate_features_into_aggregate_tuple_for_record_direct(
    #         record,
    #         self.features,
    #         self.features_auxiliary)

    def populate_features_into_aggregate_tuple_for_record_direct(self, \
        record: List[str], \
        features: Dict[str, float] = None, \
        features_auxiliary: Dict[str, float] = None) -> \
            Tuple[Dict[str, float], Dict[str, float]]:
        """
        Process one record and populate its processed features without synchronization.
        """
        if features is None:
            features: Dict[str, float] = {}
        if features_auxiliary is None:
            features_auxiliary: Dict[str, float] = {}
        individual_features, individual_features_auxiliary, weight, _ = \
            self.produce_features_into_aggregate_tuple_for_record_direct(record)
        DatatypeHelper.update_count_dictionary_direct(
            features,
            {x: weight for x in individual_features})
        if individual_features_auxiliary is not None:
            DatatypeHelper.update_count_dictionary_direct(
                individual_features_auxiliary,
                {x: weight for x in individual_features})
        return (features, features_auxiliary)

    def produce_features_into_aggregate_tuple_for_record_direct(self, \
        record: List[str]) -> \
            Tuple[List[str], List[str], float, str]:
        """
        Process one record and populate its processed features without synchronization.
        """
        weight = self.get_weight(record=record)
        identity = self.get_identity(record=record)
        individual_features = \
            self.get_features(record, self.column_index_feature_text)
        individual_features_auxiliary = None
        if self.has_feature_text_auxiliary:
            individual_features_auxiliary = \
                self.get_features(record, self.column_index_feature_text_auxiliary)
        return (individual_features, individual_features_auxiliary, weight, identity)

    def produce_feature_ids_into_aggregate_tuple_for_record_direct(self, \
        record: List[str]) -> \
            Tuple[List[int], List[int], float, str]:
        """
        Process one record and populate its processed feature ids without synchronization.
        """
        individual_features, individual_features_auxiliary, weight, identity = \
            self.produce_features_into_aggregate_tuple_for_record_direct(record=record)
        individual_feature_ids = \
            self.get_feature_combined_ids(individual_features)
        individual_feature_auxiliary_ids = None
        if self.has_feature_text_auxiliary:
            individual_feature_auxiliary_ids = \
                self.get_feature_combined_ids(individual_features_auxiliary)
        return (individual_feature_ids, individual_feature_auxiliary_ids, weight, identity)

    def prepare_records(self, \
        records: List[List[str]] = None, \
        to_use_generator: bool = True) -> \
            Tuple[List[List[str]], int]:
        """
        Prepare records, use the one in the data_manager if input is None.
        """
        count_records = -1
        if records is None:
            records = self.get_records()
            if to_use_generator and (self.record_generator is not None):
                records = self.record_generator.generate()
                count_records = self.record_generator.number_generator_records
        if records is None:
            DebuggingHelper.throw_exception(
                f'records is None')
        # ---- NOTE: for parallel progreamming purpose, if records is a generator,
        #            force materialize it into a list
        if not isinstance(records, list):
            records = [record for record in records]
        if count_records < 0:
            count_records = len(records)
        return (records, count_records)

    def get_featurizer(self):
        # ---- NOTE-ABSTRACT-FUNCTION ----
        """
        Return a featurizer.
        """
        raise NotImplementedError()
    def set_featurizer(self, tokenizer):
        # ---- NOTE-ABSTRACT-FUNCTION ----
        """
        Set a featurizer.
        """
        raise NotImplementedError()
    def serialize_featurizer(self, serialization_destination: str, dump: bool = True):
        # ---- NOTE-ABSTRACT-FUNCTION ----
        """
        Serialize a featurizer.
        """
        raise NotImplementedError()
    def deserialize_featurizer(self, serialization_destination: str):
        # ---- NOTE-ABSTRACT-FUNCTION ----
        """
        Deserialize a featurizer.
        """
        raise NotImplementedError()

    def serialize(self, serialization_destination: str, dump: bool = True):
        # ---- NOTE-CPNCREATE-BASE-FUNCTION ----
        """
        Serialize self feature manager.
        """
        return self.serialize_featurizer(
            serialization_destination=serialization_destination,
            dump=dump)
    def deserialize(self, serialization_destination: str):
        # ---- NOTE-CPNCREATE-BASE-FUNCTION ----
        """
        Deserialize self feature manager.
        """
        return self.deserialize_featurizer(
            serialization_destination=serialization_destination)

    def create_features(self, text: str) -> List[str]:
        # ---- NOTE-CONCRETE-OVERRIDE-FUNCTION ----
        """
        Create a list of features, each is a list of strings.
        An abstract function that child classes must override.
        """
        # ---- NOTE-PYLINT ---- R0201: Method could be a function (no-self-use)
        # pylint: disable=R0201
        return [text]
