# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
This module implements feature manager objects using PytorchTransformers tokenizer.
"""

from typing import Dict
from typing import List
from typing import Tuple

from joblib import Parallel, delayed

# from tqdm import tqdm

# ---- NOTE-PYLINT ---- E0611: No name 'long' in module 'torch'
# pylint: disable=E0611
from torch import tensor
from torch import long as torch_long
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler
from torch.utils.data import TensorDataset

from model.language_understanding.featurizer.base_data_feature_manager_single_label \
    import SingleLabelDataFeatureManager
from data.manager.base_record_generator \
    import BaseRecordGenerator
from data.manager.base_data_manager \
    import BaseDataManager

from utility.list_array_helper.list_array_helper \
    import ListArrayHelper
from utility.parallel_helper.parallel_helper \
    import ParallelHelper
from utility.debugging_helper.debugging_helper \
    import DebuggingHelper

class PytorchTransformersTrainingFeaturieIdizedInstance:
    """
    A single set of features of data instance.
    """
    # ---- NOTE-PYLINT ---- R0903: Too few public methods
    # pylint: disable=R0903
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

    def __repr__(self):
        return str(self.__dict__)

class PytorchTransformersTrainingFeaturieIdizedInstanceAggregate:
    """
    Aggregate of processed features and ids of data instance.
    """
    # ---- NOTE-PYLINT ---- R0903: Too few public methods
    # pylint: disable=R0903
    def __init__(self, \
        instances, \
        feature_idized_instances, \
        pytorch_transformers_all_input_ids, \
        pytorch_transformers_all_input_mask, \
        pytorch_transformers_all_segment_ids, \
        pytorch_transformers_all_label_ids, \
        pytorch_transformers_dataloader):
        """
        Init a PytorchTransformersTrainingFeaturieIdizedInstanceAggregate object.
        """
        # ---- NOTE-PYLINT ---- R0913: Too many arguments
        # pylint: disable=R0913
        self.instances = instances
        self.feature_idized_instances = feature_idized_instances
        self.pytorch_transformers_all_input_ids = pytorch_transformers_all_input_ids
        self.pytorch_transformers_all_input_mask = pytorch_transformers_all_input_mask
        self.pytorch_transformers_all_segment_ids = pytorch_transformers_all_segment_ids
        self.pytorch_transformers_all_label_ids = pytorch_transformers_all_label_ids
        self.pytorch_transformers_dataloader = pytorch_transformers_dataloader

class PytorchTransformersSingleLabelDataFeatureManager(\
    SingleLabelDataFeatureManager):
    """
    This class can create features on an input text through a tokenizer.
    """
    # ---- NOTE-PYLINT ---- R0903: Too few public methods
    # pylint: disable=R0903
    def __init__(self, \
        data_manager: BaseDataManager, \
        featurization_max_sequence_length: int, \
        featurization_dataloader_batch_size: int, \
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
        Init with a data manager object.

        Notice that default_non_existent_label_id is default to 0, as
        Pytorch Transformers CUDA code would not like a label id
        beyond the legitimate lable id range.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # ---- NOTE-PYLINT ---- R0913: Too many arguments
        # pylint: disable=R0913
        # ---- NOTE-PYLINT ---- R0914: Too many local variables
        # pylint: disable=R0914
        super(PytorchTransformersSingleLabelDataFeatureManager, self).__init__( \
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
        self.featurization_max_sequence_length = \
            featurization_max_sequence_length
        self.featurization_dataloader_batch_size = \
            featurization_dataloader_batch_size
        self.aggregate_structures_feature_idized_instances = \
            None
        # DebuggingHelper.write_line_to_system_console_out(
        #     'PytorchTransformersSingleLabelDataFeatureManager object constructed.')

    def copy_core_featurization_metadata(self, other_data_feature_manager):
        # ---- NOTE-CONCRETE-OVERRIDE-FUNCTION ----
        """
        Reference the documentation in the parent class.
        For this class, the core featurization metadata are the members being copied.
        """
        super().copy_core_featurization_metadata(
            other_data_feature_manager=other_data_feature_manager)
        other_data_feature_manager.featurization_max_sequence_length = \
            self.featurization_max_sequence_length
        other_data_feature_manager.featurization_dataloader_batch_size = \
            self.featurization_dataloader_batch_size

    def process_record_into_feature_idized_instance_aggregate_tuple(self, \
        records: List[List[str]], \
        segment_id: int, \
        number_segments: int, \
        label_map: Dict[object, int] = None, \
        output_mode: str = "classification") -> Dict[int, Tuple[List[object], List[object]]]:
        """
        Process a record into a instance and idized instance.
        """
        # ---- NOTE-PYLINT ---- R0913: Too many arguments
        # pylint: disable=R0913
        # ---- NOTE-PYLINT ---- R0914: Too many local variables
        # pylint: disable=R0914
        # ---- NOTE-PYLINT ---- R0915: Too many statements
        # pylint: disable=R0915
        instances_per_segment = []
        feature_idized_instances_per_segment = []
        number_records = len(records)
        begin_index_for_segment, end_index_for_segment, _, _, _ = \
            ListArrayHelper.get_segment_begin_end_index_tuple(
                number_records=number_records,
                number_segments=number_segments,
                segment_id=segment_id)
        for record_index in range(begin_index_for_segment, end_index_for_segment):
            instance, feature_idized_instance = \
                self.process_record_into_feature_idized_instance(
                    record=records[record_index],
                    label_map=label_map,
                    output_mode=output_mode,
                    record_index=record_index)
            instances_per_segment.append(instance)
            feature_idized_instances_per_segment.append(
                feature_idized_instance)
        return {segment_id: (instances_per_segment, feature_idized_instances_per_segment)}

    def process_record_into_feature_idized_instance(self, \
        record: List[str], \
        label_map: Dict[object, int] = None, \
        output_mode: str = "classification", \
        record_index: int = -1, \
        cls_token_at_end=False, \
        pad_on_left=False, \
        cls_token='[CLS]', \
        sep_token='[SEP]', \
        pad_token=0, \
        sequence_a_segment_id=0, \
        sequence_b_segment_id=1, \
        cls_token_segment_id=1, \
        pad_token_segment_id=0, \
        mask_padding_with_zero=True):
        """
        Process a record into a instance and idized instance.
        REVISED-FROM-OSS-REFERENCE: pytorch-transformers/examples/utils_glue.py
        """
        # ---- NOTE-PYLINT ---- W0613: Unused argument
        # pylint: disable=W0613
        # ---- NOTE-PYLINT ---- R0912: Too many branches
        # pylint: disable=R0912
        # ---- NOTE-PYLINT ---- R0913: Too many arguments
        # pylint: disable=R0913
        # ---- NOTE-PYLINT ---- R0914: Too many local variables
        # pylint: disable=R0914
        # ---- NOTE-PYLINT ---- R0915: Too many statements
        # pylint: disable=R0915
        # ----
        if label_map is None:
            label_map = self.label_map
        if label_map is None:
            DebuggingHelper.throw_exception(
                f'label_map is None')
        instance = self.get_instance(record)
        # text = instance.text
        features = instance.features
        text_auxiliary = instance.text_auxiliary
        features_auxiliary = instance.features_auxiliary
        # ----
        if text_auxiliary:
            # Modifies `features` and `features_auxiliary` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            PytorchTransformersSingleLabelDataFeatureManager.truncate_sequence_pair(
                features,
                features_auxiliary,
                self.featurization_max_sequence_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(features) > self.featurization_max_sequence_length - 2:
                features = features[:(self.featurization_max_sequence_length - 2)]
        # ----
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        # ----
        features_combined = features + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(features_combined)
        if features_auxiliary:
            features_combined += features_auxiliary + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(features_auxiliary) + 1)
        if cls_token_at_end:
            features_combined = features_combined + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            features_combined = [cls_token] + features_combined
            segment_ids = [cls_token_segment_id] + segment_ids
        # ----
        input_ids = self.get_featurizer().convert_tokens_to_ids(features_combined)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = self.featurization_max_sequence_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
        # ----
        assert len(input_ids) == self.featurization_max_sequence_length
        assert len(input_mask) == self.featurization_max_sequence_length
        assert len(segment_ids) == self.featurization_max_sequence_length
        # ----
        label = instance.label
        if output_mode == "classification":
            if label is not None:
                if label not in label_map:
                    # DebuggingHelper.throw_exception(
                    #     f'label|{label}| not in label_map|{str(label_map)}|')
                    label_id = self.default_non_existent_label_id
                else:
                    label_id = label_map[label]
            else:
                label_id = self.default_non_existent_label_id
        elif output_mode == "regression":
            if label is not None:
                label_id = float(label)
            else:
                label_id = 0 # ---- NOTE-TODO ----
        else:
            raise KeyError(output_mode)
        # ----
        feature_idized_instance = \
            PytorchTransformersTrainingFeaturieIdizedInstance(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id)
        # if record_index < 5:
        #     DebuggingHelper.write_line_to_system_console_out(
        #         "*** Example ***")
        #     # ---- DebuggingHelper.write_line_to_system_console_out(
        #     # ----     "guid: %s" % (instance.guid))
        #     DebuggingHelper.write_line_to_system_console_out(
        #         "tokens: %s" % " ".join(
        #             [str(x) for x in features_combined]))
        #     DebuggingHelper.write_line_to_system_console_out(
        #         "input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     DebuggingHelper.write_line_to_system_console_out(
        #         "input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     DebuggingHelper.write_line_to_system_console_out(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     DebuggingHelper.write_line_to_system_console_out(
        #         "label: %s (id = %d)" % (instance.label, label_id))
        return (instance, feature_idized_instance)

    def process_records_into_feature_idized_instances(self, \
        records: List[List[str]] = None, \
        batch_size: int = None, \
        to_use_generator: bool = True, \
        to_cache_output: bool = True, \
        label_map: Dict[object, int] = None, \
        output_mode: str = "classification"):
        """
        Process a list of records into a list of instances and idized instances.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # ---- NOTE-PYLINT ---- R0912: Too many branches
        # pylint: disable=R0912
        # ---- NOTE-PYLINT ---- R0913: Too many arguments
        # pylint: disable=R0913
        # ---- NOTE-PYLINT ---- R0914: Too many local variables
        # pylint: disable=R0914
        # ---- NOTE-PYLINT ---- R0915: Too many statements
        # pylint: disable=R0915
        # --------------------------------------------------------------------
        DebuggingHelper.write_line_to_system_console_out(
            'entering')
        records, count_records = \
            self.prepare_records(records=records, to_use_generator=to_use_generator)
        if label_map is None:
            label_map = self.label_map
        # --------------------------------------------------------------------
        instances = []
        feature_idized_instances = []
        DebuggingHelper.write_line_to_system_console_out(
            'ready to iterate records')
        DebuggingHelper.write_line_to_system_console_out(
            f'process_records_into_feature_idized_instances()'
            f', self.column_index_feature_text={self.column_index_feature_text}'
            f', self.column_index_feature_text_auxiliary={self.column_index_feature_text_auxiliary}'
            f', self.column_index_label={self.column_index_label}'
            f', self.column_index_weight={self.column_index_weight}'
            f', self.column_index_identity={self.column_index_identity}')
        # ---- NOTE: force materialize the record array for parallel progreamming
        records = [record for record in records]
        # ---- DebuggingHelper.write_line_to_system_console_out(
        # ----     f'process_records_into_feature_idized_instances(): number of records, len(records) = {len(records)}')
        DebuggingHelper.write_line_to_system_console_out(
            f'process_records_into_feature_idized_instances(): number of records, count_records = {count_records}')
        # ----
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # ---- NOTE: parallel with locking implementation below
        parallelism = ParallelHelper.get_parallelism()
        DebuggingHelper.write_line_to_system_console_out(
            f'process_records_into_feature_idized_instances(): parallelism={parallelism}')
        DebuggingHelper.write_line_to_system_console_out(
            f'process_records_into_feature_idized_instances(): len(records)={len(records)}')
        DebuggingHelper.write_line_to_system_console_out(
            f'process_records_into_feature_idized_instances(): CHECKPOINT-0')
        # --------------------------------------------------------------------
        # ---- NOTE: below is an sequential implementation - kept for debugging purpose.
        # parallelism = 1
        # instances_tuples_collected: Dict[int, Tuple[List[object], List[object]]] = \
        #     self.process_record_into_feature_idized_instance_aggregate_tuple(
        #         records=records,
        #         segment_id=0,
        #         number_segments=parallelism,
        #         label_map=label_map,
        #         output_mode=output_mode)
        # for segment_id in range(parallelism):
        #     instances_tuple = instances_tuples_collected[segment_id]
        #     instances.extend(instances_tuple[0])
        #     feature_idized_instances.extend(instances_tuple[1])
        # DebuggingHelper.write_line_to_system_console_out(
        #     f'process_records_into_feature_idized_instances(): CHECKPOINT-3')
        # --------------------------------------------------------------------
        instances_tuples_collected_list = \
            Parallel(n_jobs=parallelism)\
                (delayed(self.process_record_into_feature_idized_instance_aggregate_tuple)(
                    records=records,
                    segment_id=segment_id,
                    number_segments=parallelism,
                    label_map=label_map,
                    output_mode=output_mode) for segment_id in range(parallelism))
        DebuggingHelper.write_line_to_system_console_out(
            f'process_records_into_feature_idized_instances(): Parallel(): CHECKPOINT-1')
        instances_tuples_collected: Dict[int, Tuple[List[object], List[object]]] = {}
        for instances_tuples_collected_element in instances_tuples_collected_list:
            instances_tuples_collected.update(instances_tuples_collected_element)
        DebuggingHelper.write_line_to_system_console_out(
            f'process_records_into_feature_idized_instances(): Parallel(): CHECKPOINT-2')
        for segment_id in range(parallelism):
            instances_tuple = instances_tuples_collected[segment_id]
            instances.extend(instances_tuple[0])
            feature_idized_instances.extend(instances_tuple[1])
        DebuggingHelper.write_line_to_system_console_out(
            f'process_records_into_feature_idized_instances(): Parallel(): CHECKPOINT-3')
        # --------------------------------------------------------------------
        # ---- NOTE: direct sequential implementation below
        # DebuggingHelper.write_line_to_system_console_out(
        #     f'CHECKPOINT-3')
        # records_tqdm = tqdm(records, total=count_records, desc='process_records_into_feature_idized_instances()')
        # for (record_index, record) in enumerate(records_tqdm):
        #     if record_index % 10000 == 0:
        #         DebuggingHelper.write_line_to_system_console_out(
        #             f'process_records_into_feature_idized_instances()'
        #             f', record_index={record_index}'
        #             f', count_records={count_records}')
        #     instance, feature_idized_instance = \
        #         self.process_record_into_feature_idized_instance(
        #             record=record,
        #             label_map=label_map,
        #             output_mode=output_mode,
        #             record_index=record_index)
        #     instances.append(instance)
        #     feature_idized_instances.append(
        #         feature_idized_instance)
        # DebuggingHelper.write_line_to_system_console_out(
        #     f'CHECKPOINT-3-END')
        # ----
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: enable=C0301
        # ----
        DebuggingHelper.write_line_to_system_console_out(
            'ready to process torch id and mask data structures')
        if batch_size is None:
            batch_size = self.featurization_dataloader_batch_size
        DebuggingHelper.write_line_to_system_console_out(
            f'batch_size={batch_size}')
        (pytorch_transformers_dataloader,
         pytorch_transformers_all_input_ids,
         pytorch_transformers_all_input_mask,
         pytorch_transformers_all_segment_ids,
         pytorch_transformers_all_label_ids) = \
            PytorchTransformersSingleLabelDataFeatureManager.build_pytorch_transformers_dataloader(
                feature_idized_instances=\
                    feature_idized_instances,
                featurization_dataloader_batch_size=\
                    batch_size)
        # --------------------------------------------------------------------
        DebuggingHelper.write_line_to_system_console_out(
            'done processing torch id and mask data structures')
        aggregate_structures_feature_idized_instances = \
            PytorchTransformersTrainingFeaturieIdizedInstanceAggregate(
                instances=instances,
                feature_idized_instances=feature_idized_instances,
                pytorch_transformers_all_input_ids=pytorch_transformers_all_input_ids,
                pytorch_transformers_all_input_mask=pytorch_transformers_all_input_mask,
                pytorch_transformers_all_segment_ids=pytorch_transformers_all_segment_ids,
                pytorch_transformers_all_label_ids=pytorch_transformers_all_label_ids,
                pytorch_transformers_dataloader=pytorch_transformers_dataloader)
        if to_cache_output:
            self.aggregate_structures_feature_idized_instances = \
                aggregate_structures_feature_idized_instances
        return aggregate_structures_feature_idized_instances
        # --------------------------------------------------------------------

    @staticmethod
    def build_pytorch_transformers_dataloader( \
        feature_idized_instances: List[PytorchTransformersTrainingFeaturieIdizedInstance], \
        featurization_dataloader_batch_size: int):
        """
        Process a list of idized instances and build tensors.
        """
        return PytorchTransformersSingleLabelDataFeatureManager.\
            build_pytorch_transformers_dataloader_with_lists( \
                list_input_ids=\
                    [instance.input_ids for instance in feature_idized_instances],
                list_input_mask=\
                    [instance.input_mask for instance in feature_idized_instances],
                list_segment_ids=\
                    [instance.segment_ids for instance in feature_idized_instances],
                list_label_id=\
                    [instance.label_id for instance in feature_idized_instances],
                featurization_dataloader_batch_size=\
                    featurization_dataloader_batch_size)

    @staticmethod
    def build_pytorch_transformers_dataloader_with_lists( \
        list_input_ids: Tensor, \
        list_input_mask: Tensor, \
        list_segment_ids: Tensor, \
        list_label_id: Tensor, \
        featurization_dataloader_batch_size: int):
        """
        Process a list of idized instances and build tensors.
        """
        # ---- NOTE-PYLINT ---- E1102: tensor is not callable
        # pylint: disable=E1102
        pytorch_transformers_all_input_ids = \
            tensor(
                list_input_ids,
                dtype=torch_long)
        pytorch_transformers_all_input_mask = \
            tensor(
                list_input_mask,
                dtype=torch_long)
        pytorch_transformers_all_segment_ids = \
            tensor(
                list_segment_ids,
                dtype=torch_long)
        pytorch_transformers_all_label_ids = \
            tensor(
                list_label_id,
                dtype=torch_long)
        tensor_dataset = TensorDataset(
            pytorch_transformers_all_input_ids,
            pytorch_transformers_all_input_mask,
            pytorch_transformers_all_segment_ids,
            pytorch_transformers_all_label_ids)
        sampler = SequentialSampler(tensor_dataset)
        pytorch_transformers_dataloader = DataLoader(
            dataset=tensor_dataset,
            sampler=sampler,
            batch_size=featurization_dataloader_batch_size)
        return (pytorch_transformers_dataloader,
                pytorch_transformers_all_input_ids,
                pytorch_transformers_all_input_mask,
                pytorch_transformers_all_segment_ids,
                pytorch_transformers_all_label_ids)

    @staticmethod
    def truncate_sequence_pair( \
        features: List[str], \
        features_auxiliary: List[str], \
        max_length: int):
        """
        Truncates a sequence pair in place to the maximum length.
        REVISED-FROM-OSS-REFERENCE: pytorch-transformers/examples/utils_glue.py
        """
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(features) + len(features_auxiliary)
            if total_length <= max_length:
                break
            if len(features) > len(features_auxiliary):
                features.pop()
            else:
                features_auxiliary.pop()
