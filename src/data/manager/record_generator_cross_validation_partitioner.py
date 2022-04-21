# -*- coding: latin-1 -*-
"""
This module implements a cross validation record generator.
"""

from typing import List

from data.manager.base_record_generator \
    import BaseRecordGenerator

from utility.list_array_helper.list_array_helper \
    import ListArrayHelper
from utility.datatype_helper.datatype_helper \
    import DatatypeHelper
from utility.debugging_helper.debugging_helper \
    import DebuggingHelper

class CrossValidationPartitionerRecordGenerator(BaseRecordGenerator):
    """
    This class is a record generator.
    """
    # ---- NOTE-PYLINT ---- R0903: Too few public methods
    # pylint: disable=R0903
    # ---- NOTE-PYLINT ---- R0902: Too many instance attributes
    # pylint: disable=R0902
    def __init__(self, \
        records: List[object], \
        number_partitions: int = 5, \
        partition_index: int = -1, \
        to_except: bool = False):
        """
        Init a CrossValidationPartitionerRecordGenerator object.
        """
        super(CrossValidationPartitionerRecordGenerator, self).__init__()
        if DatatypeHelper.is_none_empty_whitespaces_or_nan(records):
            DebuggingHelper.throw_exception(
                'input argument, records, is None')
        if number_partitions <= 0:
            DebuggingHelper.throw_exception(
                f'input argument, '
                f'number_partitions|{number_partitions}|, is <= 0')
        if partition_index < 0:
            DebuggingHelper.throw_exception(
                f'input argument, '
                f'partition_index|{partition_index}|, is < 0')
        if partition_index >= number_partitions:
            DebuggingHelper.throw_exception(
                f'input argument, '
                f'partition_index|{partition_index}|, is >= number_partitions|{number_partitions}|')
        # ----
        self.records = records
        self.number_records = len(records)
        self.number_partitions = number_partitions
        self.partition_index = partition_index
        self.to_except = to_except
        # ----
        self.segment_begin_end_index_tuple = ListArrayHelper.get_segment_begin_end_index_tuple(
            number_records=self.number_records,
            number_segments=self.number_partitions,
            segment_id=self.partition_index)
        self.segmenting_begin_index_for_segment, \
        self.segmenting_end_index_for_segment, \
        self.segmenting_number_segment_records, \
        self.segmenting_estimated_number_per_segment, \
        self.segmenting_number_records = self.segment_begin_end_index_tuple
        # ----
        self.record_count_per_partition = self.segmenting_estimated_number_per_segment
        self.begin_index = self.segmenting_begin_index_for_segment
        self.end_index = self.segmenting_end_index_for_segment
        self.number_partition_records = self.segmenting_number_segment_records
        # ----
        if self.to_except:
            self.number_generator_records = self.number_records - self.number_partition_records
        else:
            self.number_generator_records = self.number_partition_records

    def generate(self):
        # ---- NOTE-CONCRETE-OVERRIDE-FUNCTION ----
        """
        Generator method.
        """
        if self.to_except:
            for i in range(0, self.begin_index):
                yield self.records[i]
            for i in range(self.end_index, self.number_records):
                yield self.records[i]
        else:
            for i in range(self.begin_index, self.end_index):
                yield self.records[i]
