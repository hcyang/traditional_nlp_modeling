# -*- coding: latin-1 -*-
"""
This module implements a bootstrap sampling record generator.
"""

from typing import List

import random

from data.manager.base_record_generator \
    import BaseRecordGenerator

from utility.datatype_helper.datatype_helper \
    import DatatypeHelper
from utility.debugging_helper.debugging_helper \
    import DebuggingHelper

class BootstrapSamplingPartitionerRecordGenerator(BaseRecordGenerator):
    """
    This class is a record generator.
    """
    # ---- NOTE-PYLINT ---- R0903: Too few public methods
    # pylint: disable=R0903
    # ---- NOTE-PYLINT ---- R0902: Too many instance attributes
    # pylint: disable=R0902
    # ---- NOTE-PYLINT ---- C0301: Line too long
    # pylint: disable=C0301
    def __init__(self, \
        records: List[object], \
        percentage_records: float = 1.0, \
        to_except: bool = False):
        """
        Init a BootstrapSamplingPartitionerRecordGenerator object.
        """
        super(BootstrapSamplingPartitionerRecordGenerator, self).__init__()
        if DatatypeHelper.is_none_empty_whitespaces_or_nan(records):
            DebuggingHelper.throw_exception(
                'input argument, records, is None')
        if percentage_records <= 0:
            DebuggingHelper.throw_exception(
                f'input argument, '
                f'percentage_records|{percentage_records}|, is <= 0')
        # ----
        self.records = records
        self.number_records = len(records)
        self.percentage_records = percentage_records
        self.to_except = to_except
        self.number_partition_records = int(self.number_records * percentage_records)
        # ----
        self.number_generator_records = self.number_partition_records
        # ---- NOTE: Find out the sample indexes eagerly in the constructor,
        #            so number_generator_records can be calculated
        #            before the generate() function is being called.
        self.sample_indexes = \
            [random.randint(0, self.number_records - 1) for i in range(self.number_generator_records)]
        if self.to_except:
            self.sample_indexes = \
                set(self.sample_indexes)
            self.sample_indexes = \
                [index for index in range(self.number_records) if index not in self.sample_indexes]
            self.number_generator_records = len(self.sample_indexes)

    def generate(self):
        # ---- NOTE-CONCRETE-OVERRIDE-FUNCTION ----
        """
        Generator method.
        """
        for i in self.sample_indexes:
            yield self.records[i]
