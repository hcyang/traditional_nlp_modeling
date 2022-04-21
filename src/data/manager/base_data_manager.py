# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
This module implements base data manager objects
"""

import random

from typing import List
from typing import NoReturn
from typing import Tuple

from data.manager.record_generator_cross_validation_partitioner \
    import CrossValidationPartitionerRecordGenerator
from data.manager.record_generator_bootstrap_sampling_partitioner \
    import BootstrapSamplingPartitionerRecordGenerator
from data.manager.record_generator_leave_one_out_partitioner \
    import LeaveOneOutPartitionerRecordGenerator

from utility.io_helper.io_helper \
    import IoHelper
from utility.datatype_helper.datatype_helper \
    import DatatypeHelper
from utility.debugging_helper.debugging_helper \
    import DebuggingHelper

DATA_SOURCE_MODE_DEFAULT: int = 0
DATA_SOURCE_MODE_TESTING: int = 1
DATA_SOURCE_MODE_PREDICTING: int = 2

DATASET_DEFAULT_LABEL: str = 'UNKNOWN'

DEFAULT_BASE_DATA_MANAGER_ARGUMENT_DATASET_FILENAME: str = None
def set_default_base_data_manager_argument_dataset_filename(value: str) -> str:
    """
    Reset the default argument value for 'dataset_filename'.
    """
    # ---- NOTE-PYLINT ---- W0603: Using the global statement
    # pylint: disable=W0603
    global DEFAULT_BASE_DATA_MANAGER_ARGUMENT_DATASET_FILENAME
    DEFAULT_BASE_DATA_MANAGER_ARGUMENT_DATASET_FILENAME = value
def get_default_base_data_manager_argument_dataset_filename() -> str:
    """
    Reset the default argument value for 'dataset_filename'.
    """
    # ---- NOTE-PYLINT ---- W0603: Using the global statement
    # pylint: disable=W0603
    global DEFAULT_BASE_DATA_MANAGER_ARGUMENT_DATASET_FILENAME
    return DEFAULT_BASE_DATA_MANAGER_ARGUMENT_DATASET_FILENAME

DEFAULT_BASE_DATA_MANAGER_ARGUMENT_DATASET_DELIMITER: str = '\t'
def set_default_base_data_manager_argument_dataset_delimiter(value: str) -> str:
    """
    Reset the default argument value for 'dataset_delimiter'.
    """
    # ---- NOTE-PYLINT ---- W0603: Using the global statement
    # pylint: disable=W0603
    global DEFAULT_BASE_DATA_MANAGER_ARGUMENT_DATASET_DELIMITER
    DEFAULT_BASE_DATA_MANAGER_ARGUMENT_DATASET_DELIMITER = value
def get_default_base_data_manager_argument_dataset_delimiter() -> str:
    """
    Reset the default argument value for 'dataset_delimiter'.
    """
    # ---- NOTE-PYLINT ---- W0603: Using the global statement
    # pylint: disable=W0603
    global DEFAULT_BASE_DATA_MANAGER_ARGUMENT_DATASET_DELIMITER
    return DEFAULT_BASE_DATA_MANAGER_ARGUMENT_DATASET_DELIMITER

DEFAULT_BASE_DATA_MANAGER_ARGUMENT_DATASET_QUOTECHAR: str = None
def set_default_base_data_manager_argument_dataset_quotechar(value: str) -> str:
    """
    Reset the default argument value for 'dataset_quotechar'.
    """
    # ---- NOTE-PYLINT ---- W0603: Using the global statement
    # pylint: disable=W0603
    global DEFAULT_BASE_DATA_MANAGER_ARGUMENT_DATASET_QUOTECHAR
    DEFAULT_BASE_DATA_MANAGER_ARGUMENT_DATASET_QUOTECHAR = value
def get_default_base_data_manager_argument_dataset_quotechar() -> str:
    """
    Reset the default argument value for 'dataset_quotechar'.
    """
    # ---- NOTE-PYLINT ---- W0603: Using the global statement
    # pylint: disable=W0603
    global DEFAULT_BASE_DATA_MANAGER_ARGUMENT_DATASET_QUOTECHAR
    return DEFAULT_BASE_DATA_MANAGER_ARGUMENT_DATASET_QUOTECHAR

def process_base_data_manager_arguments(parser):
    """
    To process data manager related arguments.
    """
    if parser is None:
        DebuggingHelper.throw_exception(
            'input argument, parser, is None')
    parser.add_argument(
        '--dataset_filename',
        default=DEFAULT_BASE_DATA_MANAGER_ARGUMENT_DATASET_FILENAME,
        type=str,
        required=False,
        help='Dataset filename.')
    parser.add_argument(
        '--dataset_name',
        default=None,
        type=str,
        required=False,
        help='Dataset name.')
    parser.add_argument(
        '--dataset_encoding',
        default='utf-8',
        type=str,
        required=False,
        help='encoding.')
    parser.add_argument(
        '--dataset_delimiter',
        default=DEFAULT_BASE_DATA_MANAGER_ARGUMENT_DATASET_DELIMITER,
        type=str,
        required=False,
        help='Column delimiter.')
    parser.add_argument(
        '--dataset_quotechar',
        default=DEFAULT_BASE_DATA_MANAGER_ARGUMENT_DATASET_QUOTECHAR,
        type=str,
        required=False,
        help='quotechar.')
    parser.add_argument(
        '--dataset_has_header',
        default=True,
        type=DatatypeHelper.to_bool,
        required=False,
        help='Dataset has a header or not.')
    parser.add_argument(
        '--dataset_skiprows',
        default=0,
        type=int,
        required=False,
        help='Number of rows to skip.')
    parser.add_argument(
        '--dataset_skip_errors',
        default=False,
        type=DatatypeHelper.to_bool,
        required=False,
        help='Skip erroneous records or throw an exception.')
    parser.add_argument(
        '--ngram_split_separator',
        default=' ',
        type=str,
        required=False,
        help='separator for splitting an input text.')
    parser.add_argument(
        '--dataset_ignore_empty_text',
        default=True,
        type=DatatypeHelper.to_bool,
        required=False,
        help='Skip records with empty text.')
    parser.add_argument(
        '--dataset_ignore_empty_label',
        default=True,
        type=DatatypeHelper.to_bool,
        required=False,
        help='Skip records with empty label.')
    parser.add_argument(
        '--dataset_default_label',
        default=DATASET_DEFAULT_LABEL,
        type=str,
        required=False,
        help='Default label if input label is empty.')
    return parser

def process_randomization_arguments(parser):
    """
    To process data manager related arguments.
    """
    if parser is None:
        DebuggingHelper.throw_exception(
            'input argument, parser, is None')
    parser.add_argument(
        '--random_number_generator_seed',
        default=47,
        type=int,
        required=False,
        help='shuffle randomization seed.')
    return parser

def process_base_data_manager_randomization_arguments(parser):
    """
    To process data manager related arguments.
    """
    if parser is None:
        DebuggingHelper.throw_exception(
            'input argument, parser, is None')
    parser.add_argument(
        '--randomize_dataset',
        default=True,
        type=DatatypeHelper.to_bool,
        required=False,
        help='whether to randomly shuffle records or not.')
    process_randomization_arguments(parser)
    return parser

class BaseDataManager:
    """
    This class can manage data.
    """
    # ---- NOTE-PYLINT ---- R0903: Too few public methods
    # pylint: disable=R0903
    def __init__(self, \
        filename: str, \
        name: str = None, \
        encoding: str = 'utf-8', \
        ignore_empty_text: bool = True, \
        ignore_empty_label: bool = True, \
        default_label: str = DATASET_DEFAULT_LABEL):
        """
        Init with a filename.
        """
        self.memory_based = DatatypeHelper.is_none_empty_whitespaces_or_nan(filename)
        self.file_based = not self.memory_based
        if self.file_based:
            if DatatypeHelper.is_none_empty_whitespaces_or_nan(filename):
                DebuggingHelper.throw_exception(
                    'input argument, filename, is None')
            if not IoHelper.can_open_file(filename):
                DebuggingHelper.throw_exception(
                    f'input argument, cannot open file for filename="{filename}"')
        self.filename = filename
        self.name = name
        if DatatypeHelper.is_none_empty_whitespaces_or_nan(self.name):
            self.name = self.filename
        if DatatypeHelper.is_none_empty_whitespaces_or_nan(encoding):
            encoding = 'utf-8'
        self.encoding = encoding
        self.records = None
        self.ignore_empty_text: bool = ignore_empty_text
        self.ignore_empty_label: bool = ignore_empty_label
        self.default_label: str = default_label

    def get_shape(self) -> Tuple[int, int]:
        """
        Return the number of records and fields.
        """
        if DatatypeHelper.is_none_empty_whitespaces_or_nan(self.records):
            DebuggingHelper.throw_exception(
                'input argument, self.records, is None')
        number_records = self.get_count()
        number_fields = 0
        if number_records > 0:
            number_fields = len(self.records[0])
        return (number_records, number_fields)

    def get_name(self) -> str:
        """
        Return the name for this BaseDataManager object.
        """
        return self.name

    def get_filename(self) -> str:
        """
        Return the filename used in this BaseDataManager object.
        """
        return self.filename

    def randomize_records( \
        self, \
        randomize_dataset: bool = True, \
        random_number_generator_seed: int = 47) -> NoReturn:
        """
        Create a list randomized records,
        each is a list of strings.
        """
        if DatatypeHelper.is_none_empty_whitespaces_or_nan(self.records):
            DebuggingHelper.throw_exception(
                'input argument, self.records, is None')
        if random_number_generator_seed is not None:
            random.seed(random_number_generator_seed)
        if randomize_dataset:
            random.shuffle(self.records)

    def get_record_generator(self, \
        generator_configuration_int_0: int = 5, \
        generator_configuration_int_1: int = -1, \
        generator_configuration_bool_0: bool = False) -> CrossValidationPartitionerRecordGenerator:
        """
        Return a generator for scanning the records.
        Default is to use CrossValidationPartitionerRecordGenerator,
        can be overriden by child objects.
        """
        return CrossValidationPartitionerRecordGenerator(
            records=self.records,
            number_partitions=generator_configuration_int_0,
            partition_index=generator_configuration_int_1,
            to_except=generator_configuration_bool_0)

    def get_bootstrap_sampling_record_generator(self, \
        generator_configuration_int_0: int = 5, \
        generator_configuration_int_1: int = -1, \
        generator_configuration_bool_0: bool = False) -> \
            BootstrapSamplingPartitionerRecordGenerator:
        """
        Return a generator for bootstrap sampling the records.
        """
        # ---- NOTE-PYLINT ---- W0613: Unused argument
        #      'generator_configuration_int_1' (unused-argument)
        # pylint: disable=W0613
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        return BootstrapSamplingPartitionerRecordGenerator(
            records=self.records,
            percentage_records=generator_configuration_int_0 / 100,
            to_except=generator_configuration_bool_0)

    def get_leave_one_out_record_generator(self, \
        generator_configuration_int_0: int = 5, \
        generator_configuration_int_1: int = -1, \
        generator_configuration_bool_0: bool = False) -> LeaveOneOutPartitionerRecordGenerator:
        """
        Return a generator for bootstrap sampling the records.
        """
        # ---- NOTE-PYLINT ---- W0613: Unused argument
        #      'generator_configuration_int_0' (unused-argument)
        # pylint: disable=W0613
        return LeaveOneOutPartitionerRecordGenerator(
            records=self.records,
            partition_index=generator_configuration_int_1,
            to_except=generator_configuration_bool_0)

    def create_records(self) -> List[List[str]]:
        # ---- NOTE-ABSTRACT-FUNCTION ----
        """
        Create a list of records, each is a list of strings.
        An abstract function that child classes must override.
        """
        raise NotImplementedError()

    def get_records(self) -> List[List[str]]:
        """
        Return a list of records, each is a list of strings.
        """
        if DatatypeHelper.is_none_empty_whitespaces_or_nan(self.records):
            DebuggingHelper.throw_exception('input argument, self.records, is None')
        return self.records

    def get_count(self) -> int:
        """
        Return the number of records in this dataset.
        """
        return len(self.get_records())
