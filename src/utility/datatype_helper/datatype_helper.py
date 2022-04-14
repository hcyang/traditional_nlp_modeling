# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
This module provides some helper functions for processing datatype info.
"""

import argparse
import math

from typing import Any
from typing import Dict
from typing import List

import threading

# from utility.debugging_helper.debugging_helper \
#    import DebuggingHelper

class DatatypeHelper:
    """
    This class contains some common functions for processing datatype info.
    """
    NAN = float('nan')
    INF = float('inf')
    NINF = -float('inf')

    _lock_for_update_count_dictionary = threading.Lock()
    _lock_for_update_int_count_dictionary = threading.Lock()

    @staticmethod
    def get_class_name(obj: Any):
        """
        Get the class name of an object.
        """
        return type(obj).__name__

    @staticmethod
    def flatten(list_of_lists: List[List[object]]):
        """
        Flatten a list of lists of objects into a list.
        """
        return [item for sub_list in list_of_lists for item in sub_list]

    @staticmethod
    def update_count_dictionary( \
        dictionary_first: Dict[object, float], \
        dictionary_second: Dict[object, float]) -> Dict[object, float]:
        """
        Update a count dictionary with another.
        NOTE: this locking implementation, though thread/parallel safe,
              can cause too much locking overhead.
        """
        # DebuggingHelper.write_line_to_system_console_out('entered')
        for key in dictionary_second:
            DatatypeHelper._lock_for_update_count_dictionary.acquire()
            try:
                if key in dictionary_first:
                    dictionary_first[key] = dictionary_first[key] + dictionary_second[key]
                else:
                    dictionary_first[key] = dictionary_second[key]
            finally:
                DatatypeHelper._lock_for_update_count_dictionary.release()
        return dictionary_first

    @staticmethod
    def update_count_dictionary_synchronized( \
        dictionary_first: Dict[object, float], \
        dictionary_second: Dict[object, float]) -> Dict[object, float]:
        """
        Update a count dictionary with another.
        NOTE: this locking implementation, though thread/parallel safe,
              can cause too much locking overhead.
        """
        # DebuggingHelper.write_line_to_system_console_out('entered')
        DatatypeHelper._lock_for_update_count_dictionary.acquire()
        try:
            DatatypeHelper.update_count_dictionary_direct(
                dictionary_first=dictionary_first,
                dictionary_second=dictionary_second)
        finally:
            DatatypeHelper._lock_for_update_count_dictionary.release()
        return dictionary_first

    @staticmethod
    def update_count_dictionary_direct( \
        dictionary_first: Dict[object, float], \
        dictionary_second: Dict[object, float]) -> Dict[object, float]:
        """
        Update a count dictionary with another.
        """
        # DebuggingHelper.write_line_to_system_console_out('entered')
        for key in dictionary_second:
            if key in dictionary_first:
                dictionary_first[key] = dictionary_first[key] + dictionary_second[key]
            else:
                dictionary_first[key] = dictionary_second[key]
        return dictionary_first

    @staticmethod
    def update_count_dictionaries_direct( \
        dictionary_first: Dict[object, float], \
        dictionaries: List[Dict[object, float]]) -> Dict[object, float]:
        """
        Update a count dictionary with another.
        """
        # DebuggingHelper.write_line_to_system_console_out('entered')
        for dictionary in dictionaries:
            DatatypeHelper.update_count_dictionary_direct(dictionary_first, dictionary)
        return dictionary_first

    @staticmethod
    def update_int_count_dictionary( \
        dictionary_first: Dict[object, int], \
        dictionary_second: Dict[object, int]) -> Dict[object, int]:
        """
        Update a count dictionary with another.
        NOTE: this locking implementation, though thread/parallel safe,
              can cause too much locking overhead.
        """
        # DebuggingHelper.write_line_to_system_console_out('entered')
        for key in dictionary_second:
            DatatypeHelper._lock_for_update_int_count_dictionary.acquire()
            try:
                if key in dictionary_first:
                    dictionary_first[key] = dictionary_first[key] + dictionary_second[key]
                else:
                    dictionary_first[key] = dictionary_second[key]
            finally:
                DatatypeHelper._lock_for_update_int_count_dictionary.release()
        return dictionary_first

    @staticmethod
    def update_int_count_dictionary_synchronized( \
        dictionary_first: Dict[object, int], \
        dictionary_second: Dict[object, int]) -> Dict[object, int]:
        """
        Update a count dictionary with another.
        NOTE: this locking implementation, though thread/parallel safe,
              can cause too much locking overhead.
        """
        # DebuggingHelper.write_line_to_system_console_out('entered')
        DatatypeHelper._lock_for_update_int_count_dictionary.acquire()
        try:
            DatatypeHelper.update_int_count_dictionary_direct(
                dictionary_first=dictionary_first,
                dictionary_second=dictionary_second)
        finally:
            DatatypeHelper._lock_for_update_int_count_dictionary.release()
        return dictionary_first

    @staticmethod
    def update_int_count_dictionary_direct( \
        dictionary_first: Dict[object, int], \
        dictionary_second: Dict[object, int]) -> Dict[object, int]:
        """
        Update a count dictionary with another.
        """
        # DebuggingHelper.write_line_to_system_console_out('entered')
        for key in dictionary_second:
            if key in dictionary_first:
                dictionary_first[key] = dictionary_first[key] + dictionary_second[key]
            else:
                dictionary_first[key] = dictionary_second[key]
        return dictionary_first

    @staticmethod
    def update_int_count_dictionaries_direct( \
        dictionary_first: Dict[object, int], \
        dictionaries: List[Dict[object, int]]) -> Dict[object, int]:
        """
        Update a count dictionary with another.
        """
        # DebuggingHelper.write_line_to_system_console_out('entered')
        for dictionary in dictionaries:
            DatatypeHelper.update_int_count_dictionary_direct(dictionary_first, dictionary)
        return dictionary_first

    @staticmethod
    def to_bool(value):
        """
        Convert an object to bool value
        """
        if isinstance(value, bool):
            return value
        if value.lower() in ('yes', 'true', 't', 'y', '1', 'positive', '+', 'on'):
            return True
        if value.lower() in ('no', 'false', 'f', 'n', '0', 'negative', '-', 'off'):
            return False
        raise argparse.ArgumentTypeError('Boolean value expected.')

    @staticmethod
    def none_string_if_none_empty_whitespaces_or_nan(value: str) -> str:
        """
        Return empty string if value is None.
        """
        if DatatypeHelper.is_none_empty_whitespaces_or_nan(value):
            return None
        return value

    @staticmethod
    def empty_string_if_none_empty_whitespaces_or_nan(value: str) -> str:
        """
        Return empty string if value is None.
        """
        if DatatypeHelper.is_none_empty_whitespaces_or_nan(value):
            return ''
        return value

    @staticmethod
    def is_none_empty_whitespaces_or_nan(obj) -> bool:
        """
        Detect if an object is an anomaly.
        """
        # ---- NOTE-DOES-NOT-HANDLE-float ---- return (obj is None) or (not obj)
        if obj is None:
            return True
        if isinstance(obj, str):
            obj = obj.strip()
            return not obj
        if isinstance(obj, List):
            return not obj
        if isinstance(obj, Dict):
            return not obj
        if isinstance(obj, float):
            return math.isnan(obj)
        return False

    @staticmethod
    def is_none(obj) -> bool:
        """
        Detect if an object is an anomaly.
        """
        is_none: bool = (obj is None)
        return is_none
