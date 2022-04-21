# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
This module provides some common debugging helper functions.
"""

from typing import List
from typing import Tuple

class ListArrayHelper:
    """
    This class contains some common functions for processing list or array objects.
    """

    @staticmethod
    def get_difference(first_list: List[object], second_list: List[object]) -> List[object]:
        """
        Find the items in first_list, but not in second_list.
        """
        second_set = set(second_list)
        return [item for item in first_list if item not in second_set]

    @staticmethod
    def get_segment_begin_end_index_tuple( \
        number_records: int, \
        number_segments: int, \
        segment_id: int) -> Tuple[int]:
        """
        Return the begin, end, number, and estimated number per segment.
        """
        estimated_number_per_segment = 0
        if (number_segments > 0) and (0 <= segment_id < number_segments):
            estimated_number_per_segment = number_records // number_segments
            estimated_number_per_segment += 1
            begin_index_for_segment = segment_id * estimated_number_per_segment
            end_index_for_segment = begin_index_for_segment + estimated_number_per_segment
        else:
            begin_index_for_segment = 0
            end_index_for_segment = number_records
        if begin_index_for_segment > number_records:
            begin_index_for_segment = number_records
        if end_index_for_segment > number_records:
            end_index_for_segment = number_records
        if end_index_for_segment < begin_index_for_segment:
            end_index_for_segment = begin_index_for_segment
        return (begin_index_for_segment,
                end_index_for_segment,
                (end_index_for_segment - begin_index_for_segment),
                estimated_number_per_segment,
                number_records)

    @staticmethod
    def get_first_maximum_entry(a_list: List[object]) -> Tuple[object, int]:
        """
        Return the index, entry tuple to the max entry first in the list.
        """
        max_index = -1
        max_entry = None
        for current_index, current_entry in enumerate(a_list):
            if max_entry is None:
                max_entry = current_entry
                max_index = current_index
            else:
                if max_entry < current_entry:
                    max_entry = current_entry
                    max_index = current_index
        return (max_entry, max_index)

    @staticmethod
    def get_last_maximum_entry(a_list: List[object]) -> Tuple[object, int]:
        """
        Return the index, entry tuple to the max entry last in the list.
        """
        max_index = -1
        max_entry = None
        for current_index, current_entry in enumerate(a_list):
            if max_entry is None:
                max_entry = current_entry
                max_index = current_index
            else:
                if max_entry <= current_entry:
                    max_entry = current_entry
                    max_index = current_index
        return (max_entry, max_index)

    @staticmethod
    def get_first_minimum_entry(a_list: List[object]) -> Tuple[object, int]:
        """
        Return the index, entry tuple to the min entry first in the list.
        """
        min_index = -1
        min_entry = None
        for current_index, current_entry in enumerate(a_list):
            if min_entry is None:
                min_entry = current_entry
                min_index = current_index
            else:
                if min_entry > current_entry:
                    min_entry = current_entry
                    min_index = current_index
        return (min_entry, min_index)

    @staticmethod
    def get_last_minimum_entry(a_list: List[object]) -> Tuple[object, int]:
        """
        Return the index, entry tuple to the min entry last in the list.
        """
        min_index = -1
        min_entry = None
        for current_index, current_entry in enumerate(a_list):
            if min_entry is None:
                min_entry = current_entry
                min_index = current_index
            else:
                if min_entry >= current_entry:
                    min_entry = current_entry
                    min_index = current_index
        return (min_entry, min_index)
