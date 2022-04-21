# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
This module tests common list-and-array helper functions defined in list_array_help.
"""

from utility.list_array_helper.list_array_helper import ListArrayHelper

from utility.debugging_helper.debugging_helper import DebuggingHelper

def main():
    """
    The main() function can quickly test ListArrayHelper functions
    """
    # ------------------------------------------------------------------------
    a_list = [2, 5, 1, 7, 5, 4, 8, 9, 2, 4, 9, 7, 3, 1]
    DebuggingHelper.write_line_to_system_console_out(
        f'a_list={a_list}')
    max_entry, max_index = ListArrayHelper.get_first_maximum_entry(a_list)
    DebuggingHelper.write_line_to_system_console_out(
        f'max_entry={max_entry},max_index={max_index}')
    max_entry, max_index = ListArrayHelper.get_last_maximum_entry(a_list)
    DebuggingHelper.write_line_to_system_console_out(
        f'max_entry={max_entry},max_index={max_index}')
    min_entry, min_index = ListArrayHelper.get_first_minimum_entry(a_list)
    DebuggingHelper.write_line_to_system_console_out(
        f'min_entry={min_entry},min_index={min_index}')
    min_entry, min_index = ListArrayHelper.get_last_minimum_entry(a_list)
    DebuggingHelper.write_line_to_system_console_out(
        f'min_entry={min_entry},min_index={min_index}')
    # ------------------------------------------------------------------------
    number_records = len(a_list)
    number_segments = 3
    for segment_id in range(number_segments):
        segment_begin_end_index_tuple = ListArrayHelper.get_segment_begin_end_index_tuple(
            number_records=number_records,
            number_segments=number_segments,
            segment_id=segment_id)
        DebuggingHelper.write_line_to_system_console_out(
            f'segment_id={segment_id},'
            f'segment_begin_end_index_tuple={segment_begin_end_index_tuple}')
    # ------------------------------------------------------------------------
    number_segments = 5
    for segment_id in range(number_segments):
        segment_begin_end_index_tuple = ListArrayHelper.get_segment_begin_end_index_tuple(
            number_records=number_records,
            number_segments=number_segments,
            segment_id=segment_id)
        DebuggingHelper.write_line_to_system_console_out(
            f'segment_id={segment_id},'
            f'segment_begin_end_index_tuple={segment_begin_end_index_tuple}')
    # ------------------------------------------------------------------------
    number_segments = 14
    for segment_id in range(number_segments):
        segment_begin_end_index_tuple = ListArrayHelper.get_segment_begin_end_index_tuple(
            number_records=number_records,
            number_segments=number_segments,
            segment_id=segment_id)
        DebuggingHelper.write_line_to_system_console_out(
            f'segment_id={segment_id},'
            f'segment_begin_end_index_tuple={segment_begin_end_index_tuple}')
    # ------------------------------------------------------------------------
    number_segments = 15
    for segment_id in range(number_segments):
        segment_begin_end_index_tuple = ListArrayHelper.get_segment_begin_end_index_tuple(
            number_records=number_records,
            number_segments=number_segments,
            segment_id=segment_id)
        DebuggingHelper.write_line_to_system_console_out(
            f'segment_id={segment_id},'
            f'segment_begin_end_index_tuple={segment_begin_end_index_tuple}')
    # ------------------------------------------------------------------------
    number_segments = 16
    for segment_id in range(number_segments):
        segment_begin_end_index_tuple = ListArrayHelper.get_segment_begin_end_index_tuple(
            number_records=number_records,
            number_segments=number_segments,
            segment_id=segment_id)
        DebuggingHelper.write_line_to_system_console_out(
            f'segment_id={segment_id},'
            f'segment_begin_end_index_tuple={segment_begin_end_index_tuple}')
    # ------------------------------------------------------------------------

if __name__ == '__main__':
    main()
