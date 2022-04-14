# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
This module tests TextChineseConverter objects.
"""

# from typing import List
# from typing import Set

import os
import sys

import argparse

import codecs
# import csv
# import json
# import re
# import time

# from utility.io_helper.io_helper \
#     import IoHelper

from text_processor.text_chinese_converter \
    import TextChineseConverter

from utility.datatype_helper.datatype_helper \
    import DatatypeHelper
from utility.debugging_helper.debugging_helper \
    import DebuggingHelper

def process_text_piece_arguments(parser):
    """
    To process data manager related arguments.
    """
    DebuggingHelper.write_line_to_system_console_out(
        f'Calling process_text_piece_arguments() in {__name__}')
    if parser is None:
        DebuggingHelper.throw_exception(
            'input argument, parser, is None')
    parser.add_argument(
        '--rootpath',
        type=str,
        required=True,
        help='Root path for input Wikipedia dump and output processed files.')
    parser.add_argument(
        '--filename',
        type=str,
        required=True,
        help='Wikipedia dump input filename.')
    return parser

# ---- NOTE-PYLINT ---- C0103: Function name "" doesn't conform to snake_case naming style
# pylint: disable=C0103
def example_function_TextChineseConverter():
    """
    The main function to quickly test TextChineseConverter.
    """
    # ---- NOTE-PYLINT ---- C0301: Line too long
    # pylint: disable=C0301
    # ------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    # ------------------------------------------------------------------------
    process_text_piece_arguments(parser)
    # ------------------------------------------------------------------------
    args: argparse.Namespace = parser.parse_args()
    DebuggingHelper.write_line_to_system_console_out(
        f'sys.path={str(sys.path)}')
    # DebuggingHelper.write_line_to_system_console_out(
    #     f'args={str(args)}')
    # ------------------------------------------------------------------------
    output_root_process_path: str = \
        args.rootpath
    output_dump_filename: str = \
        args.filename
    if DatatypeHelper.is_none_empty_whitespaces_or_nan(output_root_process_path):
        error_message: str = \
            f'ERROR: no process root path for the "rootpath" argument, args={str(args)}'
        DebuggingHelper.write_line_to_system_console_err(
            error_message)
        DebuggingHelper.print_in_color(error_message)
        return
    if DatatypeHelper.is_none_empty_whitespaces_or_nan(output_dump_filename):
        error_message: str = \
            f'ERROR: no output for the "filename" argument, args={str(args)}'
        DebuggingHelper.write_line_to_system_console_err(
            error_message)
        DebuggingHelper.print_in_color(error_message)
        return
    # ------------------------------------------------------------------------
    output_dump_path: str = \
        os.path.join(output_root_process_path, output_dump_filename)
    # ------------------------------------------------------------------------
    input_text: str = \
        "《老子》，又名《道德經》，是先秦時期的古籍，相傳為春秋末期思想家老子所著。《老子》為春秋戰國時期道家学派的代表性經典，亦是道教尊奉的經典。至唐代，唐太宗命人將《道德經》譯為梵語；唐玄宗时，尊此经为《道德眞經》。"
    input_text_converted: str = \
        TextChineseConverter.convert_simplified_to_traditional(input_text)
    input_text_converted_baseline: str = \
        "《老子》，又名《道德經》，是先秦時期的古籍，相傳為春秋末期思想家老子所著。《老子》為春秋戰國時期道家學派的代表性經典，亦是道教尊奉的經典。至唐代，唐太宗命人將《道德經》譯為梵語；唐玄宗時，尊此經爲《道德眞經》。"
    #    0 1 2 3 45 6 7 8 9 01 2 3 4 5 6 78 9 0 1 2 34 5 6 78 9 0 1 23 4 5 6 78 9 0 1 2 34 5 6 7 89 0 1 2 3 45 6 7 8 9 01 2 3 4 56 7 8 90 1 2 3 4 56 7 8 9 01 2 3 4 5 67 8 9 0 1 23 4 5 6 7 89 0 1 2 3 4
    # DebuggingHelper.write_line_to_system_console_out_debug(
    #     message=f'len(input_text)={len(input_text)}')
    encoding: str = "utf-8"
    with codecs.open(output_dump_path, "w", encoding) as output_dump:
        output_dump.write(input_text_converted)
        output_dump.write('\n')
        output_dump.write(input_text_converted_baseline)
        output_dump.write('\n')
    index_of_first_difference: int = \
        TextChineseConverter.index_of_first_difference(input_text_converted, input_text_converted_baseline)
    DebuggingHelper.write_line_to_system_console_out_debug(
        message=f'index_of_first_difference={index_of_first_difference}')
    DebuggingHelper.ensure(
        input_text_converted == input_text_converted_baseline,
        'input_text_converted|{}| != input_text_converted_baseline|{}|'.format(
            len(input_text_converted),
            len(input_text_converted_baseline)))
    # ------------------------------------------------------------------------

def main():
    """
    The main() function can quickly test TextChineseConverter objects.
    """
    example_function_TextChineseConverter()

if __name__ == '__main__':
    main()
    # ---- NOTE-PYLINT ---- C0301: Line too long
    # pylint: disable=C0301
    # ---- NOTE-FOR-REFERENCE ---- python app_chinese_converter.py --rootpath \tmp --filename app_chinese_converter.txt
