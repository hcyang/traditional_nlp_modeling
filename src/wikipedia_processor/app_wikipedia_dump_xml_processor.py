# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
This module tests WikipediaDumpXmlProcessor objects.
"""

from typing import Any
from typing import Set

import os
# import sys

import argparse

from utility.datatype_helper.datatype_helper \
    import DatatypeHelper
from utility.debugging_helper.debugging_helper \
    import DebuggingHelper

from wikipedia_processor.wikipedia_dump_xml_processor \
    import WikipediaDumpXmlProcessor

def process_wikipedia_processor_arguments(parser):
    """
    To process data manager related arguments.
    """
    DebuggingHelper.write_line_to_system_console_out(
        f'Calling process_wikipedia_processor_arguments() in {__name__}')
    if parser is None:
        DebuggingHelper.throw_exception(
            'input argument, parser, is None')
    parser.add_argument(
        '--input_path',
        type=str,
        required=True,
        help='Input path for the input Wikipedia dump files.')
    parser.add_argument(
        '--file_name',
        type=str,
        required=True,
        help='Wikipedia dump input file name.')
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Output path for the output processed files.')
    return parser

# ---- NOTE-PYLINT ---- C0103: Function name "" doesn't conform to snake_case naming style
# pylint: disable=C0103
def example_function_WikipediaDumpXmlProcessor():
    """
    The main function to quickly test WikipediaDumpXmlProcessor.
    """
    # ---- NOTE-PYLINT ---- R0915: Too many statements
    # pylint: disable=R0915
    # ------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    # ------------------------------------------------------------------------
    process_wikipedia_processor_arguments(parser)
    # ------------------------------------------------------------------------
    args: argparse.Namespace = parser.parse_args()
    # DebuggingHelper.write_line_to_system_console_out(
    #     f'sys.path={str(sys.path)}')
    # DebuggingHelper.write_line_to_system_console_out(
    #     f'args={str(args)}')
    # ------------------------------------------------------------------------
    wikipedia_input_process_path: str = \
        os.path.normpath(args.input_path)
    wikipedia_dump_xml_filename: str = \
        os.path.normpath(args.file_name)
    wikipedia_output_process_path: str = \
        os.path.normpath(args.output_path)
    if DatatypeHelper.is_none_empty_whitespaces_or_nan(wikipedia_input_process_path):
        error_message: str = \
            f'ERROR: no process input path for the "input_path" argument, args={str(args)}'
        DebuggingHelper.write_line_to_system_console_err(
            error_message)
        DebuggingHelper.print_in_color(error_message)
        return
    if DatatypeHelper.is_none_empty_whitespaces_or_nan(wikipedia_dump_xml_filename):
        error_message: str = \
            f'ERROR: no input for the "file_name" argument, args={str(args)}'
        DebuggingHelper.write_line_to_system_console_err(
            error_message)
        DebuggingHelper.print_in_color(error_message)
        return
    if DatatypeHelper.is_none_empty_whitespaces_or_nan(wikipedia_output_process_path):
        wikipedia_output_process_path = wikipedia_input_process_path
    # ------------------------------------------------------------------------
    wikipedia_dump_xml_processor: WikipediaDumpXmlProcessor = WikipediaDumpXmlProcessor(
        wikipedia_input_process_path=wikipedia_input_process_path,
        wikipedia_dump_xml_filename=wikipedia_dump_xml_filename,
        wikipedia_output_process_path=wikipedia_output_process_path)
    # ------------------------------------------------------------------------
    # ---- NOTE-PYLINT ---- C0301: Line too long
    # pylint: disable=C0301
    number_pages_processed_for_break: int = -1 # ---- NOTE ---- -1 for no limit, a positive number stops the processing after the designated number of page processed.
    number_pages_processed_for_progress_update: int = 1000 # ---- NOTE ---- progress reporting after a certain number of pages processed.
    # ------------------------------------------------------------------------
    debugging_title: str = None # ---- NOTE-FOR-DEBUGGING ---- limit only processing the pages with the title.
    debugging_title_alternatives: Set[str] = None # ---- NOTE-FOR-DEBUGGING ---- limit only processing the pages with the alternative titles.
    # debugging_title = \
    #     '論語'
    # debugging_title_alternatives = \
    #     {'论语'}
    # debugging_title = \
    #     '道德經'
    # debugging_title_alternatives = \
    #     {'老子_(書)', '老子 (書)', '老子 (书)', '老子_(书)', '道德经'}
    # ------------------------------------------------------------------------
    debugging_text_substring: str = None
    # debugging_text_substring = \
    #     '老子言道德之意，著书上下篇', # '老子言道德之意，著書上下篇', # '成為一冊難以看得懂的'
    # ------------------------------------------------------------------------
    json_configuration: Any = {}
    json_configuration['json_friendly_structure_to_output_text_Record'] = \
        True
    json_configuration['json_friendly_structure_to_output_lines_Record'] = \
        True
    json_configuration['json_friendly_structure_to_output_text_RecordHeading1'] = \
        False
    json_configuration['json_friendly_structure_to_output_lines_RecordHeading1'] = \
        False
    json_configuration['json_friendly_structure_to_output_text_RecordHeading2'] = \
        False
    json_configuration['json_friendly_structure_to_output_lines_RecordHeading2'] = \
        False
    json_configuration['json_friendly_structure_to_output_text_RecordHeading3'] = \
        False
    json_configuration['json_friendly_structure_to_output_lines_RecordHeading3'] = \
        False
    json_configuration['json_friendly_structure_to_output_text_RecordHeading4'] = \
        False
    json_configuration['json_friendly_structure_to_output_lines_RecordHeading4'] = \
        False
    json_configuration['json_friendly_structure_to_output_text_RecordHeading5'] = \
        False
    json_configuration['json_friendly_structure_to_output_lines_RecordHeading5'] = \
        False
    json_configuration['json_friendly_structure_to_output_text_RecordHeading6'] = \
        True
    json_configuration['json_friendly_structure_to_output_lines_RecordHeading6'] = \
        True
    json_configuration['json_friendly_structure_to_output_text_RecordTemplateExtraction'] = \
        False
    json_configuration['json_friendly_structure_to_output_lines_RecordTemplateExtraction'] = \
        False
    json_configuration['json_friendly_structure_to_output_text_RecordTemplateArgumentExtraction'] = \
        False
    json_configuration['json_friendly_structure_to_output_lines_RecordTemplateArgumentExtraction'] = \
        False
    json_configuration['json_friendly_structure_to_output_text_RecordTextPieceExtraction'] = \
        True
    json_configuration['json_friendly_structure_to_output_lines_RecordTextPieceExtraction'] = \
        True
    json_configuration['json_friendly_structure_to_output_text_RecordHtmlGenericIndividualTagExtraction'] = \
        False
    json_configuration['json_friendly_structure_to_output_lines_RecordHtmlGenericIndividualTagExtraction'] = \
        False
    json_configuration['json_friendly_structure_to_output_text_RecordHtmlReferenceExtraction'] = \
        False
    json_configuration['json_friendly_structure_to_output_lines_RecordHtmlReferenceExtraction'] = \
        False
    json_configuration['json_friendly_structure_to_output_text_RecordCommentExtraction'] = \
        False
    json_configuration['json_friendly_structure_to_output_lines_RecordCommentExtraction'] = \
        False
    # ------------------------------------------------------------------------
    wikipedia_dump_xml_processor.process( \
        number_pages_processed_for_break=number_pages_processed_for_break, \
        number_pages_processed_for_progress_update=number_pages_processed_for_progress_update, \
        debugging_title=debugging_title, \
        debugging_title_alternatives=debugging_title_alternatives, \
        debugging_text_substring=debugging_text_substring, \
        json_configuration=json_configuration)
    # ------------------------------------------------------------------------

def main():
    """
    The main() function can quickly test WikipediaDumpXmlProcessor objects.
    """
    example_function_WikipediaDumpXmlProcessor()

if __name__ == '__main__':
    main()
    # ---- NOTE-PYLINT ---- C0301: Line too long
    # pylint: disable=C0301
    # ---- NOTE-FOR-REFERENCE ---- python app_wikipedia_dump_xml_processor.py --input_path E:\data_wikipedia\dumps.wikimedia.org_zhwiki_20211020 --file_name zhwiki-20211020-pages-articles.xml --output_path E:\data_wikipedia\dumps.wikimedia.org_zhwiki_20211020_output
