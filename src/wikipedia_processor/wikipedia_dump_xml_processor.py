# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
This module is for WikipediaDumpXmlProcessor.
"""
# ---- NOTE-PYLINT ---- C0302: Too many lines in module
# pylint: disable=C0302

from typing import Any
# from typing import Dict
from typing import Generic
from typing import List
from typing import NoReturn
from typing import Set
from typing import Tuple
from typing import TypeVar

from re import Match

import bz2
import codecs
import csv
import json
import os
import re
import time

import xml.etree.ElementTree as etree

from codecs \
    import StreamReaderWriter

from text_processor.text_piece \
    import TextPieceList
from text_processor.text_piece \
    import TextPieceUtility

from text_processor.text_chinese_converter \
    import TextChineseConverter

from utility.string_helper.string_helper \
    import StringHelper

from utility.debugging_helper.debugging_helper \
    import DebuggingHelper

TypeHeadingBaseMember = TypeVar("TypeHeadingBaseMember")
TypeExtractionBaseMember = TypeVar("TypeExtractionBaseMember")

class WikipediaDumpXmlProcessorConstants:
    """
    Some constants used in this module.
    """
    # ---- NOTE-PYLINT ---- R0903: Too few public methods
    # pylint: disable=R0903
    WIKI_LINES_JOIN_STRING: str = '\n'
    WIKI_SEGMENTS_JOIN_STRING: str = ''

class WikipediaDumpXmlProcessorHelperFunctions:
    """
    Some helper functions used in this module.
    """
    @staticmethod
    def seconds_to_string(seconds: int) -> str:
        """
        Convert seconds into a readable string form.
        """
        hours: int = int(seconds / 3600)
        minutes: int = int((seconds % 3600) / 60)
        seconds: int = seconds % 60
        return "{}:{:>02}:{:>05.4f}".format(hours, minutes, seconds)
    @staticmethod
    def strip_tag_name(tag: str) -> str:
        """
        Strip a tage from
        """
        index: int = tag.rfind("}")
        if index != -1:
            tag = tag[index + 1:]
        return tag

class MatchLike:
    """
    MatchLike
    """
    def __init__(self, span_offset: int, matched_string: str):
        """
        Initialize this object
        """
        self.span_offset = span_offset
        self.matched_string: int = matched_string
    def __repr__(self) -> str:
        """
        Return a representation of this object.
        """
        # ---- NOTE-PYLINT ---- W1302: Invalid format string (bad-format-string)
        # pylint: disable=W1302
        return "{'begin_position': {}, 'end_position': {}, 'text': {}".format( \
            self.get_begin_position(), \
            self.get_end_position(), \
            self.get_matched_string())
    def set_offset(self, span_offset: int) -> NoReturn:
        """
        Reset span offset.
        """
        self.span_offset = span_offset
    def set_matched_string(self, matched_string: str) -> NoReturn:
        """
        Reset matched_string.
        """
        self.matched_string = matched_string
    def get_offset(self) -> int:
        """
        Return a tuple of matched offset and end
        """
        return self.span_offset
    def get_length(self) -> int:
        """
        Return the length of the matched_string
        """
        return len(self.matched_string)
    def get_begin_position(self) -> int:
        """
        Return the begin position
        """
        return self.get_offset()
    def get_end_position(self) -> int:
        """
        Return the end position
        """
        return self.get_offset() + self.get_length()
    def span(self) -> Tuple[int, int]:
        """
        Return a tuple of matched offset and end
        """
        return (self.get_begin_position(), self.get_end_position())
    def group0(self) -> str:
        """
        Return a matched_string, which is like re.Match.group(0)
        """
        return self.get_matched_string()
    def get_matched_string(self) -> str:
        """
        Return a matched_string
        """
        return self.matched_string

class WikipediaDumpXmlProcessorRecordExtractionBase(Generic[TypeExtractionBaseMember]):
    """
    A record for storing a processed extraction_base (something that can be extracted) structure.
    """
    def __init__(self) -> NoReturn:
        """
        Initialize a WikipediaDumpXmlProcessorRecordExtractionBase object.
        """
        self.wiki_extraction_base_text: str = None
        self.wiki_extraction_base_text_processed: str = None
        self.wiki_extraction_base_lines: List[str] = []
        self.wiki_extraction_base_components: List[TypeExtractionBaseMember] = []
    def __repr__(self) -> str:
        """
        Return a representation of this object.
        """
        return self.to_json_in_string(json_configuration={})
    def to_json_in_string(self, \
        json_configuration: Any) -> str:
        """
        Convert this/self object to a JSON string.
        """
        return json.dumps( \
            ensure_ascii=False, \
            obj=self.to_json_friendly_structure( \
                json_configuration=json_configuration))
    def to_json_friendly_structure(self, \
        json_configuration: Any) -> Any:
        # ---- NOTE-PYLINT ---- R0201: Method could be a function
        # pylint: disable=R0201
        # ---- NOTE-PYLINT ---- W0613: Unused argument
        # pylint: disable=W0613
        """
        Convert this/self object to a JSON-friendly structure.
        """
        DebuggingHelper.ensure(
            False,
            'should be implemented by a child')
        # ---- NOTE-PLACE-HOLDER ---- pass
    def write_lines_to_file(self, \
        writer: StreamReaderWriter) -> NoReturn:
        """
        Output lines hold in this object to a file.
        """
        # pass # ---- by default, do nothing, as most child objects do not own lines.
    def set_wiki_extraction_base(self, \
        input_wiki_extraction_base_text: str, \
        input_wiki_extraction_base_text_processed: str, \
        input_wiki_extraction_base_lines: List[str], \
        input_wiki_extraction_base_components: List[TypeExtractionBaseMember]) -> NoReturn:
        """
        Set the WikipediaDumpXmlProcessorRecordHeading1 data structure.
        """
        self.wiki_extraction_base_text = input_wiki_extraction_base_text
        self.wiki_extraction_base_text_processed = input_wiki_extraction_base_text_processed
        self.wiki_extraction_base_lines = input_wiki_extraction_base_lines
        self.wiki_extraction_base_components = input_wiki_extraction_base_components
    def get_wiki_extraction_base_text(self) -> str:
        """
        Get the wiki_extraction_base_text data structure.
        """
        return self.wiki_extraction_base_text
    def get_wiki_extraction_base_text_processed(self) -> str:
        """
        Get the wiki_extraction_base_text_processed data structure.
        """
        return self.wiki_extraction_base_text_processed
    def get_wiki_extraction_base_lines(self) -> List[str]:
        """
        Get the wiki_extraction_base_lines data structure.
        """
        return self.wiki_extraction_base_lines
    def get_wiki_extraction_base_components(self) -> List[TypeExtractionBaseMember]:
        """
        Get the wiki_extraction_base_components data structure.
        """
        return self.wiki_extraction_base_components

class WikipediaDumpXmlProcessorRecordTemplateExtraction( \
    WikipediaDumpXmlProcessorRecordExtractionBase[MatchLike]):
    # ---- NOTE-PYLINT ---- C0301: Line too long
    # pylint: disable=C0301
    # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> Match[str]
    """
    A record for storing a processed template_extraction structure.
    """
    WIKI_TEMPLATE_REGULAR_EXPRESSION_BEGINNING_STRING: str = '{'
    WIKI_TEMPLATE_REGULAR_EXPRESSION_ENDING_STRING: str = '}'
    WIKI_TEMPLATE_REGULAR_EXPRESSION_PATTERN = \
        re.compile("(\\{\\{|\\}\\})")
    # ---- NOTE-PYLINT ---- C0301: Line too long
    # pylint: disable=C0301
    def __init__(self) -> NoReturn:
        """
        Initialize a WikipediaDumpXmlProcessorRecordTemplateExtraction object.
        """
        super(WikipediaDumpXmlProcessorRecordTemplateExtraction, self).__init__()
        self.wiki_template_extraction_components_parts: List[Match[str]] = []
    def __repr__(self) -> str:
        return self.to_json_in_string(json_configuration={})
    def to_json_in_string(self, \
        json_configuration: Any) -> str:
        """
        Convert this/self object to a JSON string.
        """
        return json.dumps( \
            ensure_ascii=False, \
            obj=self.to_json_friendly_structure( \
                json_configuration=json_configuration))
    def to_json_friendly_structure(self, \
        json_configuration: Any) -> Any:
        """
        Convert this/self object to a JSON-friendly structure.
        """
        json_object: Any = {}
        json_friendly_structure_to_output_text: bool = \
            json_configuration['json_friendly_structure_to_output_text_RecordTemplateExtraction']
        json_friendly_structure_to_output_lines: bool = \
            json_configuration['json_friendly_structure_to_output_lines_RecordTemplateExtraction']
        if json_friendly_structure_to_output_text:
            json_object['template_extraction_text'] = \
                self.get_wiki_template_extraction_text()
        if json_friendly_structure_to_output_lines:
            json_object['template_extraction_lines'] = \
                self.get_wiki_template_extraction_lines()
        # ---- NOTE-FOR-DEBUGGING ---- json_object['template_extraction_components_parts'] = \
        # ---- NOTE-FOR-DEBUGGING ----     [{'begin_position': x.span()[0], 'end_position': x.span()[1], 'text': x.group(0)} \
        # ---- NOTE-FOR-DEBUGGING ----         for x in self.wiki_template_extraction_components_parts]
        json_object['template_extraction_components'] = \
            [{'begin_position': x.get_begin_position(), 'end_position': x.get_end_position(), 'text': x.group0()} \
                for x in self.get_wiki_template_extraction_components()]
        return json_object
    def set_wiki_template_extraction(self, \
        input_wiki_template_extraction_text: str, \
        input_wiki_template_extraction_lines: List[str], \
        input_wiki_template_extraction_components: List[MatchLike]) -> NoReturn:
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> List[Match[str]]
        """
        Set the WikipediaDumpXmlProcessorRecordTemplateExtraction data structure.
        """
        self.set_wiki_extraction_base( \
            input_wiki_extraction_base_text=input_wiki_template_extraction_text, \
            input_wiki_extraction_base_text_processed=None, \
            input_wiki_extraction_base_lines=input_wiki_template_extraction_lines, \
            input_wiki_extraction_base_components=input_wiki_template_extraction_components)
    def get_wiki_template_extraction_text(self) -> str:
        """
        Get the wiki_template_extraction_text data structure.
        """
        return self.get_wiki_extraction_base_text()
    def get_wiki_template_extraction_lines(self) -> List[str]:
        """
        Get the wiki_template_extraction_lines data structure.
        """
        return self.get_wiki_extraction_base_lines()
    def get_wiki_template_extraction_components(self) -> List[MatchLike]:
        # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> List[Match[str]]:
        """
        Get the wiki_template_extraction_components data structure.
        """
        return self.get_wiki_extraction_base_components()
    def process_template_extraction(self) -> NoReturn:
        """
        Process this WikipediaDumpXmlProcessorRecordTemplateExtraction object,
        """
        # ---- NOTE-PYLINT ---- R0912: Too many branches
        # pylint: disable=R0912
        # ---- NOTE-PYLINT ---- R0914: Too many local variables
        # pylint: disable=R0914
        wiki_template_extraction_text: str = \
            self.get_wiki_template_extraction_text()
        if wiki_template_extraction_text is None:
            wiki_template_extraction_lines: List[str] = \
                self.get_wiki_template_extraction_lines()
            if wiki_template_extraction_lines is not None:
                wiki_template_extraction_text = \
                    WikipediaDumpXmlProcessorConstants.WIKI_LINES_JOIN_STRING.join( \
                        wiki_template_extraction_lines)
        if wiki_template_extraction_text is None:
            return
        wiki_template_extraction_components: List[MatchLike] = \
            self.get_wiki_template_extraction_components()
        self.wiki_template_extraction_components_parts = \
            []
        current_position: int = 0
        while True:
            matched: Match[str] = \
                WikipediaDumpXmlProcessorRecordTemplateExtraction.WIKI_TEMPLATE_REGULAR_EXPRESSION_PATTERN.\
                    search( \
                        string=wiki_template_extraction_text, \
                        pos=current_position)
            if matched is None:
                break
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordTemplateExtraction.process_template_extraction(), ' +
            #     'matched={}'.format(matched))
            matched_span: Tuple[int, int] = matched.span()
            # matched_string: str = matched.string
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordTemplateExtraction.process_template_extraction(), ' +
            #     'matched_string={}'.format(matched_string))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordTemplateExtraction.process_template_extraction(), ' +
            #     'matched_span.__repr__()={}'.format(matched_span.__repr__()))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordTemplateExtraction.process_template_extraction(), ' +
            #     'len(matched_span)={}'.format(len(matched_span)))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordTemplateExtraction.process_template_extraction(), ' +
            #     'matched_span[0]={}'.format(matched_span[0]))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordTemplateExtraction.process_template_extraction(), ' +
            #     'matched_span[1]={}'.format(matched_span[1]))
            # matched_group_string: str = matched.group(0)
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordTemplateExtraction.process_template_extraction(), ' +
            #     'matched_group_string={}'.format(matched_group_string))
            self.wiki_template_extraction_components_parts.append(matched)
            current_position = matched_span[1]
        wiki_template_extraction_segments: List[str] = []
        current_recursion_level: int = 0
        current_segment_offset: int = 0
        current_matched_segment_offset: int = -1
        current_matched_segment_end: int = -1
        for matched in self.wiki_template_extraction_components_parts:
            matched_group_string: str = matched.group(0)
            matched_span: Tuple[int, int] = matched.span()
            if matched_group_string[0] == \
                WikipediaDumpXmlProcessorRecordTemplateExtraction.WIKI_TEMPLATE_REGULAR_EXPRESSION_BEGINNING_STRING:
                if current_recursion_level == 0:
                    # DebuggingHelper.write_line_to_system_console_out_debug( \
                    #     '---- WikipediaDumpXmlProcessorRecordTemplateExtraction.process_template_extraction(), ' +
                    #     'matched={},current_recursion_level={}'.format(matched, current_recursion_level))
                    # DebuggingHelper.write_line_to_system_console_out_debug( \
                    #     '---- WikipediaDumpXmlProcessorRecordTemplateExtraction.process_template_extraction(), ' +
                    #     'matched_span[0]={}'.format(matched_span[0]))
                    # DebuggingHelper.write_line_to_system_console_out_debug( \
                    #     '---- WikipediaDumpXmlProcessorRecordTemplateExtraction.process_template_extraction(), ' +
                    #     'matched_span[1]={}'.format(matched_span[1]))
                    # DebuggingHelper.write_line_to_system_console_out_debug( \
                    #     '---- WikipediaDumpXmlProcessorRecordTemplateExtraction.process_template_extraction(), ' +
                    #     'type(matched_span[0])={}'.format(type(matched_span[0])))
                    # DebuggingHelper.write_line_to_system_console_out_debug( \
                    #     '---- WikipediaDumpXmlProcessorRecordTemplateExtraction.process_template_extraction(), ' +
                    #     'type(matched_span[1])={}'.format(type(matched_span[1])))
                    # DebuggingHelper.write_line_to_system_console_out_debug( \
                    #     '---- WikipediaDumpXmlProcessorRecordTemplateExtraction.process_template_extraction(), ' +
                    #     'len(wiki_template_extraction_text)={}'.format(len(wiki_template_extraction_text)))
                    current_segment_end: int = matched_span[0]
                    wiki_template_extraction_segments.append( \
                        wiki_template_extraction_text[current_segment_offset:current_segment_end])
                    # ---- NOTE-PYLINT ---- R1716: Simplify chained comparison between the operands (chained-comparison)
                    # pylint: disable=R1716
                    if (current_matched_segment_offset >= 0) and (current_matched_segment_end > current_matched_segment_offset):
                        match_like: MatchLike = MatchLike( \
                            current_matched_segment_offset, \
                            wiki_template_extraction_text[current_matched_segment_offset:current_matched_segment_end])
                        wiki_template_extraction_components.append(match_like)
                        # DebuggingHelper.write_line_to_system_console_out_debug( \
                        #     '---- WikipediaDumpXmlProcessorRecordTemplateExtraction.process_template_extraction(), ' +
                        #     'match_like={}'.format(match_like))
                    current_matched_segment_offset = current_segment_end
                current_recursion_level += 1
            elif matched_group_string[-1] == \
                WikipediaDumpXmlProcessorRecordTemplateExtraction.WIKI_TEMPLATE_REGULAR_EXPRESSION_ENDING_STRING:
                # DebuggingHelper.write_line_to_system_console_out_debug( \
                #     '---- WikipediaDumpXmlProcessorRecordTemplateExtraction.process_template_extraction(), ' +
                #     'matched={},current_recursion_level={}'.format(matched, current_recursion_level))
                current_recursion_level -= 1
                if current_recursion_level < 0:
                    current_recursion_level = 0
                if current_recursion_level == 0:
                    current_segment_offset = matched_span[1]
                    current_matched_segment_end = current_segment_offset
        if current_recursion_level == 0:
            if current_segment_offset < len(wiki_template_extraction_text):
                wiki_template_extraction_segments.append( \
                    wiki_template_extraction_text[current_segment_offset:])
        else:
            current_matched_segment_end = len(wiki_template_extraction_text)
        # ---- NOTE-PYLINT ---- R1716: Simplify chained comparison between the operands (chained-comparison)
        # pylint: disable=R1716
        if (current_matched_segment_offset >= 0) and (current_matched_segment_end > current_matched_segment_offset):
            match_like: MatchLike = MatchLike( \
                current_matched_segment_offset, \
                wiki_template_extraction_text[current_matched_segment_offset:current_matched_segment_end])
            wiki_template_extraction_components.append(match_like)
        wiki_template_extraction_segments_joined: str = \
            WikipediaDumpXmlProcessorConstants.WIKI_SEGMENTS_JOIN_STRING.join( \
                wiki_template_extraction_segments)
        wiki_template_extraction_lines: List[str] = \
            wiki_template_extraction_segments_joined.splitlines()
        self.set_wiki_template_extraction( \
            input_wiki_template_extraction_text=wiki_template_extraction_text, \
            input_wiki_template_extraction_lines=wiki_template_extraction_lines, \
            input_wiki_template_extraction_components=wiki_template_extraction_components)
class WikipediaDumpXmlProcessorRecordTemplateExtractionFactory:
    """
    A factor for processing WikipediaDumpXmlProcessorRecordTemplateExtraction
    objects.
    """
    # ---- NOTE-PYLINT ---- R0903: Too few public methods
    # pylint: disable=R0903
    @staticmethod
    def process_wiki_template_extractions(input_wiki_text_lines: List[str]) -> \
        WikipediaDumpXmlProcessorRecordTemplateExtraction:
        """
        Iterate through a wiki's text lines and group the lines into groups,
        each a template_extraction.
        """
        output_wiki_template_extraction: \
            WikipediaDumpXmlProcessorRecordTemplateExtraction = \
                WikipediaDumpXmlProcessorRecordTemplateExtraction()
        output_wiki_template_extraction.\
            set_wiki_template_extraction(\
                input_wiki_template_extraction_text=None, \
                input_wiki_template_extraction_lines=input_wiki_text_lines, \
                input_wiki_template_extraction_components=[])
        output_wiki_template_extraction.\
            process_template_extraction()
        return output_wiki_template_extraction

class WikipediaDumpXmlProcessorRecordTemplateArgumentExtraction( \
    WikipediaDumpXmlProcessorRecordExtractionBase[Any]):
    # ---- NOTE-PYLINT ---- C0301: Line too long
    # pylint: disable=C0301
    # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> Match[str]
    """
    A record for storing a processed template_argument_extraction structure.
    """
    WIKI_TEMPLATE_ARGUMENT_REGULAR_EXPRESSION_PATTERN = \
        re.compile("\\{\\{\\{([^{]*?)\\}\\}\\}", \
        re.MULTILINE|re.DOTALL)
    # ---- NOTE-PYLINT ---- C0301: Line too long
    # pylint: disable=C0301
    # ---- NOTE-NO-NEED ---- def __init__(self) -> NoReturn:
    # ---- NOTE-NO-NEED ----     """
    # ---- NOTE-NO-NEED ----     Initialize a WikipediaDumpXmlProcessorRecordTemplateArgumentExtraction object.
    # ---- NOTE-NO-NEED ----     """
    # ---- NOTE-NO-NEED ----     super(WikipediaDumpXmlProcessorRecordTemplateArgumentExtraction, self).__init__()
    def __repr__(self) -> str:
        """
        Return a representation of this object.
        """
        return self.to_json_in_string(json_configuration={})
    def to_json_in_string(self, \
        json_configuration: Any) -> str:
        """
        Convert this/self object to a JSON string.
        """
        return json.dumps( \
            ensure_ascii=False, \
            obj=self.to_json_friendly_structure( \
                json_configuration=json_configuration))
    def to_json_friendly_structure(self, \
        json_configuration: Any) -> Any:
        """
        Convert this/self object to a JSON-friendly structure.
        """
        json_object: Any = {}
        json_friendly_structure_to_output_text: bool = \
            json_configuration['json_friendly_structure_to_output_text_RecordTemplateArgumentExtraction']
        json_friendly_structure_to_output_lines: bool = \
            json_configuration['json_friendly_structure_to_output_lines_RecordTemplateArgumentExtraction']
        if json_friendly_structure_to_output_text:
            json_object['template_argument_extraction_text'] = \
                self.get_wiki_template_argument_extraction_text()
        if json_friendly_structure_to_output_lines:
            json_object['template_argument_extraction_lines'] = \
                self.get_wiki_template_argument_extraction_lines()
        json_object['template_argument_extraction_components'] = \
            [{'begin_position': x.span()[0], 'end_position': x.span()[1], 'text': x.group(0)} \
                for x in self.get_wiki_template_argument_extraction_components()]
        return json_object
    def set_wiki_template_argument_extraction(self, \
        input_wiki_template_argument_extraction_text: str, \
        input_wiki_template_argument_extraction_lines: List[str], \
        input_wiki_template_argument_extraction_components: List[Any]) -> NoReturn:
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> List[Match[str]]
        """
        Set the WikipediaDumpXmlProcessorRecordTemplateArgumentExtraction data structure.
        """
        self.set_wiki_extraction_base( \
            input_wiki_extraction_base_text=input_wiki_template_argument_extraction_text, \
            input_wiki_extraction_base_text_processed=None, \
            input_wiki_extraction_base_lines=input_wiki_template_argument_extraction_lines, \
            input_wiki_extraction_base_components=input_wiki_template_argument_extraction_components)
    def get_wiki_template_argument_extraction_text(self) -> str:
        """
        Get the wiki_template_argument_extraction_text data structure.
        """
        return self.get_wiki_extraction_base_text()
    def get_wiki_template_argument_extraction_lines(self) -> List[str]:
        """
        Get the wiki_template_argument_extraction_lines data structure.
        """
        return self.get_wiki_extraction_base_lines()
    def get_wiki_template_argument_extraction_components(self) -> List[Any]:
        # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> List[Match[str]]:
        """
        Get the wiki_template_argument_extraction_components data structure.
        """
        return self.get_wiki_extraction_base_components()
    def process_template_argument_extraction(self) -> NoReturn:
        """
        Process this WikipediaDumpXmlProcessorRecordTemplateArgumentExtraction object,
        """
        wiki_template_argument_extraction_text: str = \
            self.get_wiki_template_argument_extraction_text()
        if wiki_template_argument_extraction_text is None:
            wiki_template_argument_extraction_lines: List[str] = \
                self.get_wiki_template_argument_extraction_lines()
            if wiki_template_argument_extraction_lines is not None:
                wiki_template_argument_extraction_text = \
                    WikipediaDumpXmlProcessorConstants.WIKI_LINES_JOIN_STRING.join( \
                        wiki_template_argument_extraction_lines)
        if wiki_template_argument_extraction_text is None:
            return
        wiki_template_argument_extraction_components: List[Match[str]] = \
            self.get_wiki_template_argument_extraction_components()
        current_position: int = 0
        while True:
            matched: Match[str] = \
                WikipediaDumpXmlProcessorRecordTemplateArgumentExtraction.WIKI_TEMPLATE_ARGUMENT_REGULAR_EXPRESSION_PATTERN.\
                    search( \
                        string=wiki_template_argument_extraction_text, \
                        pos=current_position)
            if matched is None:
                break
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordTemplateArgumentExtraction.process_template_argument_extraction(), ' +
            #     'matched={}'.format(matched))
            matched_span: Tuple[int, int] = matched.span()
            # matched_string: str = matched.string
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordTemplateArgumentExtraction.process_template_argument_extraction(), ' +
            #     'matched_string={}'.format(matched_string))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordTemplateArgumentExtraction.process_template_argument_extraction(), ' +
            #     'matched_span.__repr__()={}'.format(matched_span.__repr__()))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordTemplateArgumentExtraction.process_template_argument_extraction(), ' +
            #     'len(matched_span)={}'.format(len(matched_span)))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordTemplateArgumentExtraction.process_template_argument_extraction(), ' +
            #     'matched_span[0]={}'.format(matched_span[0]))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordTemplateArgumentExtraction.process_template_argument_extraction(), ' +
            #     'matched_span[1]={}'.format(matched_span[1]))
            # matched_group_string: str = matched.group(0)
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordTemplateArgumentExtraction.process_template_argument_extraction(), ' +
            #     'matched_group_string={}'.format(matched_group_string))
            wiki_template_argument_extraction_components.append(matched)
            current_position = matched_span[1]
        wiki_template_argument_extraction_segments: List[str] = []
        current_segment_offset: int = 0
        for matched in wiki_template_argument_extraction_components:
            matched_span: Tuple[int, int] = matched.span()
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordTemplateArgumentExtraction.process_template_argument_extraction(), ' +
            #     'matched_span[0]={}'.format(matched_span[0]))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordTemplateArgumentExtraction.process_template_argument_extraction(), ' +
            #     'matched_span[1]={}'.format(matched_span[1]))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordTemplateArgumentExtraction.process_template_argument_extraction(), ' +
            #     'type(matched_span[0])={}'.format(type(matched_span[0])))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordTemplateArgumentExtraction.process_template_argument_extraction(), ' +
            #     'type(matched_span[1])={}'.format(type(matched_span[1])))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordTemplateArgumentExtraction.process_template_argument_extraction(), ' +
            #     'len(wiki_template_argument_extraction_text)={}'.format(len(wiki_template_argument_extraction_text)))
            current_segment_end: int = matched_span[0]
            wiki_template_argument_extraction_segments.append( \
                wiki_template_argument_extraction_text[current_segment_offset:current_segment_end])
            current_segment_offset = matched_span[1]
        if current_segment_offset < len(wiki_template_argument_extraction_text):
            wiki_template_argument_extraction_segments.append( \
                wiki_template_argument_extraction_text[current_segment_offset:])
        wiki_template_argument_extraction_segments_joined: str = \
            WikipediaDumpXmlProcessorConstants.WIKI_SEGMENTS_JOIN_STRING.join( \
                wiki_template_argument_extraction_segments)
        wiki_template_argument_extraction_lines: List[str] = \
            wiki_template_argument_extraction_segments_joined.splitlines()
        self.set_wiki_template_argument_extraction( \
            input_wiki_template_argument_extraction_text=wiki_template_argument_extraction_text, \
            input_wiki_template_argument_extraction_lines=wiki_template_argument_extraction_lines, \
            input_wiki_template_argument_extraction_components=wiki_template_argument_extraction_components)
class WikipediaDumpXmlProcessorRecordTemplateArgumentExtractionFactory:
    """
    A factor for processing WikipediaDumpXmlProcessorRecordTemplateArgumentExtraction
    objects.
    """
    # ---- NOTE-PYLINT ---- R0903: Too few public methods
    # pylint: disable=R0903
    @staticmethod
    def process_wiki_template_argument_extractions(input_wiki_text_lines: List[str]) -> \
        WikipediaDumpXmlProcessorRecordTemplateArgumentExtraction:
        """
        Iterate through a wiki's text lines and group the lines into groups,
        each a template_argument_extraction.
        """
        output_wiki_template_argument_extraction: \
            WikipediaDumpXmlProcessorRecordTemplateArgumentExtraction = \
                WikipediaDumpXmlProcessorRecordTemplateArgumentExtraction()
        output_wiki_template_argument_extraction.\
            set_wiki_template_argument_extraction(\
                input_wiki_template_argument_extraction_text=None, \
                input_wiki_template_argument_extraction_lines=input_wiki_text_lines, \
                input_wiki_template_argument_extraction_components=[])
        output_wiki_template_argument_extraction.\
            process_template_argument_extraction()
        return output_wiki_template_argument_extraction

class WikipediaDumpXmlProcessorRecordTextPieceExtraction( \
    WikipediaDumpXmlProcessorRecordExtractionBase[Any]):
    # ---- NOTE-PYLINT ---- C0301: Line too long
    # pylint: disable=C0301
    # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> Match[str]
    """
    A record for storing a processed text_piece_extraction structure.
    """
    # ---- NOTE-PYLINT ---- C0301: Line too long
    # pylint: disable=C0301
    # ---- NOTE-NO-NEED ---- def __init__(self) -> NoReturn:
    # ---- NOTE-NO-NEED ----     """
    # ---- NOTE-NO-NEED ----     Initialize a WikipediaDumpXmlProcessorRecordTextPieceExtraction object.
    # ---- NOTE-NO-NEED ----     """
    # ---- NOTE-NO-NEED ----     super(WikipediaDumpXmlProcessorRecordTextPieceExtraction, self).__init__()
    def __repr__(self) -> str:
        return self.to_json_in_string(json_configuration={})
    def to_json_in_string(self, \
        json_configuration: Any) -> str:
        """
        Convert this/self object to a JSON string.
        """
        return json.dumps( \
            ensure_ascii=False, \
            obj=self.to_json_friendly_structure( \
                json_configuration=json_configuration))
    def to_json_friendly_structure(self, \
        json_configuration: Any) -> Any:
        """
        Convert this/self object to a JSON-friendly structure.
        """
        json_object: Any = {}
        json_friendly_structure_to_output_text: bool = \
            json_configuration['json_friendly_structure_to_output_text_RecordTextPieceExtraction']
        json_friendly_structure_to_output_lines: bool = \
            json_configuration['json_friendly_structure_to_output_lines_RecordTextPieceExtraction']
        if json_friendly_structure_to_output_text:
            json_object['text_piece_extraction_text'] = \
                self.get_wiki_text_piece_extraction_text()
        if json_friendly_structure_to_output_text:
            _wiki_text_piece_extraction_text_processed: str = \
                self.get_wiki_text_piece_extraction_text_processed()
            if _wiki_text_piece_extraction_text_processed is not None:
                json_object['text_piece_extraction_text_processed'] = \
                    _wiki_text_piece_extraction_text_processed
        if json_friendly_structure_to_output_lines:
            json_object['text_piece_extraction_lines'] = \
                self.get_wiki_text_piece_extraction_lines()
        json_object['text_piece_extraction_components'] = \
            [{'begin_position': x.get_begin_position(), 'end_position': x.get_end_position(), 'text': x.group0()} \
                for x in self.get_wiki_text_piece_extraction_components()]
        return json_object
    def write_lines_to_file(self, \
        writer: StreamReaderWriter) -> NoReturn:
        """
        Output lines hold in this object to a file.
        """
        for line in self.get_wiki_text_piece_extraction_lines():
            if not StringHelper.is_none_empty_or_whitespaces(line):
                writer.writelines([line, '\n'])
    def set_wiki_text_piece_extraction(self, \
        input_wiki_text_piece_extraction_text: str, \
        input_wiki_text_piece_extraction_text_processed: str, \
        input_wiki_text_piece_extraction_lines: List[str], \
        input_wiki_text_piece_extraction_components: List[Any]) -> NoReturn:
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> List[Match[str]]
        """
        Set the WikipediaDumpXmlProcessorRecordTextPieceExtraction data structure.
        """
        self.set_wiki_extraction_base( \
            input_wiki_extraction_base_text=input_wiki_text_piece_extraction_text, \
            input_wiki_extraction_base_text_processed=input_wiki_text_piece_extraction_text_processed, \
            input_wiki_extraction_base_lines=input_wiki_text_piece_extraction_lines, \
            input_wiki_extraction_base_components=input_wiki_text_piece_extraction_components)
    def get_wiki_text_piece_extraction_text(self) -> str:
        """
        Get the wiki_text_piece_extraction_text data structure.
        """
        return self.get_wiki_extraction_base_text()
    def get_wiki_text_piece_extraction_text_processed(self) -> str:
        """
        Get the wiki_text_piece_extraction_text_processed data structure.
        """
        return self.get_wiki_extraction_base_text_processed()
    def get_wiki_text_piece_extraction_lines(self) -> List[str]:
        """
        Get the wiki_text_piece_extraction_lines data structure.
        """
        return self.get_wiki_extraction_base_lines()
    def get_wiki_text_piece_extraction_components(self) -> List[Any]:
        # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> List[Match[str]]:
        """
        Get the wiki_text_piece_extraction_components data structure.
        """
        return self.get_wiki_extraction_base_components()
    def process_text_piece_extraction(self) -> NoReturn:
        """
        Process this WikipediaDumpXmlProcessorRecordTextPieceExtraction object,
        """
        wiki_text_piece_extraction_text: str = \
            self.get_wiki_text_piece_extraction_text()
        if wiki_text_piece_extraction_text is None:
            wiki_text_piece_extraction_lines: List[str] = \
                self.get_wiki_text_piece_extraction_lines()
            if wiki_text_piece_extraction_lines is not None:
                wiki_text_piece_extraction_text = \
                    WikipediaDumpXmlProcessorConstants.WIKI_LINES_JOIN_STRING.join( \
                        wiki_text_piece_extraction_lines)
        if wiki_text_piece_extraction_text is None:
            return
        wiki_text_piece_extraction_components: List[MatchLike] = \
            self.get_wiki_text_piece_extraction_components()
        text_piece_lists: List[TextPieceList] = \
            TextPieceUtility.process_to_text_piece_lists(input_text=wiki_text_piece_extraction_text)
        wiki_text_piece_extraction_lines_processed: List[str] = []
        current_end_position: int = 0
        for text_piece in text_piece_lists[-3].get_text_piece_list(): # ---- NOTE ---- there are several extracted TextPiece lists, but here it only uses the Wiki-Link ones
            text_piece_string: str = \
                text_piece.get_text_piece_string()
            if StringHelper.is_none_empty_or_whitespaces(input_value=text_piece_string):
                continue
            text_piece_string_converted: str = \
                TextChineseConverter.convert_simplified_to_traditional(text_piece_string)
            wiki_text_piece_extraction_lines_processed.append(text_piece_string_converted)
            wiki_text_piece_extraction_component: MatchLike = \
                MatchLike(current_end_position, text_piece_string_converted)
            if not text_piece.is_raw_text():
                wiki_text_piece_extraction_components.append(wiki_text_piece_extraction_component)
            current_end_position = wiki_text_piece_extraction_component.get_end_position()
        wiki_text_piece_extraction_lines_processed_joined: str = \
            WikipediaDumpXmlProcessorConstants.WIKI_SEGMENTS_JOIN_STRING.join( \
                wiki_text_piece_extraction_lines_processed)
        wiki_text_piece_extraction_lines: List[str] = \
            wiki_text_piece_extraction_lines_processed_joined.splitlines()
        self.set_wiki_text_piece_extraction( \
            input_wiki_text_piece_extraction_text=wiki_text_piece_extraction_text, \
            input_wiki_text_piece_extraction_text_processed=wiki_text_piece_extraction_lines_processed_joined, \
            input_wiki_text_piece_extraction_lines=wiki_text_piece_extraction_lines, \
            input_wiki_text_piece_extraction_components=wiki_text_piece_extraction_components)
class WikipediaDumpXmlProcessorRecordTextPieceExtractionFactory:
    """
    A factor for processing WikipediaDumpXmlProcessorRecordTextPieceExtraction
    objects.
    """
    # ---- NOTE-PYLINT ---- R0903: Too few public methods
    # pylint: disable=R0903
    @staticmethod
    def process_wiki_text_piece_extractions(input_wiki_text_lines: List[str]) -> \
        WikipediaDumpXmlProcessorRecordTextPieceExtraction:
        """
        Iterate through a wiki's text lines and group the lines into groups,
        each a text_piece_extraction.
        """
        output_wiki_text_piece_extraction: \
            WikipediaDumpXmlProcessorRecordTextPieceExtraction = \
                WikipediaDumpXmlProcessorRecordTextPieceExtraction()
        output_wiki_text_piece_extraction.\
            set_wiki_text_piece_extraction(\
                input_wiki_text_piece_extraction_text=None, \
                input_wiki_text_piece_extraction_text_processed=None, \
                input_wiki_text_piece_extraction_lines=input_wiki_text_lines, \
                input_wiki_text_piece_extraction_components=[])
        output_wiki_text_piece_extraction.\
            process_text_piece_extraction()
        return output_wiki_text_piece_extraction

class WikipediaDumpXmlProcessorRecordHtmlGenericIndividualTagExtraction( \
    WikipediaDumpXmlProcessorRecordExtractionBase[Any]):
    # ---- NOTE-PYLINT ---- C0301: Line too long
    # pylint: disable=C0301
    # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> Match[str]
    """
    A record for storing a processed html_generic_individual_tag_extraction structure.
    """
    WIKI_HTML_GENERIC_INDIVIDUAL_TAG_REGULAR_EXPRESSION_PATTERN = \
        re.compile("<(.*?)>", \
        re.MULTILINE|re.DOTALL)
    # ---- NOTE-PYLINT ---- C0301: Line too long
    # pylint: disable=C0301
    # ---- NOTE-NO-NEED ---- def __init__(self) -> NoReturn:
    # ---- NOTE-NO-NEED ----     """
    # ---- NOTE-NO-NEED ----     Initialize a WikipediaDumpXmlProcessorRecordHtmlGenericIndividualTagExtraction object.
    # ---- NOTE-NO-NEED ----     """
    # ---- NOTE-NO-NEED ----     super(WikipediaDumpXmlProcessorRecordHtmlGenericIndividualTagExtraction, self).__init__()
    def __repr__(self) -> str:
        return self.to_json_in_string(json_configuration={})
    def to_json_in_string(self, \
        json_configuration: Any) -> str:
        """
        Convert this/self object to a JSON string.
        """
        return json.dumps( \
            ensure_ascii=False, \
            obj=self.to_json_friendly_structure( \
                json_configuration=json_configuration))
    def to_json_friendly_structure(self, \
        json_configuration: Any) -> Any:
        """
        Convert this/self object to a JSON-friendly structure.
        """
        json_object: Any = {}
        json_friendly_structure_to_output_text: bool = \
            json_configuration['json_friendly_structure_to_output_text_RecordHtmlGenericIndividualTagExtraction']
        json_friendly_structure_to_output_lines: bool = \
            json_configuration['json_friendly_structure_to_output_lines_RecordHtmlGenericIndividualTagExtraction']
        if json_friendly_structure_to_output_text:
            json_object['html_generic_individual_tag_extraction_text'] = \
                self.get_wiki_html_generic_individual_tag_extraction_text()
        if json_friendly_structure_to_output_lines:
            json_object['html_generic_individual_tag_extraction_lines'] = \
                self.get_wiki_html_generic_individual_tag_extraction_lines()
        json_object['html_generic_individual_tag_extraction_components'] = \
            [{'begin_position': x.span()[0], 'end_position': x.span()[1], 'text': x.group(0)} \
                for x in self.get_wiki_html_generic_individual_tag_extraction_components()]
        return json_object
    def set_wiki_html_generic_individual_tag_extraction(self, \
        input_wiki_html_generic_individual_tag_extraction_text: str, \
        input_wiki_html_generic_individual_tag_extraction_lines: List[str], \
        input_wiki_html_generic_individual_tag_extraction_components: List[Any]) -> NoReturn:
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> List[Match[str]]
        """
        Set the WikipediaDumpXmlProcessorRecordHtmlGenericIndividualTagExtraction data structure.
        """
        self.set_wiki_extraction_base( \
            input_wiki_extraction_base_text=input_wiki_html_generic_individual_tag_extraction_text, \
            input_wiki_extraction_base_text_processed=None, \
            input_wiki_extraction_base_lines=input_wiki_html_generic_individual_tag_extraction_lines, \
            input_wiki_extraction_base_components=input_wiki_html_generic_individual_tag_extraction_components)
    def get_wiki_html_generic_individual_tag_extraction_text(self) -> str:
        """
        Get the wiki_html_generic_individual_tag_extraction_text data structure.
        """
        return self.get_wiki_extraction_base_text()
    def get_wiki_html_generic_individual_tag_extraction_lines(self) -> List[str]:
        """
        Get the wiki_html_generic_individual_tag_extraction_lines data structure.
        """
        return self.get_wiki_extraction_base_lines()
    def get_wiki_html_generic_individual_tag_extraction_components(self) -> List[Any]:
        # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> List[Match[str]]:
        """
        Get the wiki_html_generic_individual_tag_extraction_components data structure.
        """
        return self.get_wiki_extraction_base_components()
    def process_html_generic_individual_tag_extraction(self) -> NoReturn:
        """
        Process this WikipediaDumpXmlProcessorRecordHtmlGenericIndividualTagExtraction object,
        """
        wiki_html_generic_individual_tag_extraction_text: str = \
            self.get_wiki_html_generic_individual_tag_extraction_text()
        if wiki_html_generic_individual_tag_extraction_text is None:
            wiki_html_generic_individual_tag_extraction_lines: List[str] = \
                self.get_wiki_html_generic_individual_tag_extraction_lines()
            if wiki_html_generic_individual_tag_extraction_lines is not None:
                wiki_html_generic_individual_tag_extraction_text = \
                    WikipediaDumpXmlProcessorConstants.WIKI_LINES_JOIN_STRING.join( \
                        wiki_html_generic_individual_tag_extraction_lines)
        if wiki_html_generic_individual_tag_extraction_text is None:
            return
        wiki_html_generic_individual_tag_extraction_components: List[Match[str]] = \
            self.get_wiki_html_generic_individual_tag_extraction_components()
        current_position: int = 0
        while True:
            matched: Match[str] = \
                WikipediaDumpXmlProcessorRecordHtmlGenericIndividualTagExtraction.WIKI_HTML_GENERIC_INDIVIDUAL_TAG_REGULAR_EXPRESSION_PATTERN.\
                    search( \
                        string=wiki_html_generic_individual_tag_extraction_text, \
                        pos=current_position)
            if matched is None:
                break
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordHtmlGenericIndividualTagExtraction.process_html_generic_individual_tag_extraction(), ' +
            #     'matched={}'.format(matched))
            matched_span: Tuple[int, int] = matched.span()
            # matched_string: str = matched.string
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordHtmlGenericIndividualTagExtraction.process_html_generic_individual_tag_extraction(), ' +
            #     'matched_string={}'.format(matched_string))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordHtmlGenericIndividualTagExtraction.process_html_generic_individual_tag_extraction(), ' +
            #     'matched_span.__repr__()={}'.format(matched_span.__repr__()))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordHtmlGenericIndividualTagExtraction.process_html_generic_individual_tag_extraction(), ' +
            #     'len(matched_span)={}'.format(len(matched_span)))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordHtmlGenericIndividualTagExtraction.process_html_generic_individual_tag_extraction(), ' +
            #     'matched_span[0]={}'.format(matched_span[0]))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordHtmlGenericIndividualTagExtraction.process_html_generic_individual_tag_extraction(), ' +
            #     'matched_span[1]={}'.format(matched_span[1]))
            # matched_group_string: str = matched.group(0)
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordHtmlGenericIndividualTagExtraction.process_html_generic_individual_tag_extraction(), ' +
            #     'matched_group_string={}'.format(matched_group_string))
            wiki_html_generic_individual_tag_extraction_components.append(matched)
            current_position = matched_span[1]
        wiki_html_generic_individual_tag_extraction_segments: List[str] = []
        current_segment_offset: int = 0
        for matched in wiki_html_generic_individual_tag_extraction_components:
            matched_span: Tuple[int, int] = matched.span()
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordHtmlGenericIndividualTagExtraction.process_html_generic_individual_tag_extraction(), ' +
            #     'matched_span[0]={}'.format(matched_span[0]))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordHtmlGenericIndividualTagExtraction.process_html_generic_individual_tag_extraction(), ' +
            #     'matched_span[1]={}'.format(matched_span[1]))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordHtmlGenericIndividualTagExtraction.process_html_generic_individual_tag_extraction(), ' +
            #     'type(matched_span[0])={}'.format(type(matched_span[0])))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordHtmlGenericIndividualTagExtraction.process_html_generic_individual_tag_extraction(), ' +
            #     'type(matched_span[1])={}'.format(type(matched_span[1])))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordHtmlGenericIndividualTagExtraction.process_html_generic_individual_tag_extraction(), ' +
            #     'len(wiki_html_generic_individual_tag_extraction_text)={}'.format(len(wiki_html_generic_individual_tag_extraction_text)))
            current_segment_end: int = matched_span[0]
            wiki_html_generic_individual_tag_extraction_segments.append( \
                wiki_html_generic_individual_tag_extraction_text[current_segment_offset:current_segment_end])
            current_segment_offset = matched_span[1]
        if current_segment_offset < len(wiki_html_generic_individual_tag_extraction_text):
            wiki_html_generic_individual_tag_extraction_segments.append( \
                wiki_html_generic_individual_tag_extraction_text[current_segment_offset:])
        wiki_html_generic_individual_tag_extraction_segments_joined: str = \
            WikipediaDumpXmlProcessorConstants.WIKI_SEGMENTS_JOIN_STRING.join( \
                wiki_html_generic_individual_tag_extraction_segments)
        wiki_html_generic_individual_tag_extraction_lines: List[str] = \
            wiki_html_generic_individual_tag_extraction_segments_joined.splitlines()
        self.set_wiki_html_generic_individual_tag_extraction( \
            input_wiki_html_generic_individual_tag_extraction_text=wiki_html_generic_individual_tag_extraction_text, \
            input_wiki_html_generic_individual_tag_extraction_lines=wiki_html_generic_individual_tag_extraction_lines, \
            input_wiki_html_generic_individual_tag_extraction_components=wiki_html_generic_individual_tag_extraction_components)
class WikipediaDumpXmlProcessorRecordHtmlGenericIndividualTagExtractionFactory:
    """
    A factor for processing WikipediaDumpXmlProcessorRecordHtmlGenericIndividualTagExtraction
    objects.
    """
    # ---- NOTE-PYLINT ---- R0903: Too few public methods
    # pylint: disable=R0903
    @staticmethod
    def process_wiki_html_generic_individual_tag_extractions(input_wiki_text_lines: List[str]) -> \
        WikipediaDumpXmlProcessorRecordHtmlGenericIndividualTagExtraction:
        """
        Iterate through a wiki's text lines and group the lines into groups,
        each a html_generic_individual_tag_extraction.
        """
        output_wiki_html_generic_individual_tag_extraction: \
            WikipediaDumpXmlProcessorRecordHtmlGenericIndividualTagExtraction = \
                WikipediaDumpXmlProcessorRecordHtmlGenericIndividualTagExtraction()
        output_wiki_html_generic_individual_tag_extraction.\
            set_wiki_html_generic_individual_tag_extraction(\
                input_wiki_html_generic_individual_tag_extraction_text=None, \
                input_wiki_html_generic_individual_tag_extraction_lines=input_wiki_text_lines, \
                input_wiki_html_generic_individual_tag_extraction_components=[])
        output_wiki_html_generic_individual_tag_extraction.\
            process_html_generic_individual_tag_extraction()
        return output_wiki_html_generic_individual_tag_extraction

class WikipediaDumpXmlProcessorRecordHtmlReferenceExtraction( \
    WikipediaDumpXmlProcessorRecordExtractionBase[Any]):
    # ---- NOTE-PYLINT ---- C0301: Line too long
    # pylint: disable=C0301
    # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> Match[str]
    """
    A record for storing a processed html_reference_extraction structure.
    """
    WIKI_HTML_REFERENCE_REGULAR_EXPRESSION_PATTERN = \
        re.compile("(<ref(erence)?([^<]*)/>)|(<ref(erence)?(.*?)/ref(erence)?>)", \
            re.IGNORECASE|re.MULTILINE|re.DOTALL)
    # ---- NOTE-PYLINT ---- C0301: Line too long
    # pylint: disable=C0301
    # ---- NOTE-NO-NEED ---- def __init__(self) -> NoReturn:
    # ---- NOTE-NO-NEED ----     """
    # ---- NOTE-NO-NEED ----     Initialize a WikipediaDumpXmlProcessorRecordHtmlReferenceExtraction object.
    # ---- NOTE-NO-NEED ----     """
    # ---- NOTE-NO-NEED ----     super(WikipediaDumpXmlProcessorRecordHtmlReferenceExtraction, self).__init__()
    def __repr__(self) -> str:
        return self.to_json_in_string(json_configuration={})
    def to_json_in_string(self, \
        json_configuration: Any) -> str:
        """
        Convert this/self object to a JSON string.
        """
        return json.dumps( \
            ensure_ascii=False, \
            obj=self.to_json_friendly_structure( \
                json_configuration=json_configuration))
    def to_json_friendly_structure(self, \
        json_configuration: Any) -> Any:
        """
        Convert this/self object to a JSON-friendly structure.
        """
        json_object: Any = {}
        json_friendly_structure_to_output_text: bool = \
            json_configuration['json_friendly_structure_to_output_text_RecordHtmlReferenceExtraction']
        json_friendly_structure_to_output_lines: bool = \
            json_configuration['json_friendly_structure_to_output_lines_RecordHtmlReferenceExtraction']
        if json_friendly_structure_to_output_text:
            json_object['html_reference_extraction_text'] = \
                self.get_wiki_html_reference_extraction_text()
        if json_friendly_structure_to_output_lines:
            json_object['html_reference_extraction_lines'] = \
                self.get_wiki_html_reference_extraction_lines()
        json_object['html_reference_extraction_components'] = \
            [{'begin_position': x.span()[0], 'end_position': x.span()[1], 'text': x.group(0)} \
                for x in self.get_wiki_html_reference_extraction_components()]
        return json_object
    def set_wiki_html_reference_extraction(self, \
        input_wiki_html_reference_extraction_text: str, \
        input_wiki_html_reference_extraction_lines: List[str], \
        input_wiki_html_reference_extraction_components: List[Any]) -> NoReturn:
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> List[Match[str]]
        """
        Set the WikipediaDumpXmlProcessorRecordHtmlReferenceExtraction data structure.
        """
        self.set_wiki_extraction_base( \
            input_wiki_extraction_base_text=input_wiki_html_reference_extraction_text, \
            input_wiki_extraction_base_text_processed=None, \
            input_wiki_extraction_base_lines=input_wiki_html_reference_extraction_lines, \
            input_wiki_extraction_base_components=input_wiki_html_reference_extraction_components)
    def get_wiki_html_reference_extraction_text(self) -> str:
        """
        Get the wiki_html_reference_extraction_text data structure.
        """
        return self.get_wiki_extraction_base_text()
    def get_wiki_html_reference_extraction_lines(self) -> List[str]:
        """
        Get the wiki_html_reference_extraction_lines data structure.
        """
        return self.get_wiki_extraction_base_lines()
    def get_wiki_html_reference_extraction_components(self) -> List[Any]:
        # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> List[Match[str]]:
        """
        Get the wiki_html_reference_extraction_components data structure.
        """
        return self.get_wiki_extraction_base_components()
    def process_html_reference_extraction(self) -> NoReturn:
        """
        Process this WikipediaDumpXmlProcessorRecordHtmlReferenceExtraction object,
        """
        wiki_html_reference_extraction_text: str = \
            self.get_wiki_html_reference_extraction_text()
        if wiki_html_reference_extraction_text is None:
            wiki_html_reference_extraction_lines: List[str] = \
                self.get_wiki_html_reference_extraction_lines()
            if wiki_html_reference_extraction_lines is not None:
                wiki_html_reference_extraction_text = \
                    WikipediaDumpXmlProcessorConstants.WIKI_LINES_JOIN_STRING.join( \
                        wiki_html_reference_extraction_lines)
        if wiki_html_reference_extraction_text is None:
            return
        wiki_html_reference_extraction_components: List[Match[str]] = \
            self.get_wiki_html_reference_extraction_components()
        current_position: int = 0
        while True:
            matched: Match[str] = \
                WikipediaDumpXmlProcessorRecordHtmlReferenceExtraction.WIKI_HTML_REFERENCE_REGULAR_EXPRESSION_PATTERN.\
                    search( \
                        string=wiki_html_reference_extraction_text, \
                        pos=current_position)
            if matched is None:
                break
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordHtmlReferenceExtraction.process_html_reference_extraction(), ' +
            #     'matched={}'.format(matched))
            matched_span: Tuple[int, int] = matched.span()
            # matched_string: str = matched.string
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordHtmlReferenceExtraction.process_html_reference_extraction(), ' +
            #     'matched_string={}'.format(matched_string))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordHtmlReferenceExtraction.process_html_reference_extraction(), ' +
            #     'matched_span.__repr__()={}'.format(matched_span.__repr__()))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordHtmlReferenceExtraction.process_html_reference_extraction(), ' +
            #     'len(matched_span)={}'.format(len(matched_span)))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordHtmlReferenceExtraction.process_html_reference_extraction(), ' +
            #     'matched_span[0]={}'.format(matched_span[0]))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordHtmlReferenceExtraction.process_html_reference_extraction(), ' +
            #     'matched_span[1]={}'.format(matched_span[1]))
            # matched_group_string: str = matched.group(0)
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordHtmlReferenceExtraction.process_html_reference_extraction(), ' +
            #     'matched_group_string={}'.format(matched_group_string))
            wiki_html_reference_extraction_components.append(matched)
            current_position = matched_span[1]
        wiki_html_reference_extraction_segments: List[str] = []
        current_segment_offset: int = 0
        for matched in wiki_html_reference_extraction_components:
            matched_span: Tuple[int, int] = matched.span()
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordHtmlReferenceExtraction.process_html_reference_extraction(), ' +
            #     'matched_span[0]={}'.format(matched_span[0]))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordHtmlReferenceExtraction.process_html_reference_extraction(), ' +
            #     'matched_span[1]={}'.format(matched_span[1]))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordHtmlReferenceExtraction.process_html_reference_extraction(), ' +
            #     'type(matched_span[0])={}'.format(type(matched_span[0])))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordHtmlReferenceExtraction.process_html_reference_extraction(), ' +
            #     'type(matched_span[1])={}'.format(type(matched_span[1])))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordHtmlReferenceExtraction.process_html_reference_extraction(), ' +
            #     'len(wiki_html_reference_extraction_text)={}'.format(len(wiki_html_reference_extraction_text)))
            current_segment_end: int = matched_span[0]
            wiki_html_reference_extraction_segments.append( \
                wiki_html_reference_extraction_text[current_segment_offset:current_segment_end])
            current_segment_offset = matched_span[1]
        if current_segment_offset < len(wiki_html_reference_extraction_text):
            wiki_html_reference_extraction_segments.append( \
                wiki_html_reference_extraction_text[current_segment_offset:])
        wiki_html_reference_extraction_segments_joined: str = \
            WikipediaDumpXmlProcessorConstants.WIKI_SEGMENTS_JOIN_STRING.join( \
                wiki_html_reference_extraction_segments)
        wiki_html_reference_extraction_lines: List[str] = \
            wiki_html_reference_extraction_segments_joined.splitlines()
        self.set_wiki_html_reference_extraction( \
            input_wiki_html_reference_extraction_text=wiki_html_reference_extraction_text, \
            input_wiki_html_reference_extraction_lines=wiki_html_reference_extraction_lines, \
            input_wiki_html_reference_extraction_components=wiki_html_reference_extraction_components)
class WikipediaDumpXmlProcessorRecordHtmlReferenceExtractionFactory:
    """
    A factor for processing WikipediaDumpXmlProcessorRecordHtmlReferenceExtraction objects.
    """
    # ---- NOTE-PYLINT ---- R0903: Too few public methods
    # pylint: disable=R0903
    @staticmethod
    def process_wiki_html_reference_extractions(input_wiki_text_lines: List[str]) -> \
        WikipediaDumpXmlProcessorRecordHtmlReferenceExtraction:
        """
        Iterate through a wiki's text lines and group the lines into groups,
        each a html_reference_extraction.
        """
        output_wiki_html_reference_extraction: \
            WikipediaDumpXmlProcessorRecordHtmlReferenceExtraction = \
                WikipediaDumpXmlProcessorRecordHtmlReferenceExtraction()
        output_wiki_html_reference_extraction.set_wiki_html_reference_extraction(\
            input_wiki_html_reference_extraction_text=None, \
            input_wiki_html_reference_extraction_lines=input_wiki_text_lines, \
            input_wiki_html_reference_extraction_components=[])
        output_wiki_html_reference_extraction.process_html_reference_extraction()
        return output_wiki_html_reference_extraction

class WikipediaDumpXmlProcessorExtractionUtility:
    """
    WikipediaDumpXmlProcessorExtractionUtility is comprised of a collection of utility functions.
    """
    @staticmethod
    def process_from_text_lines(input_wiki_text_lines: List[str]) -> \
        List[WikipediaDumpXmlProcessorRecordExtractionBase[Any]]:
        """
        Iterate through a collection wiki's text lines and extract segments of
        information from them.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # ----
        # ---- NOTE-REFACTORED-TO-TextPiece-logic ---- wiki_template_argument_extraction: \
        # ---- NOTE-REFACTORED-TO-TextPiece-logic ----     WikipediaDumpXmlProcessorRecordTemplateArgumentExtraction = \
        # ---- NOTE-REFACTORED-TO-TextPiece-logic ----         WikipediaDumpXmlProcessorRecordTemplateArgumentExtractionFactory.\
        # ---- NOTE-REFACTORED-TO-TextPiece-logic ----             process_wiki_template_argument_extractions( \
        # ---- NOTE-REFACTORED-TO-TextPiece-logic ----                 input_wiki_text_lines=input_wiki_text_lines)
        # ----
        # ---- NOTE-REFACTORED-TO-TextPiece-logic ---- input_wiki_text_lines_from_template_argument: List[str] = \
        # ---- NOTE-REFACTORED-TO-TextPiece-logic ----     wiki_template_argument_extraction.get_wiki_extraction_base_lines()
        # ---- NOTE-REFACTORED-TO-TextPiece-logic ---- wiki_template_extraction: \
        # ---- NOTE-REFACTORED-TO-TextPiece-logic ----     WikipediaDumpXmlProcessorRecordTemplateExtraction = \
        # ---- NOTE-REFACTORED-TO-TextPiece-logic ----         WikipediaDumpXmlProcessorRecordTemplateExtractionFactory.\
        # ---- NOTE-REFACTORED-TO-TextPiece-logic ----             process_wiki_template_extractions( \
        # ---- NOTE-REFACTORED-TO-TextPiece-logic ----                 input_wiki_text_lines=input_wiki_text_lines_from_template_argument)
        # ----
        # ---- NOTE-REFACTORED-TO-TextPiece-logic ---- input_wiki_text_lines_from_template: List[str] = \
        # ---- NOTE-REFACTORED-TO-TextPiece-logic ----     wiki_template_extraction.get_wiki_extraction_base_lines()
        wiki_text_piece_extraction: \
            WikipediaDumpXmlProcessorRecordTextPieceExtraction = \
                WikipediaDumpXmlProcessorRecordTextPieceExtractionFactory.\
                    process_wiki_text_piece_extractions( \
                        input_wiki_text_lines=input_wiki_text_lines)
        # ----
        # ---- NOTE-REFACTORED-TO-TextPiece-logic ---- input_wiki_text_lines_from_template: List[str] = \
        # ---- NOTE-REFACTORED-TO-TextPiece-logic ----     wiki_template_extraction.get_wiki_extraction_base_lines()
        # ---- NOTE-REFACTORED-TO-TextPiece-logic ---- wiki_html_reference_extraction: \
        # ---- NOTE-REFACTORED-TO-TextPiece-logic ----     WikipediaDumpXmlProcessorRecordHtmlReferenceExtraction = \
        # ---- NOTE-REFACTORED-TO-TextPiece-logic ----         WikipediaDumpXmlProcessorRecordHtmlReferenceExtractionFactory.\
        # ---- NOTE-REFACTORED-TO-TextPiece-logic ----             process_wiki_html_reference_extractions( \
        # ---- NOTE-REFACTORED-TO-TextPiece-logic ----                 input_wiki_text_lines=input_wiki_text_lines_from_template)
        # ----
        # ---- NOTE-REFACTORED-TO-TextPiece-logic ---- input_wiki_text_lines_from_html_reference: List[str] = \
        # ---- NOTE-REFACTORED-TO-TextPiece-logic ----     wiki_html_reference_extraction.get_wiki_extraction_base_lines()
        # ---- NOTE-REFACTORED-TO-TextPiece-logic ---- wiki_html_generic_individual_tag_extraction: \
        # ---- NOTE-REFACTORED-TO-TextPiece-logic ----     WikipediaDumpXmlProcessorRecordHtmlGenericIndividualTagExtraction = \
        # ---- NOTE-REFACTORED-TO-TextPiece-logic ----         WikipediaDumpXmlProcessorRecordHtmlGenericIndividualTagExtractionFactory.\
        # ---- NOTE-REFACTORED-TO-TextPiece-logic ----             process_wiki_html_generic_individual_tag_extractions( \
        # ---- NOTE-REFACTORED-TO-TextPiece-logic ----                 input_wiki_text_lines=input_wiki_text_lines_from_html_reference)
        # ----
        return [ \
            # ---- NOTE-PYLINT ---- C0301: Line too long
            # pylint: disable=C0301
            # ---- NOTE-REFACTORED-TO-TextPiece-logic ---- wiki_template_argument_extraction, \
            # ---- NOTE-REFACTORED-TO-TextPiece-logic ---- wiki_template_extraction, \
            wiki_text_piece_extraction, \
            # ---- NOTE-REFACTORED-TO-TextPiece-logic ---- wiki_html_reference_extraction, \
            # ---- NOTE-PYLINT ---- C0301: Line too long
            # pylint: disable=C0301
            # ---- NOTE-REFACTORED-TO-TextPiece-logic ---- wiki_html_generic_individual_tag_extraction \
            ]
    @staticmethod
    def process_from_text_lines_prototype_a(input_wiki_text_lines: List[str]) -> \
        List[WikipediaDumpXmlProcessorRecordExtractionBase[Any]]:
        """
        Iterate through a collection wiki's text lines and extract segments of
        information from them.
        """
        # ----
        wiki_html_reference_extraction: \
            WikipediaDumpXmlProcessorRecordHtmlReferenceExtraction = \
                WikipediaDumpXmlProcessorRecordHtmlReferenceExtractionFactory.\
                    process_wiki_html_reference_extractions( \
                        input_wiki_text_lines=input_wiki_text_lines)
        # ----
        input_wiki_text_lines_from_html_reference: List[str] = \
            wiki_html_reference_extraction.get_wiki_extraction_base_lines()
        wiki_html_generic_individual_tag_extraction: \
            WikipediaDumpXmlProcessorRecordHtmlGenericIndividualTagExtraction = \
                WikipediaDumpXmlProcessorRecordHtmlGenericIndividualTagExtractionFactory.\
                    process_wiki_html_generic_individual_tag_extractions( \
                        input_wiki_text_lines=input_wiki_text_lines_from_html_reference)
        # ----
        input_wiki_text_lines_from_generic_individual_tag: List[str] = \
            wiki_html_generic_individual_tag_extraction.get_wiki_extraction_base_lines()
        wiki_template_argument_extraction: \
            WikipediaDumpXmlProcessorRecordTemplateArgumentExtraction = \
                WikipediaDumpXmlProcessorRecordTemplateArgumentExtractionFactory.\
                    process_wiki_template_argument_extractions( \
                        input_wiki_text_lines=input_wiki_text_lines_from_generic_individual_tag)
        # ----
        input_wiki_text_lines_from_template_argument: List[str] = \
            wiki_template_argument_extraction.get_wiki_extraction_base_lines()
        wiki_template_extraction: \
            WikipediaDumpXmlProcessorRecordTemplateExtraction = \
                WikipediaDumpXmlProcessorRecordTemplateExtractionFactory.\
                    process_wiki_template_extractions( \
                        input_wiki_text_lines=input_wiki_text_lines_from_template_argument)
        # ----
        return [ \
            wiki_html_reference_extraction, \
            wiki_html_generic_individual_tag_extraction, \
            wiki_template_argument_extraction, \
            wiki_template_extraction]

class WikipediaDumpXmlProcessorRecordHeadingBase(Generic[TypeHeadingBaseMember]):
    """
    A record for storing a processed heading_base structure.
    """
    def __init__(self) -> NoReturn:
        """
        Initialize a WikipediaDumpXmlProcessorRecordHeadingBase object.
        """
        self.wiki_heading_base: str = None
        self.wiki_heading_base_processed: str = None
        self.wiki_heading_base_lines: List[str] = []
        self.wiki_heading_base_components: List[TypeHeadingBaseMember] = []
    def __repr__(self) -> str:
        """
        Return a representation of this object.
        """
        return self.to_json_in_string(json_configuration={})
    def to_json_in_string(self, \
        json_configuration: Any) -> str:
        """
        Convert this/self object to a JSON string.
        """
        return json.dumps( \
            ensure_ascii=False, \
            obj=self.to_json_friendly_structure( \
                json_configuration=json_configuration))
    def to_json_friendly_structure(self, \
        json_configuration: Any) -> Any:
        """
        Convert this/self object to a JSON-friendly structure.
        """
        # ---- NOTE-PYLINT ---- W0613: Unused argument
        # pylint: disable=W0613
        # ---- NOTE-PYLINT ---- R0201: Method could be a function (no-self-use)
        # pylint: disable=R0201
        DebuggingHelper.ensure(
            False,
            'should be implemented by a child')
        # ---- NOTE-PLACE-HOLDER ---- pass
    def write_lines_to_file(self, \
        writer: StreamReaderWriter) -> NoReturn:
        """
        Output lines hold in this object to a file.
        """
        # ---- NOTE-PYLINT ---- W0613: Unused argument
        # pylint: disable=W0613
        # ---- NOTE-PYLINT ---- R0201: Method could be a function (no-self-use)
        # pylint: disable=R0201
        DebuggingHelper.ensure(
            False,
            'should be implemented by a child')
        # ---- NOTE-PLACE-HOLDER ---- pass
    def set_wiki_heading_base(self, \
        input_wiki_heading_base: str, \
        input_wiki_heading_base_lines: List[str], \
        input_wiki_heading_base_components: List[TypeHeadingBaseMember]) -> NoReturn:
        """
        Set the WikipediaDumpXmlProcessorRecordHeading1 data structure.
        """
        self.wiki_heading_base = input_wiki_heading_base
        self.wiki_heading_base_processed = self.process_wiki_heading_base(self.wiki_heading_base)
        self.wiki_heading_base_lines = input_wiki_heading_base_lines
        self.wiki_heading_base_components = input_wiki_heading_base_components
    def get_wiki_heading_base_processed(self) -> str:
        """
        Get the wiki_heading_base processed data structure.
        """
        return self.wiki_heading_base_processed
    def get_wiki_heading_base(self) -> str:
        """
        Get the wiki_heading_base data structure.
        """
        return self.wiki_heading_base
    def get_wiki_heading_base_lines(self) -> List[str]:
        """
        Get the wiki_heading_base_lines data structure.
        """
        return self.wiki_heading_base_lines
    def get_wiki_heading_base_components(self) -> List[TypeHeadingBaseMember]:
        """
        Get the wiki_heading_base_components data structure.
        """
        return self.wiki_heading_base_components
    def process_wiki_heading_base(self, input_text: str) -> str:
        """
        Logic to preprocess heading names
        """
        # ---- NOTE-PYLINT ---- R0201: Method could be a function (no-self-use)
        # pylint: disable=R0201
        return TextChineseConverter.convert_simplified_to_traditional(input_text)

class WikipediaDumpXmlProcessorRecordHeading6( \
    WikipediaDumpXmlProcessorRecordHeadingBase[ \
        WikipediaDumpXmlProcessorRecordExtractionBase[Any]]):
    """
    A record for storing a processed heading6 structure.
    """
    WIKI_HEADING3_REGULAR_EXPRESSION_PATTERN = \
        re.compile("^======[ ]?([^ =]+?)[ ]?======")
    # ---- NOTE-NO-NEED ---- def __init__(self) -> NoReturn:
    # ---- NOTE-NO-NEED ----     """
    # ---- NOTE-NO-NEED ----     Initialize a WikipediaDumpXmlProcessorRecordHeading6 object.
    # ---- NOTE-NO-NEED ----     """
    # ---- NOTE-NO-NEED ----     super(WikipediaDumpXmlProcessorRecordHeading6, self).__init__()
    def __repr__(self) -> str:
        """
        Return a representation of this object.
        """
        return self.to_json_in_string(json_configuration={})
    def to_json_in_string(self, \
        json_configuration: Any) -> str:
        """
        Convert this/self object to a JSON string.
        """
        return json.dumps( \
            ensure_ascii=False, \
            obj=self.to_json_friendly_structure( \
                json_configuration=json_configuration))
    def to_json_friendly_structure(self, \
        json_configuration: Any) -> Any:
        """
        Convert this/self object to a JSON-friendly structure.
        """
        json_object: Any = {}
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # ---- NOTE-PLACE-HOLDER ---- json_friendly_structure_to_output_text: bool = \
        # ---- NOTE-PLACE-HOLDER ----     json_configuration['json_friendly_structure_to_output_text_RecordHeading6']
        json_friendly_structure_to_output_lines: bool = \
            json_configuration['json_friendly_structure_to_output_lines_RecordHeading6']
        json_object['heading6_title'] = \
            self.get_wiki_heading6()
        if json_friendly_structure_to_output_lines:
            json_object['heading6_lines'] = \
                self.get_wiki_heading6_lines()
        json_object['heading6_components'] = \
            [x.to_json_friendly_structure( \
                json_configuration=json_configuration) \
            for x in self.get_wiki_heading_base_components()]
        return json_object
    def write_lines_to_file(self, \
        writer: StreamReaderWriter) -> NoReturn:
        """
        Output lines hold in this object to a file.
        """
        title: str = self.get_wiki_heading6()
        if not StringHelper.is_none_empty_or_whitespaces(title):
            writer.writelines([title, '\n'])
        for entry in self.get_wiki_heading_base_components():
            entry.write_lines_to_file(writer)
    def set_wiki_heading6(self, \
        input_wiki_heading6: str, \
        input_wiki_heading6_lines: List[str], \
        input_wiki_heading6_components: \
            List[WikipediaDumpXmlProcessorRecordExtractionBase[Any]]) -> NoReturn:
        """
        Set the WikipediaDumpXmlProcessorRecordHeading6 data structure.
        """
        self.set_wiki_heading_base( \
            input_wiki_heading_base=input_wiki_heading6, \
            input_wiki_heading_base_lines=input_wiki_heading6_lines, \
            input_wiki_heading_base_components=input_wiki_heading6_components)
    def get_wiki_heading6(self) -> str:
        """
        Get the wiki_heading6 data structure.
        """
        return self.get_wiki_heading_base_processed()
    def get_wiki_heading6_lines(self) -> List[str]:
        """
        Get the wiki_heading6_lines data structure.
        """
        return self.get_wiki_heading_base_lines()
    def get_wiki_heading6_components(self) -> \
        List[WikipediaDumpXmlProcessorRecordExtractionBase[Any]]:
        """
        Get the wiki_heading6_components data structure.
        """
        return self.get_wiki_heading_base_components()
    @staticmethod
    def process_heading6_line(input_wiki_text_line: str) -> str:
        """
        Check if an input wiki text line is the beginning of a heading6.
        If it is, return the chpater, otherwise None.
        """
        matched: Match[str] = \
            WikipediaDumpXmlProcessorRecordHeading6.WIKI_HEADING3_REGULAR_EXPRESSION_PATTERN.\
                search(input_wiki_text_line)
        if matched is None:
            return None
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     u'---- WikipediaDumpXmlProcessorRecordHeading6.process_heading6_line(), ' +
        #     'matched={}'.format(matched))
        # matched_span: Tuple[int, int] = matched.span()
        # matched_string: str = matched.string
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     '---- WikipediaDumpXmlProcessorRecordHeading6.process_heading6_line(), ' +
        #     'matched_string={}'.format(matched_string))
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     '---- WikipediaDumpXmlProcessorRecordHeading6.process_heading6_line(), ' +
        #     'matched_span.__repr__()={}'.format(matched_span.__repr__()))
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     '---- WikipediaDumpXmlProcessorRecordHeading6.process_heading6_line(), ' +
        #     'len(matched_span)={}'.format(len(matched_span)))
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     '---- WikipediaDumpXmlProcessorRecordHeading6.process_heading6_line(), ' +
        #     'matched_span[0]={}'.format(matched_span[0]))
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     '---- WikipediaDumpXmlProcessorRecordHeading6.process_heading6_line(), ' +
        #     'matched_span[1]={}'.format(matched_span[1]))
        matched_group_string: str = matched.group(1)
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     '---- WikipediaDumpXmlProcessorRecordHeading6.process_heading6_line(), ' +
        #     'matched_group_string={}'.format(matched_group_string))
        return matched_group_string
class WikipediaDumpXmlProcessorRecordHeading6Factory:
    """
    A factor for processing WikipediaDumpXmlProcessorRecordHeading6 objects.
    """
    # ---- NOTE-PYLINT ---- R0903: Too few public methods
    # pylint: disable=R0903
    @staticmethod
    def process_wiki_heading6s(input_wiki_text_lines: List[str]) -> \
        List[WikipediaDumpXmlProcessorRecordHeading6]:
        """
        Iterate through a wiki's text lines and group the lines into groups, each a heading6.
        """
        output_wiki_heading6s: List[WikipediaDumpXmlProcessorRecordHeading6] = []
        current_heading6_lines: List[str] = []
        current_heading6_name: str = ''
        for input_wiki_text_line in input_wiki_text_lines:
            return_process_heading6_line: str = \
                WikipediaDumpXmlProcessorRecordHeading6.process_heading6_line( \
                    input_wiki_text_line)
            if return_process_heading6_line is not None:
                # ----
                output_wiki_heading6: WikipediaDumpXmlProcessorRecordHeading6 = \
                    WikipediaDumpXmlProcessorRecordHeading6()
                # ----
                wiki_extractions: List[WikipediaDumpXmlProcessorRecordExtractionBase[Any]] = \
                    WikipediaDumpXmlProcessorExtractionUtility.\
                        process_from_text_lines(current_heading6_lines)
                output_wiki_heading6.set_wiki_heading6( \
                    input_wiki_heading6=current_heading6_name, \
                    input_wiki_heading6_lines=current_heading6_lines, \
                    input_wiki_heading6_components=wiki_extractions)
                # ----
                output_wiki_heading6s.append(output_wiki_heading6)
                # DebuggingHelper.write_line_to_system_console_out_debug( \
                #     '---- current_heading6_name={}'.format(current_heading6_name))
                # ----
                current_heading6_name = return_process_heading6_line
                current_heading6_lines = []
                # ----
            else:
                current_heading6_lines.append(input_wiki_text_line)
        if len(current_heading6_lines) > 0:
            # ----
            output_wiki_heading6: WikipediaDumpXmlProcessorRecordHeading6 = \
                WikipediaDumpXmlProcessorRecordHeading6()
            # ----
            wiki_extractions: List[WikipediaDumpXmlProcessorRecordExtractionBase[Any]] = \
                WikipediaDumpXmlProcessorExtractionUtility.\
                    process_from_text_lines(current_heading6_lines)
            output_wiki_heading6.set_wiki_heading6( \
                input_wiki_heading6=current_heading6_name, \
                input_wiki_heading6_lines=current_heading6_lines, \
                input_wiki_heading6_components=wiki_extractions)
            # ----
            output_wiki_heading6s.append(output_wiki_heading6)
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- current_heading6_name={}'.format(current_heading6_name))
            # ----
        return output_wiki_heading6s

class WikipediaDumpXmlProcessorRecordHeading5( \
    WikipediaDumpXmlProcessorRecordHeadingBase[ \
        WikipediaDumpXmlProcessorRecordHeading6]):
    """
    A record for storing a processed heading5 structure.
    """
    WIKI_HEADING2_REGULAR_EXPRESSION_PATTERN = \
        re.compile("^=====[ ]?([^ =]+?)[ ]?=====")
    # ---- NOTE-NO-NEED ---- def __init__(self) -> NoReturn:
    # ---- NOTE-NO-NEED ----     """
    # ---- NOTE-NO-NEED ----     Initialize a WikipediaDumpXmlProcessorRecordHeading5 object.
    # ---- NOTE-NO-NEED ----     """
    # ---- NOTE-NO-NEED ----     super(WikipediaDumpXmlProcessorRecordHeading5, self).__init__()
    def __repr__(self) -> str:
        """
        Return a representation of this object.
        """
        return self.to_json_in_string(json_configuration={})
    def to_json_in_string(self, \
        json_configuration: Any) -> str:
        """
        Convert this/self object to a JSON string.
        """
        return json.dumps( \
            ensure_ascii=False, \
            obj=self.to_json_friendly_structure( \
                json_configuration=json_configuration))
    def to_json_friendly_structure(self, \
        json_configuration: Any) -> Any:
        """
        Convert this/self object to a JSON-friendly structure.
        """
        json_object: Any = {}
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # ---- NOTE-PLACE-HOLDER ---- json_friendly_structure_to_output_text: bool = \
        # ---- NOTE-PLACE-HOLDER ----     json_configuration['json_friendly_structure_to_output_text_RecordHeading5']
        json_friendly_structure_to_output_lines: bool = \
            json_configuration['json_friendly_structure_to_output_lines_RecordHeading5']
        json_object['heading5_title'] = \
            self.get_wiki_heading5()
        if json_friendly_structure_to_output_lines:
            json_object['heading5_lines'] = \
                self.get_wiki_heading5_lines()
        json_object['heading5_heading6s'] = \
            [x.to_json_friendly_structure( \
                json_configuration=json_configuration) \
            for x in self.get_wiki_heading5_heading6s()]
        return json_object
    def write_lines_to_file(self, \
        writer: StreamReaderWriter) -> NoReturn:
        """
        Output lines hold in this object to a file.
        """
        title: str = self.get_wiki_heading5()
        if not StringHelper.is_none_empty_or_whitespaces(title):
            writer.writelines([title, '\n'])
        for entry in self.get_wiki_heading5_heading6s():
            entry.write_lines_to_file(writer)
    def set_wiki_heading5(self, \
        input_wiki_heading5: str, \
        input_wiki_heading5_lines: List[str], \
        input_wiki_heading5_components: List[WikipediaDumpXmlProcessorRecordHeading6]) -> NoReturn:
        """
        Set the WikipediaDumpXmlProcessorRecordHeading5 data structure.
        """
        self.set_wiki_heading_base( \
            input_wiki_heading_base=input_wiki_heading5, \
            input_wiki_heading_base_lines=input_wiki_heading5_lines, \
            input_wiki_heading_base_components=input_wiki_heading5_components)
    def get_wiki_heading5(self) -> str:
        """
        Get the wiki_heading5 data structure.
        """
        return self.get_wiki_heading_base_processed()
    def get_wiki_heading5_lines(self) -> List[str]:
        """
        Get the wiki_heading5_lines data structure.
        """
        return self.get_wiki_heading_base_lines()
    def get_wiki_heading5_heading6s(self) -> List[WikipediaDumpXmlProcessorRecordHeading6]:
        """
        Get the wiki_heading5_heading6s data structure.
        """
        return self.get_wiki_heading_base_components()
    @staticmethod
    def process_heading5_line(input_wiki_text_line: str) -> str:
        """
        Check if an input wiki text line is the beginning of a heading5.
        If it is, return the chpater, otherwise None.
        """
        matched: Match[str] = \
            WikipediaDumpXmlProcessorRecordHeading5.\
                WIKI_HEADING2_REGULAR_EXPRESSION_PATTERN.search( \
                    input_wiki_text_line)
        if matched is None:
            return None
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     u'---- WikipediaDumpXmlProcessorRecordHeading5.process_heading5_line(), ' +
        #     'matched={}'.format(matched))
        # matched_span: Tuple[int, int] = matched.span()
        # matched_string: str = matched.string
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     '---- WikipediaDumpXmlProcessorRecordHeading5.process_heading5_line(), ' +
        #     'matched_string={}'.format(matched_string))
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     '---- WikipediaDumpXmlProcessorRecordHeading5.process_heading5_line(), ' +
        #     'matched_span.__repr__()={}'.format(matched_span.__repr__()))
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     '---- WikipediaDumpXmlProcessorRecordHeading5.process_heading5_line(), ' +
        #     'len(matched_span)={}'.format(len(matched_span)))
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     '---- WikipediaDumpXmlProcessorRecordHeading5.process_heading5_line(), ' +
        #     'matched_span[0]={}'.format(matched_span[0]))
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     '---- WikipediaDumpXmlProcessorRecordHeading5.process_heading5_line(), ' +
        #     'matched_span[1]={}'.format(matched_span[1]))
        matched_group_string: str = matched.group(1)
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     '---- WikipediaDumpXmlProcessorRecordHeading5.process_heading5_line(), ' +
        #     'matched_group_string={}'.format(matched_group_string))
        return matched_group_string
class WikipediaDumpXmlProcessorRecordHeading5Factory:
    """
    A factor for processing WikipediaDumpXmlProcessorRecordHeading5 objects.
    """
    # ---- NOTE-PYLINT ---- R0903: Too few public methods
    # pylint: disable=R0903
    @staticmethod
    def process_wiki_heading5s(input_wiki_text_lines: List[str]) -> \
        List[WikipediaDumpXmlProcessorRecordHeading5]:
        """
        Iterate through a wiki's text lines and group the lines into groups, each a heading5.
        """
        output_wiki_heading5s: List[WikipediaDumpXmlProcessorRecordHeading5] = []
        current_heading5_lines: List[str] = []
        current_heading5_name: str = ''
        for input_wiki_text_line in input_wiki_text_lines:
            return_process_heading5_line: str = \
                WikipediaDumpXmlProcessorRecordHeading5.process_heading5_line(input_wiki_text_line)
            if return_process_heading5_line is not None:
                output_wiki_heading5: WikipediaDumpXmlProcessorRecordHeading5 = \
                    WikipediaDumpXmlProcessorRecordHeading5()
                output_wiki_heading6s: List[WikipediaDumpXmlProcessorRecordHeading6] = \
                    WikipediaDumpXmlProcessorRecordHeading6Factory.process_wiki_heading6s( \
                        current_heading5_lines)
                output_wiki_heading5.set_wiki_heading5( \
                    input_wiki_heading5=current_heading5_name, \
                    input_wiki_heading5_lines=current_heading5_lines, \
                    input_wiki_heading5_components=output_wiki_heading6s)
                output_wiki_heading5s.append(output_wiki_heading5)
                current_heading5_name = return_process_heading5_line
                current_heading5_lines = []
                # DebuggingHelper.write_line_to_system_console_out_debug( \
                #     '---- current_heading5_name={}'.format(current_heading5_name))
            else:
                current_heading5_lines.append(input_wiki_text_line)
        if len(current_heading5_lines) > 0:
            output_wiki_heading5: WikipediaDumpXmlProcessorRecordHeading5 = \
                WikipediaDumpXmlProcessorRecordHeading5()
            output_wiki_heading6s: List[WikipediaDumpXmlProcessorRecordHeading6] = \
                WikipediaDumpXmlProcessorRecordHeading6Factory.process_wiki_heading6s( \
                    current_heading5_lines)
            output_wiki_heading5.set_wiki_heading5( \
                input_wiki_heading5=current_heading5_name, \
                input_wiki_heading5_lines=current_heading5_lines, \
                input_wiki_heading5_components=output_wiki_heading6s)
            output_wiki_heading5s.append(output_wiki_heading5)
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- current_heading5_name={}'.format(current_heading5_name))
        return output_wiki_heading5s

class WikipediaDumpXmlProcessorRecordHeading4( \
    WikipediaDumpXmlProcessorRecordHeadingBase[ \
        WikipediaDumpXmlProcessorRecordHeading5]):
    """
    A record for storing a processed heading4 structure.
    """
    WIKI_HEADING2_REGULAR_EXPRESSION_PATTERN = \
        re.compile("^====[ ]?([^ =]+?)[ ]?====")
    # ---- NOTE-NO-NEED ---- def __init__(self) -> NoReturn:
    # ---- NOTE-NO-NEED ----     """
    # ---- NOTE-NO-NEED ----     Initialize a WikipediaDumpXmlProcessorRecordHeading4 object.
    # ---- NOTE-NO-NEED ----     """
    # ---- NOTE-NO-NEED ----     super(WikipediaDumpXmlProcessorRecordHeading4, self).__init__()
    def __repr__(self) -> str:
        """
        Return a representation of this object.
        """
        return self.to_json_in_string(json_configuration={})
    def to_json_in_string(self, \
        json_configuration: Any) -> str:
        """
        Convert this/self object to a JSON string.
        """
        return json.dumps( \
            ensure_ascii=False, \
            obj=self.to_json_friendly_structure( \
                json_configuration=json_configuration))
    def to_json_friendly_structure(self, \
        json_configuration: Any) -> Any:
        """
        Convert this/self object to a JSON-friendly structure.
        """
        json_object: Any = {}
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # ---- NOTE-PLACE-HOLDER ---- json_friendly_structure_to_output_text: bool = \
        # ---- NOTE-PLACE-HOLDER ----     json_configuration['json_friendly_structure_to_output_text_RecordHeading4']
        json_friendly_structure_to_output_lines: bool = \
            json_configuration['json_friendly_structure_to_output_lines_RecordHeading4']
        json_object['heading4_title'] = \
            self.get_wiki_heading4()
        if json_friendly_structure_to_output_lines:
            json_object['heading4_lines'] = \
                self.get_wiki_heading4_lines()
        json_object['heading4_heading5s'] = \
            [x.to_json_friendly_structure( \
                json_configuration=json_configuration) \
            for x in self.get_wiki_heading4_heading5s()]
        return json_object
    def write_lines_to_file(self, \
        writer: StreamReaderWriter) -> NoReturn:
        """
        Output lines hold in this object to a file.
        """
        title: str = self.get_wiki_heading4()
        if not StringHelper.is_none_empty_or_whitespaces(title):
            writer.writelines([title, '\n'])
        for entry in self.get_wiki_heading4_heading5s():
            entry.write_lines_to_file(writer)
    def set_wiki_heading4(self, \
        input_wiki_heading4: str, \
        input_wiki_heading4_lines: List[str], \
        input_wiki_heading4_components: List[WikipediaDumpXmlProcessorRecordHeading5]) -> NoReturn:
        """
        Set the WikipediaDumpXmlProcessorRecordHeading4 data structure.
        """
        self.set_wiki_heading_base( \
            input_wiki_heading_base=input_wiki_heading4, \
            input_wiki_heading_base_lines=input_wiki_heading4_lines, \
            input_wiki_heading_base_components=input_wiki_heading4_components)
    def get_wiki_heading4(self) -> str:
        """
        Get the wiki_heading4 data structure.
        """
        return self.get_wiki_heading_base_processed()
    def get_wiki_heading4_lines(self) -> List[str]:
        """
        Get the wiki_heading4_lines data structure.
        """
        return self.get_wiki_heading_base_lines()
    def get_wiki_heading4_heading5s(self) -> List[WikipediaDumpXmlProcessorRecordHeading5]:
        """
        Get the wiki_heading4_heading5s data structure.
        """
        return self.get_wiki_heading_base_components()
    @staticmethod
    def process_heading4_line(input_wiki_text_line: str) -> str:
        """
        Check if an input wiki text line is the beginning of a heading4.
        If it is, return the chpater, otherwise None.
        """
        matched: Match[str] = \
            WikipediaDumpXmlProcessorRecordHeading4.\
                WIKI_HEADING2_REGULAR_EXPRESSION_PATTERN.search( \
                    input_wiki_text_line)
        if matched is None:
            return None
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     u'---- WikipediaDumpXmlProcessorRecordHeading4.process_heading4_line(), ' +
        #     'matched={}'.format(matched))
        # matched_span: Tuple[int, int] = matched.span()
        # matched_string: str = matched.string
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     '---- WikipediaDumpXmlProcessorRecordHeading4.process_heading4_line(), ' +
        #     'matched_string={}'.format(matched_string))
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     '---- WikipediaDumpXmlProcessorRecordHeading4.process_heading4_line(), ' +
        #     'matched_span.__repr__()={}'.format(matched_span.__repr__()))
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     '---- WikipediaDumpXmlProcessorRecordHeading4.process_heading4_line(), ' +
        #     'len(matched_span)={}'.format(len(matched_span)))
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     '---- WikipediaDumpXmlProcessorRecordHeading4.process_heading4_line(), ' +
        #     'matched_span[0]={}'.format(matched_span[0]))
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     '---- WikipediaDumpXmlProcessorRecordHeading4.process_heading4_line(), ' +
        #     'matched_span[1]={}'.format(matched_span[1]))
        matched_group_string: str = matched.group(1)
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     '---- WikipediaDumpXmlProcessorRecordHeading4.process_heading4_line(), ' +
        #     'matched_group_string={}'.format(matched_group_string))
        return matched_group_string
class WikipediaDumpXmlProcessorRecordHeading4Factory:
    """
    A factor for processing WikipediaDumpXmlProcessorRecordHeading4 objects.
    """
    # ---- NOTE-PYLINT ---- R0903: Too few public methods
    # pylint: disable=R0903
    @staticmethod
    def process_wiki_heading4s(input_wiki_text_lines: List[str]) -> \
        List[WikipediaDumpXmlProcessorRecordHeading4]:
        """
        Iterate through a wiki's text lines and group the lines into groups, each a heading4.
        """
        output_wiki_heading4s: List[WikipediaDumpXmlProcessorRecordHeading4] = []
        current_heading4_lines: List[str] = []
        current_heading4_name: str = ''
        for input_wiki_text_line in input_wiki_text_lines:
            return_process_heading4_line: str = \
                WikipediaDumpXmlProcessorRecordHeading4.process_heading4_line(input_wiki_text_line)
            if return_process_heading4_line is not None:
                output_wiki_heading4: WikipediaDumpXmlProcessorRecordHeading4 = \
                    WikipediaDumpXmlProcessorRecordHeading4()
                output_wiki_heading5s: List[WikipediaDumpXmlProcessorRecordHeading5] = \
                    WikipediaDumpXmlProcessorRecordHeading5Factory.process_wiki_heading5s( \
                        current_heading4_lines)
                output_wiki_heading4.set_wiki_heading4( \
                    input_wiki_heading4=current_heading4_name, \
                    input_wiki_heading4_lines=current_heading4_lines, \
                    input_wiki_heading4_components=output_wiki_heading5s)
                output_wiki_heading4s.append(output_wiki_heading4)
                current_heading4_name = return_process_heading4_line
                current_heading4_lines = []
                # DebuggingHelper.write_line_to_system_console_out_debug( \
                #     '---- current_heading4_name={}'.format(current_heading4_name))
            else:
                current_heading4_lines.append(input_wiki_text_line)
        if len(current_heading4_lines) > 0:
            output_wiki_heading4: WikipediaDumpXmlProcessorRecordHeading4 = \
                WikipediaDumpXmlProcessorRecordHeading4()
            output_wiki_heading5s: List[WikipediaDumpXmlProcessorRecordHeading5] = \
                WikipediaDumpXmlProcessorRecordHeading5Factory.process_wiki_heading5s( \
                    current_heading4_lines)
            output_wiki_heading4.set_wiki_heading4( \
                input_wiki_heading4=current_heading4_name, \
                input_wiki_heading4_lines=current_heading4_lines, \
                input_wiki_heading4_components=output_wiki_heading5s)
            output_wiki_heading4s.append(output_wiki_heading4)
          # DebuggingHelper.write_line_to_system_console_out_debug( \
          #     '---- current_heading4_name={}'.format(current_heading4_name))
        return output_wiki_heading4s

class WikipediaDumpXmlProcessorRecordHeading3( \
    WikipediaDumpXmlProcessorRecordHeadingBase[ \
        WikipediaDumpXmlProcessorRecordHeading4]):
    """
    A record for storing a processed heading3 structure.
    """
    WIKI_HEADING2_REGULAR_EXPRESSION_PATTERN = \
        re.compile("^===[ ]?([^ =]+?)[ ]?===")
    # ---- NOTE-NO-NEED ---- def __init__(self) -> NoReturn:
    # ---- NOTE-NO-NEED ----     """
    # ---- NOTE-NO-NEED ----     Initialize a WikipediaDumpXmlProcessorRecordHeading3 object.
    # ---- NOTE-NO-NEED ----     """
    # ---- NOTE-NO-NEED ----     super(WikipediaDumpXmlProcessorRecordHeading3, self).__init__()
    def __repr__(self) -> str:
        """
        Return a representation of this object.
        """
        return self.to_json_in_string(json_configuration={})
    def to_json_in_string(self, \
        json_configuration: Any) -> str:
        """
        Convert this/self object to a JSON string.
        """
        return json.dumps( \
            ensure_ascii=False, \
            obj=self.to_json_friendly_structure( \
                json_configuration=json_configuration))
    def to_json_friendly_structure(self, \
        json_configuration: Any) -> Any:
        """
        Convert this/self object to a JSON-friendly structure.
        """
        json_object: Any = {}
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # ---- NOTE-PLACE-HOLDER ---- json_friendly_structure_to_output_text: bool = \
        # ---- NOTE-PLACE-HOLDER ----     json_configuration['json_friendly_structure_to_output_text_RecordHeading3']
        json_friendly_structure_to_output_lines: bool = \
            json_configuration['json_friendly_structure_to_output_lines_RecordHeading3']
        json_object['heading3_title'] = \
            self.get_wiki_heading3()
        if json_friendly_structure_to_output_lines:
            json_object['heading3_lines'] = \
                self.get_wiki_heading3_lines()
        json_object['heading3_heading4s'] = \
            [x.to_json_friendly_structure( \
                json_configuration=json_configuration) \
            for x in self.get_wiki_heading3_heading4s()]
        return json_object
    def write_lines_to_file(self, \
        writer: StreamReaderWriter) -> NoReturn:
        """
        Output lines hold in this object to a file.
        """
        title: str = self.get_wiki_heading3()
        if not StringHelper.is_none_empty_or_whitespaces(title):
            writer.writelines([title, '\n'])
        for entry in self.get_wiki_heading3_heading4s():
            entry.write_lines_to_file(writer)
    def set_wiki_heading3(self, \
        input_wiki_heading3: str, \
        input_wiki_heading3_lines: List[str], \
        input_wiki_heading3_components: List[WikipediaDumpXmlProcessorRecordHeading4]) -> NoReturn:
        """
        Set the WikipediaDumpXmlProcessorRecordHeading3 data structure.
        """
        self.set_wiki_heading_base( \
            input_wiki_heading_base=input_wiki_heading3, \
            input_wiki_heading_base_lines=input_wiki_heading3_lines, \
            input_wiki_heading_base_components=input_wiki_heading3_components)
    def get_wiki_heading3(self) -> str:
        """
        Get the wiki_heading3 data structure.
        """
        return self.get_wiki_heading_base_processed()
    def get_wiki_heading3_lines(self) -> List[str]:
        """
        Get the wiki_heading3_lines data structure.
        """
        return self.get_wiki_heading_base_lines()
    def get_wiki_heading3_heading4s(self) -> List[WikipediaDumpXmlProcessorRecordHeading4]:
        """
        Get the wiki_heading3_heading4s data structure.
        """
        return self.get_wiki_heading_base_components()
    @staticmethod
    def process_heading3_line(input_wiki_text_line: str) -> str:
        """
        Check if an input wiki text line is the beginning of a heading3.
        If it is, return the chpater, otherwise None.
        """
        matched: Match[str] = \
            WikipediaDumpXmlProcessorRecordHeading3.\
                WIKI_HEADING2_REGULAR_EXPRESSION_PATTERN.search( \
                    input_wiki_text_line)
        if matched is None:
            return None
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     u'---- WikipediaDumpXmlProcessorRecordHeading3.process_heading3_line(), ' +
        #     'matched={}'.format(matched))
        # matched_span: Tuple[int, int] = matched.span()
        # matched_string: str = matched.string
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     '---- WikipediaDumpXmlProcessorRecordHeading3.process_heading3_line(), ' +
        #     'matched_string={}'.format(matched_string))
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     '---- WikipediaDumpXmlProcessorRecordHeading3.process_heading3_line(), ' +
        #     'matched_span.__repr__()={}'.format(matched_span.__repr__()))
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     '---- WikipediaDumpXmlProcessorRecordHeading3.process_heading3_line(), ' +
        #     'len(matched_span)={}'.format(len(matched_span)))
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     '---- WikipediaDumpXmlProcessorRecordHeading3.process_heading3_line(), ' +
        #     'matched_span[0]={}'.format(matched_span[0]))
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     '---- WikipediaDumpXmlProcessorRecordHeading3.process_heading3_line(), ' +
        #     'matched_span[1]={}'.format(matched_span[1]))
        matched_group_string: str = matched.group(1)
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     '---- WikipediaDumpXmlProcessorRecordHeading3.process_heading3_line(), ' +
        #     'matched_group_string={}'.format(matched_group_string))
        return matched_group_string
class WikipediaDumpXmlProcessorRecordHeading3Factory:
    """
    A factor for processing WikipediaDumpXmlProcessorRecordHeading3 objects.
    """
    # ---- NOTE-PYLINT ---- R0903: Too few public methods
    # pylint: disable=R0903
    @staticmethod
    def process_wiki_heading3s(input_wiki_text_lines: List[str]) -> \
        List[WikipediaDumpXmlProcessorRecordHeading3]:
        """
        Iterate through a wiki's text lines and group the lines into groups, each a heading3.
        """
        output_wiki_heading3s: List[WikipediaDumpXmlProcessorRecordHeading3] = []
        current_heading3_lines: List[str] = []
        current_heading3_name: str = ''
        for input_wiki_text_line in input_wiki_text_lines:
            return_process_heading3_line: str = \
                WikipediaDumpXmlProcessorRecordHeading3.process_heading3_line(input_wiki_text_line)
            if return_process_heading3_line is not None:
                output_wiki_heading3: WikipediaDumpXmlProcessorRecordHeading3 = \
                    WikipediaDumpXmlProcessorRecordHeading3()
                output_wiki_heading4s: List[WikipediaDumpXmlProcessorRecordHeading4] = \
                    WikipediaDumpXmlProcessorRecordHeading4Factory.process_wiki_heading4s( \
                        current_heading3_lines)
                output_wiki_heading3.set_wiki_heading3( \
                    input_wiki_heading3=current_heading3_name, \
                    input_wiki_heading3_lines=current_heading3_lines, \
                    input_wiki_heading3_components=output_wiki_heading4s)
                output_wiki_heading3s.append(output_wiki_heading3)
                current_heading3_name = return_process_heading3_line
                current_heading3_lines = []
                # DebuggingHelper.write_line_to_system_console_out_debug( \
                #     '---- current_heading3_name={}'.format(current_heading3_name))
            else:
                current_heading3_lines.append(input_wiki_text_line)
        if len(current_heading3_lines) > 0:
            output_wiki_heading3: WikipediaDumpXmlProcessorRecordHeading3 = \
                WikipediaDumpXmlProcessorRecordHeading3()
            output_wiki_heading4s: List[WikipediaDumpXmlProcessorRecordHeading4] = \
                WikipediaDumpXmlProcessorRecordHeading4Factory.process_wiki_heading4s( \
                    current_heading3_lines)
            output_wiki_heading3.set_wiki_heading3( \
                input_wiki_heading3=current_heading3_name, \
                input_wiki_heading3_lines=current_heading3_lines, \
                input_wiki_heading3_components=output_wiki_heading4s)
            output_wiki_heading3s.append(output_wiki_heading3)
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- current_heading3_name={}'.format(current_heading3_name))
        return output_wiki_heading3s

class WikipediaDumpXmlProcessorRecordHeading2( \
    WikipediaDumpXmlProcessorRecordHeadingBase[ \
        WikipediaDumpXmlProcessorRecordHeading3]):
    """
    A record for storing a processed heading2 structure.
    """
    WIKI_HEADING2_REGULAR_EXPRESSION_PATTERN = \
        re.compile("^==[ ]?([^ =]+?)[ ]?==")
    # ---- NOTE-NO-NEED ---- def __init__(self) -> NoReturn:
    # ---- NOTE-NO-NEED ----     """
    # ---- NOTE-NO-NEED ----     Initialize a WikipediaDumpXmlProcessorRecordHeading2 object.
    # ---- NOTE-NO-NEED ----     """
    # ---- NOTE-NO-NEED ----     super(WikipediaDumpXmlProcessorRecordHeading2, self).__init__()
    def __repr__(self) -> str:
        """
        Return a representation of this object.
        """
        return self.to_json_in_string(json_configuration={})
    def to_json_in_string(self, \
        json_configuration: Any) -> str:
        """
        Convert this/self object to a JSON string.
        """
        return json.dumps( \
            ensure_ascii=False, \
            obj=self.to_json_friendly_structure( \
                json_configuration=json_configuration))
    def to_json_friendly_structure(self, \
        json_configuration: Any) -> Any:
        """
        Convert this/self object to a JSON-friendly structure.
        """
        json_object: Any = {}
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # ---- NOTE-PLACE-HOLDER ---- json_friendly_structure_to_output_text: bool = \
        # ---- NOTE-PLACE-HOLDER ----     json_configuration['json_friendly_structure_to_output_text_RecordHeading2']
        json_friendly_structure_to_output_lines: bool = \
            json_configuration['json_friendly_structure_to_output_lines_RecordHeading2']
        json_object['heading2_title'] = \
            self.get_wiki_heading2()
        if json_friendly_structure_to_output_lines:
            json_object['heading2_lines'] = \
                self.get_wiki_heading2_lines()
        json_object['heading2_heading3s'] = \
            [x.to_json_friendly_structure( \
                json_configuration=json_configuration) \
            for x in self.get_wiki_heading2_heading3s()]
        return json_object
    def write_lines_to_file(self, \
        writer: StreamReaderWriter) -> NoReturn:
        """
        Output lines hold in this object to a file.
        """
        title: str = self.get_wiki_heading2()
        if not StringHelper.is_none_empty_or_whitespaces(title):
            writer.writelines([title, '\n'])
        for entry in self.get_wiki_heading2_heading3s():
            entry.write_lines_to_file(writer)
    def set_wiki_heading2(self, \
        input_wiki_heading2: str, \
        input_wiki_heading2_lines: List[str], \
        input_wiki_heading2_components: List[WikipediaDumpXmlProcessorRecordHeading3]) -> NoReturn:
        """
        Set the WikipediaDumpXmlProcessorRecordHeading2 data structure.
        """
        self.set_wiki_heading_base( \
            input_wiki_heading_base=input_wiki_heading2, \
            input_wiki_heading_base_lines=input_wiki_heading2_lines, \
            input_wiki_heading_base_components=input_wiki_heading2_components)
    def get_wiki_heading2(self) -> str:
        """
        Get the wiki_heading2 data structure.
        """
        return self.get_wiki_heading_base_processed()
    def get_wiki_heading2_lines(self) -> List[str]:
        """
        Get the wiki_heading2_lines data structure.
        """
        return self.get_wiki_heading_base_lines()
    def get_wiki_heading2_heading3s(self) -> List[WikipediaDumpXmlProcessorRecordHeading3]:
        """
        Get the wiki_heading2_heading3s data structure.
        """
        return self.get_wiki_heading_base_components()
    @staticmethod
    def process_heading2_line(input_wiki_text_line: str) -> str:
        """
        Check if an input wiki text line is the beginning of a heading2.
        If it is, return the chpater, otherwise None.
        """
        matched: Match[str] = \
            WikipediaDumpXmlProcessorRecordHeading2.\
                WIKI_HEADING2_REGULAR_EXPRESSION_PATTERN.search( \
                    input_wiki_text_line)
        if matched is None:
            return None
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     u'---- WikipediaDumpXmlProcessorRecordHeading2.process_heading2_line(), ' +
        #     'matched={}'.format(matched))
        # matched_span: Tuple[int, int] = matched.span()
        # matched_string: str = matched.string
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     '---- WikipediaDumpXmlProcessorRecordHeading2.process_heading2_line(), ' +
        #     'matched_string={}'.format(matched_string))
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     '---- WikipediaDumpXmlProcessorRecordHeading2.process_heading2_line(), ' +
        #     'matched_span.__repr__()={}'.format(matched_span.__repr__()))
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     '---- WikipediaDumpXmlProcessorRecordHeading2.process_heading2_line(), ' +
        #     'len(matched_span)={}'.format(len(matched_span)))
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     '---- WikipediaDumpXmlProcessorRecordHeading2.process_heading2_line(), ' +
        #     'matched_span[0]={}'.format(matched_span[0]))
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     '---- WikipediaDumpXmlProcessorRecordHeading2.process_heading2_line(), ' +
        #     'matched_span[1]={}'.format(matched_span[1]))
        matched_group_string: str = matched.group(1)
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     '---- WikipediaDumpXmlProcessorRecordHeading2.process_heading2_line(), ' +
        #     'matched_group_string={}'.format(matched_group_string))
        return matched_group_string
class WikipediaDumpXmlProcessorRecordHeading2Factory:
    """
    A factor for processing WikipediaDumpXmlProcessorRecordHeading2 objects.
    """
    # ---- NOTE-PYLINT ---- R0903: Too few public methods
    # pylint: disable=R0903
    @staticmethod
    def process_wiki_heading2s(input_wiki_text_lines: List[str]) -> \
        List[WikipediaDumpXmlProcessorRecordHeading2]:
        """
        Iterate through a wiki's text lines and group the lines into groups, each a heading2.
        """
        output_wiki_heading2s: List[WikipediaDumpXmlProcessorRecordHeading2] = []
        current_heading2_lines: List[str] = []
        current_heading2_name: str = ''
        for input_wiki_text_line in input_wiki_text_lines:
            return_process_heading2_line: str = \
                WikipediaDumpXmlProcessorRecordHeading2.process_heading2_line(input_wiki_text_line)
            if return_process_heading2_line is not None:
                output_wiki_heading2: WikipediaDumpXmlProcessorRecordHeading2 = \
                    WikipediaDumpXmlProcessorRecordHeading2()
                output_wiki_heading3s: List[WikipediaDumpXmlProcessorRecordHeading3] = \
                    WikipediaDumpXmlProcessorRecordHeading3Factory.process_wiki_heading3s( \
                        current_heading2_lines)
                output_wiki_heading2.set_wiki_heading2( \
                    input_wiki_heading2=current_heading2_name, \
                    input_wiki_heading2_lines=current_heading2_lines, \
                    input_wiki_heading2_components=output_wiki_heading3s)
                output_wiki_heading2s.append(output_wiki_heading2)
                current_heading2_name = return_process_heading2_line
                current_heading2_lines = []
                # DebuggingHelper.write_line_to_system_console_out_debug( \
                #     '---- current_heading2_name={}'.format(current_heading2_name))
            else:
                current_heading2_lines.append(input_wiki_text_line)
        if len(current_heading2_lines) > 0:
            output_wiki_heading2: WikipediaDumpXmlProcessorRecordHeading2 = \
                WikipediaDumpXmlProcessorRecordHeading2()
            output_wiki_heading3s: List[WikipediaDumpXmlProcessorRecordHeading3] = \
                WikipediaDumpXmlProcessorRecordHeading3Factory.process_wiki_heading3s( \
                    current_heading2_lines)
            output_wiki_heading2.set_wiki_heading2( \
                input_wiki_heading2=current_heading2_name, \
                input_wiki_heading2_lines=current_heading2_lines, \
                input_wiki_heading2_components=output_wiki_heading3s)
            output_wiki_heading2s.append(output_wiki_heading2)
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- current_heading2_name={}'.format(current_heading2_name))
        return output_wiki_heading2s

class WikipediaDumpXmlProcessorRecordHeading1( \
    WikipediaDumpXmlProcessorRecordHeadingBase[ \
        WikipediaDumpXmlProcessorRecordHeading2]):
    """
    A record for storing a processed heading1 structure.
    """
    WIKI_HEADING1_REGULAR_EXPRESSION_PATTERN = \
        re.compile("^=[ ]?([^ =]+?)[ ]?=")
    # ---- NOTE-NO-NEED ---- def __init__(self) -> NoReturn:
    # ---- NOTE-NO-NEED ----     """
    # ---- NOTE-NO-NEED ----     Initialize a WikipediaDumpXmlProcessorRecordHeading1 object.
    # ---- NOTE-NO-NEED ----     """
    # ---- NOTE-NO-NEED ----     super(WikipediaDumpXmlProcessorRecordHeading1, self).__init__()
    def __repr__(self) -> str:
        """
        Return a representation of this object.
        """
        return self.to_json_in_string(json_configuration={})
    def to_json_in_string(self, \
        json_configuration: Any) -> str:
        """
        Convert this/self object to a JSON string.
        """
        return json.dumps( \
            ensure_ascii=False, \
            obj=self.to_json_friendly_structure( \
                json_configuration=json_configuration))
    def to_json_friendly_structure(self, \
        json_configuration: Any) -> Any:
        """
        Convert this/self object to a JSON-friendly structure.
        """
        json_object: Any = {}
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # ---- NOTE-PLACE-HOLDER ---- json_friendly_structure_to_output_text: bool = \
        # ---- NOTE-PLACE-HOLDER ----     json_configuration['json_friendly_structure_to_output_text_RecordHeading1']
        json_friendly_structure_to_output_lines: bool = \
            json_configuration['json_friendly_structure_to_output_lines_RecordHeading1']
        json_object['heading1_title'] = \
            self.get_wiki_heading1()
        if json_friendly_structure_to_output_lines:
            json_object['heading1_lines'] = \
                self.get_wiki_heading1_lines()
        json_object['heading1_heading2s'] = \
            [x.to_json_friendly_structure( \
                json_configuration=json_configuration) \
            for x in self.get_wiki_heading1_heading2s()]
        return json_object
    def write_lines_to_file(self, \
        writer: StreamReaderWriter) -> NoReturn:
        """
        Output lines hold in this object to a file.
        """
        title: str = self.get_wiki_heading1()
        if not StringHelper.is_none_empty_or_whitespaces(title):
            writer.writelines([title, '\n'])
        for entry in self.get_wiki_heading1_heading2s():
            entry.write_lines_to_file(writer)
    def set_wiki_heading1(self, \
        input_wiki_heading1: str, \
        input_wiki_heading1_lines: List[str], \
        input_wiki_heading1_components: List[WikipediaDumpXmlProcessorRecordHeading2]) -> NoReturn:
        """
        Set the WikipediaDumpXmlProcessorRecordHeading1 data structure.
        """
        self.set_wiki_heading_base( \
            input_wiki_heading_base=input_wiki_heading1, \
            input_wiki_heading_base_lines=input_wiki_heading1_lines, \
            input_wiki_heading_base_components=input_wiki_heading1_components)
    def get_wiki_heading1(self) -> str:
        """
        Get the wiki_heading1 data structure.
        """
        return self.get_wiki_heading_base_processed()
    def get_wiki_heading1_lines(self) -> List[str]:
        """
        Get the wiki_heading1_lines data structure.
        """
        return self.get_wiki_heading_base_lines()
    def get_wiki_heading1_heading2s(self) -> List[WikipediaDumpXmlProcessorRecordHeading2]:
        """
        Get the wiki_heading1_heading2s data structure.
        """
        return self.get_wiki_heading_base_components()
    @staticmethod
    def process_heading1_line(input_wiki_text_line: str) -> str:
        """
        Check if an input wiki text line is the beginning of a heading1.
        If it is, return the chpater, otherwise None.
        """
        matched: Match[str] = \
            WikipediaDumpXmlProcessorRecordHeading1.\
                WIKI_HEADING1_REGULAR_EXPRESSION_PATTERN.search(\
                    input_wiki_text_line)
        if matched is None:
            return None
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     u'---- WikipediaDumpXmlProcessorRecordHeading1.process_heading1_line(), ' +
        #     'matched={}'.format(matched))
        # matched_span: Tuple[int, int] = matched.span()
        # matched_string: str = matched.string
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     '---- WikipediaDumpXmlProcessorRecordHeading1.process_heading1_line(), ' +
        #     'matched_string={}'.format(matched_string))
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     '---- WikipediaDumpXmlProcessorRecordHeading1.process_heading1_line(), ' +
        #     'matched_span.__repr__()={}'.format(matched_span.__repr__()))
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     '---- WikipediaDumpXmlProcessorRecordHeading1.process_heading1_line(), ' +
        #     'len(matched_span)={}'.format(len(matched_span)))
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     '---- WikipediaDumpXmlProcessorRecordHeading1.process_heading1_line(), ' +
        #     'matched_span[0]={}'.format(matched_span[0]))
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     '---- WikipediaDumpXmlProcessorRecordHeading1.process_heading1_line(), ' +
        #     'matched_span[1]={}'.format(matched_span[1]))
        matched_group_string: str = matched.group(1)
        # DebuggingHelper.write_line_to_system_console_out_debug( \
        #     '---- WikipediaDumpXmlProcessorRecordHeading1.process_heading1_line(), ' +
        #     'matched_group_string={}'.format(matched_group_string))
        return matched_group_string
class WikipediaDumpXmlProcessorRecordHeading1Factory:
    """
    A factor for processing WikipediaDumpXmlProcessorRecordHeading1 objects.
    """
    # ---- NOTE-PYLINT ---- R0903: Too few public methods
    # pylint: disable=R0903
    @staticmethod
    def process_wiki_heading1s(input_wiki_text_lines: List[str]) -> \
        List[WikipediaDumpXmlProcessorRecordHeading1]:
        """
        Iterate through a wiki's text lines and group the lines into groups, each a heading1.
        """
        output_wiki_heading1s: List[WikipediaDumpXmlProcessorRecordHeading1] = []
        current_heading1_lines: List[str] = []
        current_heading1_name: str = ''
        for input_wiki_text_line in input_wiki_text_lines:
            return_process_heading1_line: str = \
                WikipediaDumpXmlProcessorRecordHeading1.process_heading1_line(input_wiki_text_line)
            if return_process_heading1_line is not None:
                output_wiki_heading1: WikipediaDumpXmlProcessorRecordHeading1 = \
                    WikipediaDumpXmlProcessorRecordHeading1()
                output_wiki_heading2s: List[WikipediaDumpXmlProcessorRecordHeading2] = \
                    WikipediaDumpXmlProcessorRecordHeading2Factory.process_wiki_heading2s( \
                        current_heading1_lines)
                output_wiki_heading1.set_wiki_heading1( \
                    input_wiki_heading1=current_heading1_name, \
                    input_wiki_heading1_lines=current_heading1_lines, \
                    input_wiki_heading1_components=output_wiki_heading2s)
                output_wiki_heading1s.append(output_wiki_heading1)
                current_heading1_name = return_process_heading1_line
                current_heading1_lines = []
                # DebuggingHelper.write_line_to_system_console_out_debug( \
                #     '---- current_heading1_name={}'.format(current_heading1_name))
            else:
                current_heading1_lines.append(input_wiki_text_line)
        if len(current_heading1_lines) > 0:
            output_wiki_heading1: WikipediaDumpXmlProcessorRecordHeading1 = \
                WikipediaDumpXmlProcessorRecordHeading1()
            output_wiki_heading2s: List[WikipediaDumpXmlProcessorRecordHeading2] = \
                WikipediaDumpXmlProcessorRecordHeading2Factory.process_wiki_heading2s( \
                    current_heading1_lines)
            output_wiki_heading1.set_wiki_heading1( \
                input_wiki_heading1=current_heading1_name, \
                input_wiki_heading1_lines=current_heading1_lines, \
                input_wiki_heading1_components=output_wiki_heading2s)
            output_wiki_heading1s.append(output_wiki_heading1)
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- current_heading1_name={}'.format(current_heading1_name))
        return output_wiki_heading1s

class WikipediaDumpXmlProcessorRecordCommentExtraction( \
    WikipediaDumpXmlProcessorRecordExtractionBase[Any]):
    # ---- NOTE-PYLINT ---- C0301: Line too long
    # pylint: disable=C0301
    # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> Match[str]
    """
    A record for storing a processed comment_extraction structure.
    """
    WIKI_COMMENT_REGULAR_EXPRESSION_PATTERN = \
        re.compile("<!--(.*?)(-->|$)", re.MULTILINE|re.DOTALL)
    # ---- NOTE-PYLINT ---- C0301: Line too long
    # pylint: disable=C0301
    # ---- NOTE-NO-NEED ---- def __init__(self) -> NoReturn:
    # ---- NOTE-NO-NEED ----     """
    # ---- NOTE-NO-NEED ----     Initialize a WikipediaDumpXmlProcessorRecordCommentExtraction object.
    # ---- NOTE-NO-NEED ----     """
    # ---- NOTE-NO-NEED ----     super(WikipediaDumpXmlProcessorRecordCommentExtraction, self).__init__()
    def __repr__(self) -> str:
        """
        Return a representation of this object.
        """
        return self.to_json_in_string(json_configuration={})
    def to_json_in_string(self, \
        json_configuration: Any) -> str:
        """
        Convert this/self object to a JSON string.
        """
        return json.dumps( \
            ensure_ascii=False, \
            obj=self.to_json_friendly_structure( \
                json_configuration=json_configuration))
    def to_json_friendly_structure(self, \
        json_configuration: Any) -> Any:
        """
        Convert this/self object to a JSON-friendly structure.
        """
        json_object: Any = {}
        json_friendly_structure_to_output_text: bool = \
            json_configuration['json_friendly_structure_to_output_text_RecordCommentExtraction']
        json_friendly_structure_to_output_lines: bool = \
            json_configuration['json_friendly_structure_to_output_lines_RecordCommentExtraction']
        if json_friendly_structure_to_output_text:
            json_object['comment_extraction_text'] = \
                self.get_wiki_comment_extraction_text()
        if json_friendly_structure_to_output_lines:
            json_object['comment_extraction_lines'] = \
                self.get_wiki_comment_extraction_lines()
        json_object['comment_extraction_components'] = \
            [{'begin_position': x.span()[0], 'end_position': x.span()[1], 'text': x.group(0)} \
                for x in self.get_wiki_comment_extraction_components()]
        return json_object
    def set_wiki_comment_extraction(self, \
        input_wiki_comment_extraction_text: str, \
        input_wiki_comment_extraction_lines: List[str], \
        input_wiki_comment_extraction_components: List[Any]) -> NoReturn:
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> List[Match[str]]
        """
        Set the WikipediaDumpXmlProcessorRecordCommentExtraction data structure.
        """
        self.set_wiki_extraction_base( \
            input_wiki_extraction_base_text=input_wiki_comment_extraction_text, \
            input_wiki_extraction_base_text_processed=None, \
            input_wiki_extraction_base_lines=input_wiki_comment_extraction_lines, \
            input_wiki_extraction_base_components=input_wiki_comment_extraction_components)
    def get_wiki_comment_extraction_text(self) -> str:
        """
        Get the wiki_comment_extraction_text data structure.
        """
        return self.get_wiki_extraction_base_text()
    def get_wiki_comment_extraction_lines(self) -> List[str]:
        """
        Get the wiki_comment_extraction_lines data structure.
        """
        return self.get_wiki_extraction_base_lines()
    def get_wiki_comment_extraction_components(self) -> List[Any]:
        # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> List[Match[str]]:
        """
        Get the wiki_comment_extraction_components data structure.
        """
        return self.get_wiki_extraction_base_components()
    def process_comment_extraction(self) -> NoReturn:
        """
        Process this WikipediaDumpXmlProcessorRecordCommentExtraction object,
        """
        wiki_comment_extraction_text: str = \
            self.get_wiki_comment_extraction_text()
        if wiki_comment_extraction_text is None:
            wiki_comment_extraction_lines: List[str] = \
                self.get_wiki_comment_extraction_lines()
            if wiki_comment_extraction_lines is not None:
                wiki_comment_extraction_text = \
                    WikipediaDumpXmlProcessorConstants.WIKI_LINES_JOIN_STRING.join( \
                        wiki_comment_extraction_lines)
        if wiki_comment_extraction_text is None:
            return
        wiki_comment_extraction_components: List[Match[str]] = \
            self.get_wiki_comment_extraction_components()
        current_position: int = 0
        while True:
            matched: Match[str] = \
                WikipediaDumpXmlProcessorRecordCommentExtraction.WIKI_COMMENT_REGULAR_EXPRESSION_PATTERN.\
                    search( \
                        string=wiki_comment_extraction_text, \
                        pos=current_position)
            if matched is None:
                break
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordCommentExtraction.process_comment_extraction(), ' +
            #     'matched={}'.format(matched))
            matched_span: Tuple[int, int] = matched.span()
            # matched_string: str = matched.string
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordCommentExtraction.process_comment_extraction(), ' +
            #     'matched_string={}'.format(matched_string))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordCommentExtraction.process_comment_extraction(), ' +
            #     'matched_span.__repr__()={}'.format(matched_span.__repr__()))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordCommentExtraction.process_comment_extraction(), ' +
            #     'len(matched_span)={}'.format(len(matched_span)))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordCommentExtraction.process_comment_extraction(), ' +
            #     'matched_span[0]={}'.format(matched_span[0]))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordCommentExtraction.process_comment_extraction(), ' +
            #     'matched_span[1]={}'.format(matched_span[1]))
            # matched_group_string: str = matched.group(0)
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordCommentExtraction.process_comment_extraction(), ' +
            #     'matched_group_string={}'.format(matched_group_string))
            wiki_comment_extraction_components.append(matched)
            current_position = matched_span[1]
        wiki_comment_extraction_segments: List[str] = []
        current_segment_offset: int = 0
        for matched in wiki_comment_extraction_components:
            matched_span: Tuple[int, int] = matched.span()
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordCommentExtraction.process_comment_extraction(), ' +
            #     'matched_span[0]={}'.format(matched_span[0]))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordCommentExtraction.process_comment_extraction(), ' +
            #     'matched_span[1]={}'.format(matched_span[1]))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordCommentExtraction.process_comment_extraction(), ' +
            #     'type(matched_span[0])={}'.format(type(matched_span[0])))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordCommentExtraction.process_comment_extraction(), ' +
            #     'type(matched_span[1])={}'.format(type(matched_span[1])))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- WikipediaDumpXmlProcessorRecordCommentExtraction.process_comment_extraction(), ' +
            #     'len(wiki_comment_extraction_text)={}'.format(len(wiki_comment_extraction_text)))
            current_segment_end: int = matched_span[0]
            wiki_comment_extraction_segments.append( \
                wiki_comment_extraction_text[current_segment_offset:current_segment_end])
            current_segment_offset = matched_span[1]
        if current_segment_offset < len(wiki_comment_extraction_text):
            wiki_comment_extraction_segments.append( \
                wiki_comment_extraction_text[current_segment_offset:])
        wiki_comment_extraction_segments_joined: str = \
            WikipediaDumpXmlProcessorConstants.WIKI_SEGMENTS_JOIN_STRING.join( \
                wiki_comment_extraction_segments)
        wiki_comment_extraction_lines: List[str] = \
            wiki_comment_extraction_segments_joined.splitlines()
        self.set_wiki_comment_extraction( \
            input_wiki_comment_extraction_text=wiki_comment_extraction_text, \
            input_wiki_comment_extraction_lines=wiki_comment_extraction_lines, \
            input_wiki_comment_extraction_components=wiki_comment_extraction_components)
class WikipediaDumpXmlProcessorRecordCommentExtractionFactory:
    """
    A factor for processing WikipediaDumpXmlProcessorRecordCommentExtraction objects.
    """
    # ---- NOTE-PYLINT ---- R0903: Too few public methods
    # pylint: disable=R0903
    @staticmethod
    def process_wiki_comment_extractions(input_wiki_text_lines: List[str]) -> \
        WikipediaDumpXmlProcessorRecordCommentExtraction:
        """
        Iterate through a wiki's text lines and group the lines into groups,
        each a comment_extraction.
        """
        output_wiki_comment_extraction: \
            WikipediaDumpXmlProcessorRecordCommentExtraction = \
                WikipediaDumpXmlProcessorRecordCommentExtraction()
        output_wiki_comment_extraction.set_wiki_comment_extraction(\
            None, \
            input_wiki_text_lines, \
            [])
        output_wiki_comment_extraction.process_comment_extraction()
        return output_wiki_comment_extraction

class WikipediaDumpXmlProcessorRecord:
    """
    A record for storing processed structure for a Wiki text.
    """
    def __init__(self) -> NoReturn:
        """
        Initialize a WikipediaDumpXmlProcessorRecord object.
        """
        self.wiki_text: str = None
        self.wiki_text_lines: List[str] = []
        self.wiki_heading1s: List[WikipediaDumpXmlProcessorRecordHeading1] = []
        self.wiki_comment_extractions: List[WikipediaDumpXmlProcessorRecordCommentExtraction] = []
        self.web_page_title: str = ""
    def __repr__(self) -> str:
        """
        Return a representation of this object.
        """
        return self.to_json_in_string(json_configuration={})
    def to_json_in_string(self, \
        json_configuration: Any) -> str:
        """
        Convert this/self object to a JSON string.
        """
        return json.dumps( \
            ensure_ascii=False, \
            obj=self.to_json_friendly_structure( \
                web_page_title=self.get_web_page_title(), \
                json_configuration=json_configuration))
    def to_json_friendly_structure(self, \
        web_page_title: str, \
        json_configuration: Any) -> Any:
        """
        Convert this/self object to a JSON-friendly structure.
        """
        json_object: Any = {}
        json_friendly_structure_to_output_text: bool = \
            json_configuration['json_friendly_structure_to_output_text_Record']
        json_friendly_structure_to_output_lines: bool = \
            json_configuration['json_friendly_structure_to_output_lines_Record']
        if json_friendly_structure_to_output_text:
            json_object['wiki_page_text'] = \
                self.get_wiki_text()
        if json_friendly_structure_to_output_lines:
            json_object['wiki_page_text_lines'] = \
                self.get_wiki_text_lines()
        json_object['web_page_title'] = \
            web_page_title
        json_object['web_page_heading1s'] = \
            [x.to_json_friendly_structure( \
                json_configuration=json_configuration) \
            for x in self.get_wiki_heading1s()]
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # ---- NOTE-PLACE-HOLDER-IGNORE-FOR-NOW ---- json_object['wiki_page_comment_extractions'] = \
        # ---- NOTE-PLACE-HOLDER-IGNORE-FOR-NOW ----     [x.to_json_friendly_structure( \
        # ---- NOTE-PLACE-HOLDER-IGNORE-FOR-NOW ----         json_configuration=json_configuration) \
        # ---- NOTE-PLACE-HOLDER-IGNORE-FOR-NOW ----     for x in self.get_wiki_comment_extractions()]
        return json_object
    def write_lines_to_file(self, \
        web_page_title: str, \
        writer: StreamReaderWriter) -> NoReturn:
        """
        Output lines hold in this object to a file.
        """
        title: str = web_page_title
        if not StringHelper.is_none_empty_or_whitespaces(title):
            writer.writelines([title, '\n'])
        for entry in self.get_wiki_heading1s():
            entry.write_lines_to_file(writer)
    def set_wiki_text(self, input_wiki_text) -> NoReturn:
        """
        Set the wiki_text data structure.
        """
        self.wiki_text = input_wiki_text
    def get_wiki_text(self) -> str:
        """
        Get the wiki_text data structure.
        """
        return self.wiki_text
    def set_wiki_text_lines(self, input_wiki_text_lines) -> NoReturn:
        """
        Set the wiki_text_lines data structure.
        """
        self.wiki_text_lines = input_wiki_text_lines
    def get_wiki_text_lines(self) -> List[str]:
        """
        Get the wiki_text_lines data structure.
        """
        return self.wiki_text_lines
    def set_wiki_heading1s(self, \
        input_wiki_heading1s: List[WikipediaDumpXmlProcessorRecordHeading1]) -> NoReturn:
        """
        Set the wiki_heading1s data structure.
        """
        self.wiki_heading1s = input_wiki_heading1s
    def get_wiki_heading1s(self) -> List[WikipediaDumpXmlProcessorRecordHeading1]:
        """
        Get the wiki_heading1s data structure.
        """
        return self.wiki_heading1s
    def set_wiki_comment_extractions(self, \
        input_wiki_comment_extractions: \
            List[WikipediaDumpXmlProcessorRecordCommentExtraction]) -> NoReturn:
        """
        Set the wiki_comment_extractions data structure.
        """
        self.wiki_comment_extractions = input_wiki_comment_extractions
    def get_wiki_comment_extractions(self) -> \
        List[WikipediaDumpXmlProcessorRecordCommentExtraction]:
        """
        Get the wiki_comment_extractions data structure.
        """
        return self.wiki_comment_extractions
    def set_web_page_title(self, input_web_page_title) -> NoReturn:
        """
        Set the web_page_title data structure.
        """
        self.web_page_title = input_web_page_title
    def get_web_page_title(self) -> str:
        """
        Get the web_page_title data structure.
        """
        return self.web_page_title
class WikipediaDumpXmlProcessorRecordFactory:
    """
    A factor for processing WikipediaDumpXmlProcessorRecord objects.
    """
    # ---- NOTE-PYLINT ---- R0903: Too few public methods
    # pylint: disable=R0903
    @staticmethod
    def process_wiki_text(record: WikipediaDumpXmlProcessorRecord, input_wiki_text: str) -> \
        WikipediaDumpXmlProcessorRecord:
        """
        Iterate through a wiki's text lines and group the lines into groups, each a heading1.
        """
        if record is None:
            return None
        if input_wiki_text is None:
            return record
        record.set_wiki_text(input_wiki_text)
        input_wiki_text_defined_spaces_removed: str = \
            StringHelper.replace_defined_spaces_with_space_except_newline_carriage_return( \
                input_wiki_text)
        output_text_lines: List[str] = \
            input_wiki_text_defined_spaces_removed.splitlines()
        record.set_wiki_text_lines(output_text_lines)
        output_wiki_comment_extraction: WikipediaDumpXmlProcessorRecordCommentExtraction = \
            WikipediaDumpXmlProcessorRecordCommentExtractionFactory.\
                process_wiki_comment_extractions( \
                    output_text_lines)
        record.set_wiki_comment_extractions([output_wiki_comment_extraction])
        output_text_lines = \
            output_wiki_comment_extraction.get_wiki_comment_extraction_lines()
        output_wiki_heading1s: List[WikipediaDumpXmlProcessorRecordHeading1] = \
            WikipediaDumpXmlProcessorRecordHeading1Factory.process_wiki_heading1s(output_text_lines)
        record.set_wiki_heading1s(input_wiki_heading1s=output_wiki_heading1s)
        return record

class WikipediaDumpXmlProcessor:
    # ---- NOTE-PYLINT ---- C0301: Line too long
    # pylint: disable=C0301
    """
    This class can process a Wikipedia dump file.
        REFERENCE:
            https://www.mediawiki.org/wiki/MediaWiki
            https://www.mediawiki.org/wiki/Download
            https://www.mediawiki.org/w/index.php?title=Manual:Database_layout/diagram&action=render
            https://meta.wikimedia.org/wiki/Data_dumps/Other_tools
            https://meta.wikimedia.org/wiki/WikiXRay
            https://en.wikipedia.org/wiki/Wikipedia:Redirect
            https://en.wikipedia.org/wiki/Help:Redirect
            https://en.wikipedia.org/wiki/Wikipedia:Templates
            https://en.wikipedia.org/wiki/Help:Template
            https://en.wikipedia.org/wiki/Wikipedia:Revision
            <!-- https://en.wikipedia.org/wiki/Help:Revision -->
            https://en.wikipedia.org/wiki/Help:Page_history
            https://en.wikipedia.org/wiki/Wikipedia:Namespace
            https://en.wikipedia.org/wiki/Wikipedia:NS

            https://www.mediawiki.org/wiki/Preprocessor_ABNF
            ; START = start of string
            ; END = end of string
            ; LINE-START = start of line
            ; LINE-END = end of line
            ;
            ; The string starts with LINE-START. An LF input produces the tokens
            ; LINE-END LF LINE-START, and the string ends with LINE-END.
            ;
            ; The starting symbol of the grammar is wikitext-L1.

            xml-char = %x9 / %xA / %xD / %x20-D7FF / %xE000-FFFD / %x10000-10FFFF
            sptab = SP / HTAB

            ; everything except ">" (%x3E)
            attr-char = %x9 / %xA / %xD / %x20-3D / %x3F-D7FF / %xE000-FFFD / %x10000-10FFFF

            literal         = *xml-char
            title           = wikitext-L3
            part-name       = wikitext-L3
            part-value      = wikitext-L3
            part            = ( part-name "=" part-value ) / ( part-value )
            parts           = [ title *( "|" part ) ]
            tplarg          = "{{{" parts "}}}"
            template        = "{{" parts "}}"
            link            = "[[" wikitext-L3 "]]"

            comment         = "<!--" literal "-->"
            unclosed-comment = "<!--" literal END
            ; the + in the line-eating-comment rule was absent between MW 1.12 and MW 1.22
            line-eating-comment = LF LINE-START *SP +( comment *SP ) LINE-END

            attr            = *attr-char
            nowiki-element  = "<nowiki" attr ( "/>" / ( ">" literal ( "</nowiki>" / END ) ) )
            ; ...and similar rules added by XML-style extensions.

            xmlish-element  = nowiki-element / ... extensions ...

            heading = LINE-START heading-inner [ *sptab comment ] *sptab LINE-END

            heading-inner   =       "=" wikitext-L3 "="                /
                                    "==" wikitext-L3 "=="              /
                                    "===" wikitext-L3 "==="            /
                                    "====" wikitext-L3 "===="          /
                                    "=====" wikitext-L3 "====="        /
                                    "======" wikitext-L3 "======"

            ; wikitext-L1 is a simple proxy to wikitext-L2, except in inclusion mode, where it
            ; has a role in <onlyinclude> syntax (see below)
            wikitext-L1     = wikitext-L2 / *wikitext-L1
            wikitext-L2     = heading / wikitext-L3 / *wikitext-L2
            wikitext-L3     = literal / template / tplarg / link / comment /
                              line-eating-comment / unclosed-comment / xmlish-element /
                              *wikitext-L3

            https://www.mediawiki.org/xml/export-0.11.xsd
            Wikipedia XML XSD:
            <!--
                This is an XML Schema description of the format
                output by MediaWiki's Special:Export system.

                Version 0.2 adds optional basic file upload info support,
                which is used by our OAI export/import submodule.

                Version 0.3 adds some site configuration information such
                as a list of defined namespaces.

                Version 0.4 adds per-revision delete flags, log exports,
                discussion threading data, a per-page redirect flag, and
                per-namespace capitalization.

                Version 0.5 adds byte count per revision.

                Version 0.6 adds a separate namespace tag, and resolves the
                redirect target and adds a separate sha1 tag for each revision.

                Version 0.7 adds a unique identity constraint for both page and
                revision identifiers. See also bug 4220.
                Fix type for <ns> from "positiveInteger" to "nonNegativeInteger" to allow 0
                Moves <logitem> to its right location.
                Add parentid to revision.
                Fix type for <id> within <contributor> to "nonNegativeInteger"

                Version 0.8 adds support for a <model> and a <format> tag for
                each revision. See contenthandler.md.

                Version 0.9 adds the database name to the site information.

                Version 0.10 moved the <model> and <format> tags before the <text> tag.

                Version 0.11 introduced <content> tag.

                The canonical URL to the schema document is:
                http://www.mediawiki.org/xml/export-0.11.xsd

                Use the namespace:
                http://www.mediawiki.org/xml/export-0.11/
             -->
            <schema xmlns="http://www.w3.org/2001/XMLSchema" xmlns:mw="http://www.mediawiki.org/xml/export-0.11/" targetNamespace="http://www.mediawiki.org/xml/export-0.11/" elementFormDefault="qualified">
                <annotation>
                    <documentation xml:lang="en"> MediaWiki's page export format </documentation>
                </annotation>
                <!--  Need this to reference xml:lang  -->
                <import namespace="http://www.w3.org/XML/1998/namespace" schemaLocation="http://www.w3.org/2001/xml.xsd"/>
                <!--  Our root element  -->
                <element name="mediawiki" type="mw:MediaWikiType">
                    <!--  Page ID contraint, see bug 4220  -->
                    <unique name="PageIDConstraint">
                        <selector xpath="mw:page"/>
                        <field xpath="mw:id"/>
                    </unique>
                    <!--  Revision ID contraint, see bug 4220  -->
                    <unique name="RevIDConstraint">
                        <selector xpath="mw:page/mw:revision"/>
                        <field xpath="mw:id"/>
                    </unique>
                </element>
                <complexType name="MediaWikiType">
                    <sequence>
                        <element name="siteinfo" type="mw:SiteInfoType" minOccurs="0" maxOccurs="1"/>
                        <element name="page" type="mw:PageType" minOccurs="0" maxOccurs="unbounded"/>
                        <element name="logitem" type="mw:LogItemType" minOccurs="0" maxOccurs="unbounded"/>
                    </sequence>
                    <attribute name="version" type="string" use="required"/>
                    <attribute ref="xml:lang" use="required"/>
                </complexType>
                <complexType name="SiteInfoType">
                    <sequence>
                        <element name="sitename" type="string" minOccurs="0"/>
                        <element name="dbname" type="string" minOccurs="0"/>
                        <element name="base" type="anyURI" minOccurs="0"/>
                        <element name="generator" type="string" minOccurs="0"/>
                        <element name="case" type="mw:CaseType" minOccurs="0"/>
                        <element name="namespaces" type="mw:NamespacesType" minOccurs="0"/>
                    </sequence>
                </complexType>
                <simpleType name="CaseType">
                    <restriction base="NMTOKEN">
                        <!--  Cannot have two titles differing only by case of first letter.  -->
                        <!--  Default behavior through 1.5, $wgCapitalLinks = true  -->
                        <enumeration value="first-letter"/>
                        <!--  Complete title is case-sensitive  -->
                        <!--  Behavior when $wgCapitalLinks = false  -->
                        <enumeration value="case-sensitive"/>
                        <!--  Cannot have non-case senstitive titles eg [[FOO]] == [[Foo]]  -->
                        <!--  Not yet implemented as of MediaWiki 1.18  -->
                        <enumeration value="case-insensitive"/>
                    </restriction>
                </simpleType>
                <simpleType name="DeletedFlagType">
                    <restriction base="NMTOKEN">
                        <enumeration value="deleted"/>
                    </restriction>
                </simpleType>
                <complexType name="NamespacesType">
                    <sequence>
                        <element name="namespace" type="mw:NamespaceType" minOccurs="0" maxOccurs="unbounded"/>
                    </sequence>
                </complexType>
                <complexType name="NamespaceType">
                    <simpleContent>
                        <extension base="string">
                            <attribute name="key" type="integer"/>
                            <attribute name="case" type="mw:CaseType"/>
                        </extension>
                    </simpleContent>
                </complexType>
                <complexType name="RedirectType">
                    <simpleContent>
                        <extension base="string">
                            <attribute name="title" type="string"/>
                        </extension>
                    </simpleContent>
                </complexType>
                <simpleType name="ContentModelType">
                    <restriction base="string">
                        <pattern value="[a-zA-Z][-+./a-zA-Z0-9]*"/>
                    </restriction>
                </simpleType>
                <simpleType name="ContentFormatType">
                    <restriction base="string">
                        <pattern value="[a-zA-Z][-+.a-zA-Z0-9]*/[a-zA-Z][-+.a-zA-Z0-9]*"/>
                    </restriction>
                </simpleType>
                <complexType name="PageType">
                    <sequence>
                        <!--  Title in text form. (Using spaces, not underscores; with namespace )  -->
                        <element name="title" type="string"/>
                        <!--  Namespace in canonical form  -->
                        <element name="ns" type="nonNegativeInteger"/>
                        <!--  optional page ID number  -->
                        <element name="id" type="positiveInteger"/>
                        <!--  flag if the current revision is a redirect  -->
                        <element name="redirect" type="mw:RedirectType" minOccurs="0" maxOccurs="1"/>
                        <!--  comma-separated list of string tokens, if present  -->
                        <element name="restrictions" type="string" minOccurs="0"/>
                        <!--  Zero or more sets of revision or upload data  -->
                        <choice minOccurs="0" maxOccurs="unbounded">
                            <element name="revision" type="mw:RevisionType"/>
                            <element name="upload" type="mw:UploadType"/>
                        </choice>
                        <!--  Zero or One sets of discussion threading data  -->
                        <element name="discussionthreadinginfo" minOccurs="0" maxOccurs="1" type="mw:DiscussionThreadingInfo"/>
                    </sequence>
                </complexType>
                <complexType name="RevisionType">
                    <sequence>
                        <element name="id" type="positiveInteger"/>
                        <element name="parentid" type="positiveInteger" minOccurs="0" maxOccurs="1"/>
                        <element name="timestamp" type="dateTime"/>
                        <element name="contributor" type="mw:ContributorType"/>
                        <element name="minor" minOccurs="0" maxOccurs="1"/>
                        <element name="comment" type="mw:CommentType"/>
                        <!--  corresponds to slot origin for the main slot  -->
                        <element name="origin" type="positiveInteger"/>
                        <!--  the main slot's content model  -->
                        <element name="model" type="mw:ContentModelType"/>
                        <!--  the main slot's serialization format  -->
                        <element name="format" type="mw:ContentFormatType"/>
                        <!--  the main slot's serialized content  -->
                        <element name="text" type="mw:TextType"/>
                        <element name="content" type="mw:ContentType" minOccurs="0" maxOccurs="unbounded"/>
                        <!--  sha1 of the revision, a combined sha1 of content in all slots  -->
                        <element name="sha1" type="string"/>
                    </sequence>
                </complexType>
                <complexType name="ContentType">
                    <sequence>
                        <!--  corresponds to slot role_name  -->
                        <element name="role" type="mw:SlotRoleType"/>
                        <!--  corresponds to slot origin  -->
                        <element name="origin" type="positiveInteger"/>
                        <element name="model" type="mw:ContentModelType"/>
                        <element name="format" type="mw:ContentFormatType"/>
                        <element name="text" type="mw:ContentTextType"/>
                    </sequence>
                </complexType>
                <simpleType name="SlotRoleType">
                    <restriction base="string">
                        <pattern value="[a-zA-Z][-+./a-zA-Z0-9]*"/>
                    </restriction>
                </simpleType>
                <complexType name="ContentTextType">
                    <simpleContent>
                        <extension base="string">
                            <attribute ref="xml:space" default="preserve"/>
                            <!--  This allows deleted=deleted on non-empty elements, but XSD is not omnipotent  -->
                            <attribute name="deleted" type="mw:DeletedFlagType"/>
                            <attribute name="location" type="anyURI"/>
                            <attribute name="sha1" type="string"/>
                            <attribute name="bytes" type="nonNegativeInteger"/>
                        </extension>
                    </simpleContent>
                </complexType>
                <complexType name="LogItemType">
                    <sequence>
                        <element name="id" type="positiveInteger"/>
                        <element name="timestamp" type="dateTime"/>
                        <element name="contributor" type="mw:ContributorType"/>
                        <element name="comment" type="mw:CommentType" minOccurs="0"/>
                        <element name="type" type="string"/>
                        <element name="action" type="string"/>
                        <element name="text" type="mw:LogTextType" minOccurs="0" maxOccurs="1"/>
                        <element name="logtitle" type="string" minOccurs="0" maxOccurs="1"/>
                        <element name="params" type="mw:LogParamsType" minOccurs="0" maxOccurs="1"/>
                    </sequence>
                </complexType>
                <complexType name="CommentType">
                    <simpleContent>
                        <extension base="string">
                        <!--  This allows deleted=deleted on non-empty elements, but XSD is not omnipotent  -->
                        <attribute name="deleted" type="mw:DeletedFlagType"/>
                        </extension>
                    </simpleContent>
                </complexType>
                <complexType name="TextType">
                    <simpleContent>
                        <extension base="string">
                            <attribute ref="xml:space" default="preserve"/>
                            <!--  This allows deleted=deleted on non-empty elements, but XSD is not omnipotent  -->
                            <attribute name="deleted" type="mw:DeletedFlagType"/>
                            <!--  This isn't a good idea; we should be using "ID" instead of "NMTOKEN"  -->
                            <!--  However, "NMTOKEN" is strictest definition that is both compatible with existing  -->
                            <!--  usage ([0-9]+) and with the "ID" type.  -->
                            <attribute name="id" type="NMTOKEN"/>
                            <attribute name="location" type="anyURI"/>
                            <attribute name="sha1" type="string"/>
                            <attribute name="bytes" type="nonNegativeInteger"/>
                        </extension>
                    </simpleContent>
                </complexType>
                <complexType name="LogTextType">
                    <simpleContent>
                    <extension base="string">
                    <!--  This allows deleted=deleted on non-empty elements, but XSD is not omnipotent  -->
                    <attribute name="deleted" type="mw:DeletedFlagType"/>
                    </extension>
                    </simpleContent>
                </complexType>
                <complexType name="LogParamsType">
                    <simpleContent>
                    <extension base="string">
                    <attribute ref="xml:space" default="preserve"/>
                    </extension>
                    </simpleContent>
                </complexType>
                <complexType name="ContributorType">
                    <sequence>
                    <element name="username" type="string" minOccurs="0"/>
                    <element name="id" type="nonNegativeInteger" minOccurs="0"/>
                    <element name="ip" type="string" minOccurs="0"/>
                    </sequence>
                    <!--  This allows deleted=deleted on non-empty elements, but XSD is not omnipotent  -->
                    <attribute name="deleted" type="mw:DeletedFlagType"/>
                </complexType>
                <complexType name="UploadType">
                    <sequence>
                        <!--  Revision-style data...  -->
                        <element name="timestamp" type="dateTime"/>
                        <element name="contributor" type="mw:ContributorType"/>
                        <element name="comment" type="string" minOccurs="0"/>
                        <!--  Filename. (Using underscores, not spaces. No 'File:' namespace marker.)  -->
                        <element name="filename" type="string"/>
                        <!--  URI at which this resource can be obtained  -->
                        <element name="src" type="anyURI"/>
                        <element name="size" type="positiveInteger"/>
                        <!--  TODO: add other metadata fields  -->
                    </sequence>
                </complexType>
                <!--  Discussion threading data for LiquidThreads  -->
                <complexType name="DiscussionThreadingInfo">
                    <sequence>
                        <element name="ThreadSubject" type="string"/>
                        <element name="ThreadParent" type="positiveInteger"/>
                        <element name="ThreadAncestor" type="positiveInteger"/>
                        <element name="ThreadPage" type="string"/>
                        <element name="ThreadID" type="positiveInteger"/>
                        <element name="ThreadAuthor" type="string"/>
                        <element name="ThreadEditStatus" type="string"/>
                        <element name="ThreadType" type="string"/>
                    </sequence>
                </complexType>
            </schema>

            https://www.mediawiki.org/xml/export-0.10.xsd
            Wikipedia XML XSD:
            <!--
                This is an XML Schema description of the format
                output by MediaWiki's Special:Export system.

                Version 0.2 adds optional basic file upload info support,
                which is used by our OAI export/import submodule.

                Version 0.3 adds some site configuration information such
                as a list of defined namespaces.

                Version 0.4 adds per-revision delete flags, log exports,
                discussion threading data, a per-page redirect flag, and
                per-namespace capitalization.

                Version 0.5 adds byte count per revision.

                Version 0.6 adds a separate namespace tag, and resolves the
                redirect target and adds a separate sha1 tag for each revision.

                Version 0.7 adds a unique identity constraint for both page and
                revision identifiers. See also bug 4220.
                Fix type for <ns> from "positiveInteger" to "nonNegativeInteger" to allow 0
                Moves <logitem> to its right location.
                Add parentid to revision.
                Fix type for <id> within <contributor> to "nonNegativeInteger"

                Version 0.8 adds support for a <model> and a <format> tag for
                each revision. See contenthandler.txt.

                Version 0.9 adds the database name to the site information.

                Version 0.10 moved the <model> and <format> tags before the <text> tag.

                The canonical URL to the schema document is:
                http://www.mediawiki.org/xml/export-0.10.xsd

                Use the namespace:
                http://www.mediawiki.org/xml/export-0.10/
             -->
            <schema xmlns="http://www.w3.org/2001/XMLSchema" xmlns:mw="http://www.mediawiki.org/xml/export-0.10/" targetNamespace="http://www.mediawiki.org/xml/export-0.10/" elementFormDefault="qualified">
                <annotation>
                    <documentation xml:lang="en"> MediaWiki's page export format </documentation>
                </annotation>
                <!--  Need this to reference xml:lang  -->
                <import namespace="http://www.w3.org/XML/1998/namespace" schemaLocation="http://www.w3.org/2001/xml.xsd"/>
                <!--  Our root element  -->
                <element name="mediawiki" type="mw:MediaWikiType">
                    <!--  Page ID contraint, see bug 4220  -->
                    <unique name="PageIDConstraint">
                        <selector xpath="mw:page"/>
                        <field xpath="mw:id"/>
                    </unique>
                    <!--  Revision ID contraint, see bug 4220  -->
                    <unique name="RevIDConstraint">
                        <selector xpath="mw:page/mw:revision"/>
                        <field xpath="mw:id"/>
                    </unique>
                </element>
                <complexType name="MediaWikiType">
                    <sequence>
                        <element name="siteinfo" type="mw:SiteInfoType" minOccurs="0" maxOccurs="1"/>
                        <element name="page" type="mw:PageType" minOccurs="0" maxOccurs="unbounded"/>
                        <element name="logitem" type="mw:LogItemType" minOccurs="0" maxOccurs="unbounded"/>
                    </sequence>
                    <attribute name="version" type="string" use="required"/>
                    <attribute ref="xml:lang" use="required"/>
                </complexType>
                <complexType name="SiteInfoType">
                    <sequence>
                        <element name="sitename" type="string" minOccurs="0"/>
                        <element name="dbname" type="string" minOccurs="0"/>
                        <element name="base" type="anyURI" minOccurs="0"/>
                        <element name="generator" type="string" minOccurs="0"/>
                        <element name="case" type="mw:CaseType" minOccurs="0"/>
                        <element name="namespaces" type="mw:NamespacesType" minOccurs="0"/>
                    </sequence>
                </complexType>
                <simpleType name="CaseType">
                    <restriction base="NMTOKEN">
                        <!--  Cannot have two titles differing only by case of first letter.  -->
                        <!--  Default behavior through 1.5, $wgCapitalLinks = true  -->
                        <enumeration value="first-letter"/>
                        <!--  Complete title is case-sensitive  -->
                        <!--  Behavior when $wgCapitalLinks = false  -->
                        <enumeration value="case-sensitive"/>
                        <!--  Cannot have non-case senstitive titles eg [[FOO]] == [[Foo]]  -->
                        <!--  Not yet implemented as of MediaWiki 1.18  -->
                        <enumeration value="case-insensitive"/>
                    </restriction>
                </simpleType>
                <simpleType name="DeletedFlagType">
                    <restriction base="NMTOKEN">
                        <enumeration value="deleted"/>
                    </restriction>
                </simpleType>
                <complexType name="NamespacesType">
                    <sequence>
                        <element name="namespace" type="mw:NamespaceType" minOccurs="0" maxOccurs="unbounded"/>
                    </sequence>
                </complexType>
                <complexType name="NamespaceType">
                    <simpleContent>
                        <extension base="string">
                        <attribute name="key" type="integer"/>
                        <attribute name="case" type="mw:CaseType"/>
                        </extension>
                    </simpleContent>
                </complexType>
                <complexType name="RedirectType">
                    <simpleContent>
                        <extension base="string">
                            <attribute name="title" type="string"/>
                        </extension>
                    </simpleContent>
                </complexType>
                <simpleType name="ContentModelType">
                    <restriction base="string">
                        <pattern value="[a-zA-Z][-+./a-zA-Z0-9]*"/>
                    </restriction>
                </simpleType>
                <simpleType name="ContentFormatType">
                    <restriction base="string">
                        <pattern value="[a-zA-Z][-+.a-zA-Z0-9]*/[a-zA-Z][-+.a-zA-Z0-9]*"/>
                    </restriction>
                </simpleType>
                <complexType name="PageType">
                    <sequence>
                        <!--  Title in text form. (Using spaces, not underscores; with namespace )  -->
                        <element name="title" type="string"/>
                        <!--  Namespace in canonical form  -->
                        <element name="ns" type="nonNegativeInteger"/>
                        <!--  optional page ID number  -->
                        <element name="id" type="positiveInteger"/>
                        <!--  flag if the current revision is a redirect  -->
                        <element name="redirect" type="mw:RedirectType" minOccurs="0" maxOccurs="1"/>
                        <!--  comma-separated list of string tokens, if present  -->
                        <element name="restrictions" type="string" minOccurs="0"/>
                        <!--  Zero or more sets of revision or upload data  -->
                        <choice minOccurs="0" maxOccurs="unbounded">
                            <element name="revision" type="mw:RevisionType"/>
                            <element name="upload" type="mw:UploadType"/>
                        </choice>
                        <!--  Zero or One sets of discussion threading data  -->
                        <element name="discussionthreadinginfo" minOccurs="0" maxOccurs="1" type="mw:DiscussionThreadingInfo"/>
                    </sequence>
                </complexType>
                <complexType name="RevisionType">
                    <sequence>
                        <element name="id" type="positiveInteger"/>
                        <element name="parentid" type="positiveInteger" minOccurs="0"/>
                        <element name="timestamp" type="dateTime"/>
                        <element name="contributor" type="mw:ContributorType"/>
                        <element name="minor" minOccurs="0" maxOccurs="1"/>
                        <element name="comment" type="mw:CommentType" minOccurs="0" maxOccurs="1"/>
                        <element name="model" type="mw:ContentModelType"/>
                        <element name="format" type="mw:ContentFormatType"/>
                        <element name="text" type="mw:TextType"/>
                        <element name="sha1" type="string"/>
                    </sequence>
                </complexType>
                <complexType name="LogItemType">
                    <sequence>
                        <element name="id" type="positiveInteger"/>
                        <element name="timestamp" type="dateTime"/>
                        <element name="contributor" type="mw:ContributorType"/>
                        <element name="comment" type="mw:CommentType" minOccurs="0"/>
                        <element name="type" type="string"/>
                        <element name="action" type="string"/>
                        <element name="text" type="mw:LogTextType" minOccurs="0" maxOccurs="1"/>
                        <element name="logtitle" type="string" minOccurs="0" maxOccurs="1"/>
                        <element name="params" type="mw:LogParamsType" minOccurs="0" maxOccurs="1"/>
                    </sequence>
                </complexType>
                <complexType name="CommentType">
                    <simpleContent>
                        <extension base="string">
                            <!--  This allows deleted=deleted on non-empty elements, but XSD is not omnipotent  -->
                            <attribute name="deleted" use="optional" type="mw:DeletedFlagType"/>
                        </extension>
                    </simpleContent>
                </complexType>
                <complexType name="TextType">
                    <simpleContent>
                        <extension base="string">
                            <attribute ref="xml:space" use="optional" default="preserve"/>
                            <!--  This allows deleted=deleted on non-empty elements, but XSD is not omnipotent  -->
                            <attribute name="deleted" use="optional" type="mw:DeletedFlagType"/>
                            <!--  This isn't a good idea; we should be using "ID" instead of "NMTOKEN"  -->
                            <!--  However, "NMTOKEN" is strictest definition that is both compatible with existing  -->
                            <!--  usage ([0-9]+) and with the "ID" type.  -->
                            <attribute name="id" type="NMTOKEN"/>
                            <attribute name="bytes" use="optional" type="nonNegativeInteger"/>
                        </extension>
                    </simpleContent>
                </complexType>
                <complexType name="LogTextType">
                    <simpleContent>
                        <extension base="string">
                            <!--  This allows deleted=deleted on non-empty elements, but XSD is not omnipotent  -->
                            <attribute name="deleted" use="optional" type="mw:DeletedFlagType"/>
                        </extension>
                    </simpleContent>
                </complexType>
                <complexType name="LogParamsType">
                    <simpleContent>
                        <extension base="string">
                            <attribute ref="xml:space" use="optional" default="preserve"/>
                        </extension>
                    </simpleContent>
                </complexType>
                <complexType name="ContributorType">
                    <sequence>
                        <element name="username" type="string" minOccurs="0"/>
                        <element name="id" type="nonNegativeInteger" minOccurs="0"/>
                        <element name="ip" type="string" minOccurs="0"/>
                    </sequence>
                    <!--  This allows deleted=deleted on non-empty elements, but XSD is not omnipotent  -->
                    <attribute name="deleted" use="optional" type="mw:DeletedFlagType"/>
                </complexType>
                <complexType name="UploadType">
                    <sequence>
                        <!--  Revision-style data...  -->
                        <element name="timestamp" type="dateTime"/>
                        <element name="contributor" type="mw:ContributorType"/>
                        <element name="comment" type="string" minOccurs="0"/>
                        <!--  Filename. (Using underscores, not spaces. No 'File:' namespace marker.)  -->
                        <element name="filename" type="string"/>
                        <!--  URI at which this resource can be obtained  -->
                        <element name="src" type="anyURI"/>
                        <element name="size" type="positiveInteger"/>
                        <!--  TODO: add other metadata fields  -->
                    </sequence>
                </complexType>
                <!--  Discussion threading data for LiquidThreads  -->
                <complexType name="DiscussionThreadingInfo">
                    <sequence>
                        <element name="ThreadSubject" type="string"/>
                        <element name="ThreadParent" type="positiveInteger"/>
                        <element name="ThreadAncestor" type="positiveInteger"/>
                        <element name="ThreadPage" type="string"/>
                        <element name="ThreadID" type="positiveInteger"/>
                        <element name="ThreadAuthor" type="string"/>
                        <element name="ThreadEditStatus" type="string"/>
                        <element name="ThreadType" type="string"/>
                    </sequence>
                </complexType>
            </schema>

            https://www.mediawiki.org/wiki/Help:Export#Export_format
            Wikipedia XML DTD:
                <!ELEMENT mediawiki (siteinfo,page*)>
                <!-- version contains the version number of the format (currently 0.3) -->
                <!ATTLIST mediawiki
                  version  CDATA  #REQUIRED
                  xmlns CDATA #FIXED "https://www.mediawiki.org/xml/export-0.3/"
                  xmlns:xsi CDATA #FIXED "http://www.w3.org/2001/XMLSchema-instance"
                  xsi:schemaLocation CDATA #FIXED
                    "https://www.mediawiki.org/xml/export-0.3/ https://www.mediawiki.org/xml/export-0.3.xsd"
                  xml:lang  CDATA #IMPLIED
                >
                <!ELEMENT siteinfo (sitename,base,generator,case,namespaces)>
                <!ELEMENT sitename (#PCDATA)>      <!-- name of the wiki -->
                <!ELEMENT base (#PCDATA)>          <!-- url of the main page -->
                <!ELEMENT generator (#PCDATA)>     <!-- MediaWiki version string -->
                <!ELEMENT case (#PCDATA)>          <!-- how cases in page names are handled -->
                   <!-- possible values: 'first-letter' | 'case-sensitive'
                                         'case-insensitive' option is reserved for future -->
                <!ELEMENT namespaces (namespace+)> <!-- list of namespaces and prefixes -->
                  <!ELEMENT namespace (#PCDATA)>     <!-- contains namespace prefix -->
                  <!ATTLIST namespace key CDATA #REQUIRED> <!-- internal namespace number -->
                <!ELEMENT page (title,id?,restrictions?,(revision|upload)*)>
                  <!ELEMENT title (#PCDATA)>         <!-- Title with namespace prefix -->
                  <!ELEMENT id (#PCDATA)>
                  <!ELEMENT restrictions (#PCDATA)>  <!-- optional page restrictions -->
                <!ELEMENT revision (id?,timestamp,contributor,minor?,comment?,text)>
                  <!ELEMENT timestamp (#PCDATA)>     <!-- according to ISO8601 -->
                  <!ELEMENT minor EMPTY>             <!-- minor flag -->
                  <!ELEMENT comment (#PCDATA)>
                  <!ELEMENT text (#PCDATA)>          <!-- Wikisyntax -->
                  <!ATTLIST text xml:space CDATA  #FIXED "preserve">
                <!ELEMENT contributor ((username,id) | ip)>
                  <!ELEMENT username (#PCDATA)>
                  <!ELEMENT ip (#PCDATA)>
                <!ELEMENT upload (timestamp,contributor,comment?,filename,src,size)>
                  <!ELEMENT filename (#PCDATA)>
                  <!ELEMENT src (#PCDATA)>
                  <!ELEMENT size (#PCDATA)>

            Namespaces:
                <namespaces>
                  <namespace key="-2" case="first-letter">Media</namespace>
                  <namespace key="-1" case="first-letter">Special</namespace>
                  <namespace key="0" case="first-letter" />
                  <namespace key="1" case="first-letter">Talk</namespace>
                  <namespace key="2" case="first-letter">User</namespace>
                  <namespace key="3" case="first-letter">User talk</namespace>
                  <namespace key="4" case="first-letter">Wikipedia</namespace>
                  <namespace key="5" case="first-letter">Wikipedia talk</namespace>
                  <namespace key="6" case="first-letter">File</namespace>
                  <namespace key="7" case="first-letter">File talk</namespace>
                  <namespace key="8" case="first-letter">MediaWiki</namespace>
                  <namespace key="9" case="first-letter">MediaWiki talk</namespace>
                  <namespace key="10" case="first-letter">Template</namespace>
                  <namespace key="11" case="first-letter">Template talk</namespace>
                  <namespace key="12" case="first-letter">Help</namespace>
                  <namespace key="13" case="first-letter">Help talk</namespace>
                  <namespace key="14" case="first-letter">Category</namespace>
                  <namespace key="15" case="first-letter">Category talk</namespace>
                  <namespace key="100" case="first-letter">Portal</namespace>
                  <namespace key="101" case="first-letter">Portal talk</namespace>
                  <namespace key="102" case="first-letter">WikiProject</namespace>
                  <namespace key="103" case="first-letter">WikiProject talk</namespace>
                  <namespace key="118" case="first-letter">Draft</namespace>
                  <namespace key="119" case="first-letter">Draft talk</namespace>
                  <namespace key="828" case="first-letter">Module</namespace>
                  <namespace key="829" case="first-letter">Module talk</namespace>
                  <namespace key="2300" case="first-letter">Gadget</namespace>
                  <namespace key="2301" case="first-letter">Gadget talk</namespace>
                  <namespace key="2302" case="case-sensitive">Gadget definition</namespace>
                  <namespace key="2303" case="case-sensitive">Gadget definition talk</namespace>
                  <namespace key="2600" case="first-letter">Topic</namespace>
                </namespaces>
    """
    # ---- NOTE-PYLINT ---- R0903: Too few public methods
    # pylint: disable=R0903
    def __init__(self, \
        wikipedia_input_process_path: str, \
        wikipedia_dump_xml_filename: str, \
        wikipedia_output_process_path: str):
        """
        Initialize a WikipediaDumpXmlProcessor object.
        """
        self.wikipedia_input_process_path = wikipedia_input_process_path
        self.wikipedia_dump_xml_filename = wikipedia_dump_xml_filename
        self.wikipedia_output_process_path = wikipedia_output_process_path

    def process(self, \
        json_configuration: Any, \
        number_pages_processed_for_break: int = -1, \
        number_pages_processed_for_progress_update: int = 10000, \
        debugging_title: str = None, \
        debugging_title_alternatives: Set[str] = None, \
        debugging_text_substring: str = None):
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # ---- NOTE-PYLINT ---- R0912: Too many branches (28/12) (too-many-branches)
        # pylint: disable=R0912
        # ---- NOTE-PYLINT ---- R0913: Too many arguments
        # pylint: disable=R0913
        # ---- NOTE-PYLINT ---- R0914: Too many local variables (43/15) (too-many-locals)
        # pylint: disable=R0914
        # ---- NOTE-PYLINT ---- R0915: Too many statements (109/50) (too-many-statements)
        # pylint: disable=R0915
        """
        Process the input Wikipedia dump file and create processed outputs.
        """
        # --------------------------------------------------------------------
        encoding: str = "utf-8"
        # --------------------------------------------------------------------
        processed_output_filename_article_redirects: str = 'article_redirects.tsv'
        processed_output_filename_article_revisions: str = 'article_revisions.tsv'
        processed_output_filename_templates: str = 'article_templates.tsv'
        processed_output_filename_article_redirect_json_entry: str = 'article_redirect_entry_{}_{}.json'
        processed_output_filename_article_revision_json_entry: str = 'article_revision_entry_{}_{}.json'
        processed_output_filename_template_json_entry: str = 'article_template_entry_{}_{}.json'
        processed_output_filename_article_redirect_txt_entry: str = 'article_redirect_entry_{}_{}.txt'
        processed_output_filename_article_revision_txt_entry: str = 'article_revision_entry_{}_{}.txt'
        processed_output_filename_template_txt_entry: str = 'article_template_entry_{}_{}.txt'
        processed_output_filename_article_redirect_exception_entry: str = 'article_redirect_exception_entry_{}_{}.txt'
        processed_output_filename_article_revision_exception_entry: str = 'article_revision_exception_entry_{}_{}.txt'
        processed_output_filename_template_exception_entry: str = 'article_template_entry_exception_{}_{}.txt'
        wikipedia_dump_xml_path: str = \
            os.path.join(self.wikipedia_input_process_path, self.wikipedia_dump_xml_filename)
        processed_output_path_article_redirects: str = \
            os.path.join(self.wikipedia_output_process_path, processed_output_filename_article_redirects)
        processed_output_path_article_revisions: str = \
            os.path.join(self.wikipedia_output_process_path, processed_output_filename_article_revisions)
        processed_output_path_templates: str = \
            os.path.join(self.wikipedia_output_process_path, processed_output_filename_templates)
        # --------------------------------------------------------------------
        count_total_pages: int = 0
        count_article_pages: int = 0
        count_redirect_pages: int = 0
        count_template_pages: int = 0
        count_article_page_rows_written: int = 0
        count_redirect_page_rows_written: int = 0
        count_template_page_rows_written: int = 0
        # --------------------------------------------------------------------
        count_article_page_revisions: int = 0
        # --------------------------------------------------------------------
        current_page_title: str = None
        current_page_ns: int = None
        current_page_id: int = None
        current_page_redirect_title: str = None
        current_page_restrictions: str = None
        current_page_revision_ids: List[int] = []
        current_page_revision_parent_ids: List[int] = []
        current_page_revision_texts: List[str] = []
        current_page_revision_timestamps: List[str] = []
        current_revision_id: int = None
        current_revision_parent_id: int = None
        current_revision_timestamp: str = None
        flag_is_in_page: bool = False
        flag_page_is_in_revision: bool = False
        # --------------------------------------------------------------------
        process_start_time = time.time()
        # --------------------------------------------------------------------
        if not os.path.exists(self.wikipedia_output_process_path):
            os.makedirs(self.wikipedia_output_process_path)
        with codecs.open(filename=processed_output_path_article_redirects, mode="w", encoding=encoding) as \
                processed_output_file_handle_article_redirects, \
            codecs.open(filename=processed_output_path_article_revisions, mode="w", encoding=encoding) as \
                processed_output_file_handle_article_revisions, \
            codecs.open(filename=processed_output_path_templates, mode="w", encoding=encoding) as \
                processed_output_file_handle_templates:
            templates_writer: csv.writer = csv.writer( \
                processed_output_file_handle_templates, \
                quoting=csv.QUOTE_MINIMAL, delimiter='\t')
            article_redirects_writer: csv.writer = csv.writer( \
                processed_output_file_handle_article_redirects, \
                quoting=csv.QUOTE_MINIMAL, delimiter='\t')
            article_revisions_writer: csv.writer = csv.writer( \
                processed_output_file_handle_article_revisions, \
                quoting=csv.QUOTE_MINIMAL, delimiter='\t')
            templates_writer.writerow([
                'row_index',
                'id',
                'ns',
                'title',
                'text_heading1s',
                'text_lines',
                'restricitions'])
            article_redirects_writer.writerow([
                'row_index',
                'id',
                'ns',
                'revision_id',
                'revision_parent_id',
                'title',
                'redirect_title',
                'revision_timestamp',
                'text_heading1s',
                'text_lines',
                'restricitions'])
            article_revisions_writer.writerow([
                'row_index',
                'id',
                'ns',
                'revision_id',
                'revision_parent_id',
                'title',
                'revision_timestamp',
                'text_heading1s',
                'text_lines',
                'restricitions'])
            # ---- NOTE-PYLINT ---- R1702: Too many nested blocks (6/5) (too-many-nested-blocks)
            # pylint: disable=R1702
            # ---- NOTE-FOR-REFERENCE ---- def iterparse(source, events=None, parser=None):
            # ---- NOTE-FOR-REFERENCE ----     """Incrementally parse XML document into ElementTree.
            # ---- NOTE-FOR-REFERENCE ----
            # ---- NOTE-FOR-REFERENCE ----     This class also reports what's going on to the user based on the
            # ---- NOTE-FOR-REFERENCE ----     *events* it is initialized with.  The supported events are the strings
            # ---- NOTE-FOR-REFERENCE ----     "start", "end", "start-ns" and "end-ns" (the "ns" events are used to get
            # ---- NOTE-FOR-REFERENCE ----     detailed namespace information).  If *events* is omitted, only
            # ---- NOTE-FOR-REFERENCE ----     "end" events are reported.
            # ---- NOTE-FOR-REFERENCE ----
            # ---- NOTE-FOR-REFERENCE ----     *source* is a filename or file object containing XML data, *events* is
            # ---- NOTE-FOR-REFERENCE ----     a list of events to report back, *parser* is an optional parser instance.
            # ---- NOTE-FOR-REFERENCE ----
            # ---- NOTE-FOR-REFERENCE ----     Returns an iterator providing (event, elem) pairs.
            # ---- NOTE-FOR-REFERENCE ----
            # ---- NOTE-FOR-REFERENCE ----     """
            with bz2.open(filename=wikipedia_dump_xml_path, mode='rt', encoding=encoding) if wikipedia_dump_xml_path.endswith('.bz2') else codecs.open(filename=wikipedia_dump_xml_path, mode='r', encoding=encoding) as wikipedia_dump_xml_fd:
                for event, element in etree.iterparse(wikipedia_dump_xml_fd, events=('start', 'end')):
                    tag_name: str = WikipediaDumpXmlProcessorHelperFunctions.strip_tag_name(element.tag)
                    if event == 'start': # ---- 'start' event
                        if tag_name == 'page':
                            current_page_title = ''
                            current_page_ns = 0
                            current_page_id = -1
                            current_page_redirect_title = ''
                            current_page_restrictions = ''
                            current_page_revision_ids = []
                            current_page_revision_parent_ids = []
                            current_page_revision_texts = []
                            current_page_revision_timestamps = []
                            current_revision_id = -1
                            current_revision_parent_id = -1
                            current_revision_timestamp = ''
                            flag_is_in_page = True
                            flag_page_is_in_revision = False
                        elif tag_name == 'revision':
                            current_page_revision_ids = []
                            current_page_revision_parent_ids = []
                            current_page_revision_texts = []
                            current_page_revision_timestamps = []
                            current_revision_id = -1
                            current_revision_parent_id = -1
                            current_revision_timestamp = ''
                            flag_page_is_in_revision = True
                    else: # ---- 'end' event
                        if tag_name == 'page':
                            flag_is_in_page = False
                            count_total_pages += 1
                            # ----------------------------------------------------
                            # ---- NOTE ---- this area is used to control limited output for debugging purpose.
                            to_output: bool = True
                            to_output = ((debugging_title is None) and (debugging_title_alternatives is None) and (debugging_text_substring is None)) or \
                                ((debugging_title is not None) and (current_page_title == debugging_title)) or \
                                ((debugging_title_alternatives is not None) and (current_page_title in debugging_title_alternatives)) or \
                                ((debugging_text_substring is not None) and StringHelper.is_in_substring(debugging_text_substring, current_page_revision_texts))
                            # ----------------------------------------------------
                            if to_output:
                                current_page_has_redirect_title: bool = len(current_page_redirect_title) > 0
                                if current_page_ns == 10: # ---- ns=10 is a template page.
                                    count_template_pages += 1
                                    for index in range(len(current_page_revision_ids)):
                                        current_page_revision_text: str = current_page_revision_texts[index]
                                        record: WikipediaDumpXmlProcessorRecord = \
                                            WikipediaDumpXmlProcessorRecord()
                                        try:
                                            WikipediaDumpXmlProcessorRecordFactory.process_wiki_text( \
                                                record=record,
                                                input_wiki_text=current_page_revision_text)
                                        except RuntimeError:
                                            DebuggingHelper.write_line_to_system_console_out(\
                                                "==== EXCEPTION-THROWN-WikipediaDumpXmlProcessorRecordFactory.process_wiki_text: current_page_id={}".format(\
                                                current_page_id))
                                            processed_output_path_template_exception_entry: str = \
                                                os.path.join( \
                                                    self.wikipedia_output_process_path, \
                                                    processed_output_filename_template_exception_entry.format(count_template_page_rows_written, current_page_id))
                                            with codecs.open( \
                                                filename=processed_output_path_template_exception_entry, \
                                                mode="w", \
                                                encoding=encoding) as processed_output_filename_template_exception_entry_writer:
                                                processed_output_filename_template_exception_entry_writer.write(current_page_revision_text)
                                            continue # ---- NOTE ---- ignore this case
                                        record_json_friendly_structure: Any = \
                                            record.to_json_friendly_structure( \
                                                web_page_title=current_page_title, \
                                                json_configuration=json_configuration)
                                        try:
                                            templates_writer.writerow(\
                                                [count_template_page_rows_written, \
                                                current_page_id, \
                                                current_page_ns, \
                                                current_page_title, \
                                                record_json_friendly_structure, \
                                                record.get_wiki_text_lines(), \
                                                current_page_restrictions])
                                        except:
                                            DebuggingHelper.write_line_to_system_console_out(\
                                                "==== EXCEPTION-THROWN-templates_writer: current_page_id={}".format(\
                                                current_page_id))
                                            raise
                                        processed_output_path_template_json_entry: str = \
                                            os.path.join( \
                                                self.wikipedia_output_process_path, \
                                                processed_output_filename_template_json_entry.format(count_template_page_rows_written, current_page_id))
                                        with codecs.open( \
                                            filename=processed_output_path_template_json_entry, \
                                            mode="w", \
                                            encoding=encoding) as processed_output_filename_template_json_entry_writer:
                                            json.dump( \
                                                ensure_ascii=False, \
                                                obj=record_json_friendly_structure, \
                                                fp=processed_output_filename_template_json_entry_writer, \
                                                indent=2)
                                        processed_output_path_template_txt_entry: str = \
                                            os.path.join( \
                                                self.wikipedia_output_process_path, \
                                                processed_output_filename_template_txt_entry.format(count_template_page_rows_written, current_page_id))
                                        with codecs.open( \
                                            filename=processed_output_path_template_txt_entry, \
                                            mode="w", \
                                            encoding=encoding) as processed_output_filename_template_txt_entry_writer:
                                            record.write_lines_to_file( \
                                                web_page_title=current_page_title, \
                                                writer=processed_output_filename_template_txt_entry_writer)
                                        count_template_page_rows_written += 1
                                        # ---- NOTE-FOR-DEBUGGING ---- templates_writer.writerow(current_page_revision_text)
                                elif current_page_has_redirect_title:
                                    count_redirect_pages += 1
                                    # ---- NOTE-PYLINT ---- C0200: Consider using enumerate instead of iterating with range and len (consider-using-enumerate)
                                    # pylint: disable=C0200
                                    for index in range(len(current_page_revision_ids)):
                                        current_page_revision_text: str = current_page_revision_texts[index]
                                        record: WikipediaDumpXmlProcessorRecord = \
                                            WikipediaDumpXmlProcessorRecord()
                                        try:
                                            WikipediaDumpXmlProcessorRecordFactory.process_wiki_text( \
                                                record=record,
                                                input_wiki_text=current_page_revision_text)
                                        except RuntimeError:
                                            DebuggingHelper.write_line_to_system_console_out(\
                                                "==== EXCEPTION-THROWN-WikipediaDumpXmlProcessorRecordFactory.process_wiki_text: current_page_id={}".format(\
                                                current_page_id))
                                            processed_output_path_article_redirect_exception_entry: str = \
                                                os.path.join( \
                                                    self.wikipedia_output_process_path, \
                                                    processed_output_filename_article_redirect_exception_entry.format(count_redirect_page_rows_written, current_page_id))
                                            with codecs.open( \
                                                filename=processed_output_path_article_redirect_exception_entry, \
                                                mode="w", \
                                                encoding=encoding) as processed_output_filename_article_redirect_exception_entry_writer:
                                                processed_output_filename_article_redirect_exception_entry_writer.write(current_page_revision_text)
                                            continue # ---- NOTE ---- ignore this case
                                        record_json_friendly_structure: Any = \
                                            record.to_json_friendly_structure( \
                                                web_page_title=current_page_title, \
                                                json_configuration=json_configuration)
                                        try:
                                            article_redirects_writer.writerow(\
                                                [count_redirect_page_rows_written, \
                                                current_page_id, \
                                                current_page_ns, \
                                                current_page_revision_ids[index], \
                                                current_page_revision_parent_ids[index], \
                                                current_page_title, \
                                                current_page_redirect_title, \
                                                current_page_revision_timestamps[index], \
                                                record_json_friendly_structure, \
                                                record.get_wiki_text_lines(), \
                                                current_page_restrictions])
                                        except:
                                            DebuggingHelper.write_line_to_system_console_out(\
                                                "==== EXCEPTION-THROWN-article_redirects_writer: current_page_id={}".format(\
                                                current_page_id))
                                            raise
                                        processed_output_path_article_redirect_json_entry: str = \
                                            os.path.join( \
                                                self.wikipedia_output_process_path, \
                                                processed_output_filename_article_redirect_json_entry.format(count_redirect_page_rows_written, current_page_id))
                                        with codecs.open( \
                                            filename=processed_output_path_article_redirect_json_entry, \
                                            mode="w", \
                                            encoding=encoding) as processed_output_filename_article_redirect_json_entry_writer:
                                            json.dump( \
                                                ensure_ascii=False, \
                                                obj=record_json_friendly_structure, \
                                                fp=processed_output_filename_article_redirect_json_entry_writer, \
                                                indent=2)
                                        processed_output_path_article_redirect_txt_entry: str = \
                                            os.path.join( \
                                                self.wikipedia_output_process_path, \
                                                processed_output_filename_article_redirect_txt_entry.format(count_redirect_page_rows_written, current_page_id))
                                        with codecs.open( \
                                            filename=processed_output_path_article_redirect_txt_entry, \
                                            mode="w", \
                                            encoding=encoding) as processed_output_filename_article_redirect_txt_entry_writer:
                                            record.write_lines_to_file( \
                                                web_page_title=current_page_title, \
                                                writer=processed_output_filename_article_redirect_txt_entry_writer)
                                        count_redirect_page_rows_written += 1
                                        # ---- NOTE-FOR-DEBUGGING ---- article_redirects_writer.writerow(current_page_revision_text)
                                else:
                                    count_article_pages += 1
                                    # ---- NOTE-PYLINT ---- C0200: Consider using enumerate instead of iterating with range and len (consider-using-enumerate)
                                    # pylint: disable=C0200
                                    for index in range(len(current_page_revision_ids)):
                                        current_page_revision_text: str = current_page_revision_texts[index]
                                        # DebuggingHelper.write_line_to_system_console_out_debug( \
                                        #     '---- current_page_revision_text={}'.format(current_page_revision_text))
                                        record: WikipediaDumpXmlProcessorRecord = \
                                            WikipediaDumpXmlProcessorRecord()
                                        try:
                                            WikipediaDumpXmlProcessorRecordFactory.process_wiki_text( \
                                                record=record,
                                                input_wiki_text=current_page_revision_text)
                                        except RuntimeError:
                                            DebuggingHelper.write_line_to_system_console_out(\
                                                "==== EXCEPTION-THROWN-WikipediaDumpXmlProcessorRecordFactory.process_wiki_text: current_page_id={}".format(\
                                                current_page_id))
                                            processed_output_path_article_revision_exception_entry: str = \
                                                os.path.join( \
                                                    self.wikipedia_output_process_path, \
                                                    processed_output_filename_article_revision_exception_entry.format(count_article_page_rows_written, current_page_id))
                                            with codecs.open( \
                                                filename=processed_output_path_article_revision_exception_entry, \
                                                mode="w", \
                                                encoding=encoding) as processed_output_filename_article_revision_exception_entry_writer:
                                                processed_output_filename_article_revision_exception_entry_writer.write(current_page_revision_text)
                                            continue # ---- NOTE ---- ignore this case
                                        record_json_friendly_structure: Any = \
                                            record.to_json_friendly_structure( \
                                                web_page_title=current_page_title, \
                                                json_configuration=json_configuration)
                                        try:
                                            article_revisions_writer.writerow(\
                                                [count_article_page_rows_written, \
                                                current_page_id, \
                                                current_page_ns, \
                                                current_page_revision_ids[index], \
                                                current_page_revision_parent_ids[index], \
                                                current_page_title, \
                                                current_page_revision_timestamps[index], \
                                                record_json_friendly_structure, \
                                                record.get_wiki_text_lines(), \
                                                current_page_restrictions])
                                        except:
                                            DebuggingHelper.write_line_to_system_console_out(\
                                                "==== EXCEPTION-THROWN-article_revisions_writer: current_page_id={}".format(\
                                                current_page_id))
                                            raise
                                        processed_output_path_article_revision_json_entry: str = \
                                            os.path.join( \
                                                self.wikipedia_output_process_path, \
                                                processed_output_filename_article_revision_json_entry.format(count_article_page_rows_written, current_page_id))
                                        with codecs.open( \
                                            filename=processed_output_path_article_revision_json_entry, \
                                            mode="w", \
                                            encoding=encoding) as processed_output_filename_article_revision_json_entry_writer:
                                            json.dump( \
                                                ensure_ascii=False, \
                                                obj=record_json_friendly_structure, \
                                                fp=processed_output_filename_article_revision_json_entry_writer, \
                                                indent=2)
                                        processed_output_path_article_revision_txt_entry: str = \
                                            os.path.join( \
                                                self.wikipedia_output_process_path, \
                                                processed_output_filename_article_revision_txt_entry.format(count_article_page_rows_written, current_page_id))
                                        with codecs.open( \
                                            filename=processed_output_path_article_revision_txt_entry, \
                                            mode="w", \
                                            encoding=encoding) as processed_output_filename_article_revision_txt_entry_writer:
                                            record.write_lines_to_file( \
                                                web_page_title=current_page_title, \
                                                writer=processed_output_filename_article_revision_txt_entry_writer)
                                        count_article_page_rows_written += 1
                                        # ---- NOTE-FOR-DEBUGGING ---- article_revisions_writer.writerow(current_page_revision_text)
                            if (number_pages_processed_for_progress_update > 0) and \
                                ((count_total_pages % number_pages_processed_for_progress_update) == 0):
                                DebuggingHelper.write_line_to_system_console_out(\
                                    "---- number of pages processed: {}".format(count_total_pages))
                        if flag_is_in_page:
                            if flag_page_is_in_revision:
                                if tag_name == 'parentid': # ---- RevisionType element
                                    current_revision_parent_id = int(element.text)
                                elif tag_name == 'timestamp': # ---- RevisionType element
                                    current_revision_timestamp = element.text
                                elif tag_name == 'text': # ---- RevisionType element
                                    current_revision_text: str = element.text
                                    if current_revision_text is not None:
                                        count_article_page_revisions += 1
                                        current_page_revision_ids.append(current_revision_id)
                                        current_page_revision_parent_ids.append(current_revision_parent_id)
                                        current_page_revision_texts.append(current_revision_text)
                                        current_page_revision_timestamps.append(current_revision_timestamp)
                            if tag_name == 'title': # ---- PageType element
                                current_page_title = element.text
                            elif tag_name == 'ns': # ---- PageType element
                                current_page_ns = int(element.text)
                            elif tag_name == 'id': # ---- PageType or RevisionType element
                                if flag_page_is_in_revision:
                                    current_revision_id = int(element.text)
                                else:
                                    current_page_id = int(element.text)
                            elif tag_name == 'redirect': # ---- PageType element
                                current_page_redirect_title = element.attrib['title']
                            elif tag_name == 'restrictions': # ---- PageType element
                                current_page_restrictions = element.text
                            elif tag_name == 'revision': # ---- PageType element
                                flag_page_is_in_revision = False
                        # ---- NOTE-PYLINT ---- R1716: Simplify chained comparison between the operands (chained-comparison)
                        # pylint: disable=R1716
                        if (number_pages_processed_for_break >= 0) and \
                            (count_total_pages >= number_pages_processed_for_break):
                            break
                        element.clear() # ---- NOTE ---- need to clear the element, otherwise the system might run out of memory.
        process_duration = time.time() - process_start_time
        DebuggingHelper.write_line_to_system_console_out(\
            "---- total number of pages processed: {}".format(\
            count_total_pages))
        DebuggingHelper.write_line_to_system_console_out(\
            "---- total number of article pages processed: {}".format(\
            count_article_pages))
        DebuggingHelper.write_line_to_system_console_out(\
            "---- total number of redirect pages processed: {}".format(\
            count_redirect_pages))
        DebuggingHelper.write_line_to_system_console_out(\
            "---- total number of template pages processed: {}".format(\
            count_template_pages))
        DebuggingHelper.write_line_to_system_console_out(\
            "---- total number of article page rows written: {}".format(\
            count_article_page_rows_written))
        DebuggingHelper.write_line_to_system_console_out(\
            "---- total number of redirect page rows written: {}".format(\
            count_redirect_page_rows_written))
        DebuggingHelper.write_line_to_system_console_out(\
            "---- total number of template page rows written: {}".format(\
            count_template_page_rows_written))
        DebuggingHelper.write_line_to_system_console_out(\
            "---- total number of article revisions processed: {}".format(\
            count_article_page_revisions))
        DebuggingHelper.write_line_to_system_console_out(\
            "---- total processing time: {}".format(\
            WikipediaDumpXmlProcessorHelperFunctions.seconds_to_string(process_duration)))
