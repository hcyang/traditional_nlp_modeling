# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
This module is for TextPiece for parsing textual input.
"""
# ---- NOTE-PYLINT ---- C0302: Too many lines in module
# pylint: disable=C0302

from typing import Any
# from typing import Dict
from typing import Generic
from typing import List
from typing import NoReturn
# from typing import Set
from typing import Tuple
from typing import TypeVar

from re import Match

# import codecs
# import csv
import functools
import json
# import os
import re
# import time

# from utility.string_helper.string_helper \
#     import StringHelper

from utility.debugging_helper.debugging_helper \
    import DebuggingHelper

TextPieceMetadata = TypeVar("TextPieceMetadata")

class TextPieceConstants:
    """
    Some constants used by TextPiece objects.
    """
    # ---- NOTE-PYLINT ---- R0903: Too few public methods
    # pylint: disable=R0903
    TEXT_PIECE_EMPTY_STRING: str = ''
    TEXT_PIECE_SPACE_STRING: str = ' '

class InterfaceTextPiece:
    """
    InterfaceTextPiece is an interface with a function to return
    a piece of text that is either a plain sub-text, or an
    extracted text captured with some meta-data.
    """
    # ---- NOTE-PYLINT ---- R0903: Too few public methods
    # pylint: disable=R0903
    def get_text_piece_string_from_input_text(self, input_text: str) -> str:
        """
        This function must be overridden by a child.
        """
        # ---- NOTE-PYLINT ---- R0201: Method could be a function (no-self-use)
        # pylint: disable=R0201
        # ---- NOTE-PYLINT ---- W0613: Unused argument
        # pylint: disable=W0613
        DebuggingHelper.ensure(
            condition=False,
            message='should be implemented by a child')
        # ---- NOTE-PLACE-HOLDER ---- pass

# ---- NOTE-TODO ----
class TextPieceSpan(InterfaceTextPiece):
    """
    TextPieceSpan represents a span of text that is either a plain sub-text, or an
    extracted text captured with some meta-data.
    """
    def __init__(self, \
        offset: int, \
        length: int) -> None:
        self.set_span( \
            offset=offset, \
            length=length)
    def get_span(self) -> Tuple[int, int]:
        """
        Return the span as a tuple.
        """
        return tuple(self.get_span_offset(), self.get_span_length())
    def get_span_offset(self) -> int:
        """
        Return the span offset.
        """
        return self.__offset
    def get_span_length(self) -> int:
        """
        Return the span length.
        """
        return self.__length
    def set_span(self, \
        offset: int, \
        length: int) -> NoReturn:
        """
        Set the span offset and length.
        """
        self.__offset = offset
        self.__length = length
    def get_text_piece_string_from_input_text(self, input_text: str) -> str:
        """
        Get the text piece sub-text using the stored span information.
        """
        if input_text is None:
            return None
        span_offset: int = self.get_span_offset()
        span_end_position: int = span_offset + self.get_span_length()
        return input_text[span_offset:span_end_position]
    def is_raw_text(self) -> bool:
        """
        Return whether this object contains raw text or not. Should be only True for the
        base TextPieceSpan object and False for all the children.
        """
        # ---- NOTE-PYLINT ---- R0201: Method could be a function (no-self-use)
        # pylint: disable=R0201
        return True

class TextPiece(InterfaceTextPiece):
    """
    TextPiece represents a piece of text that is either a plain sub-text, or an
    extracted text captured with some meta-data.
    """
    def __init__(self, \
        raw_text_string: str) -> None:
        self.set_raw_text_string(raw_text_string=raw_text_string)
    def __repr__(self) -> str:
        """
        Return a representation of a TextPiece object.
        """
        return self.to_json_in_string()
    def to_json_in_string(self) -> str:
        """
        Convert this/self object to a JSON string.
        """
        return json.dumps( \
            ensure_ascii=False, \
            obj=self.to_json_friendly_structure())
    def to_json_friendly_structure(self) -> Any:
        """
        Convert this/self object to a JSON-friendly structure.
        """
        json_object: Any = {}
        json_object['length_raw_text_string'] = \
            len(self.get_raw_text_string())
        json_object['length_text_piece_string'] = \
            len(self.get_text_piece_string())
        json_object['raw_text_string'] = \
            self.get_raw_text_string()
        json_object['text_piece_string'] = \
            self.get_text_piece_string()
        json_object['is_raw_text'] = \
            self.is_raw_text()
        return json_object
    def get_raw_text_string(self) -> str:
        """
        Return the raw text this object covers.
        """
        return self.__raw_text_string
    def set_raw_text_string(self, raw_text_string: str) -> NoReturn:
        """
        Set the raw text this object covers.
        """
        self.__raw_text_string = raw_text_string
    def get_text_piece_string(self) -> str:
        """
        This function should be overridden by a child.
        In the most simple form, a TextPiece object simply represents the raw text it covers.
        """
        return self.get_text_piece_string_from_input_text(self.get_raw_text_string())
    def get_text_piece_string_from_input_text(self, input_text: str) -> str:
        """
        This function should be overridden by a child.
        In the most simple form, a TextPiece object simply represents the raw text it covers.
        """
        return input_text
    def is_raw_text(self) -> bool:
        """
        Return whether this object is raw text or not. Should be only True for the
        base TextPiece object and False for all the children.
        """
        # ---- NOTE-PYLINT ---- R0201: Method could be a function (no-self-use)
        # pylint: disable=R0201
        return True
class TextPieceWithMetadata( \
    TextPiece, \
    Generic[TextPieceMetadata]):
    """
    TextPieceWithMetadata represents a piece of text that is either a plain text, or an
    extracted text with some meta-data.
    """
    def __init__(self, \
        raw_text_string: str, \
        text_piece_meta_data: TextPieceMetadata = None) -> None:
        super(TextPieceWithMetadata, self).__init__( \
            raw_text_string=raw_text_string)
        self.set_text_piece_meta_data(text_piece_meta_data=text_piece_meta_data)
    def to_json_friendly_structure(self) -> Any:
        """
        Convert this/self object to a JSON-friendly structure.
        """
        json_object: Any = super().to_json_friendly_structure()
        json_object['text_piece_meta_data'] = \
            self.get_text_piece_meta_data()
        return json_object
    def get_text_piece_meta_data(self) -> TextPieceMetadata:
        """
        Return the meta data used to represent the text piece string.
        """
        return self.__text_piece_meta_data
    def set_text_piece_meta_data(self, text_piece_meta_data: TextPieceMetadata) -> NoReturn:
        """
        Set the meta data used to represent the text piece string.
        """
        self.__text_piece_meta_data = text_piece_meta_data
    def is_raw_text(self) -> bool:
        """
        Return whether this object contains raw text or not. Should be only True for the
        base TextPiece object and False for all the children.
        """
        return False
class TextPieceWithMetadataExtracted( \
    TextPieceWithMetadata[TextPieceMetadata], Generic[TextPieceMetadata]):
    """
    TextPieceWithMetadataFiltered is a TextPieceWithMetadata object that was extracted from a text.
    """
    def __init__(self, \
        raw_text_string: str, \
        extracted_text_string: str, \
        text_piece_meta_data: TextPieceMetadata = None) -> None:
        super(TextPieceWithMetadataExtracted, self).__init__( \
            raw_text_string=raw_text_string, \
            text_piece_meta_data=text_piece_meta_data)
        self.set_extracted_text_string(extracted_text_string=extracted_text_string)
    def get_text_piece_string(self) -> str:
        """
        As extracted text, return the extracted text.
        """
        return self.get_text_piece_string_from_input_text(self.get_extracted_text_string())
    def get_extracted_text_string(self) -> str:
        """
        As extracted text, return the extracted text.
        """
        return self.__extracted_text_string
    def set_extracted_text_string(self, extracted_text_string: str) -> NoReturn:
        """
        Set the extracted text.
        """
        self.__extracted_text_string = extracted_text_string
class TextPieceWithMetadataFiltered( \
    TextPieceWithMetadataExtracted[TextPieceMetadata], Generic[TextPieceMetadata]):
    """
    TextPieceWithMetadataFiltered is a TextPieceWithMetadata object that was filtered from a text,
    thus this TextPieceWithMetadata object should always return an empty string or
    a specified empty string.
    """
    def __init__(self, \
        raw_text_string: str, \
        text_piece_meta_data: TextPieceMetadata = None, \
        empty_text_string: str = TextPieceConstants.TEXT_PIECE_EMPTY_STRING) -> None:
        super(TextPieceWithMetadataFiltered, self).__init__( \
            raw_text_string=raw_text_string, \
            extracted_text_string=empty_text_string, \
            text_piece_meta_data=text_piece_meta_data)
class TextPieceWithMetadataExtractedWithRegularExpressionMatchObject(\
    TextPieceWithMetadataExtracted[Any]):
    # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> Match[str]
    """
    TextPieceWithMetadataExtractedWithRegularExpressionMatchObject is a
    TextPieceWithMetadataExtracted object.
    """
    # ---- NOTE-PYLINT ---- R0903: Too few public methods
    # pylint: disable=R0903
    def __init__(self, \
        raw_text_string: str, \
        extracted_text_string: str, \
        text_piece_meta_data: Any) -> None:
        # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> Match[str]
        super(TextPieceWithMetadataExtractedWithRegularExpressionMatchObject, self).__init__( \
            raw_text_string=raw_text_string, \
            extracted_text_string=extracted_text_string, \
            text_piece_meta_data=text_piece_meta_data)
class TextPieceWithMetadataFilteredWithRegularExpressionMatchObject( \
    TextPieceWithMetadataFiltered[Any]):
    # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> Match[str]
    """
    TextPieceWithMetadataFilteredWithRegularExpressionMatchObject is a
    TextPieceWithMetadataFiltered object.
    """
    # ---- NOTE-PYLINT ---- R0903: Too few public methods
    # pylint: disable=R0903
    def __init__(self, \
        raw_text_string: str, \
        text_piece_meta_data: Any, \
        empty_text_string: str = TextPieceConstants.TEXT_PIECE_EMPTY_STRING) -> None:
        # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> Match[str]
        super(TextPieceWithMetadataFilteredWithRegularExpressionMatchObject, self).__init__( \
            raw_text_string=raw_text_string, \
            text_piece_meta_data=text_piece_meta_data, \
            empty_text_string=empty_text_string)
class TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectWikiLink(\
    TextPieceWithMetadataExtractedWithRegularExpressionMatchObject):
    # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> Match[str]
    """
    TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectWikiLink is a
    TextPieceWithMetadataExtractedWithRegularExpressionMatchObject object.
    """
    def __init__(self, \
        raw_text_string: str, \
        text_piece_meta_data: Any) -> None:
        # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> Match[str]
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        super(TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectWikiLink, self).__init__( \
            raw_text_string=raw_text_string, \
            extracted_text_string=TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectWikiLink.extract_text_piece_string(raw_text_string), \
            text_piece_meta_data=text_piece_meta_data)
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # ---- NOTE-PLACE-HOLDER ---- extracted_text_string: str = None
        # ---- NOTE-PLACE-HOLDER ---- if self.get_text_piece_meta_data() is not None:
        # ---- NOTE-PLACE-HOLDER ----     extracted_text_string = \
        # ---- NOTE-PLACE-HOLDER ----         TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectWikiLink.extract_text_piece_string(raw_text_string)
        # ---- NOTE-PLACE-HOLDER ---- else:
        # ---- NOTE-PLACE-HOLDER ----     extracted_text_string = \
        # ---- NOTE-PLACE-HOLDER ----         TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectWikiLink.extract_open_text_piece_string(raw_text_string)
        # ---- NOTE-PLACE-HOLDER ---- self.set_extracted_text_string(extracted_text_string)
    def is_open(self) -> bool:
        """
        Return whether this TextPiece is open or not.
        """
        return self.get_text_piece_meta_data() is None
    @staticmethod
    def check_if_wiki_link(input_text: str) -> bool:
        """
        Check if the input string is a wiki link or not.
        """
        # ---- NOTE-PYLINT ---- R0911: Too many return statements
        # pylint: disable=R0911
        if input_text is None:
            return False
        if len(input_text) < 4:
            return False
        if input_text[0] != '[':
            return False
        if input_text[1] != '[':
            return False
        if input_text[-1] != ']':
            return False
        if input_text[-2] != ']':
            return False
        return True
    @staticmethod
    def check_if_open_wiki_link(input_text: str) -> bool:
        """
        Check if the input string is a wiki link or not.
        """
        if input_text is None:
            return False
        if len(input_text) < 2:
            return False
        if input_text[0] != '[':
            return False
        if input_text[1] != '[':
            return False
        # ---- NOTE-FOR-REFERENCE-ONLY ---- if input_text[-1] != ']':
        # ---- NOTE-FOR-REFERENCE-ONLY ----     return False
        # ---- NOTE-FOR-REFERENCE-ONLY ---- if input_text[-2] != ']':
        # ---- NOTE-FOR-REFERENCE-ONLY ----     return False
        return True
    @staticmethod
    def extract_text_piece_string(input_text: str) -> str:
        """
        Extract the text piece string from a wiki link.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        DebuggingHelper.ensure(
            condition=TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectWikiLink.check_if_wiki_link(input_text),
            message='The input ${}$ is not a Wiki Link'.format(input_text))
        input_text = input_text[2:len(input_text)-2]
        input_text_splitted: List[str] = input_text.split('|')
        if len(input_text_splitted) <= 0:
            return ''
        if len(input_text_splitted) > 2: # ---- NOTE ---- ignore a wiki link which standards alone and not part of a paragraph.
            return ''
        return input_text_splitted[-1]
    @staticmethod
    def extract_open_text_piece_string(input_text: str) -> str:
        """
        Extract the text piece string from a wiki link.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        DebuggingHelper.ensure(
            condition=TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectWikiLink.check_if_open_wiki_link(input_text),
            message='The input ${}$ is not a Wiki Link'.format(input_text))
        input_text = input_text[2:len(input_text)]
        input_text_splitted: List[str] = input_text.split('|')
        if len(input_text_splitted) <= 0:
            return ''
        return input_text_splitted[-1]
class TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectWikiEmphasisBoldItalic(\
    TextPieceWithMetadataExtractedWithRegularExpressionMatchObject):
    # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> Match[str]
    """
    TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectWikiEmphasisBoldItalic is a
    TextPieceWithMetadataExtractedWithRegularExpressionMatchObject object.
    """
    def __init__(self, \
        raw_text_string: str, \
        text_piece_meta_data: Any) -> None:
        # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> Match[str]
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        super(TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectWikiEmphasisBoldItalic, self).__init__( \
            raw_text_string=raw_text_string, \
            extracted_text_string=TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectWikiEmphasisBoldItalic.extract_text_piece_string(raw_text_string), \
            text_piece_meta_data=text_piece_meta_data)
    @staticmethod
    def check_if_wiki_emphasis_bold_italic(input_text: str) -> bool:
        """
        Check if the input string is a wiki link or not.
        """
        # ---- NOTE-PYLINT ---- R0911: Too many return statements
        # pylint: disable=R0911
        # ---- NOTE-PYLINT ---- R0912: Too many branches
        # pylint: disable=R0912
        if input_text is None:
            return False
        if len(input_text) < 12:
            return False
        if input_text[0] != '\'':
            return False
        if input_text[1] != '\'':
            return False
        if input_text[2] != '\'':
            return False
        if input_text[3] != '\'':
            return False
        if input_text[4] != '\'':
            return False
        if input_text[5] != '\'':
            return False
        if input_text[-1] != '\'':
            return False
        if input_text[-2] != '\'':
            return False
        if input_text[-3] != '\'':
            return False
        if input_text[-4] != '\'':
            return False
        if input_text[-5] != '\'':
            return False
        if input_text[-6] != '\'':
            return False
        return True
    @staticmethod
    def extract_text_piece_string(input_text: str) -> str:
        """
        Extract the text piece string from a wiki link.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        DebuggingHelper.ensure(
            condition=TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectWikiEmphasisBoldItalic.check_if_wiki_emphasis_bold_italic(input_text),
            message='The input ${}$ is not a Wiki Bold Italic text'.format(input_text))
        input_text = input_text[6:len(input_text)-6]
        return input_text
class TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectWikiEmphasisBold(\
    TextPieceWithMetadataExtractedWithRegularExpressionMatchObject):
    # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> Match[str]
    """
    TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectWikiEmphasisBold is a
    TextPieceWithMetadataExtractedWithRegularExpressionMatchObject object.
    """
    def __init__(self, \
        raw_text_string: str, \
        text_piece_meta_data: Any) -> None:
        # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> Match[str]
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        super(TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectWikiEmphasisBold, self).__init__( \
            raw_text_string=raw_text_string, \
            extracted_text_string=TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectWikiEmphasisBold.extract_text_piece_string(raw_text_string), \
            text_piece_meta_data=text_piece_meta_data)
    @staticmethod
    def check_if_wiki_emphasis_bold(input_text: str) -> bool:
        """
        Check if the input string is a wiki link or not.
        """
        # ---- NOTE-PYLINT ---- R0911: Too many return statements
        # pylint: disable=R0911
        if input_text is None:
            return False
        if len(input_text) < 6:
            return False
        if input_text[0] != '\'':
            return False
        if input_text[1] != '\'':
            return False
        if input_text[2] != '\'':
            return False
        if input_text[-1] != '\'':
            return False
        if input_text[-2] != '\'':
            return False
        if input_text[-3] != '\'':
            return False
        return True
    @staticmethod
    def extract_text_piece_string(input_text: str) -> str:
        """
        Extract the text piece string from a wiki link.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        DebuggingHelper.ensure(
            condition=TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectWikiEmphasisBold.check_if_wiki_emphasis_bold(input_text),
            message='The input ${}$ is not a Wiki Bold text'.format(input_text))
        input_text = input_text[3:len(input_text)-3]
        return input_text
class TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectChineseBookEnclosure(\
    TextPieceWithMetadataExtractedWithRegularExpressionMatchObject):
    # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> Match[str]
    """
    TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectChineseBookEnclosure is a
    TextPieceWithMetadataExtractedWithRegularExpressionMatchObject object.
    """
    def __init__(self, \
        raw_text_string: str, \
        text_piece_meta_data: Any) -> None:
        # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> Match[str]
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        super(TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectChineseBookEnclosure, self).__init__( \
            raw_text_string=raw_text_string, \
            extracted_text_string=TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectChineseBookEnclosure.extract_text_piece_string(raw_text_string), \
            text_piece_meta_data=text_piece_meta_data)
    @staticmethod
    def check_if_chinese_book_enclosure(input_text: str) -> bool:
        """
        Check if the input string is a Chinese book enclosure or not.
        """
        if input_text is None:
            return False
        if len(input_text) < 2:
            return False
        if input_text[0] != '《':
            return False
        if input_text[-1] != '》':
            return False
        return True
    @staticmethod
    def extract_text_piece_string(input_text: str) -> str:
        """
        Extract the text piece string from a Chinese book enclosure.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        DebuggingHelper.ensure(
            condition=TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectChineseBookEnclosure.check_if_chinese_book_enclosure(input_text),
            message='The input ${}$ is not a Chinese book enclosure text'.format(input_text))
        return input_text
class TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectChineseBookEnclosureSingle(\
    TextPieceWithMetadataExtractedWithRegularExpressionMatchObject):
    # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> Match[str]
    """
    TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectChineseBookEnclosureSingle is a
    TextPieceWithMetadataExtractedWithRegularExpressionMatchObject object.
    """
    def __init__(self, \
        raw_text_string: str, \
        text_piece_meta_data: Any) -> None:
        # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> Match[str]
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        super(TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectChineseBookEnclosureSingle, self).__init__( \
            raw_text_string=raw_text_string, \
            extracted_text_string=TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectChineseBookEnclosureSingle.extract_text_piece_string(raw_text_string), \
            text_piece_meta_data=text_piece_meta_data)
    @staticmethod
    def check_if_chinese_book_enclosure_single(input_text: str) -> bool:
        """
        Check if the input string is a Chinese book enclosure single or not.
        """
        if input_text is None:
            return False
        if len(input_text) < 2:
            return False
        if input_text[0] != '〈':
            return False
        if input_text[-1] != '〉':
            return False
        return True
    @staticmethod
    def extract_text_piece_string(input_text: str) -> str:
        """
        Extract the text piece string from a Chinese book enclosure.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        DebuggingHelper.ensure(
            condition=TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectChineseBookEnclosureSingle.check_if_chinese_book_enclosure_single(input_text),
            message='The input ${}$ is not a Chinese book enclosure single text'.format(input_text))
        return input_text

class TextPieceList(TextPiece):
    """
    A TextPieceList object processes an input raw text and
    create a List of TextPiece objects that can be referenced later.
    Itself is a TextPiece.
    """
    def __init__(self, \
        raw_text_string: str, \
        join_string: str = TextPieceConstants.TEXT_PIECE_EMPTY_STRING) -> None:
        super(TextPieceList, self).__init__(raw_text_string=raw_text_string)
        self.__text_piece_list: List[TextPiece] = \
            self.generate_text_piece_list_from_raw_text_string( \
                raw_text_string=self.get_raw_text_string())
        self.__text_piece_string: str = \
            self.join_text_piece_list_into_text_piece_string( \
                text_piece_list=self.get_text_piece_list(), \
                join_string=join_string)
    def get_text_piece_list(self) -> List[TextPiece]:
        """
        Return the processed TextPiece list.
        """
        return self.__text_piece_list
    def get_text_piece_string(self) -> str:
        """
        This function should be overridden by a child.
        In the most simple form, a TextPiece object simply represents the raw text it covers.
        """
        return self.get_text_piece_string_from_input_text(self.__text_piece_string)
    def generate_text_piece_list_from_raw_text_string(self, \
        raw_text_string: str) -> List[TextPiece]:
        """
        This function should be overridden by a child that the logic
        converts an input raw text into a list of TextPiece objects.
        In the most simple form, it simply return one TextPiece object
        with the input raw text in a list.
        """
        # ---- NOTE-PYLINT ---- R0201: Method could be a function (no-self-use)
        # pylint: disable=R0201
        return [TextPiece(raw_text_string)]
    def join_text_piece_list_into_text_piece_string(self, \
        text_piece_list: List[TextPiece], \
        join_string: str = TextPieceConstants.TEXT_PIECE_EMPTY_STRING) -> str:
        """
        This function should be overridden by a child that the logic
        converts a list of TextPiece objects into a single string.
        """
        # ---- NOTE-PYLINT ---- R0201: Method could be a function (no-self-use)
        # pylint: disable=R0201
        DebuggingHelper.ensure(
            condition=text_piece_list is not None,
            message='input "text_piece_list" argument should not be None')
        # DebuggingHelper.write_line_to_system_console_out_debug(
        #     f'text_piece_list={text_piece_list}')
        return functools.reduce( \
            lambda x, y: x + join_string + y.get_text_piece_string(), \
            text_piece_list, \
            '')
class TextPieceListExtractedChineseBookEnclosureSingle(TextPieceList):
    # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> Match[str]
    """
    TextPieceListExtractedChineseBookEnclosureSingle is a TextPieceList
    object representing
    a comment text that was filtered from a text.
    This object should always return an empty string or
    a specified empty string.
    """
    CHINESE_BOOK_ENCLOSURE_SINGLE_REGULAR_EXPRESSION_PATTERN = \
        re.compile("〈(.*?)〉", \
        re.MULTILINE|re.DOTALL)
    def __init__(self, \
        raw_text_string: str, \
        join_string: str = TextPieceConstants.TEXT_PIECE_EMPTY_STRING) -> None:
        super(TextPieceListExtractedChineseBookEnclosureSingle, self).__init__( \
            raw_text_string=raw_text_string, \
            join_string=join_string)
    def generate_text_piece_list_from_raw_text_string(self, \
        raw_text_string: str) -> List[TextPiece]:
        """
        Process an input raw_text_string and generate a list of TextPiece objects.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        current_position: int = 0
        text_piece_list: List[TextPiece] = []
        while True:
            matched: Match[str] = \
                TextPieceListExtractedChineseBookEnclosureSingle.CHINESE_BOOK_ENCLOSURE_SINGLE_REGULAR_EXPRESSION_PATTERN.\
                    search( \
                        string=raw_text_string, \
                        pos=current_position)
            if matched is None:
                break
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedChineseBookEnclosureSingle.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched={}'.format(matched))
            matched_span: Tuple[int, int] = matched.span()
            # matched_string: str = matched.string
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedChineseBookEnclosureSingle.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_string={}'.format(matched_string))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedChineseBookEnclosureSingle.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_span.__repr__()={}'.format(matched_span.__repr__()))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedChineseBookEnclosureSingle.generate_text_piece_list_from_raw_text_string(), ' +
            #     'len(matched_span)={}'.format(len(matched_span)))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedChineseBookEnclosureSingle.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_span[0]={}'.format(matched_span[0]))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedChineseBookEnclosureSingle.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_span[1]={}'.format(matched_span[1]))
            # matched_group_string: str = matched.group(0)
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedChineseBookEnclosureSingle.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_group_string={}'.format(matched_group_string))
            plain_sub_text_end_position: int = \
                matched_span[0]
            if plain_sub_text_end_position > current_position:
                text_piece_plain_sub_text: TextPiece = \
                    TextPiece(raw_text_string=raw_text_string[current_position:plain_sub_text_end_position])
                text_piece_list.append(text_piece_plain_sub_text)
            next_position: int = matched_span[1]
            text_piece_extracted_chinese_book_enclosure_single: TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectChineseBookEnclosureSingle = \
                TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectChineseBookEnclosureSingle( \
                    raw_text_string=raw_text_string[plain_sub_text_end_position:next_position], \
                    text_piece_meta_data=matched_span)
            text_piece_list.append(text_piece_extracted_chinese_book_enclosure_single)
            current_position = next_position
        raw_text_string_end_position: int = len(raw_text_string)
        if current_position < raw_text_string_end_position:
            text_piece_end_plain_sub_text: TextPiece = \
                TextPiece(raw_text_string=raw_text_string[current_position:raw_text_string_end_position])
            text_piece_list.append(text_piece_end_plain_sub_text)
        return text_piece_list
class TextPieceListExtractedChineseBookEnclosure(TextPieceList):
    # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> Match[str]
    """
    TextPieceListExtractedChineseBookEnclosure is a TextPieceList
    object representing
    a comment text that was filtered from a text.
    This object should always return an empty string or
    a specified empty string.
    """
    CHINESE_BOOK_ENCLOSURE_REGULAR_EXPRESSION_PATTERN = \
        re.compile("《(.*?)》", \
        re.MULTILINE|re.DOTALL)
    def __init__(self, \
        raw_text_string: str, \
        join_string: str = TextPieceConstants.TEXT_PIECE_EMPTY_STRING) -> None:
        super(TextPieceListExtractedChineseBookEnclosure, self).__init__( \
            raw_text_string=raw_text_string, \
            join_string=join_string)
    def generate_text_piece_list_from_raw_text_string(self, \
        raw_text_string: str) -> List[TextPiece]:
        """
        Process an input raw_text_string and generate a list of TextPiece objects.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        current_position: int = 0
        text_piece_list: List[TextPiece] = []
        while True:
            matched: Match[str] = \
                TextPieceListExtractedChineseBookEnclosure.CHINESE_BOOK_ENCLOSURE_REGULAR_EXPRESSION_PATTERN.\
                    search( \
                        string=raw_text_string, \
                        pos=current_position)
            if matched is None:
                break
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedChineseBookEnclosure.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched={}'.format(matched))
            matched_span: Tuple[int, int] = matched.span()
            # matched_string: str = matched.string
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedChineseBookEnclosure.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_string={}'.format(matched_string))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedChineseBookEnclosure.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_span.__repr__()={}'.format(matched_span.__repr__()))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedChineseBookEnclosure.generate_text_piece_list_from_raw_text_string(), ' +
            #     'len(matched_span)={}'.format(len(matched_span)))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedChineseBookEnclosure.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_span[0]={}'.format(matched_span[0]))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedChineseBookEnclosure.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_span[1]={}'.format(matched_span[1]))
            # matched_group_string: str = matched.group(0)
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedChineseBookEnclosure.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_group_string={}'.format(matched_group_string))
            plain_sub_text_end_position: int = \
                matched_span[0]
            if plain_sub_text_end_position > current_position:
                text_piece_plain_sub_text: TextPiece = \
                    TextPiece(raw_text_string=raw_text_string[current_position:plain_sub_text_end_position])
                text_piece_list.append(text_piece_plain_sub_text)
            next_position: int = matched_span[1]
            text_piece_extracted_chinese_book_enclosure: TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectChineseBookEnclosure = \
                TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectChineseBookEnclosure( \
                    raw_text_string=raw_text_string[plain_sub_text_end_position:next_position], \
                    text_piece_meta_data=matched_span)
            text_piece_list.append(text_piece_extracted_chinese_book_enclosure)
            current_position = next_position
        raw_text_string_end_position: int = len(raw_text_string)
        if current_position < raw_text_string_end_position:
            text_piece_end_plain_sub_text: TextPiece = \
                TextPiece(raw_text_string=raw_text_string[current_position:raw_text_string_end_position])
            text_piece_list.append(text_piece_end_plain_sub_text)
        return text_piece_list
class TextPieceListExtractedWikiEmphasisBold(TextPieceList):
    # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> Match[str]
    """
    TextPieceListExtractedWikiEmphasisBold is a TextPieceList
    object representing
    a comment text that was filtered from a text.
    This object should always return an empty string or
    a specified empty string.
    """
    WIKI_EMPHASIS_BOLD_REGULAR_EXPRESSION_PATTERN = \
        re.compile("'''(.*?)'''", \
        re.MULTILINE|re.DOTALL)
    def __init__(self, \
        raw_text_string: str, \
        join_string: str = TextPieceConstants.TEXT_PIECE_EMPTY_STRING) -> None:
        super(TextPieceListExtractedWikiEmphasisBold, self).__init__( \
            raw_text_string=raw_text_string, \
            join_string=join_string)
    def generate_text_piece_list_from_raw_text_string(self, \
        raw_text_string: str) -> List[TextPiece]:
        """
        Process an input raw_text_string and generate a list of TextPiece objects.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        current_position: int = 0
        text_piece_list: List[TextPiece] = []
        while True:
            matched: Match[str] = \
                TextPieceListExtractedWikiEmphasisBold.WIKI_EMPHASIS_BOLD_REGULAR_EXPRESSION_PATTERN.\
                    search( \
                        string=raw_text_string, \
                        pos=current_position)
            if matched is None:
                break
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedWikiEmphasisBold.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched={}'.format(matched))
            matched_span: Tuple[int, int] = matched.span()
            # matched_string: str = matched.string
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedWikiEmphasisBold.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_string={}'.format(matched_string))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedWikiEmphasisBold.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_span.__repr__()={}'.format(matched_span.__repr__()))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedWikiEmphasisBold.generate_text_piece_list_from_raw_text_string(), ' +
            #     'len(matched_span)={}'.format(len(matched_span)))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedWikiEmphasisBold.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_span[0]={}'.format(matched_span[0]))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedWikiEmphasisBold.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_span[1]={}'.format(matched_span[1]))
            # matched_group_string: str = matched.group(0)
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedWikiEmphasisBold.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_group_string={}'.format(matched_group_string))
            plain_sub_text_end_position: int = \
                matched_span[0]
            if plain_sub_text_end_position > current_position:
                text_piece_plain_sub_text: TextPiece = \
                    TextPiece(raw_text_string=raw_text_string[current_position:plain_sub_text_end_position])
                text_piece_list.append(text_piece_plain_sub_text)
            next_position: int = matched_span[1]
            text_piece_extracted_wiki_emphasis_bold: TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectWikiEmphasisBold = \
                TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectWikiEmphasisBold( \
                    raw_text_string=raw_text_string[plain_sub_text_end_position:next_position], \
                    text_piece_meta_data=matched_span)
            text_piece_list.append(text_piece_extracted_wiki_emphasis_bold)
            current_position = next_position
        raw_text_string_end_position: int = len(raw_text_string)
        if current_position < raw_text_string_end_position:
            text_piece_end_plain_sub_text: TextPiece = \
                TextPiece(raw_text_string=raw_text_string[current_position:raw_text_string_end_position])
            text_piece_list.append(text_piece_end_plain_sub_text)
        return text_piece_list
class TextPieceListExtractedWikiEmphasisBoldItalic(TextPieceList):
    # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> Match[str]
    """
    TextPieceListExtractedWikiEmphasisBoldItalic is a TextPieceList
    object representing
    a comment text that was filtered from a text.
    This object should always return an empty string or
    a specified empty string.
    """
    WIKI_EMPHASIS_BOLD_ITALIC_REGULAR_EXPRESSION_PATTERN = \
        re.compile("''''''(.*?)''''''", \
        re.MULTILINE|re.DOTALL)
    def __init__(self, \
        raw_text_string: str, \
        join_string: str = TextPieceConstants.TEXT_PIECE_EMPTY_STRING) -> None:
        super(TextPieceListExtractedWikiEmphasisBoldItalic, self).__init__( \
            raw_text_string=raw_text_string, \
            join_string=join_string)
    def generate_text_piece_list_from_raw_text_string(self, \
        raw_text_string: str) -> List[TextPiece]:
        """
        Process an input raw_text_string and generate a list of TextPiece objects.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        current_position: int = 0
        text_piece_list: List[TextPiece] = []
        while True:
            matched: Match[str] = \
                TextPieceListExtractedWikiEmphasisBoldItalic.WIKI_EMPHASIS_BOLD_ITALIC_REGULAR_EXPRESSION_PATTERN.\
                    search( \
                        string=raw_text_string, \
                        pos=current_position)
            if matched is None:
                break
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedWikiEmphasisBoldItalic.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched={}'.format(matched))
            matched_span: Tuple[int, int] = matched.span()
            # matched_string: str = matched.string
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedWikiEmphasisBoldItalic.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_string={}'.format(matched_string))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedWikiEmphasisBoldItalic.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_span.__repr__()={}'.format(matched_span.__repr__()))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedWikiEmphasisBoldItalic.generate_text_piece_list_from_raw_text_string(), ' +
            #     'len(matched_span)={}'.format(len(matched_span)))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedWikiEmphasisBoldItalic.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_span[0]={}'.format(matched_span[0]))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedWikiEmphasisBoldItalic.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_span[1]={}'.format(matched_span[1]))
            # matched_group_string: str = matched.group(0)
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedWikiEmphasisBoldItalic.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_group_string={}'.format(matched_group_string))
            plain_sub_text_end_position: int = \
                matched_span[0]
            if plain_sub_text_end_position > current_position:
                text_piece_plain_sub_text: TextPiece = \
                    TextPiece(raw_text_string=raw_text_string[current_position:plain_sub_text_end_position])
                text_piece_list.append(text_piece_plain_sub_text)
            next_position: int = matched_span[1]
            text_piece_extracted_wiki_emphasis_bold_italic: TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectWikiEmphasisBoldItalic = \
                TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectWikiEmphasisBoldItalic( \
                    raw_text_string=raw_text_string[plain_sub_text_end_position:next_position], \
                    text_piece_meta_data=matched_span)
            text_piece_list.append(text_piece_extracted_wiki_emphasis_bold_italic)
            current_position = next_position
        raw_text_string_end_position: int = len(raw_text_string)
        if current_position < raw_text_string_end_position:
            text_piece_end_plain_sub_text: TextPiece = \
                TextPiece(raw_text_string=raw_text_string[current_position:raw_text_string_end_position])
            text_piece_list.append(text_piece_end_plain_sub_text)
        return text_piece_list
class TextPieceListExtractedWikiLink(TextPieceList):
    # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> Match[str]
    """
    TextPieceListExtractedWikiLink is a TextPieceList
    object representing
    a comment text that was filtered from a text.
    This object should always return an empty string or
    a specified empty string.
    """
    WIKI_LINK_REGULAR_EXPRESSION_BEGINNING_STRING: str = '['
    WIKI_LINK_REGULAR_EXPRESSION_ENDING_STRING: str = ']'
    WIKI_LINK_REGULAR_EXPRESSION_PATTERN = \
        re.compile("(\\[\\[|\\]\\])")
    def __init__(self, \
        raw_text_string: str, \
        join_string: str = TextPieceConstants.TEXT_PIECE_EMPTY_STRING) -> None:
        super(TextPieceListExtractedWikiLink, self).__init__( \
            raw_text_string=raw_text_string, \
            join_string=join_string)
    def generate_text_piece_list_from_raw_text_string(self, \
        raw_text_string: str) -> List[TextPiece]:
        """
        Process an input raw_text_string and generate a list of TextPiece objects.
        """
        # ---- NOTE-PYLINT ---- R0914: Too many local variables
        # pylint: disable=R0914
        # ---- NOTE-PYLINT ---- R0912: Too many branches
        # pylint: disable=R0912
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        current_position: int = 0
        text_piece_list: List[TextPiece] = []
        wiki_link_extraction_components_parts: List[Match[str]] = []
        while True:
            matched: Match[str] = \
                TextPieceListExtractedWikiLink.WIKI_LINK_REGULAR_EXPRESSION_PATTERN.\
                    search( \
                        string=raw_text_string, \
                        pos=current_position)
            if matched is None:
                break
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedWikiLink.process_link_extraction(), ' +
            #     'matched={}'.format(matched))
            matched_span: Tuple[int, int] = matched.span()
            # matched_string: str = matched.string
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedWikiLink.process_link_extraction(), ' +
            #     'matched_string={}'.format(matched_string))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedWikiLink.process_link_extraction(), ' +
            #     'matched_span.__repr__()={}'.format(matched_span.__repr__()))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedWikiLink.process_link_extraction(), ' +
            #     'len(matched_span)={}'.format(len(matched_span)))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedWikiLink.process_link_extraction(), ' +
            #     'matched_span[0]={}'.format(matched_span[0]))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedWikiLink.process_link_extraction(), ' +
            #     'matched_span[1]={}'.format(matched_span[1]))
            # matched_group_string: str = matched.group(0)
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedWikiLink.process_link_extraction(), ' +
            #     'matched_group_string={}'.format(matched_group_string))
            wiki_link_extraction_components_parts.append(matched)
            current_position = matched_span[1]
        current_recursion_level: int = 0
        current_segment_offset: int = 0
        current_matched_segment_offset: int = -1
        current_matched_segment_end: int = -1
        previous_matched_span: Tuple[int, int] = None
        for matched in wiki_link_extraction_components_parts:
            matched_group_string: str = matched.group(0)
            matched_span: Tuple[int, int] = matched.span()
            if matched_group_string[0] == \
                TextPieceListExtractedWikiLink.WIKI_LINK_REGULAR_EXPRESSION_BEGINNING_STRING:
                if current_recursion_level == 0:
                    # ----
                    # DebuggingHelper.write_line_to_system_console_out_debug( \
                    #     '---- TextPieceListExtractedWikiLink.process_link_extraction(), ' +
                    #     'matched={},current_recursion_level={}'.format(matched, current_recursion_level))
                    # DebuggingHelper.write_line_to_system_console_out_debug( \
                    #     '---- TextPieceListExtractedWikiLink.process_link_extraction(), ' +
                    #     'matched_span[0]={}'.format(matched_span[0]))
                    # DebuggingHelper.write_line_to_system_console_out_debug( \
                    #     '---- TextPieceListExtractedWikiLink.process_link_extraction(), ' +
                    #     'matched_span[1]={}'.format(matched_span[1]))
                    # DebuggingHelper.write_line_to_system_console_out_debug( \
                    #     '---- TextPieceListExtractedWikiLink.process_link_extraction(), ' +
                    #     'type(matched_span[0])={}'.format(type(matched_span[0])))
                    # DebuggingHelper.write_line_to_system_console_out_debug( \
                    #     '---- TextPieceListExtractedWikiLink.process_link_extraction(), ' +
                    #     'type(matched_span[1])={}'.format(type(matched_span[1])))
                    # DebuggingHelper.write_line_to_system_console_out_debug( \
                    #     '---- TextPieceListExtractedWikiLink.process_link_extraction(), ' +
                    #     'len(raw_text_string)={}'.format(len(raw_text_string)))
                    # ----
                    # ---- NOTE-PYLINT ---- R1716: Simplify chained comparison between the operands (chained-comparison)
                    # pylint: disable=R1716
                    if (current_matched_segment_offset >= 0) and (current_matched_segment_end > current_matched_segment_offset):
                        text_piece_extracted_wiki_link: TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectWikiLink = \
                            TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectWikiLink( \
                                raw_text_string=raw_text_string[current_matched_segment_offset:current_matched_segment_end], \
                                text_piece_meta_data=matched_span)
                        text_piece_list.append(text_piece_extracted_wiki_link)
                        # DebuggingHelper.write_line_to_system_console_out_debug( \
                        #     '---- TextPieceListExtractedWikiLink.process_link_extraction(), ' +
                        #     'match_like={}'.format(match_like))
                    # ----
                    current_segment_end: int = matched_span[0]
                    text_piece_plain_sub_text: TextPiece = \
                        TextPiece(raw_text_string=raw_text_string[current_segment_offset:current_segment_end])
                    text_piece_list.append(text_piece_plain_sub_text)
                    # ----
                    current_matched_segment_offset = current_segment_end
                    # ----
                current_recursion_level += 1
            elif matched_group_string[-1] == \
                TextPieceListExtractedWikiLink.WIKI_LINK_REGULAR_EXPRESSION_ENDING_STRING:
                # DebuggingHelper.write_line_to_system_console_out_debug( \
                #     '---- TextPieceListExtractedWikiLink.process_link_extraction(), ' +
                #     'matched={},current_recursion_level={}'.format(matched, current_recursion_level))
                current_recursion_level -= 1
                if current_recursion_level < 0:
                    current_recursion_level = 0
                if current_recursion_level == 0:
                    current_segment_offset = matched_span[1]
                    current_matched_segment_end = current_segment_offset
            previous_matched_span = matched_span
        if current_recursion_level == 0:
            # ----
            # ---- NOTE-PYLINT ---- R1716: Simplify chained comparison between the operands (chained-comparison)
            # pylint: disable=R1716
            if (current_matched_segment_offset >= 0) and (current_matched_segment_end > current_matched_segment_offset):
                text_piece_extracted_wiki_link: TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectWikiLink = \
                    TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectWikiLink( \
                        raw_text_string=raw_text_string[current_matched_segment_offset:current_matched_segment_end], \
                        text_piece_meta_data=previous_matched_span)
                text_piece_list.append(text_piece_extracted_wiki_link)
            # ----
            if current_segment_offset < len(raw_text_string):
                text_piece_plain_sub_text: TextPiece = \
                    TextPiece(raw_text_string=raw_text_string[current_segment_offset:])
                text_piece_list.append(text_piece_plain_sub_text)
            # ----
        else:
            # ----
            current_matched_segment_end = len(raw_text_string)
            # ----
            # ---- NOTE-PYLINT ---- R1716: Simplify chained comparison between the operands (chained-comparison)
            # pylint: disable=R1716
            if (current_matched_segment_offset >= 0) and (current_matched_segment_end > current_matched_segment_offset):
                text_piece_extracted_wiki_link: TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectWikiLink = \
                    TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectWikiLink( \
                        raw_text_string=raw_text_string[current_matched_segment_offset:current_matched_segment_end], \
                        text_piece_meta_data=None)
                text_piece_list.append(text_piece_extracted_wiki_link)
            # ----
        return text_piece_list
class TextPieceListExtractedWikiLinkSingular(TextPieceList):
    # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> Match[str]
    """
    TextPieceListExtractedWikiLinkSingular is a TextPieceList
    object representing
    a comment text that was filtered from a text.
    This object should always return an empty string or
    a specified empty string.
    Note as wiki links can be recursive, this REGEX is limited and not able to
    deal with recusrive situation.
    """
    WIKI_LINK_REGULAR_EXPRESSION_PATTERN = \
        re.compile("\\[\\[(.*?)\\]\\]", \
        re.MULTILINE|re.DOTALL)
    def __init__(self, \
        raw_text_string: str, \
        join_string: str = TextPieceConstants.TEXT_PIECE_EMPTY_STRING) -> None:
        super(TextPieceListExtractedWikiLinkSingular, self).__init__( \
            raw_text_string=raw_text_string, \
            join_string=join_string)
    def generate_text_piece_list_from_raw_text_string(self, \
        raw_text_string: str) -> List[TextPiece]:
        """
        Process an input raw_text_string and generate a list of TextPiece objects.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        current_position: int = 0
        text_piece_list: List[TextPiece] = []
        while True:
            matched: Match[str] = \
                TextPieceListExtractedWikiLinkSingular.WIKI_LINK_REGULAR_EXPRESSION_PATTERN.\
                    search( \
                        string=raw_text_string, \
                        pos=current_position)
            if matched is None:
                break
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedWikiLinkSingular.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched={}'.format(matched))
            matched_span: Tuple[int, int] = matched.span()
            # matched_string: str = matched.string
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedWikiLinkSingular.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_string={}'.format(matched_string))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedWikiLinkSingular.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_span.__repr__()={}'.format(matched_span.__repr__()))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedWikiLinkSingular.generate_text_piece_list_from_raw_text_string(), ' +
            #     'len(matched_span)={}'.format(len(matched_span)))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedWikiLinkSingular.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_span[0]={}'.format(matched_span[0]))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedWikiLinkSingular.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_span[1]={}'.format(matched_span[1]))
            # matched_group_string: str = matched.group(0)
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListExtractedWikiLinkSingular.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_group_string={}'.format(matched_group_string))
            plain_sub_text_end_position: int = \
                matched_span[0]
            if plain_sub_text_end_position > current_position:
                text_piece_plain_sub_text: TextPiece = \
                    TextPiece(raw_text_string=raw_text_string[current_position:plain_sub_text_end_position])
                text_piece_list.append(text_piece_plain_sub_text)
            next_position: int = matched_span[1]
            text_piece_extracted_wiki_link: TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectWikiLink = \
                TextPieceWithMetadataExtractedWithRegularExpressionMatchObjectWikiLink( \
                    raw_text_string=raw_text_string[plain_sub_text_end_position:next_position], \
                    text_piece_meta_data=matched_span)
            text_piece_list.append(text_piece_extracted_wiki_link)
            current_position = next_position
        raw_text_string_end_position: int = len(raw_text_string)
        if current_position < raw_text_string_end_position:
            text_piece_end_plain_sub_text: TextPiece = \
                TextPiece(raw_text_string=raw_text_string[current_position:raw_text_string_end_position])
            text_piece_list.append(text_piece_end_plain_sub_text)
        return text_piece_list
class TextPieceListFilteredWikiTemplateArgument(TextPieceList):
    # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> Match[str]
    """
    TextPieceListFilteredWikiTemplateArgument is a TextPieceList
    object representing
    a comment text that was filtered from a text.
    This object should always return an empty string or
    a specified empty string.
    """
    WIKI_TEMPLATE_ARGUMENT_REGULAR_EXPRESSION_PATTERN = \
        re.compile("\\{\\{\\{([^{]*?)\\}\\}\\}", \
        re.MULTILINE|re.DOTALL)
    def __init__(self, \
        raw_text_string: str, \
        join_string: str = TextPieceConstants.TEXT_PIECE_EMPTY_STRING) -> None:
        super(TextPieceListFilteredWikiTemplateArgument, self).__init__( \
            raw_text_string=raw_text_string, \
            join_string=join_string)
    def generate_text_piece_list_from_raw_text_string(self, \
        raw_text_string: str) -> List[TextPiece]:
        """
        Process an input raw_text_string and generate a list of TextPiece objects.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        current_position: int = 0
        text_piece_list: List[TextPiece] = []
        while True:
            matched: Match[str] = \
                TextPieceListFilteredWikiTemplateArgument.WIKI_TEMPLATE_ARGUMENT_REGULAR_EXPRESSION_PATTERN.\
                    search( \
                        string=raw_text_string, \
                        pos=current_position)
            if matched is None:
                break
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredWikiTemplateArgument.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched={}'.format(matched))
            matched_span: Tuple[int, int] = matched.span()
            # matched_string: str = matched.string
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredWikiTemplateArgument.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_string={}'.format(matched_string))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredWikiTemplateArgument.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_span.__repr__()={}'.format(matched_span.__repr__()))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredWikiTemplateArgument.generate_text_piece_list_from_raw_text_string(), ' +
            #     'len(matched_span)={}'.format(len(matched_span)))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredWikiTemplateArgument.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_span[0]={}'.format(matched_span[0]))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredWikiTemplateArgument.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_span[1]={}'.format(matched_span[1]))
            # matched_group_string: str = matched.group(0)
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredWikiTemplateArgument.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_group_string={}'.format(matched_group_string))
            plain_sub_text_end_position: int = \
                matched_span[0]
            if plain_sub_text_end_position > current_position:
                text_piece_plain_sub_text: TextPiece = \
                    TextPiece(raw_text_string=raw_text_string[current_position:plain_sub_text_end_position])
                text_piece_list.append(text_piece_plain_sub_text)
            next_position: int = matched_span[1]
            text_piece_filtered_wiki_template_argument: TextPieceWithMetadataFilteredWithRegularExpressionMatchObject = \
                TextPieceWithMetadataFilteredWithRegularExpressionMatchObject( \
                    raw_text_string=raw_text_string[plain_sub_text_end_position:next_position], \
                    text_piece_meta_data=matched_span)
            text_piece_list.append(text_piece_filtered_wiki_template_argument)
            current_position = next_position
        raw_text_string_end_position: int = len(raw_text_string)
        if current_position < raw_text_string_end_position:
            text_piece_end_plain_sub_text: TextPiece = \
                TextPiece(raw_text_string=raw_text_string[current_position:raw_text_string_end_position])
            text_piece_list.append(text_piece_end_plain_sub_text)
        return text_piece_list
class TextPieceListFilteredWikiTemplate(TextPieceList):
    # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> Match[str]
    """
    TextPieceListFilteredWikiTemplate is a TextPieceList
    object representing
    a comment text that was filtered from a text.
    This object should always return an empty string or
    a specified empty string.
    """
    WIKI_TEMPLATE_REGULAR_EXPRESSION_BEGINNING_STRING: str = '{'
    WIKI_TEMPLATE_REGULAR_EXPRESSION_ENDING_STRING: str = '}'
    WIKI_TEMPLATE_REGULAR_EXPRESSION_PATTERN = \
        re.compile("(\\{\\{|\\}\\})")
    def __init__(self, \
        raw_text_string: str, \
        join_string: str = TextPieceConstants.TEXT_PIECE_EMPTY_STRING) -> None:
        super(TextPieceListFilteredWikiTemplate, self).__init__( \
            raw_text_string=raw_text_string, \
            join_string=join_string)
    def generate_text_piece_list_from_raw_text_string(self, \
        raw_text_string: str) -> List[TextPiece]:
        """
        Process an input raw_text_string and generate a list of TextPiece objects.
        """
        # ---- NOTE-PYLINT ---- R0914: Too many local variables
        # pylint: disable=R0914
        # ---- NOTE-PYLINT ---- R0912: Too many branches (15/12)
        # pylint: disable=R0912
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        current_position: int = 0
        text_piece_list: List[TextPiece] = []
        wiki_template_extraction_components_parts: List[Match[str]] = []
        while True:
            matched: Match[str] = \
                TextPieceListFilteredWikiTemplate.WIKI_TEMPLATE_REGULAR_EXPRESSION_PATTERN.\
                    search( \
                        string=raw_text_string, \
                        pos=current_position)
            if matched is None:
                break
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredWikiTemplate.process_template_extraction(), ' +
            #     'matched={}'.format(matched))
            matched_span: Tuple[int, int] = matched.span()
            # matched_string: str = matched.string
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredWikiTemplate.process_template_extraction(), ' +
            #     'matched_string={}'.format(matched_string))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredWikiTemplate.process_template_extraction(), ' +
            #     'matched_span.__repr__()={}'.format(matched_span.__repr__()))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredWikiTemplate.process_template_extraction(), ' +
            #     'len(matched_span)={}'.format(len(matched_span)))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredWikiTemplate.process_template_extraction(), ' +
            #     'matched_span[0]={}'.format(matched_span[0]))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredWikiTemplate.process_template_extraction(), ' +
            #     'matched_span[1]={}'.format(matched_span[1]))
            # matched_group_string: str = matched.group(0)
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredWikiTemplate.process_template_extraction(), ' +
            #     'matched_group_string={}'.format(matched_group_string))
            wiki_template_extraction_components_parts.append(matched)
            current_position = matched_span[1]
        current_recursion_level: int = 0
        current_segment_offset: int = 0
        current_matched_segment_offset: int = -1
        current_matched_segment_end: int = -1
        previous_matched_span: Tuple[int, int] = None
        for matched in wiki_template_extraction_components_parts:
            matched_group_string: str = matched.group(0)
            matched_span: Tuple[int, int] = matched.span()
            if matched_group_string[0] == \
                TextPieceListFilteredWikiTemplate.WIKI_TEMPLATE_REGULAR_EXPRESSION_BEGINNING_STRING:
                if current_recursion_level == 0:
                    # ----
                    # DebuggingHelper.write_line_to_system_console_out_debug( \
                    #     '---- TextPieceListFilteredWikiTemplate.process_template_extraction(), ' +
                    #     'matched={},current_recursion_level={}'.format(matched, current_recursion_level))
                    # DebuggingHelper.write_line_to_system_console_out_debug( \
                    #     '---- TextPieceListFilteredWikiTemplate.process_template_extraction(), ' +
                    #     'matched_span[0]={}'.format(matched_span[0]))
                    # DebuggingHelper.write_line_to_system_console_out_debug( \
                    #     '---- TextPieceListFilteredWikiTemplate.process_template_extraction(), ' +
                    #     'matched_span[1]={}'.format(matched_span[1]))
                    # DebuggingHelper.write_line_to_system_console_out_debug( \
                    #     '---- TextPieceListFilteredWikiTemplate.process_template_extraction(), ' +
                    #     'type(matched_span[0])={}'.format(type(matched_span[0])))
                    # DebuggingHelper.write_line_to_system_console_out_debug( \
                    #     '---- TextPieceListFilteredWikiTemplate.process_template_extraction(), ' +
                    #     'type(matched_span[1])={}'.format(type(matched_span[1])))
                    # DebuggingHelper.write_line_to_system_console_out_debug( \
                    #     '---- TextPieceListFilteredWikiTemplate.process_template_extraction(), ' +
                    #     'len(raw_text_string)={}'.format(len(raw_text_string)))
                    # ----
                    # ---- NOTE-PYLINT ---- R1716: Simplify chained comparison between the operands (chained-comparison)
                    # pylint: disable=R1716
                    if (current_matched_segment_offset >= 0) and (current_matched_segment_end > current_matched_segment_offset):
                        text_piece_filtered_wiki_template: TextPieceWithMetadataFilteredWithRegularExpressionMatchObject = \
                            TextPieceWithMetadataFilteredWithRegularExpressionMatchObject( \
                                raw_text_string=raw_text_string[current_matched_segment_offset:current_matched_segment_end], \
                                text_piece_meta_data=matched_span)
                        text_piece_list.append(text_piece_filtered_wiki_template)
                        # DebuggingHelper.write_line_to_system_console_out_debug( \
                        #     '---- TextPieceListFilteredWikiTemplate.process_template_extraction(), ' +
                        #     'match_like={}'.format(match_like))
                    # ----
                    current_segment_end: int = matched_span[0]
                    text_piece_plain_sub_text: TextPiece = \
                        TextPiece(raw_text_string=raw_text_string[current_segment_offset:current_segment_end])
                    text_piece_list.append(text_piece_plain_sub_text)
                    # ----
                    current_matched_segment_offset = current_segment_end
                    # ----
                current_recursion_level += 1
            elif matched_group_string[-1] == \
                TextPieceListFilteredWikiTemplate.WIKI_TEMPLATE_REGULAR_EXPRESSION_ENDING_STRING:
                # DebuggingHelper.write_line_to_system_console_out_debug( \
                #     '---- TextPieceListFilteredWikiTemplate.process_template_extraction(), ' +
                #     'matched={},current_recursion_level={}'.format(matched, current_recursion_level))
                current_recursion_level -= 1
                if current_recursion_level < 0:
                    current_recursion_level = 0
                if current_recursion_level == 0:
                    current_segment_offset = matched_span[1]
                    current_matched_segment_end = current_segment_offset
            previous_matched_span = matched_span
        if current_recursion_level == 0:
            # ----
            # ---- NOTE-PYLINT ---- R1716: Simplify chained comparison between the operands (chained-comparison)
            # pylint: disable=R1716
            if (current_matched_segment_offset >= 0) and (current_matched_segment_end > current_matched_segment_offset):
                text_piece_filtered_wiki_template: TextPieceWithMetadataFilteredWithRegularExpressionMatchObject = \
                    TextPieceWithMetadataFilteredWithRegularExpressionMatchObject( \
                        raw_text_string=raw_text_string[current_matched_segment_offset:current_matched_segment_end], \
                        text_piece_meta_data=previous_matched_span)
                text_piece_list.append(text_piece_filtered_wiki_template)
            # ----
            if current_segment_offset < len(raw_text_string):
                text_piece_plain_sub_text: TextPiece = \
                    TextPiece(raw_text_string=raw_text_string[current_segment_offset:])
                text_piece_list.append(text_piece_plain_sub_text)
            # ----
        else:
            # ----
            current_matched_segment_end = len(raw_text_string)
            # ----
            # ---- NOTE-PYLINT ---- R1716: Simplify chained comparison between the operands (chained-comparison)
            # pylint: disable=R1716
            if (current_matched_segment_offset >= 0) and (current_matched_segment_end > current_matched_segment_offset):
                text_piece_filtered_wiki_template: TextPieceWithMetadataFilteredWithRegularExpressionMatchObject = \
                    TextPieceWithMetadataFilteredWithRegularExpressionMatchObject( \
                        raw_text_string=raw_text_string[current_matched_segment_offset:current_matched_segment_end], \
                        text_piece_meta_data=None)
                text_piece_list.append(text_piece_filtered_wiki_template)
            # ----
        return text_piece_list
class TextPieceListFilteredWikiTemplateSingular(TextPieceList):
    # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> Match[str]
    """
    TextPieceListFilteredWikiTemplateSingular is a TextPieceList
    object representing
    a comment text that was filtered from a text.
    This object should always return an empty string or
    a specified empty string.
    Note as wiki templates can be recursive, this REGEX is limited and not able to
    deal with recusrive situation.
    """
    WIKI_TEMPLATE_SINGULAR_REGULAR_EXPRESSION_PATTERN = \
        re.compile("\\{\\{([^{]*?)\\}\\}", \
        re.MULTILINE|re.DOTALL)
    def __init__(self, \
        raw_text_string: str, \
        join_string: str = TextPieceConstants.TEXT_PIECE_EMPTY_STRING) -> None:
        super(TextPieceListFilteredWikiTemplateSingular, self).__init__( \
            raw_text_string=raw_text_string, \
            join_string=join_string)
    def generate_text_piece_list_from_raw_text_string(self, \
        raw_text_string: str) -> List[TextPiece]:
        """
        Process an input raw_text_string and generate a list of TextPiece objects.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        current_position: int = 0
        text_piece_list: List[TextPiece] = []
        while True:
            matched: Match[str] = \
                TextPieceListFilteredWikiTemplateSingular.WIKI_TEMPLATE_SINGULAR_REGULAR_EXPRESSION_PATTERN.\
                    search( \
                        string=raw_text_string, \
                        pos=current_position)
            if matched is None:
                break
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredWikiTemplateSingular.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched={}'.format(matched))
            matched_span: Tuple[int, int] = matched.span()
            # matched_string: str = matched.string
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredWikiTemplateSingular.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_string={}'.format(matched_string))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredWikiTemplateSingular.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_span.__repr__()={}'.format(matched_span.__repr__()))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredWikiTemplateSingular.generate_text_piece_list_from_raw_text_string(), ' +
            #     'len(matched_span)={}'.format(len(matched_span)))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredWikiTemplateSingular.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_span[0]={}'.format(matched_span[0]))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredWikiTemplateSingular.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_span[1]={}'.format(matched_span[1]))
            # matched_group_string: str = matched.group(0)
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredWikiTemplateSingular.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_group_string={}'.format(matched_group_string))
            plain_sub_text_end_position: int = \
                matched_span[0]
            if plain_sub_text_end_position > current_position:
                text_piece_plain_sub_text: TextPiece = \
                    TextPiece(raw_text_string=raw_text_string[current_position:plain_sub_text_end_position])
                text_piece_list.append(text_piece_plain_sub_text)
            next_position: int = matched_span[1]
            text_piece_filtered_wiki_template: TextPieceWithMetadataFilteredWithRegularExpressionMatchObject = \
                TextPieceWithMetadataFilteredWithRegularExpressionMatchObject( \
                    raw_text_string=raw_text_string[plain_sub_text_end_position:next_position], \
                    text_piece_meta_data=matched_span)
            text_piece_list.append(text_piece_filtered_wiki_template)
            current_position = next_position
        raw_text_string_end_position: int = len(raw_text_string)
        if current_position < raw_text_string_end_position:
            text_piece_end_plain_sub_text: TextPiece = \
                TextPiece(raw_text_string=raw_text_string[current_position:raw_text_string_end_position])
            text_piece_list.append(text_piece_end_plain_sub_text)
        return text_piece_list
class TextPieceListFilteredHtmlGenericIndividualTag(TextPieceList):
    # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> Match[str]
    """
    TextPieceListFilteredHtmlGenericIndividualTag is a TextPieceList
    object representing
    a comment text that was filtered from a text.
    This object should always return an empty string or
    a specified empty string.
    """
    WIKI_HTML_GENERIC_INDIVIDUAL_TAG_REGULAR_EXPRESSION_PATTERN = \
        re.compile("<(.*?)>", re.MULTILINE|re.DOTALL)
    def __init__(self, \
        raw_text_string: str, \
        join_string: str = TextPieceConstants.TEXT_PIECE_EMPTY_STRING) -> None:
        super(TextPieceListFilteredHtmlGenericIndividualTag, self).__init__( \
            raw_text_string=raw_text_string, \
            join_string=join_string)
    def generate_text_piece_list_from_raw_text_string(self, \
        raw_text_string: str) -> List[TextPiece]:
        """
        Process an input raw_text_string and generate a list of TextPiece objects.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        current_position: int = 0
        text_piece_list: List[TextPiece] = []
        while True:
            matched: Match[str] = \
                TextPieceListFilteredHtmlGenericIndividualTag.WIKI_HTML_GENERIC_INDIVIDUAL_TAG_REGULAR_EXPRESSION_PATTERN.\
                    search( \
                        string=raw_text_string, \
                        pos=current_position)
            if matched is None:
                break
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredHtmlGenericIndividualTag.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched={}'.format(matched))
            matched_span: Tuple[int, int] = matched.span()
            # matched_string: str = matched.string
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredHtmlGenericIndividualTag.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_string={}'.format(matched_string))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredHtmlGenericIndividualTag.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_span.__repr__()={}'.format(matched_span.__repr__()))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredHtmlGenericIndividualTag.generate_text_piece_list_from_raw_text_string(), ' +
            #     'len(matched_span)={}'.format(len(matched_span)))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredHtmlGenericIndividualTag.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_span[0]={}'.format(matched_span[0]))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredHtmlGenericIndividualTag.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_span[1]={}'.format(matched_span[1]))
            # matched_group_string: str = matched.group(0)
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredHtmlGenericIndividualTag.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_group_string={}'.format(matched_group_string))
            plain_sub_text_end_position: int = \
                matched_span[0]
            if plain_sub_text_end_position > current_position:
                text_piece_plain_sub_text: TextPiece = \
                    TextPiece(raw_text_string=raw_text_string[current_position:plain_sub_text_end_position])
                text_piece_list.append(text_piece_plain_sub_text)
            next_position: int = matched_span[1]
            text_piece_filtered_html_generic_individual_tag: TextPieceWithMetadataFilteredWithRegularExpressionMatchObject = \
                TextPieceWithMetadataFilteredWithRegularExpressionMatchObject( \
                    raw_text_string=raw_text_string[plain_sub_text_end_position:next_position], \
                    text_piece_meta_data=matched_span)
            text_piece_list.append(text_piece_filtered_html_generic_individual_tag)
            current_position = next_position
        raw_text_string_end_position: int = len(raw_text_string)
        if current_position < raw_text_string_end_position:
            text_piece_end_plain_sub_text: TextPiece = \
                TextPiece(raw_text_string=raw_text_string[current_position:raw_text_string_end_position])
            text_piece_list.append(text_piece_end_plain_sub_text)
        return text_piece_list
class TextPieceListFilteredHtmlReference(TextPieceList):
    # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> Match[str]
    """
    TextPieceListFilteredHtmlReference is a TextPieceList
    object representing
    a comment text that was filtered from a text.
    This object should always return an empty string or
    a specified empty string.
    """
    WIKI_HTML_REFERENCE_REGULAR_EXPRESSION_PATTERN = \
        re.compile("(<ref(erence)?([^<]*)/>)|(<ref(erence)?(.*?)/ref(erence)?>)", \
            re.IGNORECASE|re.MULTILINE|re.DOTALL)
    def __init__(self, \
        raw_text_string: str, \
        join_string: str = TextPieceConstants.TEXT_PIECE_EMPTY_STRING) -> None:
        super(TextPieceListFilteredHtmlReference, self).__init__( \
            raw_text_string=raw_text_string, \
            join_string=join_string)
    def generate_text_piece_list_from_raw_text_string(self, \
        raw_text_string: str) -> List[TextPiece]:
        """
        Process an input raw_text_string and generate a list of TextPiece objects.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        current_position: int = 0
        text_piece_list: List[TextPiece] = []
        while True:
            matched: Match[str] = \
                TextPieceListFilteredHtmlReference.WIKI_HTML_REFERENCE_REGULAR_EXPRESSION_PATTERN.\
                    search( \
                        string=raw_text_string, \
                        pos=current_position)
            if matched is None:
                break
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredHtmlReference.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched={}'.format(matched))
            matched_span: Tuple[int, int] = matched.span()
            # matched_string: str = matched.string
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredHtmlReference.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_string={}'.format(matched_string))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredHtmlReference.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_span.__repr__()={}'.format(matched_span.__repr__()))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredHtmlReference.generate_text_piece_list_from_raw_text_string(), ' +
            #     'len(matched_span)={}'.format(len(matched_span)))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredHtmlReference.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_span[0]={}'.format(matched_span[0]))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredHtmlReference.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_span[1]={}'.format(matched_span[1]))
            # matched_group_string: str = matched.group(0)
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredHtmlReference.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_group_string={}'.format(matched_group_string))
            plain_sub_text_end_position: int = \
                matched_span[0]
            if plain_sub_text_end_position > current_position:
                text_piece_plain_sub_text: TextPiece = \
                    TextPiece(raw_text_string=raw_text_string[current_position:plain_sub_text_end_position])
                text_piece_list.append(text_piece_plain_sub_text)
            next_position: int = matched_span[1]
            text_piece_filtered_html_reference: TextPieceWithMetadataFilteredWithRegularExpressionMatchObject = \
                TextPieceWithMetadataFilteredWithRegularExpressionMatchObject( \
                    raw_text_string=raw_text_string[plain_sub_text_end_position:next_position], \
                    text_piece_meta_data=matched_span)
            text_piece_list.append(text_piece_filtered_html_reference)
            current_position = next_position
        raw_text_string_end_position: int = len(raw_text_string)
        if current_position < raw_text_string_end_position:
            text_piece_end_plain_sub_text: TextPiece = \
                TextPiece(raw_text_string=raw_text_string[current_position:raw_text_string_end_position])
            text_piece_list.append(text_piece_end_plain_sub_text)
        return text_piece_list
class TextPieceListFilteredComment(TextPieceList):
    # ---- NOTE-MAY-LEAD-TO-TypeError: 'type' object is not subscriptable ---- -> Match[str]
    """
    TextPieceListFilteredComment is a TextPieceList
    object representing
    a comment text that was filtered from a text.
    This object should always return an empty string or
    a specified empty string.
    """
    WIKI_COMMENT_REGULAR_EXPRESSION_PATTERN = \
        re.compile("<!--(.*?)(-->|$)", re.MULTILINE|re.DOTALL)
    def __init__(self, \
        raw_text_string: str, \
        join_string: str = TextPieceConstants.TEXT_PIECE_EMPTY_STRING) -> None:
        super(TextPieceListFilteredComment, self).__init__( \
            raw_text_string=raw_text_string, \
            join_string=join_string)
    def generate_text_piece_list_from_raw_text_string(self, \
        raw_text_string: str) -> List[TextPiece]:
        """
        Process an input raw_text_string and generate a list of TextPiece objects.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        current_position: int = 0
        text_piece_list: List[TextPiece] = []
        while True:
            matched: Match[str] = \
                TextPieceListFilteredComment.WIKI_COMMENT_REGULAR_EXPRESSION_PATTERN.\
                    search( \
                        string=raw_text_string, \
                        pos=current_position)
            if matched is None:
                break
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredComment.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched={}'.format(matched))
            matched_span: Tuple[int, int] = matched.span()
            # matched_string: str = matched.string
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredComment.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_string={}'.format(matched_string))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredComment.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_span.__repr__()={}'.format(matched_span.__repr__()))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredComment.generate_text_piece_list_from_raw_text_string(), ' +
            #     'len(matched_span)={}'.format(len(matched_span)))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredComment.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_span[0]={}'.format(matched_span[0]))
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredComment.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_span[1]={}'.format(matched_span[1]))
            # matched_group_string: str = matched.group(0)
            # DebuggingHelper.write_line_to_system_console_out_debug( \
            #     '---- TextPieceListFilteredComment.generate_text_piece_list_from_raw_text_string(), ' +
            #     'matched_group_string={}'.format(matched_group_string))
            plain_sub_text_end_position: int = \
                matched_span[0]
            if plain_sub_text_end_position > current_position:
                text_piece_plain_sub_text: TextPiece = \
                    TextPiece(raw_text_string=raw_text_string[current_position:plain_sub_text_end_position])
                text_piece_list.append(text_piece_plain_sub_text)
            next_position: int = matched_span[1]
            text_piece_filtered_comment: TextPieceWithMetadataFilteredWithRegularExpressionMatchObject = \
                TextPieceWithMetadataFilteredWithRegularExpressionMatchObject( \
                    raw_text_string=raw_text_string[plain_sub_text_end_position:next_position], \
                    text_piece_meta_data=matched_span)
            text_piece_list.append(text_piece_filtered_comment)
            current_position = next_position
        raw_text_string_end_position: int = len(raw_text_string)
        if current_position < raw_text_string_end_position:
            text_piece_end_plain_sub_text: TextPiece = \
                TextPiece(raw_text_string=raw_text_string[current_position:raw_text_string_end_position])
            text_piece_list.append(text_piece_end_plain_sub_text)
        return text_piece_list

class TextPieceUtility:
    """
    TextPieceUtility contains a bunch of integration of the TextPiece classes as
    utility functions.
    """
    @staticmethod
    def process_to_text_piece_string_lists(input_text: str) -> List[List[str]]:
        """
        Generate a list of TextPiece strings from processing an input text.
        """
        text_piece_list_lists: List[List[TextPiece]] = \
            TextPieceUtility.process_to_text_piece_list_lists(input_text=input_text)
        text_piece_string_lists: List[List[str]] = [[
            item.get_text_piece_string() \
            for item in text_piece_list_list] \
            for text_piece_list_list in text_piece_list_lists]
        return text_piece_string_lists
    @staticmethod
    def process_to_text_piece_list_lists(input_text: str) -> List[List[TextPiece]]:
        """
        Generate a list of list of TextPiece objects from processing an input text.
        """
        text_piece_lists: List[TextPieceList] = \
            TextPieceUtility.process_to_text_piece_lists(input_text=input_text)
        text_piece_list_lists: List[List[TextPiece]] = [ \
            text_piece_list.get_text_piece_list() for text_piece_list in text_piece_lists]
        return text_piece_list_lists
    @staticmethod
    def logging_dump(text_piece_list: List[TextPiece], header_message: str = '') -> NoReturn:
        """
        Used for debugging/tracing purpose.
        """
        for index, text_piece in enumerate(text_piece_list):
            DebuggingHelper.write_line_to_system_console_out_debug(
                f'==== {header_message}, index={index}, text_piece={text_piece}')
            # ---- NOTE-PYLINT ---- C0301: Line too long
            # pylint: disable=C0301
            # ---- NOTE-FOR-DEBUGGING ---- DebuggingHelper.write_line_to_system_console_out_debug(
            # ---- NOTE-FOR-DEBUGGING ----     f'==== {header_message}, index={index}, text_piece.get_text_piece_string()=${text_piece.get_text_piece_string()}$')
    @staticmethod
    def process_to_text_piece_lists(input_text: str) -> List[TextPieceList]:
        """
        Generate a list of TextPieceList objects from processing an input text.
        """
        # ---- NOTE-PYLINT ---- R0914: Too many local variables
        # pylint: disable=R0914
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # --------------------------------------------------------------------
        # ---- NOTE-FOR-DEBUGGING ---- input_text: str = \
        # ---- NOTE-FOR-DEBUGGING ----     "《'''老子'''》，又名《'''道德經'''》，是[[先秦]]時期的古籍，相傳為[[春秋時期|春秋]]末期思想家[[老子]]所著<ref>阎纯德：《汉学研究》 第8期 第459页 9 《马王堆本〈老子〉及其文献流传的线索》 中华书局, 2004</ref>。《老子》為[[东周|春秋戰國]]時期[[道家]]学派的代表性經典，亦是[[道教]]尊奉的經典。至唐代，[[唐太宗]]命人將《道德經》譯為[[梵語]]；[[唐玄宗]]时，尊此经为《'''道德眞經'''》。"
        # --------------------------------------------------------------------
        text_piece_list_filtered_comment: TextPieceListFilteredComment = \
            TextPieceListFilteredComment(raw_text_string=input_text)
        # text_piece_list_after_filtered_comment: List[TextPiece] = \
        #     text_piece_list_filtered_comment.get_text_piece_list()
        # TextPieceUtility.logging_dump( \
        #     text_piece_list=text_piece_list_after_filtered_comment, \
        #     header_message='filtered_comment')
        # --------------------------------------------------------------------
        text_piece_string_after_filtered_comment: str = \
            text_piece_list_filtered_comment.get_text_piece_string()
        # --------------------------------------------------------------------
        text_piece_list_filtered_wiki_template: TextPieceListFilteredWikiTemplate = \
            TextPieceListFilteredWikiTemplate(raw_text_string=text_piece_string_after_filtered_comment)
        # text_piece_list_after_filtered_wiki_template: List[TextPiece] = \
        #     text_piece_list_filtered_wiki_template.get_text_piece_list()
        # ---- NOTE-PLACE-HOLDER ---- text_piece_list_after_filtered_wiki_template: List[TextPiece] = [ \
        # ---- NOTE-PLACE-HOLDER ----     item \
        # ---- NOTE-PLACE-HOLDER ----         for sublist in \
        # ---- NOTE-PLACE-HOLDER ----         [TextPieceListFilteredWikiTemplate(raw_text_string=element.get_text_piece_string()).get_text_piece_list() \
        # ---- NOTE-PLACE-HOLDER ----             for element in text_piece_list_after_filtered_comment] \
        # ---- NOTE-PLACE-HOLDER ----     for item in sublist]
        # TextPieceUtility.logging_dump( \
        #     text_piece_list=text_piece_list_after_filtered_wiki_template, \
        #     header_message='filtered_wiki_template')
        # --------------------------------------------------------------------
        text_piece_string_after_filtered_wiki_template: str = \
            text_piece_list_filtered_wiki_template.get_text_piece_string()
        # --------------------------------------------------------------------
        text_piece_list_filtered_html_reference: TextPieceListFilteredHtmlReference = \
            TextPieceListFilteredHtmlReference(raw_text_string=text_piece_string_after_filtered_wiki_template)
        # text_piece_list_after_filtered_html_reference: List[TextPiece] = \
        #     text_piece_list_filtered_html_reference.get_text_piece_list()
        # ---- NOTE-PLACE-HOLDER ---- text_piece_list_after_filtered_html_reference: List[TextPiece] = [ \
        # ---- NOTE-PLACE-HOLDER ----     item \
        # ---- NOTE-PLACE-HOLDER ----         for sublist in \
        # ---- NOTE-PLACE-HOLDER ----         [TextPieceListFilteredHtmlReference(raw_text_string=element.get_text_piece_string()).get_text_piece_list() \
        # ---- NOTE-PLACE-HOLDER ----             for element in text_piece_list_after_filtered_wiki_template] \
        # ---- NOTE-PLACE-HOLDER ----     for item in sublist]
        # TextPieceUtility.logging_dump( \
        #     text_piece_list=text_piece_list_after_filtered_html_reference, \
        #     header_message='filtered_html_reference')
        # --------------------------------------------------------------------
        text_piece_string_after_filtered_html_reference: str = \
            text_piece_list_filtered_html_reference.get_text_piece_string()
        # --------------------------------------------------------------------
        text_piece_list_filtered_html_generic_individual_tag: TextPieceListFilteredHtmlGenericIndividualTag = \
            TextPieceListFilteredHtmlGenericIndividualTag(raw_text_string=text_piece_string_after_filtered_html_reference)
        # text_piece_list_after_filtered_html_generic_individual_tag: List[TextPiece] = \
        #     text_piece_list_filtered_html_generic_individual_tag.get_text_piece_list()
        # ---- NOTE-PLACE-HOLDER ---- text_piece_list_after_filtered_html_generic_individual_tag: List[TextPiece] = [ \
        # ---- NOTE-PLACE-HOLDER ----     item \
        # ---- NOTE-PLACE-HOLDER ----         for sublist in \
        # ---- NOTE-PLACE-HOLDER ----         [TextPieceListFilteredHtmlGenericIndividualTag(raw_text_string=element.get_text_piece_string()).get_text_piece_list() \
        # ---- NOTE-PLACE-HOLDER ----             for element in text_piece_list_after_filtered_html_reference] \
        # ---- NOTE-PLACE-HOLDER ----     for item in sublist]
        # TextPieceUtility.logging_dump( \
        #     text_piece_list=text_piece_list_after_filtered_html_generic_individual_tag, \
        #     header_message='filtered_html_generic_individual_tag')
        # --------------------------------------------------------------------
        text_piece_string_after_filtered_html_generic_individual_tag: str = \
            text_piece_list_filtered_html_generic_individual_tag.get_text_piece_string()
        # --------------------------------------------------------------------
        text_piece_list_extracted_wiki_emphasis_bold_italic: TextPieceListExtractedWikiEmphasisBoldItalic = \
            TextPieceListExtractedWikiEmphasisBoldItalic(raw_text_string=text_piece_string_after_filtered_html_generic_individual_tag)
        # text_piece_list_after_extracted_wiki_emphasis_bold_italic: List[TextPiece] = \
        #     text_piece_list_extracted_wiki_emphasis_bold_italic.get_text_piece_list()
        # ---- NOTE-PLACE-HOLDER ---- text_piece_list_after_extracted_wiki_emphasis_bold_italic: List[TextPiece] = [ \
        # ---- NOTE-PLACE-HOLDER ----     item \
        # ---- NOTE-PLACE-HOLDER ----         for sublist in \
        # ---- NOTE-PLACE-HOLDER ----         [TextPieceListExtractedWikiEmphasisBoldItalic(raw_text_string=element.get_text_piece_string()).get_text_piece_list() \
        # ---- NOTE-PLACE-HOLDER ----             for element in text_piece_list_after_filtered_html_generic_individual_tag] \
        # ---- NOTE-PLACE-HOLDER ----     for item in sublist]
        # TextPieceUtility.logging_dump( \
        #     text_piece_list=text_piece_list_after_extracted_wiki_emphasis_bold_italic, \
        #     header_message='extracted_wiki_emphasis_bold_italic')
        # --------------------------------------------------------------------
        text_piece_string_after_extracted_wiki_emphasis_bold_italic: str = \
            text_piece_list_extracted_wiki_emphasis_bold_italic.get_text_piece_string()
        # --------------------------------------------------------------------
        text_piece_list_extracted_wiki_emphasis_bold: TextPieceListExtractedWikiEmphasisBold = \
            TextPieceListExtractedWikiEmphasisBold(raw_text_string=text_piece_string_after_extracted_wiki_emphasis_bold_italic)
        # text_piece_list_after_extracted_wiki_emphasis_bold: List[TextPiece] = \
        #     text_piece_list_extracted_wiki_emphasis_bold.get_text_piece_list()
        # ---- NOTE-PLACE-HOLDER ---- text_piece_list_after_extracted_wiki_emphasis_bold: List[TextPiece] = [ \
        # ---- NOTE-PLACE-HOLDER ----     item \
        # ---- NOTE-PLACE-HOLDER ----         for sublist in \
        # ---- NOTE-PLACE-HOLDER ----         [TextPieceListExtractedWikiEmphasisBold(raw_text_string=element.get_text_piece_string()).get_text_piece_list() \
        # ---- NOTE-PLACE-HOLDER ----             for element in text_piece_list_after_extracted_wiki_emphasis_bold_italic] \
        # ---- NOTE-PLACE-HOLDER ----     for item in sublist]
        # TextPieceUtility.logging_dump( \
        #     text_piece_list=text_piece_list_after_extracted_wiki_emphasis_bold, \
        #     header_message='filtered_wiki_emphasis_bold')
        # --------------------------------------------------------------------
        text_piece_string_after_extracted_wiki_emphasis_bold: str = \
            text_piece_list_extracted_wiki_emphasis_bold.get_text_piece_string()
        # --------------------------------------------------------------------
        text_piece_list_extracted_wiki_link: TextPieceListExtractedWikiLink = \
            TextPieceListExtractedWikiLink(raw_text_string=text_piece_string_after_extracted_wiki_emphasis_bold)
        # text_piece_list_after_extracted_wiki_link: List[TextPiece] = \
        #     text_piece_list_extracted_wiki_link.get_text_piece_list()
        # ---- NOTE-PLACE-HOLDER ---- text_piece_list_after_extracted_wiki_link: List[TextPiece] = [ \
        # ---- NOTE-PLACE-HOLDER ----     item \
        # ---- NOTE-PLACE-HOLDER ----         for sublist in \
        # ---- NOTE-PLACE-HOLDER ----         [TextPieceListExtractedWikiLink(raw_text_string=element.get_text_piece_string()).get_text_piece_list() \
        # ---- NOTE-PLACE-HOLDER ----             for element in text_piece_list_after_extracted_wiki_emphasis_bold] \
        # ---- NOTE-PLACE-HOLDER ----     for item in sublist]
        # TextPieceUtility.logging_dump( \
        #     text_piece_list=text_piece_list_after_extracted_wiki_link, \
        #     header_message='extracted_wiki_link')
        # --------------------------------------------------------------------
        text_piece_string_after_extracted_wiki_link: str = \
            text_piece_list_extracted_wiki_link.get_text_piece_string()
        # --------------------------------------------------------------------
        text_piece_list_extracted_chinese_book_enclosure: TextPieceListExtractedChineseBookEnclosure = \
            TextPieceListExtractedChineseBookEnclosure(raw_text_string=text_piece_string_after_extracted_wiki_link)
        # text_piece_list_after_extracted_chinese_book_enclosure: List[TextPiece] = \
        #     text_piece_list_extracted_chinese_book_enclosure.get_text_piece_list()
        # ---- NOTE-PLACE-HOLDER ---- text_piece_list_after_extracted_chinese_book_enclosure: List[TextPiece] = [ \
        # ---- NOTE-PLACE-HOLDER ----     item \
        # ---- NOTE-PLACE-HOLDER ----         for sublist in \
        # ---- NOTE-PLACE-HOLDER ----         [TextPieceListExtractedChineseBookEnclosure(raw_text_string=element.get_text_piece_string()).get_text_piece_list() \
        # ---- NOTE-PLACE-HOLDER ----             for element in text_piece_list_after_extracted_wiki_link] \
        # ---- NOTE-PLACE-HOLDER ----     for item in sublist]
        # TextPieceUtility.logging_dump( \
        #     text_piece_list=text_piece_list_after_extracted_chinese_book_enclosure, \
        #     header_message='filtered_chinese_book_enclosure')
        # --------------------------------------------------------------------
        text_piece_string_after_extracted_chinese_book_enclosure: str = \
            text_piece_list_extracted_chinese_book_enclosure.get_text_piece_string()
        # --------------------------------------------------------------------
        text_piece_list_extracted_chinese_book_enclosure_single: TextPieceListExtractedChineseBookEnclosureSingle = \
            TextPieceListExtractedChineseBookEnclosureSingle(raw_text_string=text_piece_string_after_extracted_chinese_book_enclosure)
        # text_piece_list_after_extracted_chinese_book_enclosure_single: List[TextPiece] = \
        #     text_piece_list_extracted_chinese_book_enclosure_single.get_text_piece_list()
        # ---- NOTE-PLACE-HOLDER ---- text_piece_list_after_extracted_chinese_book_enclosure_single: List[TextPiece] = [ \
        # ---- NOTE-PLACE-HOLDER ----     item \
        # ---- NOTE-PLACE-HOLDER ----         for sublist in \
        # ---- NOTE-PLACE-HOLDER ----         [TextPieceListExtractedChineseBookEnclosureSingle(raw_text_string=element.get_text_piece_string()).get_text_piece_list() \
        # ---- NOTE-PLACE-HOLDER ----             for element in text_piece_list_after_extracted_chinese_book_enclosure] \
        # ---- NOTE-PLACE-HOLDER ----     for item in sublist]
        # TextPieceUtility.logging_dump( \
        #     text_piece_list=text_piece_list_after_extracted_chinese_book_enclosure_single, \
        #     header_message='filtered_chinese_book_enclosure_single')
        # --------------------------------------------------------------------
        return [
            text_piece_list_filtered_comment,
            text_piece_list_filtered_wiki_template,
            text_piece_list_filtered_html_reference,
            text_piece_list_filtered_html_generic_individual_tag,
            text_piece_list_extracted_wiki_emphasis_bold_italic,
            text_piece_list_extracted_wiki_emphasis_bold,
            text_piece_list_extracted_wiki_link,
            text_piece_list_extracted_chinese_book_enclosure,
            text_piece_list_extracted_chinese_book_enclosure_single]
        # ---- NOTE-PLACE-HOLDER ---- return [
        # ---- NOTE-PLACE-HOLDER ----     text_piece_list_after_filtered_comment,
        # ---- NOTE-PLACE-HOLDER ----     text_piece_list_after_filtered_wiki_template,
        # ---- NOTE-PLACE-HOLDER ----     text_piece_list_after_filtered_html_reference,
        # ---- NOTE-PLACE-HOLDER ----     text_piece_list_after_filtered_html_generic_individual_tag,
        # ---- NOTE-PLACE-HOLDER ----     text_piece_list_after_extracted_wiki_emphasis_bold_italic,
        # ---- NOTE-PLACE-HOLDER ----     text_piece_list_after_extracted_wiki_emphasis_bold,
        # ---- NOTE-PLACE-HOLDER ----     text_piece_list_after_extracted_wiki_link,
        # ---- NOTE-PLACE-HOLDER ----     text_piece_list_after_extracted_chinese_book_enclosure,
        # ---- NOTE-PLACE-HOLDER ----     text_piece_list_after_extracted_chinese_book_enclosure_single ]
        # --------------------------------------------------------------------
