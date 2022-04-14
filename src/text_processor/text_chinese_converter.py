# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
This module is for converting text between Unicode systems.
"""
# ---- NOTE-PYLINT ---- C0302: Too many lines in module
# pylint: disable=C0302

# from typing import Any
# from typing import Dict
# from typing import Generic
from typing import List
# from typing import NoReturn
# from typing import Set
# from typing import Tuple
# from typing import TypeVar

import opencc

# from utility.string_helper.string_helper \
#     import StringHelper

# from utility.debugging_helper.debugging_helper \
#     import DebuggingHelper

class TextChineseConverter:
    """
    Converting text between Unicode systems.
    """
    CONVERTER_simplified_to_traditional = opencc.OpenCC('s2t.json')
    @staticmethod
    def convert_simplified_to_traditional(input_text: str) -> str:
        """
        Convert any simplified phrases or characters into tranditional counterpart.
        """
        return TextChineseConverter.CONVERTER_simplified_to_traditional.convert(input_text)
    @staticmethod
    def index_of_first_difference(input_first: str, input_second: str) -> int:
        """
        The index of the first different character.
        """
        length_input_first: int = len(input_first)
        length_input_second: int = len(input_second)
        length_to_check: int = min(length_input_first, length_input_second)
        for i in range(length_to_check):
            if input_first[i] != input_second[i]:
                return i
        return length_to_check

class TextChineseConverterUtility:
    """
    TextChineseConverterUtility contains a bunch of integration of the TextChineseConverter
    classes as utility functions.
    """
    # ---- NOTE-PYLINT ---- R0903: Too few public methods
    # pylint: disable=R0903
    @staticmethod
    def process(input_texts: List[str]) -> List[str]:
        """
        Generate a list of str from processing an input text.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        input_texts_converted: List[str] = \
            [TextChineseConverter.convert_simplified_to_traditional(item) for item in input_texts]
        return input_texts_converted
