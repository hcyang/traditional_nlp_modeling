# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
This module provides some common string helper functions.
"""
# ---- NOTE-PYLINT ---- C0302: Too many lines in module
# pylint: disable=C0302

from typing import List
from typing import Set

from utility.datatype_helper.datatype_helper \
    import DatatypeHelper

class StringHelper:
    """
    This class contains some common functions for processing strings.
    """

    LanguageTokenVariableLegitimateLetters: Set[str] = { \
        '0', \
        '1', \
        '2', \
        '3', \
        '4', \
        '5', \
        '6', \
        '7', \
        '8', \
        '9', \
        'A', \
        'B', \
        'C', \
        'D', \
        'E', \
        'F', \
        'G', \
        'H', \
        'I', \
        'J', \
        'K', \
        'L', \
        'M', \
        'N', \
        'O', \
        'P', \
        'Q', \
        'R', \
        'S', \
        'T', \
        'U', \
        'V', \
        'W', \
        'X', \
        'Y', \
        'Z', \
        '_', \
        'a', \
        'b', \
        'c', \
        'd', \
        'e', \
        'f', \
        'g', \
        'h', \
        'i', \
        'j', \
        'k', \
        'l', \
        'm', \
        'n', \
        'o', \
        'p', \
        'q', \
        'r', \
        's', \
        't', \
        'u', \
        'v', \
        'w', \
        'x', \
        'y', \
        'z', \
    }

    LanguageTokenPunctuationDelimiters: List[str] = [ \
        # ---- // ---- '\0', \
        # ---- // ---- '\u0001', \
        # ---- // ---- '\u0002', \
        # ---- // ---- '\u0003', \
        # ---- // ---- '\u0004', \
        # ---- // ---- '\u0005', \
        # ---- // ---- '\u0006', \
        # ---- // ---- '\a', \
        # ---- // ---- '\b', \
        # ---- // ---- '\t', \
        # ---- // ---- '\n', \
        # ---- // ---- '\v', \
        # ---- // ---- '\f', \
        # ---- // ---- '\r', \
        # ---- // ---- '\u000E', \
        # ---- // ---- '\u000F', \
        # ---- // ---- '\u0010', \
        # ---- // ---- '\u0011', \
        # ---- // ---- '\u0012', \
        # ---- // ---- '\u0013', \
        # ---- // ---- '\u0014', \
        # ---- // ---- '\u0015', \
        # ---- // ---- '\u0016', \
        # ---- // ---- '\u0017', \
        # ---- // ---- '\u0018', \
        # ---- // ---- '\u0019', \
        # ---- // ---- '\u001A', \
        # ---- // ---- '\u001B', \
        # ---- // ---- '\u001C', \
        # ---- // ---- '\u001D', \
        # ---- // ---- '\u001E', \
        # ---- // ---- '\u001F', \
        # ---- // ---- ' ', \
        '!', \
        '\"', \
        '#', \
        '$', \
        '%', \
        '&', \
        '\'', \
        '(', \
        ')', \
        '*', \
        '+', \
        ',', \
        '-', \
        '.', \
        '/', \
        ':', \
        ';', \
        '<', \
        '=', \
        '>', \
        '?', \
        '@', \
        '[', \
        '\\', \
        ']', \
        '^', \
        '`', \
        '{', \
        '|', \
        '}', \
        '~', \
        # ---- // ---- '\u007F'
        ]

    LanguageTokenSpaceMinimalListDelimiters: List[str] = [ \
        # ---- '\0', \
        # ---- '\u0001', \
        # ---- '\u0002', \
        # ---- '\u0003', \
        # ---- '\u0004', \
        # ---- '\u0005', \
        # ---- '\u0006', \
        # ---- '\a', \
        # ---- '\b', \
        '\t', \
        '\n', \
        # ---- '\v', \
        # ---- '\f', \
        '\r', \
        # ---- '\u000E', \
        # ---- '\u000F', \
        # ---- '\u0010', \
        # ---- '\u0011', \
        # ---- '\u0012', \
        # ---- '\u0013', \
        # ---- '\u0014', \
        # ---- '\u0015', \
        # ---- '\u0016', \
        # ---- '\u0017', \
        # ---- '\u0018', \
        # ---- '\u0019', \
        # ---- '\u001A', \
        # ---- '\u001B', \
        # ---- '\u001C', \
        # ---- '\u001D', \
        # ---- '\u001E', \
        # ---- '\u001F', \
        ' ', \
        # ---- // ---- '!', \
        # ---- // ---- '\"', \
        # ---- // ---- '#', \
        # ---- // ---- '$', \
        # ---- // ---- '%', \
        # ---- // ---- '&', \
        # ---- // ---- '\'', \
        # ---- // ---- '(', \
        # ---- // ---- ')', \
        # ---- // ---- '*', \
        # ---- // ---- '+', \
        # ---- // ---- ',', \
        # ---- // ---- '-', \
        # ---- // ---- '.', \
        # ---- // ---- '/', \
        # ---- // ---- ':', \
        # ---- // ---- ';', \
        # ---- // ---- '<', \
        # ---- // ---- '=', \
        # ---- // ---- '>', \
        # ---- // ---- '?', \
        # ---- // ---- '@', \
        # ---- // ---- '[', \
        # ---- // ---- '\\', \
        # ---- // ---- ']', \
        # ---- // ---- '^', \
        # ---- // ---- '`', \
        # ---- // ---- '{', \
        # ---- // ---- '|', \
        # ---- // ---- '}', \
        # ---- // ---- '~', \
        # ---- '\u007F'
        ]

    LanguageTokenSpaceDelimiters: List[str] = [ \
        '\0', \
        '\u0001', \
        '\u0002', \
        '\u0003', \
        '\u0004', \
        '\u0005', \
        '\u0006', \
        '\a', \
        '\b', \
        '\t', \
        '\n', \
        '\v', \
        '\f', \
        '\r', \
        '\u000E', \
        '\u000F', \
        '\u0010', \
        '\u0011', \
        '\u0012', \
        '\u0013', \
        '\u0014', \
        '\u0015', \
        '\u0016', \
        '\u0017', \
        '\u0018', \
        '\u0019', \
        '\u001A', \
        '\u001B', \
        '\u001C', \
        '\u001D', \
        '\u001E', \
        '\u001F', \
        ' ', \
        # ---- // ---- '!', \
        # ---- // ---- '\"', \
        # ---- // ---- '#', \
        # ---- // ---- '$', \
        # ---- // ---- '%', \
        # ---- // ---- '&', \
        # ---- // ---- '\'', \
        # ---- // ---- '(', \
        # ---- // ---- ')', \
        # ---- // ---- '*', \
        # ---- // ---- '+', \
        # ---- // ---- ',', \
        # ---- // ---- '-', \
        # ---- // ---- '.', \
        # ---- // ---- '/', \
        # ---- // ---- ':', \
        # ---- // ---- ';', \
        # ---- // ---- '<', \
        # ---- // ---- '=', \
        # ---- // ---- '>', \
        # ---- // ---- '?', \
        # ---- // ---- '@', \
        # ---- // ---- '[', \
        # ---- // ---- '\\', \
        # ---- // ---- ']', \
        # ---- // ---- '^', \
        # ---- // ---- '`', \
        # ---- // ---- '{', \
        # ---- // ---- '|', \
        # ---- // ---- '}', \
        # ---- // ---- '~', \
        '\u007F'
        ]

    LanguageTokenSpaceExceptNewlineCarriageReturnDelimiters: List[str] = [ \
        '\0', \
        '\u0001', \
        '\u0002', \
        '\u0003', \
        '\u0004', \
        '\u0005', \
        '\u0006', \
        '\a', \
        '\b', \
        '\t', \
        # ---- '\n', \
        '\v', \
        '\f', \
        # ---- '\r', \
        '\u000E', \
        '\u000F', \
        '\u0010', \
        '\u0011', \
        '\u0012', \
        '\u0013', \
        '\u0014', \
        '\u0015', \
        '\u0016', \
        '\u0017', \
        '\u0018', \
        '\u0019', \
        '\u001A', \
        '\u001B', \
        '\u001C', \
        '\u001D', \
        '\u001E', \
        '\u001F', \
        ' ', \
        # ---- // ---- '!', \
        # ---- // ---- '\"', \
        # ---- // ---- '#', \
        # ---- // ---- '$', \
        # ---- // ---- '%', \
        # ---- // ---- '&', \
        # ---- // ---- '\'', \
        # ---- // ---- '(', \
        # ---- // ---- ')', \
        # ---- // ---- '*', \
        # ---- // ---- '+', \
        # ---- // ---- ',', \
        # ---- // ---- '-', \
        # ---- // ---- '.', \
        # ---- // ---- '/', \
        # ---- // ---- ':', \
        # ---- // ---- ';', \
        # ---- // ---- '<', \
        # ---- // ---- '=', \
        # ---- // ---- '>', \
        # ---- // ---- '?', \
        # ---- // ---- '@', \
        # ---- // ---- '[', \
        # ---- // ---- '\\', \
        # ---- // ---- ']', \
        # ---- // ---- '^', \
        # ---- // ---- '`', \
        # ---- // ---- '{', \
        # ---- // ---- '|', \
        # ---- // ---- '}', \
        # ---- // ---- '~', \
        '\u007F'
        ]

    @staticmethod
    def remove_spaces_except_ending_newline( \
        input_value: str, \
        input_space_array: List[str] = None) -> str:
        """
        Remove a space with a replacement string.
        """
        if input_value is None:
            return input_value
        if len(input_value) <= 0:
            return input_value
        is_input_value_ending_with_newline: bool = StringHelper.is_ending_with_newline(input_value)
        input_value_spaces_removed: str = StringHelper.replace_spaces_with_empty(
            input_value=input_value,
            input_space_array=input_space_array)
        if is_input_value_ending_with_newline and len(input_value_spaces_removed) > 0:
            return input_value_spaces_removed + '\n'
        else:
            return input_value_spaces_removed
    @staticmethod
    def is_ending_with_newline( \
        input_value: str) -> bool:
        """
        Test if an input str is ended with a newline.
        """
        if input_value is None:
            return False
        if len(input_value) <= 0:
            return False
        # ---- NOTE-FOR-REFERENCE ---- https://docs.python.org/3/library/os.html#os.linesep
        if input_value[-1] == '\n':
            return True
        return False

    @staticmethod
    def is_none_empty_or_whitespaces( \
        input_value: str) -> bool:
        """
        Test if an input str is in a set of strings.
        """
        if input_value is None:
            return True
        if len(input_value) <= 0:
            return True
        input_value_spaces_removed: str = StringHelper.remove_spaces( \
            input_value=input_value, \
            input_space_array=StringHelper.LanguageTokenSpaceMinimalListDelimiters)
        if len(input_value_spaces_removed) <= 0:
            return True
        return False

    @staticmethod
    def is_in( \
        input_value: str, \
        input_set: Set[str]) -> bool:
        """
        Test if an input str is in a set of strings.
        """
        return input_value in input_set

    @staticmethod
    def is_in_substring( \
        input_value: str, \
        input_set: Set[str]) -> bool:
        """
        Test if an input str is a substring of a set of strings.
        """
        for input_set_element in input_set:
            if input_set_element.find(input_value) != -1:
                return True
        return False

    @staticmethod
    def remove_punctuations_and_spaces( \
        input_value: str, \
        input_punctuation_array: List[str] = None, \
        input_space_array: List[str] = None) -> str:
        """
        Replace space and punctuations from an input string.
        """
        input_value = StringHelper.replace_punctuations_with_empty(
            input_value=input_value,
            input_punctuation_array=input_punctuation_array)
        input_value = StringHelper.replace_spaces_with_empty(
            input_value=input_value,
            input_space_array=input_space_array)
        return input_value
    @staticmethod
    def remove_defined_punctuations_and_spaces( \
        input_value: str) -> str:
        """
        Replace space and punctuations from an input string.
        """
        input_value = StringHelper.replace_defined_punctuations_with_empty(
            input_value=input_value)
        input_value = StringHelper.replace_defined_spaces_with_empty(
            input_value=input_value)
        return input_value

    @staticmethod
    def replace_punctuations_and_spaces( \
        input_value: str, \
        input_punctuation_array: List[str] = None, \
        input_space_array: List[str] = None, \
        replacement: str = ' ') -> str:
        """
        Replace space and punctuations from an input string.
        """
        input_value = StringHelper.replace_punctuations(
            input_value=input_value,
            input_punctuation_array=input_punctuation_array,
            replacement=replacement)
        input_value = StringHelper.replace_spaces(
            input_value=input_value,
            input_space_array=input_space_array,
            replacement=replacement)
        return input_value
    @staticmethod
    def replace_punctuations_and_spaces_with_space( \
        input_value: str, \
        input_punctuation_array: List[str] = None, \
        input_space_array: List[str] = None) -> str:
        """
        Replace space and punctuations from an input string.
        """
        input_value = StringHelper.replace_punctuations_with_space(
            input_value=input_value,
            input_punctuation_array=input_punctuation_array)
        input_value = StringHelper.replace_spaces_with_space(
            input_value=input_value,
            input_space_array=input_space_array)
        return input_value
    @staticmethod
    def replace_punctuations_and_spaces_with_empty( \
        input_value: str, \
        input_punctuation_array: List[str] = None, \
        input_space_array: List[str] = None) -> str:
        """
        Replace space and punctuations from an input string.
        """
        input_value = StringHelper.replace_punctuations_with_empty(
            input_value=input_value,
            input_punctuation_array=input_punctuation_array)
        input_value = StringHelper.replace_spaces_with_empty(
            input_value=input_value,
            input_space_array=input_space_array)
        return input_value
    @staticmethod
    def replace_defiend_punctuations_and_spaces( \
        input_value: str, \
        replacement: str = ' ') -> str:
        """
        Replace space and punctuations from an input string.
        """
        input_value = StringHelper.replace_defined_punctuations(
            input_value=input_value,
            replacement=replacement)
        input_value = StringHelper.replace_defined_spaces(
            input_value=input_value,
            replacement=replacement)
        return input_value
    @staticmethod
    def replace_defiend_punctuations_and_spaces_with_space( \
        input_value: str) -> str:
        """
        Replace space and punctuations from an input string.
        """
        input_value = StringHelper.replace_defined_punctuations_with_space(
            input_value=input_value)
        input_value = StringHelper.replace_defined_spaces_with_space(
            input_value=input_value)
        return input_value
    @staticmethod
    def replace_defiend_punctuations_and_spaces_with_empty( \
        input_value: str) -> str:
        """
        Replace space and punctuations from an input string.
        """
        input_value = StringHelper.replace_defined_punctuations_with_empty(
            input_value=input_value)
        input_value = StringHelper.replace_defined_spaces_with_empty(
            input_value=input_value)
        return input_value

    @staticmethod
    def contains_punctuations_or_spaces( \
        input_value: str, \
        input_punctuation_array: List[str] = None, \
        input_space_array: List[str] = None) -> bool:
        """
        Replace space and punctuations from an input string.
        """
        has_punctuation: bool = StringHelper.contains_punctuations(
            input_value=input_value,
            input_punctuation_array=input_punctuation_array)
        if has_punctuation:
            return True
        has_space: bool = StringHelper.contains_spaces(
            input_value=input_value,
            input_space_array=input_space_array)
        if has_space:
            return True
        return False
    @staticmethod
    def contains_defined_punctuations_or_spaces( \
        input_value: str) -> bool:
        """
        Replace space and punctuations from an input string.
        """
        has_punctuation: bool = StringHelper.contains_defined_punctuations(
            input_value=input_value)
        if has_punctuation:
            return True
        has_space: bool = StringHelper.contains_defined_spaces(
            input_value=input_value)
        if has_space:
            return True
        return False

    @staticmethod
    def remove_punctuations_and_spaces_except_newline_carriage_return( \
        input_value: str, \
        input_punctuation_array: List[str] = None, \
        input_space_array: List[str] = None) -> str:
        """
        Replace space and punctuations from an input string.
        """
        input_value = StringHelper.replace_punctuations_with_empty(
            input_value=input_value,
            input_punctuation_array=input_punctuation_array)
        input_value = StringHelper.replace_spaces_with_empty_except_newline_carriage_return(
            input_value=input_value,
            input_space_array=input_space_array)
        return input_value
    @staticmethod
    def remove_defined_punctuations_and_spaces_except_newline_carriage_return( \
        input_value: str) -> str:
        """
        Replace space and punctuations from an input string.
        """
        input_value = StringHelper.replace_defined_punctuations_with_empty(
            input_value=input_value)
        input_value = StringHelper.replace_defined_spaces_with_empty_except_newline_carriage_return(
            input_value=input_value)
        return input_value

    @staticmethod
    def replace_punctuations_and_spaces_except_newline_carriage_return( \
        input_value: str, \
        input_punctuation_array: List[str] = None, \
        input_space_array: List[str] = None, \
        replacement: str = ' ') -> str:
        """
        Replace space and punctuations from an input string.
        """
        input_value = StringHelper.replace_punctuations(
            input_value=input_value,
            input_punctuation_array=input_punctuation_array,
            replacement=replacement)
        input_value = StringHelper.replace_spaces_except_newline_carriage_return(
            input_value=input_value,
            input_space_array=input_space_array,
            replacement=replacement)
        return input_value
    @staticmethod
    def replace_punctuations_and_spaces_with_space_except_newline_carriage_return( \
        input_value: str, \
        input_punctuation_array: List[str] = None, \
        input_space_array: List[str] = None) -> str:
        """
        Replace space and punctuations from an input string.
        """
        input_value = StringHelper.replace_punctuations_with_space(
            input_value=input_value,
            input_punctuation_array=input_punctuation_array)
        input_value = StringHelper.replace_spaces_with_space_except_newline_carriage_return(
            input_value=input_value,
            input_space_array=input_space_array)
        return input_value
    @staticmethod
    def replace_punctuations_and_spaces_with_empty_except_newline_carriage_return( \
        input_value: str, \
        input_punctuation_array: List[str] = None, \
        input_space_array: List[str] = None) -> str:
        """
        Replace space and punctuations from an input string.
        """
        input_value = StringHelper.replace_punctuations_with_empty(
            input_value=input_value,
            input_punctuation_array=input_punctuation_array)
        input_value = StringHelper.replace_spaces_with_empty_except_newline_carriage_return(
            input_value=input_value,
            input_space_array=input_space_array)
        return input_value
    @staticmethod
    def replace_defiend_punctuations_and_spaces_except_newline_carriage_return( \
        input_value: str, \
        replacement: str = ' ') -> str:
        """
        Replace space and punctuations from an input string.
        """
        input_value = StringHelper.replace_defined_punctuations(
            input_value=input_value,
            replacement=replacement)
        input_value = StringHelper.replace_defined_spaces_except_newline_carriage_return(
            input_value=input_value,
            replacement=replacement)
        return input_value
    @staticmethod
    def replace_defiend_punctuations_and_spaces_with_space_except_newline_carriage_return( \
        input_value: str) -> str:
        """
        Replace space and punctuations from an input string.
        """
        input_value = StringHelper.replace_defined_punctuations_with_space(
            input_value=input_value)
        input_value = StringHelper.replace_defined_spaces_with_space_except_newline_carriage_return(
            input_value=input_value)
        return input_value
    @staticmethod
    def replace_defiend_punctuations_and_spaces_with_empty_except_newline_carriage_return( \
        input_value: str) -> str:
        """
        Replace space and punctuations from an input string.
        """
        input_value = StringHelper.replace_defined_punctuations_with_empty(
            input_value=input_value)
        input_value = StringHelper.replace_defined_spaces_with_empty_except_newline_carriage_return(
            input_value=input_value)
        return input_value

    @staticmethod
    def contains_punctuations_or_spaces_except_newline_carriage_return( \
        input_value: str, \
        input_punctuation_array: List[str] = None, \
        input_space_array: List[str] = None) -> bool:
        """
        Replace space and punctuations from an input string.
        """
        has_punctuation: bool = StringHelper.contains_punctuations(
            input_value=input_value,
            input_punctuation_array=input_punctuation_array)
        if has_punctuation:
            return True
        has_space: bool = StringHelper.contains_spaces_except_newline_carriage_return(
            input_value=input_value,
            input_space_array=input_space_array)
        if has_space:
            return True
        return False
    @staticmethod
    def contains_defined_punctuations_or_spaces_except_newline_carriage_return( \
        input_value: str) -> bool:
        """
        Replace space and punctuations from an input string.
        """
        has_punctuation: bool = StringHelper.contains_defined_punctuations(
            input_value=input_value)
        if has_punctuation:
            return True
        has_space: bool = StringHelper.contains_defined_spaces_except_newline_carriage_return(
            input_value=input_value)
        if has_space:
            return True
        return False

    @staticmethod
    def remove_non_variable_legitimate_letters( \
        input_value: str, \
        input_variable_legitimate_letter_set: Set[str] = None) -> str:
        """
        Remove a non-variable-legal letter with a replacement string.
        """
        return StringHelper.replace_non_variable_legitimate_letters_with_empty(
            input_value=input_value,
            input_variable_legitimate_letter_set=input_variable_legitimate_letter_set)
    @staticmethod
    def remove_defined_non_variable_legitimate_letters( \
        input_value: str) -> str:
        """
        Remove a non-variable-legal letter with a replacement string.
        """
        return StringHelper.replace_defined_non_variable_legitimate_letters_with_empty(
            input_value=input_value)
    @staticmethod
    def remove_punctuations( \
        input_value: str, \
        input_punctuation_array: List[str] = None) -> str:
        """
        Remove a punctuation with a replacement string.
        """
        return StringHelper.replace_punctuations_with_empty(
            input_value=input_value,
            input_punctuation_array=input_punctuation_array)
    @staticmethod
    def remove_defined_punctuations( \
        input_value: str) -> str:
        """
        Remove a punctuation with a replacement string.
        """
        return StringHelper.replace_defined_punctuations_with_empty(
            input_value=input_value)
    @staticmethod
    def remove_spaces( \
        input_value: str, \
        input_space_array: List[str] = None) -> str:
        """
        Remove a space with a replacement string.
        """
        return StringHelper.replace_spaces_with_empty(
            input_value=input_value,
            input_space_array=input_space_array)
    @staticmethod
    def remove_defined_spaces( \
        input_value: str) -> str:
        """
        Remove a space with a replacement string.
        """
        return StringHelper.replace_defined_spaces_with_empty(
            input_value=input_value)
    @staticmethod
    def remove_spaces_except_newline_carriage_return( \
        input_value: str, \
        input_space_array: List[str] = None) -> str:
        """
        Remove a space with a replacement string.
        """
        return StringHelper.replace_spaces_with_empty_except_newline_carriage_return(
            input_value=input_value,
            input_space_array=input_space_array)
    @staticmethod
    def remove_defined_spaces_except_newline_carriage_return( \
        input_value: str) -> str:
        """
        Remove a space with a replacement string.
        """
        return StringHelper.replace_defined_spaces_with_empty_except_newline_carriage_return(
            input_value=input_value)

    @staticmethod
    def replace_non_variable_legitimate_letters( \
        input_value: str, \
        input_variable_legitimate_letter_set: Set[str] = None, \
        replacement: str = ' ') -> str:
        """
        Replace a non-variable-legal letter with a replacement string.
        """
        if input_variable_legitimate_letter_set is None:
            input_variable_legitimate_letter_set = \
                StringHelper.LanguageTokenVariableLegitimateLetters
        input_value = map(
            lambda x: x if x in input_variable_legitimate_letter_set \
                else replacement,
            input_value)
        return ''.join(input_value)
    @staticmethod
    def replace_non_variable_legitimate_letters_with_space( \
        input_value: str, \
        input_variable_legitimate_letter_set: Set[str] = None) -> str:
        """
        Replace a non-variable-legal letter with a replacement string.
        """
        if input_variable_legitimate_letter_set is None:
            input_variable_legitimate_letter_set = \
                StringHelper.LanguageTokenVariableLegitimateLetters
        input_value = map(
            lambda x: x if x in input_variable_legitimate_letter_set \
                else ' ',
            input_value)
        return ''.join(input_value)
    @staticmethod
    def replace_non_variable_legitimate_letters_with_empty( \
        input_value: str, \
        input_variable_legitimate_letter_set: Set[str] = None) -> str:
        """
        Replace a non-variable-legal letter with a replacement string.
        """
        if input_variable_legitimate_letter_set is None:
            input_variable_legitimate_letter_set = \
                StringHelper.LanguageTokenVariableLegitimateLetters
        input_value = map(
            lambda x: x if x in input_variable_legitimate_letter_set \
                else '',
            input_value)
        return ''.join(input_value)
    @staticmethod
    def replace_defined_non_variable_legitimate_letters( \
        input_value: str, \
        replacement: str = ' ') -> str:
        """
        Replace a non-variable-legal letter with a replacement string.
        """
        input_value = map(
            lambda x: x if x in StringHelper.LanguageTokenVariableLegitimateLetters \
                else replacement,
            input_value)
        return ''.join(input_value)
    @staticmethod
    def replace_defined_non_variable_legitimate_letters_with_space( \
        input_value: str) -> str:
        """
        Replace a non-variable-legal letter with a replacement string.
        """
        input_value = map(
            lambda x: x if x in StringHelper.LanguageTokenVariableLegitimateLetters \
                else ' ',
            input_value)
        return ''.join(input_value)
    @staticmethod
    def replace_defined_non_variable_legitimate_letters_with_empty( \
        input_value: str) -> str:
        """
        Replace a non-variable-legal letter with a replacement string.
        """
        input_value = map(
            lambda x: x if x in StringHelper.LanguageTokenVariableLegitimateLetters \
                else '',
            input_value)
        return ''.join(input_value)
    @staticmethod
    def replace_punctuations( \
        input_value: str, \
        input_punctuation_array: List[str] = None, \
        replacement: str = ' ') -> str:
        """
        Replace a punctuation with a replacement string.
        """
        if input_punctuation_array is None:
            input_punctuation_array = \
                StringHelper.LanguageTokenPunctuationDelimiters
        for punctuation in input_punctuation_array:
            input_value = input_value.replace(punctuation, replacement)
        return input_value
    @staticmethod
    def replace_punctuations_with_space( \
        input_value: str, \
        input_punctuation_array: List[str] = None) -> str:
        """
        Replace a punctuation with a replacement string.
        """
        if input_punctuation_array is None:
            input_punctuation_array = \
                StringHelper.LanguageTokenPunctuationDelimiters
        for punctuation in input_punctuation_array:
            input_value = input_value.replace(punctuation, ' ')
        return input_value
    @staticmethod
    def replace_punctuations_with_empty( \
        input_value: str, \
        input_punctuation_array: List[str] = None) -> str:
        """
        Replace a punctuation with a replacement string.
        """
        if input_punctuation_array is None:
            input_punctuation_array = \
                StringHelper.LanguageTokenPunctuationDelimiters
        for punctuation in input_punctuation_array:
            input_value = input_value.replace(punctuation, '')
        return input_value
    @staticmethod
    def replace_defined_punctuations( \
        input_value: str, \
        replacement: str = ' ') -> str:
        """
        Replace a punctuation with a replacement string.
        """
        for punctuation in StringHelper.LanguageTokenPunctuationDelimiters:
            input_value = input_value.replace(punctuation, replacement)
        return input_value
    @staticmethod
    def replace_defined_punctuations_with_space( \
        input_value: str) -> str:
        """
        Replace a punctuation with a replacement string.
        """
        for punctuation in StringHelper.LanguageTokenPunctuationDelimiters:
            input_value = input_value.replace(punctuation, ' ')
        return input_value
    @staticmethod
    def replace_defined_punctuations_with_empty( \
        input_value: str) -> str:
        """
        Replace a punctuation with a replacement string.
        """
        for punctuation in StringHelper.LanguageTokenPunctuationDelimiters:
            input_value = input_value.replace(punctuation, '')
        return input_value
    @staticmethod
    def replace_spaces( \
        input_value: str, \
        input_space_array: List[str] = None, \
        replacement: str = ' ') -> str:
        """
        Replace a space with a replacement string.
        """
        if input_space_array is None:
            input_space_array = \
                StringHelper.LanguageTokenSpaceDelimiters
        for space in input_space_array:
            input_value = input_value.replace(space, replacement)
        return input_value
    @staticmethod
    def replace_spaces_with_space( \
        input_value: str, \
        input_space_array: List[str] = None) -> str:
        """
        Replace a space with a replacement string.
        """
        if input_space_array is None:
            input_space_array = \
                StringHelper.LanguageTokenSpaceDelimiters
        for space in input_space_array:
            input_value = input_value.replace(space, ' ')
        return input_value
    @staticmethod
    def replace_spaces_with_empty( \
        input_value: str, \
        input_space_array: List[str] = None) -> str:
        """
        Replace a space with a replacement string.
        """
        if input_space_array is None:
            input_space_array = \
                StringHelper.LanguageTokenSpaceDelimiters
        for space in input_space_array:
            input_value = input_value.replace(space, '')
        return input_value
    @staticmethod
    def replace_defined_spaces( \
        input_value: str, \
        replacement: str = ' ') -> str:
        """
        Replace a space with a replacement string.
        """
        for space in StringHelper.LanguageTokenSpaceDelimiters:
            input_value = input_value.replace(space, replacement)
        return input_value
    @staticmethod
    def replace_defined_spaces_with_space( \
        input_value: str) -> str:
        """
        Replace a space with a replacement string.
        """
        for space in StringHelper.LanguageTokenSpaceDelimiters:
            input_value = input_value.replace(space, ' ')
        return input_value
    @staticmethod
    def replace_defined_spaces_with_empty( \
        input_value: str) -> str:
        """
        Replace a space with a replacement string.
        """
        for space in StringHelper.LanguageTokenSpaceDelimiters:
            input_value = input_value.replace(space, '')
        return input_value
    @staticmethod
    def replace_spaces_except_newline_carriage_return( \
        input_value: str, \
        input_space_array: List[str] = None, \
        replacement: str = ' ') -> str:
        """
        Replace a space with a replacement string.
        """
        if input_space_array is None:
            input_space_array = \
                StringHelper.LanguageTokenSpaceExceptNewlineCarriageReturnDelimiters
        for space in input_space_array:
            input_value = input_value.replace(space, replacement)
        return input_value
    @staticmethod
    def replace_spaces_with_space_except_newline_carriage_return( \
        input_value: str, \
        input_space_array: List[str] = None) -> str:
        """
        Replace a space with a replacement string.
        """
        if input_space_array is None:
            input_space_array = \
                StringHelper.LanguageTokenSpaceExceptNewlineCarriageReturnDelimiters
        for space in input_space_array:
            input_value = input_value.replace(space, ' ')
        return input_value
    @staticmethod
    def replace_spaces_with_empty_except_newline_carriage_return( \
        input_value: str, \
        input_space_array: List[str] = None) -> str:
        """
        Replace a space with a replacement string.
        """
        if input_space_array is None:
            input_space_array = \
                StringHelper.LanguageTokenSpaceExceptNewlineCarriageReturnDelimiters
        for space in input_space_array:
            input_value = input_value.replace(space, '')
        return input_value
    @staticmethod
    def replace_defined_spaces_except_newline_carriage_return( \
        input_value: str, \
        replacement: str = ' ') -> str:
        """
        Replace a space with a replacement string.
        """
        for space in StringHelper.LanguageTokenSpaceExceptNewlineCarriageReturnDelimiters:
            input_value = input_value.replace(space, replacement)
        return input_value
    @staticmethod
    def replace_defined_spaces_with_space_except_newline_carriage_return( \
        input_value: str) -> str:
        """
        Replace a space with a replacement string.
        """
        for space in StringHelper.LanguageTokenSpaceExceptNewlineCarriageReturnDelimiters:
            input_value = input_value.replace(space, ' ')
        return input_value
    @staticmethod
    def replace_defined_spaces_with_empty_except_newline_carriage_return( \
        input_value: str) -> str:
        """
        Replace a space with a replacement string.
        """
        for space in StringHelper.LanguageTokenSpaceExceptNewlineCarriageReturnDelimiters:
            input_value = input_value.replace(space, '')
        return input_value

    @staticmethod
    def contains_non_variable_legitimate_letters( \
        input_value: str, \
        input_variable_legitimate_letter_set: Set[str] = None) -> bool:
        """
        Check if the input string contains non-variable-legal letters.
        """
        if input_variable_legitimate_letter_set is None:
            input_variable_legitimate_letter_set = \
                StringHelper.LanguageTokenVariableLegitimateLetters
        for letter in input_value:
            if letter not in input_variable_legitimate_letter_set:
                return True
        return False
    @staticmethod
    def contains_defined_non_variable_legitimate_letters( \
        input_value: str) -> bool:
        """
        Check if the input string contains non-variable-legal letters.
        """
        for letter in input_value:
            if letter not in StringHelper.LanguageTokenVariableLegitimateLetters:
                return True
        return False
    @staticmethod
    def contains_punctuations( \
        input_value: str, \
        input_punctuation_array: List[str] = None) -> bool:
        """
        Check if the input string contains punctuations.
        """
        if input_punctuation_array is None:
            input_punctuation_array = \
                StringHelper.LanguageTokenPunctuationDelimiters
        for punctuation in input_punctuation_array:
            if punctuation in input_value:
                return True
        return False
    @staticmethod
    def contains_defined_punctuations( \
        input_value: str) -> bool:
        """
        Check if the input string contains punctuations.
        """
        for punctuation in StringHelper.LanguageTokenPunctuationDelimiters:
            if punctuation in input_value:
                return True
        return False
    @staticmethod
    def contains_spaces( \
        input_value: str, \
        input_space_array: List[str] = None) -> bool:
        """
        Check if the input string contains spaces.
        """
        if input_space_array is None:
            input_space_array = \
                StringHelper.LanguageTokenSpaceDelimiters
        for space in input_space_array:
            if space in input_value:
                return True
        return False
    @staticmethod
    def contains_defined_spaces( \
        input_value: str) -> bool:
        """
        Check if the input string contains spaces.
        """
        for space in StringHelper.LanguageTokenSpaceDelimiters:
            if space in input_value:
                return True
        return False
    @staticmethod
    def contains_spaces_except_newline_carriage_return( \
        input_value: str, \
        input_space_array: List[str] = None) -> bool:
        """
        Check if the input string contains spaces.
        """
        if input_space_array is None:
            input_space_array = \
                StringHelper.LanguageTokenSpaceExceptNewlineCarriageReturnDelimiters
        for space in input_space_array:
            if space in input_value:
                return True
        return False
    @staticmethod
    def contains_defined_spaces_except_newline_carriage_return( \
        input_value: str) -> bool:
        """
        Check if the input string contains spaces.
        """
        for space in StringHelper.LanguageTokenSpaceExceptNewlineCarriageReturnDelimiters:
            if space in input_value:
                return True
        return False

    @staticmethod
    def csv_line_to_list(csv_line: str) -> List[str]:
        """
        Convert a CSV line to a list of strings.
        """
        string_list: List[str] = []
        if not DatatypeHelper.is_none_empty_whitespaces_or_nan(csv_line):
            string_list = csv_line.split(",")
        return string_list
    @staticmethod
    def csv_line_to_set(csv_line: str) -> Set[str]:
        """
        Convert a CSV line to a set of strings.
        """
        string_set: Set[str] = {}
        if not DatatypeHelper.is_none_empty_whitespaces_or_nan(csv_line):
            string_set = set(csv_line.split(","))
        return string_set

    @staticmethod
    def get_revised_removing_upto_last_substring( \
        input_string: str, \
        last_substring: str = '.') -> str:
        """
        get_revised_removing_upto_last_substring()
        """
        last_index_to_substring: str = \
            input_string.rfind(last_substring)
        if last_index_to_substring < 0:
            return input_string
        return input_string[:last_index_to_substring]
    @staticmethod
    def get_last_substring( \
        input_string: str, \
        last_substring: str = '.') -> str:
        """
        get_last_substring()
        """
        last_index_to_substring: str = \
            input_string.rfind(last_substring)
        if last_index_to_substring < 0:
            return ''
        return input_string[last_index_to_substring:]

    @staticmethod
    def get_revised_replacing_upto_last_substring( \
        input_string: str, \
        replacement_substring: str, \
        last_substring: str = '.') -> str:
        """
        get_revised_replacing_upto_last_substring()
        """
        revised: str = StringHelper.get_revised_removing_upto_last_substring( \
            input_string, \
            last_substring)
        return revised + replacement_substring

    # ---- NOTE-PLACE-HOLDER ---- @staticmethod
    # ---- NOTE-PLACE-HOLDER ---- def get_unicode_length_from_utf8_string( \
    # ---- NOTE-PLACE-HOLDER ----     input_string: str) -> str:
    # ---- NOTE-PLACE-HOLDER ----     """
    # ---- NOTE-PLACE-HOLDER ----     get_unicode_length_from_utf8_string()
    # ---- NOTE-PLACE-HOLDER ----     """
    # ---- NOTE-PLACE-HOLDER ----     return len(input_string)
    # ---- NOTE-PLACE-HOLDER ----     # ---- NOTE-PLACE-HOLDER-FOR-REFERENCE-FOR-BINARY-ENCODING ----  return len(input_string.encode().decode('utf-8'))
