# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
This module tests common string helper functions defined in string_helper.
"""

from typing import Set

from utility.string_helper.string_helper import StringHelper

from utility.debugging_helper.debugging_helper import DebuggingHelper

def main_test_string_normalization():
    """
    The main_test_string_normalization() function can quickly test StringHelper functions
    """
    test_punctuation_string: str = 'hello, world.'
    DebuggingHelper.write_line_to_system_console_out(
        f'test_punctuation_string=${test_punctuation_string}$')
    test_punctuation_string_replaced = \
        StringHelper.replace_punctuations_and_spaces(
            test_punctuation_string)
    DebuggingHelper.write_line_to_system_console_out(
        f'test_punctuation_string_replaced=${test_punctuation_string_replaced}$')
    test_punctuation_string_removed = \
        StringHelper.remove_punctuations_and_spaces(
            test_punctuation_string)
    DebuggingHelper.write_line_to_system_console_out(
        f'test_punctuation_string_removed=${test_punctuation_string_removed}$')
    test_punctuation_string = 'hel\'lo, world.'
    DebuggingHelper.write_line_to_system_console_out(
        f'test_punctuation_string=${test_punctuation_string}$')
    test_punctuation_string_replaced = \
        StringHelper.replace_punctuations_and_spaces(
            test_punctuation_string)
    DebuggingHelper.write_line_to_system_console_out(
        f'test_punctuation_string_replaced=${test_punctuation_string_replaced}$')
    test_punctuation_string_removed = \
        StringHelper.remove_punctuations_and_spaces(
            test_punctuation_string)
    DebuggingHelper.write_line_to_system_console_out(
        f'test_punctuation_string_removed=${test_punctuation_string_removed}$')
    test_non_variable_legitimate_letters = 'hel\'lo, wor’ld.'
    DebuggingHelper.write_line_to_system_console_out(
        f'test_non_variable_legitimate_letters=${test_non_variable_legitimate_letters}$')
    test_non_variable_legitimate_letters_replaced = \
        StringHelper.replace_non_variable_legitimate_letters(
            test_non_variable_legitimate_letters)
    # ---- NOTE-PYLINT ---- C0301: Line too long
    # pylint: disable=C0301
    DebuggingHelper.write_line_to_system_console_out(
        f'test_non_variable_legitimate_letters_replaced=${test_non_variable_legitimate_letters_replaced}$')
    test_non_variable_legitimate_letters_removed = \
        StringHelper.remove_non_variable_legitimate_letters(
            test_non_variable_legitimate_letters)
    # ---- NOTE-PYLINT ---- C0301: Line too long
    # pylint: disable=C0301
    DebuggingHelper.write_line_to_system_console_out(
        f'test_non_variable_legitimate_letters_removed=${test_non_variable_legitimate_letters_removed}$')

def main_test_is_in():
    """
    The main_test_is_in() function can quickly test StringHelper functions
    """
    input_set: Set[str] = {'a', 'A'}
    input_value_a: str = 'a'
    is_in_a: bool = StringHelper.is_in(input_value_a, input_set)
    DebuggingHelper.write_line_to_system_console_out(
        f'is_in_a=${is_in_a}$')
    input_value_b: str = 'b'
    is_in_b: bool = StringHelper.is_in(input_value_b, input_set)
    DebuggingHelper.write_line_to_system_console_out(
        f'is_in_b=${is_in_b}$')

def main():
    """
    The main() function can quickly test StringHelper functions
    """
    main_test_string_normalization()
    main_test_is_in()

if __name__ == '__main__':
    main()
