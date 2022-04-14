# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
This module tests common datatype helper functions defined in datatype_help.
"""

from utility.datatype_helper.datatype_helper \
    import DatatypeHelper
from utility.debugging_helper.debugging_helper \
    import DebuggingHelper

def main():
    """
    The main() function can quickly test DatatypeHelper functions
    """
    example_function_DatatypeHelper_is_none_empty_whitespaces_or_nan()
    example_function_DatatypeHelper_update_count_dictionary()

def example_function_DatatypeHelper_update_count_dictionary():
    """
    The test function to quickly test DatatypeHelper functions
    """
    # ---- NOTE-PYLINT ---- C0103: Function name "" doesn't conform to snake_case naming style
    # pylint: disable=C0103
    a = {'a':2, 'b':3, 'c':4}
    b = {'a':4, 'd':5}
    DebuggingHelper.write_line_to_system_console_out(
        f'a={str(a)}')
    DatatypeHelper.update_count_dictionary(a, b)
    DebuggingHelper.write_line_to_system_console_out(
        f'a={str(a)}')
    DebuggingHelper.write_line_to_system_console_out(
        f'b={str(b)}')
    a = {'a':2, 'b':3, 'c':4}
    b = {'a':4, 'd':5}
    DebuggingHelper.write_line_to_system_console_out(
        f'a={str(a)}')
    DatatypeHelper.update_int_count_dictionary(a, b)
    DebuggingHelper.write_line_to_system_console_out(
        f'a={str(a)}')
    DebuggingHelper.write_line_to_system_console_out(
        f'b={str(b)}')

def example_function_DatatypeHelper_is_none_empty_whitespaces_or_nan():
    """
    The test function to quickly test DatatypeHelper functions
    """
    # ---- NOTE-PYLINT ---- C0103: Function name "" doesn't conform to snake_case naming style
    # pylint: disable=C0103
    a = None
    DebuggingHelper.write_line_to_system_console_out(
        f'is_none_empty_whitespaces_or_nan({a})='
        f'{DatatypeHelper.is_none_empty_whitespaces_or_nan(a)}')
    a = DatatypeHelper.NAN
    DebuggingHelper.write_line_to_system_console_out(
        f'is_none_empty_whitespaces_or_nan({a})='
        f'{DatatypeHelper.is_none_empty_whitespaces_or_nan(a)}')
    a = 'a'
    DebuggingHelper.write_line_to_system_console_out(
        f'is_none_empty_whitespaces_or_nan({a})='
        f'{DatatypeHelper.is_none_empty_whitespaces_or_nan(a)}')
    a = ''
    DebuggingHelper.write_line_to_system_console_out(
        f'is_none_empty_whitespaces_or_nan({a})='
        f'{DatatypeHelper.is_none_empty_whitespaces_or_nan(a)}')
    a = 1
    DebuggingHelper.write_line_to_system_console_out(
        f'is_none_empty_whitespaces_or_nan({a})='
        f'{DatatypeHelper.is_none_empty_whitespaces_or_nan(a)}')

if __name__ == '__main__':
    main()
