# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
This module tests common debugging helper functions defined in debugging_help.
"""

from colorama import Fore

from utility.debugging_helper.debugging_helper import DebuggingHelper

def example_function_set_up_specific_directory_middle():
    """
    The main() function can quickly test DebuggingHelper functions
    """
    DebuggingHelper.write_line_to_system_console_err(
        f'0 - DebuggingHelper.LOGGER_DICTIONARY={DebuggingHelper.LOGGER_DICTIONARY}')
    DebuggingHelper.write_line_to_system_console_err(
        f'0 - DebuggingHelper.LOGGER_DIRECTORY={DebuggingHelper.LOGGER_DIRECTORY}')
    DebuggingHelper.write_line_to_system_console_err(
        f'0 - DebuggingHelper.LOGGER_TAG={DebuggingHelper.LOGGER_TAG}')
    DebuggingHelper.write_line_to_system_console_out('test-stdout')
    DebuggingHelper.write_line_to_system_console_err('test-stderr')
    DebuggingHelper.write_line_to_system_console_err(
        f'1 - DebuggingHelper.LOGGER_DICTIONARY={DebuggingHelper.LOGGER_DICTIONARY}')
    DebuggingHelper.write_line_to_system_console_err(
        f'1 - DebuggingHelper.LOGGER_DIRECTORY={DebuggingHelper.LOGGER_DIRECTORY}')
    DebuggingHelper.write_line_to_system_console_err(
        f'1 - DebuggingHelper.LOGGER_TAG={DebuggingHelper.LOGGER_TAG}')
    DebuggingHelper.reset_log_and_all_working_directories(
        root_directory_set_already=False,
        root_directory=None,
        base_directory=None,
        stamp='example_function_set_up_specific_directory_middle',
        clear_up_root_directory=True)
    DebuggingHelper.write_line_to_system_console_err(
        f'2 - DebuggingHelper.LOGGER_DICTIONARY={DebuggingHelper.LOGGER_DICTIONARY}')
    DebuggingHelper.write_line_to_system_console_err(
        f'2 - DebuggingHelper.LOGGER_DIRECTORY={DebuggingHelper.LOGGER_DIRECTORY}')
    DebuggingHelper.write_line_to_system_console_err(
        f'2 - DebuggingHelper.LOGGER_TAG={DebuggingHelper.LOGGER_TAG}')
    DebuggingHelper.write_line_to_system_console_out('test-stdout-0')
    DebuggingHelper.write_line_to_system_console_err('test-stderr-0')
    DebuggingHelper.write_line_to_system_console_err(
        f'3 - DebuggingHelper.LOGGER_DICTIONARY={DebuggingHelper.LOGGER_DICTIONARY}')
    DebuggingHelper.write_line_to_system_console_err(
        f'3 - DebuggingHelper.LOGGER_DIRECTORY={DebuggingHelper.LOGGER_DIRECTORY}')
    DebuggingHelper.write_line_to_system_console_err(
        f'3 - DebuggingHelper.LOGGER_TAG={DebuggingHelper.LOGGER_TAG}')
    DebuggingHelper.write_line_to_system_console_err(
        f'3 - DebuggingHelper.LOGGER_TAG={DebuggingHelper.LOGGER_TAG}',
        color_code=Fore.RED)

def example_function_set_up_specific_directory_first():
    """
    The main() function can quickly test DebuggingHelper functions
    """
    DebuggingHelper.reset_log_and_all_working_directories(
        root_directory_set_already=False,
        root_directory=None,
        base_directory=None,
        stamp='example_function_set_up_specific_directory_first',
        clear_up_root_directory=True)
    DebuggingHelper.write_line_to_system_console_err(
        f'0 - DebuggingHelper.LOGGER_DICTIONARY={DebuggingHelper.LOGGER_DICTIONARY}')
    DebuggingHelper.write_line_to_system_console_err(
        f'0 - DebuggingHelper.LOGGER_DIRECTORY={DebuggingHelper.LOGGER_DIRECTORY}')
    DebuggingHelper.write_line_to_system_console_err(
        f'0 - DebuggingHelper.LOGGER_TAG={DebuggingHelper.LOGGER_TAG}')
    DebuggingHelper.write_line_to_system_console_out('test-stdout')
    DebuggingHelper.write_line_to_system_console_err('test-stderr')
    DebuggingHelper.write_line_to_system_console_err(
        f'1 - DebuggingHelper.LOGGER_DICTIONARY={DebuggingHelper.LOGGER_DICTIONARY}')
    DebuggingHelper.write_line_to_system_console_err(
        f'1 - DebuggingHelper.LOGGER_DIRECTORY={DebuggingHelper.LOGGER_DIRECTORY}')
    DebuggingHelper.write_line_to_system_console_err(
        f'1 - DebuggingHelper.LOGGER_TAG={DebuggingHelper.LOGGER_TAG}')
    DebuggingHelper.write_line_to_system_console_err(
        f'2 - DebuggingHelper.LOGGER_DICTIONARY={DebuggingHelper.LOGGER_DICTIONARY}')
    DebuggingHelper.write_line_to_system_console_err(
        f'2 - DebuggingHelper.LOGGER_DIRECTORY={DebuggingHelper.LOGGER_DIRECTORY}')
    DebuggingHelper.write_line_to_system_console_err(
        f'2 - DebuggingHelper.LOGGER_TAG={DebuggingHelper.LOGGER_TAG}')
    DebuggingHelper.write_line_to_system_console_out('test-stdout-0')
    DebuggingHelper.write_line_to_system_console_err('test-stderr-0')
    DebuggingHelper.write_line_to_system_console_err(
        f'3 - DebuggingHelper.LOGGER_DICTIONARY={DebuggingHelper.LOGGER_DICTIONARY}')
    DebuggingHelper.write_line_to_system_console_err(
        f'3 - DebuggingHelper.LOGGER_DIRECTORY={DebuggingHelper.LOGGER_DIRECTORY}')
    DebuggingHelper.write_line_to_system_console_err(
        f'3 - DebuggingHelper.LOGGER_TAG={DebuggingHelper.LOGGER_TAG}')
    DebuggingHelper.write_line_to_system_console_err(
        f'3 - DebuggingHelper.LOGGER_TAG={DebuggingHelper.LOGGER_TAG}',
        color_code=Fore.RED)

def main():
    """
    The main() function can quickly test DebuggingHelper functions
    """
    example_function_set_up_specific_directory_first()
    # example_function_set_up_specific_directory_middle()

if __name__ == '__main__':
    main()
