# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
This module tests common configuration helper functions defined in configuration_help.
"""

from utility.configuration_helper.configuration_helper import ConfigurationHelper

from utility.debugging_helper.debugging_helper import DebuggingHelper

def main():
    """
    The main() function can quickly test ConfigurationHelper functions
    """
    ConfigurationHelper.reset_all_working_directories(
        root_directory_set_already=False,
        root_directory=None,
        base_directory=None,
        stamp=None,
        clear_up_root_directory=True)
    DebuggingHelper.write_line_to_system_console_out(
        f'ConfigurationHelper.ROOT_DIRECTORY_STAMPED='
        f'{ConfigurationHelper.ROOT_DIRECTORY_STAMPED}')
    ConfigurationHelper.reset_all_working_directories(
        root_directory_set_already=False,
        root_directory=None,
        base_directory=None,
        stamp='app_configuration_helper',
        clear_up_root_directory=True)
    DebuggingHelper.write_line_to_system_console_out(
        f'ConfigurationHelper.ROOT_DIRECTORY_STAMPED='
        f'{ConfigurationHelper.ROOT_DIRECTORY_STAMPED}')

if __name__ == '__main__':
    main()
