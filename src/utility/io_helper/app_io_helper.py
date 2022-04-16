# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
This module tests common IO helper functions defined in io_help.
"""

import inspect
import os

from utility.io_helper.io_helper import IoHelper

from utility.debugging_helper.debugging_helper import DebuggingHelper

def main():
    """
    The main() function can quickly test IoHelper functions
    """
    filename = 'sdfwsdfqwefg'
    can_open_file = IoHelper.can_open_file(filename)
    DebuggingHelper.write_line_to_system_console_out(
        f'can_open_file("{filename}") = {can_open_file}')
    filename = inspect.getfile(inspect.currentframe())
    directoryname = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    DebuggingHelper.write_line_to_system_console_out(
        f'filename="{filename}"') # script filename (usually with path)
    DebuggingHelper.write_line_to_system_console_out(
        f'directoryname="{directoryname}"') # script directoryname (usually with path)
    can_open_file = IoHelper.can_open_file(filename)
    DebuggingHelper.write_line_to_system_console_out(
        f'can_open_file("{filename}") = {can_open_file}')
    path_exists = IoHelper.exists(directoryname)
    DebuggingHelper.write_line_to_system_console_out(
        f'path_exists("{directoryname}") = {path_exists}')
    DebuggingHelper.write_line_to_system_console_out(
        f'IoHelper.get_executable_path()={IoHelper.get_executable_path()}')
    DebuggingHelper.write_line_to_system_console_out(
        f'IoHelper.get_root_path()={IoHelper.get_root_path()}')
    DebuggingHelper.write_line_to_system_console_out(
        f'IoHelper.get_root()={IoHelper.get_root()}')
    absolute_path = os.path.join(directoryname, filename)
    DebuggingHelper.write_line_to_system_console_out(
        f'absolute_path={absolute_path}')
    DebuggingHelper.write_line_to_system_console_out(
        f'IoHelper.basename(absolute_path)={IoHelper.basename(absolute_path)}')
    DebuggingHelper.write_line_to_system_console_out(
        f'IoHelper.head(absolute_path)={IoHelper.head(absolute_path)}')
    DebuggingHelper.write_line_to_system_console_out(
        f'IoHelper.tail(absolute_path)={IoHelper.tail(absolute_path)}')
    DebuggingHelper.write_line_to_system_console_out(
        f'IoHelper.splitext_head(absolute_path)={IoHelper.splitext_head(absolute_path)}')
    DebuggingHelper.write_line_to_system_console_out(
        f'IoHelper.splitext_tail(absolute_path)={IoHelper.splitext_tail(absolute_path)}')
    absolute_path = directoryname
    DebuggingHelper.write_line_to_system_console_out(
        f'absolute_path={absolute_path}')
    DebuggingHelper.write_line_to_system_console_out(
        f'IoHelper.basename(absolute_path)={IoHelper.basename(absolute_path)}')
    DebuggingHelper.write_line_to_system_console_out(
        f'IoHelper.head(absolute_path)={IoHelper.head(absolute_path)}')
    DebuggingHelper.write_line_to_system_console_out(
        f'IoHelper.tail(absolute_path)={IoHelper.tail(absolute_path)}')
    DebuggingHelper.write_line_to_system_console_out(
        f'IoHelper.splitext_head(absolute_path)={IoHelper.splitext_head(absolute_path)}')
    DebuggingHelper.write_line_to_system_console_out(
        f'IoHelper.splitext_tail(absolute_path)={IoHelper.splitext_tail(absolute_path)}')

if __name__ == '__main__':
    main()
