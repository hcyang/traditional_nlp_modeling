# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
This module provides some common debugging helper functions.
"""

from typing import Any

import os
import sys
import inspect
import datetime
import logging
import shutil
from typing import NoReturn

# ---- NOTE-PYLINT ---- W0611: Unused Back imported from colorama (unused-import)
# ---- NOTE-PYLINT ---- W0611: Unused Style imported from colorama (unused-import)
# pylint: disable=W0611
from colorama import Fore, Back, Style
from colorama import init as init_colorama
# pylint: enable=W0611

from utility.configuration_helper.configuration_helper \
    import ConfigurationHelper

init_colorama(autoreset=True)

class DebuggingHelperBase:
    """
    DebuggingHelperBase
    """
    # ---- NOTE-PYLINT ---- R0903: Too few public methods
    # pylint: disable=R0903
    LOGGER_DIRECTORY_BASE = 'log'

class DebuggingHelper:
    """
    This class contains some common functions for debugging purpose.
    REFERENCE: https://docs.python.org/3/library/logging.html#levels
    """
    # ---- NOTE-PYLINT ---- R0904: Too many public methods
    # pylint: disable=R0904

    LOGGER_DICTIONARY = {}
    LOGGER_DIRECTORY = os.path.join(
        ConfigurationHelper.ROOT_DIRECTORY_STAMPED, DebuggingHelperBase.LOGGER_DIRECTORY_BASE)
    LOGGER_TAG = ''

    @staticmethod
    def print_in_color(message: str, color_code: int = Fore.RED):
        """
        Print message in color, default to red.
        """
        print(color_code + message)

    @staticmethod
    def print_stderr_in_color(message: str, color_code: int = Fore.RED):
        """
        Print message in color, default to red.
        """
        print(color_code + message, file=sys.stderr)

    @staticmethod
    def reset_log_and_all_working_directories( \
        root_directory_set_already: bool = True, \
        root_directory: str = None, \
        base_directory: str = None, \
        stamp=None, \
        clear_up_root_directory: bool = False, \
        use_numbered_log_directory: bool = True):
        """
        Reset all directories using new root_directory, base_directory,
        and stamp.
        """
        # ---- NOTE-PYLINT ---- R0913: Too many arguments
        # pylint: disable=R0913
        ConfigurationHelper.reset_all_working_directories(
            root_directory_set_already=root_directory_set_already,
            root_directory=root_directory,
            base_directory=base_directory,
            stamp=stamp,
            clear_up_root_directory=clear_up_root_directory)
        if use_numbered_log_directory:
            DebuggingHelper.reset_logger_numbered_directory()
        else:
            DebuggingHelper.reset_logger_directory()

    @staticmethod
    def clear_logger_dictionary():
        """
        Reset the logger stamped directory.
        """
        for entry in DebuggingHelper.LOGGER_DICTIONARY:
            logger = DebuggingHelper.LOGGER_DICTIONARY[entry]
            handlers = logger.handlers[:]
            for handler in handlers:
                handler.close()
                logger.removeHandler(handler)
        DebuggingHelper.LOGGER_DICTIONARY = {}
        return DebuggingHelper.LOGGER_DICTIONARY

    @staticmethod
    def reset_logger_directory( \
        logger_directory_base: str = DebuggingHelperBase.LOGGER_DIRECTORY_BASE):
        """
        Reset the logger stamped directory.
        """
        if logger_directory_base is None:
            logger_directory_base = DebuggingHelperBase.LOGGER_DIRECTORY_BASE
        DebuggingHelper.clear_logger_dictionary()
        DebuggingHelper.LOGGER_DIRECTORY = os.path.join(
            ConfigurationHelper.ROOT_DIRECTORY_STAMPED,
            logger_directory_base)
        shutil.rmtree(
            DebuggingHelper.LOGGER_DIRECTORY,
            ignore_errors=True)
        return DebuggingHelper.LOGGER_DIRECTORY

    @staticmethod
    def reset_logger_numbered_directory( \
        logger_directory_base: str = DebuggingHelperBase.LOGGER_DIRECTORY_BASE):
        """
        Reset the logger stamped directory.
        """
        if logger_directory_base is None:
            logger_directory_base = DebuggingHelperBase.LOGGER_DIRECTORY_BASE
        max_log_directory_number: int = 0
        root_log_directory = os.path.join(
            ConfigurationHelper.ROOT_DIRECTORY_STAMPED,
            logger_directory_base)
        if os.path.exists(root_log_directory):
            for filename in os.listdir(root_log_directory):
                if filename.startswith(logger_directory_base):
                    # print(f'FILENAME='
                    #       f'{os.path.join(root_log_directory, filename)}')
                    number_string = filename[len(logger_directory_base)+1:]
                    try:
                        number = int(number_string)
                        if number >= max_log_directory_number:
                            max_log_directory_number = number + 1
                    except ValueError:
                        # print(f'EXCEPTION')
                        continue
        logger_directory: str = f'{logger_directory_base}_{max_log_directory_number}'
        DebuggingHelper.clear_logger_dictionary()
        DebuggingHelper.LOGGER_DIRECTORY = os.path.join(
            root_log_directory,
            logger_directory)
        shutil.rmtree(
            DebuggingHelper.LOGGER_DIRECTORY,
            ignore_errors=True)
        return DebuggingHelper.LOGGER_DIRECTORY

    @staticmethod
    def reset_logger_tag(logging_tag: str = None):
        """
        Reset logger tag.
        """
        DebuggingHelper.LOGGER_TAG = logging_tag

    @staticmethod
    def get_loggers( \
        name: str = __name__, \
        log_filename: str = None, \
        logging_level=logging.DEBUG, \
        to_initialize_log_file: bool = False, \
        initialize_logger_with_encoding: str = 'utf-8'):
        """
        Return a named logger with the default logger.
        """
        default_logger, default_file_handler = DebuggingHelper.get_logger( \
            name='_default_', \
            log_filename=None, \
            logging_level=logging.DEBUG, \
            to_initialize_log_file=to_initialize_log_file, \
            initialize_logger_with_encoding=initialize_logger_with_encoding)
        designated_logger, designated_file_handler = DebuggingHelper.get_logger( \
            name=name, \
            log_filename=log_filename, \
            logging_level=logging_level, \
            to_initialize_log_file=to_initialize_log_file, \
            initialize_logger_with_encoding=initialize_logger_with_encoding)
        return (default_logger, designated_logger, default_file_handler, designated_file_handler)

    @staticmethod
    def get_logger( \
        name: str = __name__, \
        log_filename: str = None, \
        logging_level=logging.DEBUG, \
        to_initialize_log_file: bool = True, \
        initialize_logger_with_encoding: str = 'utf-8'):
        """
        Return a configured logger.
        """
        log_file_basename: str = name
        if name is None:
            name = ''
            log_file_basename = name
        else:
            log_file_basename, _ = os.path.splitext(name)
        if name in DebuggingHelper.LOGGER_DICTIONARY:
            return DebuggingHelper.LOGGER_DICTIONARY[name]
        log_file_basename: str = os.path.basename(log_file_basename)
        if log_filename is None:
            log_filename = f'logging_{DebuggingHelper.LOGGER_TAG}_{log_file_basename}.log'
        if DebuggingHelper.LOGGER_DIRECTORY is not None:
            if not os.path.exists(DebuggingHelper.LOGGER_DIRECTORY):
                os.makedirs(DebuggingHelper.LOGGER_DIRECTORY)
            log_filename = os.path.join(DebuggingHelper.LOGGER_DIRECTORY, log_filename)
        if to_initialize_log_file:
            if os.path.exists(log_filename):
                os.remove(log_filename)
        logging_format = logging.Formatter('{%(asctime)s - %(name)s - %(levelname)s}: %(message)s')
        # stream_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(log_filename, encoding=initialize_logger_with_encoding)
        # stream_handler.setFormatter(logging_format)
        file_handler.setFormatter(logging_format)
        # stream_handler.setLevel(logging_level)
        file_handler.setLevel(logging_level)
        logger = logging.getLogger(name)
        # logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
        logger.setLevel(logging_level)
        DebuggingHelper.LOGGER_DICTIONARY[name] = (logger, file_handler)
        # print(f'CHECKPOINT-logger={str(logger)}')
        DebuggingHelper.log(
            logging_level=logging_level,
            message=f'LOGGER-STARTED, name=[{name}]'
                    f', logging_level=[{logging.getLevelName(logging_level)}:{logging_level}]'
                    f', log_filename=[{log_filename}]',
            logger=logger)
        file_handler.flush()
        return (logger, file_handler)

    CALLER_FRAME_INDEX_RELATIVE_TO_CURRENT_FRAME_FOR_get_caller_arguments: int = 2

    @staticmethod
    def get_caller_arguments(
            # ---- NOTE-PYLINT ---- C0301: Line too long
            # pylint: disable=C0301
            caller_frame_index_relative_to_current_frame:int = \
                CALLER_FRAME_INDEX_RELATIVE_TO_CURRENT_FRAME_FOR_get_caller_arguments):
        """
        Get caller's arguments.
        """
        if caller_frame_index_relative_to_current_frame is None:
            caller_frame_index_relative_to_current_frame=\
                DebuggingHelper.CALLER_FRAME_INDEX_RELATIVE_TO_CURRENT_FRAME_FOR_get_caller_arguments
        current_frame = inspect.currentframe()
        caller_frame = current_frame
        i = 0
        while i < caller_frame_index_relative_to_current_frame:
            caller_frame = caller_frame.f_back
            i = i + 1
        # ---- NOTE-PYLINT ---- W1505: Using deprecated method getargvalues() (deprecated-method)
        # pylint: disable=W1505
        args, _, _, locals_data = inspect.getargvalues(caller_frame)
        # NOTE: > pylint warns that getargvalues() is deprecated, but from
        #         https://docs.python.org/3/library/inspect.html
        #           Note This function was inadvertently marked as deprecated in Python 3.5.
        #               inspect.getargvalues(frame)
        #                   Get information about arguments passed into a particular frame.
        #                   A named tuple ArgInfo(args, varargs, keywords, locals) is returned.
        #                   args is a list of the argument names.
        #                   varargs and keywords are the names of the * and ** arguments or None.
        #                   locals is the locals dictionary of the given frame.
        #       > locals_data has been reamed from locals from a pylint warning.
        #
        return [(i, locals_data[i]) for i in args]

    CALLER_FRAME_INDEX_RELATIVE_TO_CURRENT_FRAME_FOR_get_caller_information_in_string: int = 2

    @staticmethod
    def get_caller_information_in_string(
            # ---- NOTE-PYLINT ---- C0301: Line too long
            # pylint: disable=C0301
            caller_frame_index_relative_to_current_frame:int = \
                CALLER_FRAME_INDEX_RELATIVE_TO_CURRENT_FRAME_FOR_get_caller_information_in_string):
        """
        Get caller information in string.
        """
        if caller_frame_index_relative_to_current_frame is None:
            caller_frame_index_relative_to_current_frame=\
                DebuggingHelper.CALLER_FRAME_INDEX_RELATIVE_TO_CURRENT_FRAME_FOR_get_caller_information_in_string
        current_frame = inspect.currentframe()
        caller_frame = current_frame
        i = 0
        while i < caller_frame_index_relative_to_current_frame:
            caller_frame = caller_frame.f_back
            i = i + 1
        (filename, line_number, function_name, lines, index) = inspect.getframeinfo(caller_frame)
        # NOTE: getframeinfo(frame, context=1):
        #           Get information about a frame or traceback object.
        #           A tuple of five things is returned: the filename, the line number of
        #           the current line, the function name, a list of lines of context from
        #           the source code, and the index of the current line within that list.
        #           The optional second argument specifies the number of lines of context
        #           to return, which are centered around the current line.
        caller_information_in_string = '[{} @ {}:{}]'.format(function_name, filename, line_number)
        return (filename, line_number, function_name, lines, index, caller_information_in_string)

    @staticmethod
    def get_timestamp_in_string(now=None):
        """
        Get timestamp information in string.
        """
        if now is None:
            now = datetime.datetime.now()
        return '[{}{}{}-{}:{}:{}.{}]'.format(
            now.year,
            now.month,
            now.day,
            now.hour,
            now.minute,
            now.second,
            now.microsecond)
    @staticmethod
    def get_timestamp_in_iso_format_string(now=None):
        """
        Get timestamp information in ISO-format string.
        """
        if now is None:
            now = datetime.datetime.now()
        return f'[{now.isoformat()}]'

    CALLER_FRAME_INDEX_RELATIVE_TO_CURRENT_FRAME_FOR_throw_exception: int = 2

    @staticmethod
    def ensure(condition: bool, message: str, \
        utf8_conversion: bool = False, \
        caller_frame_index_relative_to_current_frame:int = \
            CALLER_FRAME_INDEX_RELATIVE_TO_CURRENT_FRAME_FOR_throw_exception):
        """
        Assert a condition. Throw if the condition is not met.
        """
        if not condition:
            DebuggingHelper.throw_exception( \
                input_message=message, \
                utf8_conversion=utf8_conversion, \
                caller_frame_index_relative_to_current_frame=caller_frame_index_relative_to_current_frame + 1)

    @staticmethod
    def throw_exception(input_message: str, \
        utf8_conversion: bool = False, \
        caller_frame_index_relative_to_current_frame:int = \
            CALLER_FRAME_INDEX_RELATIVE_TO_CURRENT_FRAME_FOR_throw_exception):
        """
        Write a debugging information message to console standard output.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        message: str = input_message
        if utf8_conversion:
            message = DebuggingHelper.str_utf8(input_object=input_message)
        if caller_frame_index_relative_to_current_frame is None:
            caller_frame_index_relative_to_current_frame=\
                DebuggingHelper.CALLER_FRAME_INDEX_RELATIVE_TO_CURRENT_FRAME_FOR_throw_exception
        timestamp_in_string = DebuggingHelper.get_timestamp_in_iso_format_string()
        (filename, _, _, _, _, caller_information_in_string) = \
            DebuggingHelper.get_caller_information_in_string(
                caller_frame_index_relative_to_current_frame=\
                    caller_frame_index_relative_to_current_frame)
        logging_level = logging.CRITICAL
        message = \
            f'EXCEPTION: {timestamp_in_string}{caller_information_in_string}' \
            f'[{logging.getLevelName(logging_level)}:{logging_level}]: {message}'
        default_logger, designated_logger, default_file_handler, designated_file_handler = DebuggingHelper.get_loggers(
            name=filename)
        DebuggingHelper.log(
            logging_level=logging_level,
            message=message,
            logger=default_logger)
        DebuggingHelper.log(
            logging_level=logging_level,
            message=message,
            logger=designated_logger)
        default_file_handler.flush()
        designated_file_handler.flush()
        raise Exception(message)

    CALLER_FRAME_INDEX_RELATIVE_TO_CURRENT_FRAME_FOR_write_line_to_system_console_out: int = 2

    @staticmethod
    def write_line_to_system_console_out_notset_in_red( \
        message: str, \
        utf8_conversion: bool = False, \
        caller_frame_index_relative_to_current_frame: int = \
            CALLER_FRAME_INDEX_RELATIVE_TO_CURRENT_FRAME_FOR_write_line_to_system_console_out):
        """
        Write a debugging information message to console standard output at NOTSET level.
        """
        DebuggingHelper.write_line_to_system_console_out_notset(
            message=message,
            color_code=Fore.RED,
            utf8_conversion=utf8_conversion,
            caller_frame_index_relative_to_current_frame=caller_frame_index_relative_to_current_frame + 1)
    @staticmethod
    def write_line_to_system_console_out_notset( \
        message: str, \
        color_code: int = None, \
        utf8_conversion: bool = False, \
        caller_frame_index_relative_to_current_frame: int = \
            CALLER_FRAME_INDEX_RELATIVE_TO_CURRENT_FRAME_FOR_write_line_to_system_console_out):
        """
        Write a debugging information message to console standard output at NOTSET level.
        """
        DebuggingHelper.write_line_to_system_console_out(
            input_message=message,
            logging_level=logging.NOTSET,
            color_code=color_code,
            utf8_conversion=utf8_conversion,
            caller_frame_index_relative_to_current_frame=caller_frame_index_relative_to_current_frame + 1)
    @staticmethod
    def write_line_to_system_console_out_debug_in_red( \
        message: str, \
        utf8_conversion: bool = False, \
        caller_frame_index_relative_to_current_frame: int = \
            CALLER_FRAME_INDEX_RELATIVE_TO_CURRENT_FRAME_FOR_write_line_to_system_console_out):
        """
        Write a debugging information message to console standard output at DEBUG level.
        """
        DebuggingHelper.write_line_to_system_console_out_debug(
            message=message,
            color_code=Fore.RED,
            utf8_conversion=utf8_conversion,
            caller_frame_index_relative_to_current_frame=caller_frame_index_relative_to_current_frame + 1)
    @staticmethod
    def write_line_to_system_console_out_debug( \
        message: str, \
        color_code: int = None, \
        utf8_conversion: bool = False, \
        caller_frame_index_relative_to_current_frame: int = \
            CALLER_FRAME_INDEX_RELATIVE_TO_CURRENT_FRAME_FOR_write_line_to_system_console_out):
        """
        Write a debugging information message to console standard output at DEBUG level.
        """
        DebuggingHelper.write_line_to_system_console_out(
            input_message=message,
            logging_level=logging.DEBUG,
            color_code=color_code,
            utf8_conversion=utf8_conversion,
            caller_frame_index_relative_to_current_frame=caller_frame_index_relative_to_current_frame + 1)
    @staticmethod
    def write_line_to_system_console_out_info_in_red( \
        message: str, \
        utf8_conversion: bool = False, \
        caller_frame_index_relative_to_current_frame: int = \
            CALLER_FRAME_INDEX_RELATIVE_TO_CURRENT_FRAME_FOR_write_line_to_system_console_out):
        """
        Write a debugging information message to console standard output at INFO level.
        """
        DebuggingHelper.write_line_to_system_console_out_info(
            message=message,
            color_code=Fore.RED,
            utf8_conversion=utf8_conversion,
            caller_frame_index_relative_to_current_frame=caller_frame_index_relative_to_current_frame + 1)
    @staticmethod
    def write_line_to_system_console_out_info( \
        message: str, \
        color_code: int = None, \
        utf8_conversion: bool = False, \
        caller_frame_index_relative_to_current_frame: int = \
            CALLER_FRAME_INDEX_RELATIVE_TO_CURRENT_FRAME_FOR_write_line_to_system_console_out):
        """
        Write a debugging information message to console standard output at INFO level.
        """
        DebuggingHelper.write_line_to_system_console_out(
            input_message=message,
            logging_level=logging.INFO,
            color_code=color_code,
            utf8_conversion=utf8_conversion,
            caller_frame_index_relative_to_current_frame=caller_frame_index_relative_to_current_frame + 1)
    @staticmethod
    def write_line_to_system_console_out_in_red( \
        message: str, \
        logging_level=logging.DEBUG, \
        utf8_conversion: bool = False, \
        caller_frame_index_relative_to_current_frame: int = \
            CALLER_FRAME_INDEX_RELATIVE_TO_CURRENT_FRAME_FOR_write_line_to_system_console_out):
        """
        Write a debugging information message to console standard output.
        """
        DebuggingHelper.write_line_to_system_console_out(
            input_message=message,
            logging_level=logging_level,
            color_code=Fore.RED,
            utf8_conversion=utf8_conversion,
            caller_frame_index_relative_to_current_frame=caller_frame_index_relative_to_current_frame + 1)
    @staticmethod
    def write_line_to_system_console_out( \
        input_message: str, \
        logging_level=logging.DEBUG, \
        color_code: int = None, \
        utf8_conversion: bool = False, \
        caller_frame_index_relative_to_current_frame: int = \
            CALLER_FRAME_INDEX_RELATIVE_TO_CURRENT_FRAME_FOR_write_line_to_system_console_out):
        """
        Write a debugging information message to console standard output.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        message: str = input_message
        if utf8_conversion:
            message = DebuggingHelper.str_utf8(input_object=input_message)
        if caller_frame_index_relative_to_current_frame is None:
            caller_frame_index_relative_to_current_frame=\
                DebuggingHelper.CALLER_FRAME_INDEX_RELATIVE_TO_CURRENT_FRAME_FOR_write_line_to_system_console_out
        timestamp_in_string = DebuggingHelper.get_timestamp_in_iso_format_string()
        (filename, _, _, _, _, caller_information_in_string) = \
            DebuggingHelper.get_caller_information_in_string(
                caller_frame_index_relative_to_current_frame=\
                    caller_frame_index_relative_to_current_frame)
        message = \
            f'STDOUT-INFO: {timestamp_in_string}{caller_information_in_string}' \
            f'[{logging.getLevelName(logging_level)}:{logging_level}]: {message}'
        # print(f'CHECKPOINT-write_line_to_system_console_out-0')
        default_logger, designated_logger, default_file_handler, designated_file_handler = DebuggingHelper.get_loggers(
            name=filename)
        # print(f'CHECKPOINT-write_line_to_system_console_out-1')
        DebuggingHelper.log(
            logging_level=logging_level,
            message=message,
            logger=default_logger)
        DebuggingHelper.log(
            logging_level=logging_level,
            message=message,
            logger=designated_logger)
        default_file_handler.flush()
        designated_file_handler.flush()
        if color_code is None:
            print(message)
        else:
            DebuggingHelper.print_in_color(message=message, color_code=color_code)
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # ---- NOTE: non-ASCII code in message may blow it up with an error like
        # ----       UnicodeEncodeError: 'ascii' codec can't encode character '\xa7' in position 436: ordinal not in range(128)

    CALLER_FRAME_INDEX_RELATIVE_TO_CURRENT_FRAME_FOR_write_line_to_system_console_err: int = 2

    @staticmethod
    def write_line_to_system_console_err_warning_in_red( \
        message: str, \
        utf8_conversion: bool = False, \
        caller_frame_index_relative_to_current_frame: int = \
            CALLER_FRAME_INDEX_RELATIVE_TO_CURRENT_FRAME_FOR_write_line_to_system_console_err):
        """
        Write a debugging information message to console standard error at WARNING level.
        """
        DebuggingHelper.write_line_to_system_console_err_warning(
            message=message,
            color_code=Fore.RED,
            utf8_conversion=utf8_conversion,
            caller_frame_index_relative_to_current_frame=caller_frame_index_relative_to_current_frame + 1)
    @staticmethod
    def write_line_to_system_console_err_warning( \
        message: str, \
        color_code: int = None, \
        utf8_conversion: bool = False, \
        caller_frame_index_relative_to_current_frame: int = \
            CALLER_FRAME_INDEX_RELATIVE_TO_CURRENT_FRAME_FOR_write_line_to_system_console_err):
        """
        Write a debugging information message to console standard error at WARNING level.
        """
        DebuggingHelper.write_line_to_system_console_err(
            input_message=message,
            logging_level=logging.WARNING,
            color_code=color_code,
            utf8_conversion=utf8_conversion,
            caller_frame_index_relative_to_current_frame=caller_frame_index_relative_to_current_frame + 1)
    @staticmethod
    def write_line_to_system_console_err_error_in_red( \
        message: str, \
        utf8_conversion: bool = False, \
        caller_frame_index_relative_to_current_frame: int = \
            CALLER_FRAME_INDEX_RELATIVE_TO_CURRENT_FRAME_FOR_write_line_to_system_console_err):
        """
        Write a debugging information message to console standard error at ERROR level.
        """
        DebuggingHelper.write_line_to_system_console_err_error(
            message=message,
            color_code=Fore.RED,
            utf8_conversion=utf8_conversion,
            caller_frame_index_relative_to_current_frame=caller_frame_index_relative_to_current_frame + 1)
    @staticmethod
    def write_line_to_system_console_err_error( \
        message: str, \
        color_code: int = None, \
        utf8_conversion: bool = False, \
        caller_frame_index_relative_to_current_frame: int = \
            CALLER_FRAME_INDEX_RELATIVE_TO_CURRENT_FRAME_FOR_write_line_to_system_console_err):
        """
        Write a debugging information message to console standard error at ERROR level.
        """
        DebuggingHelper.write_line_to_system_console_err(
            input_message=message,
            logging_level=logging.ERROR,
            color_code=color_code,
            utf8_conversion=utf8_conversion,
            caller_frame_index_relative_to_current_frame=caller_frame_index_relative_to_current_frame + 1)
    @staticmethod
    def write_line_to_system_console_err_critical_in_red( \
        message: str, \
        utf8_conversion: bool = False, \
        caller_frame_index_relative_to_current_frame: int = \
            CALLER_FRAME_INDEX_RELATIVE_TO_CURRENT_FRAME_FOR_write_line_to_system_console_err):
        """
        Write a debugging information message to console standard error ar CRITICAL level.
        """
        DebuggingHelper.write_line_to_system_console_err_critical(
            message=message,
            color_code=Fore.RED,
            utf8_conversion=utf8_conversion,
            caller_frame_index_relative_to_current_frame=caller_frame_index_relative_to_current_frame + 1)
    @staticmethod
    def write_line_to_system_console_err_critical( \
        message: str, \
        color_code: int = None, \
        utf8_conversion: bool = False, \
        caller_frame_index_relative_to_current_frame: int = \
            CALLER_FRAME_INDEX_RELATIVE_TO_CURRENT_FRAME_FOR_write_line_to_system_console_err):
        """
        Write a debugging information message to console standard error ar CRITICAL level.
        """
        DebuggingHelper.write_line_to_system_console_err(
            input_message=message,
            logging_level=logging.CRITICAL,
            color_code=color_code,
            utf8_conversion=utf8_conversion,
            caller_frame_index_relative_to_current_frame=caller_frame_index_relative_to_current_frame + 1)
    @staticmethod
    def write_line_to_system_console_err_in_red( \
        message: str, \
        logging_level=logging.WARNING, \
        utf8_conversion: bool = False, \
        caller_frame_index_relative_to_current_frame: int = \
            CALLER_FRAME_INDEX_RELATIVE_TO_CURRENT_FRAME_FOR_write_line_to_system_console_err):
        """
        Write a debugging information message to console standard error.
        """
        DebuggingHelper.write_line_to_system_console_err(
            input_message=message,
            logging_level=logging_level,
            color_code=Fore.RED,
            utf8_conversion=utf8_conversion,
            caller_frame_index_relative_to_current_frame=caller_frame_index_relative_to_current_frame + 1)
    @staticmethod
    def write_line_to_system_console_err( \
        input_message: str, \
        logging_level=logging.WARNING, \
        color_code: int = None, \
        utf8_conversion: bool = False, \
        caller_frame_index_relative_to_current_frame: int = \
            CALLER_FRAME_INDEX_RELATIVE_TO_CURRENT_FRAME_FOR_write_line_to_system_console_err):
        """
        Write a debugging information message to console standard error.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        message: str = input_message
        if utf8_conversion:
            message = DebuggingHelper.str_utf8(input_object=input_message)
        if caller_frame_index_relative_to_current_frame is None:
            caller_frame_index_relative_to_current_frame=\
                DebuggingHelper.CALLER_FRAME_INDEX_RELATIVE_TO_CURRENT_FRAME_FOR_write_line_to_system_console_err
        timestamp_in_string = DebuggingHelper.get_timestamp_in_iso_format_string()
        (filename, _, _, _, _, caller_information_in_string) = \
            DebuggingHelper.get_caller_information_in_string(
                caller_frame_index_relative_to_current_frame=\
                    caller_frame_index_relative_to_current_frame)
        message = \
            f'STDERR-INFO: {timestamp_in_string}{caller_information_in_string}' \
            f'[{logging.getLevelName(logging_level)}:{logging_level}]: {message}'
        # print(f'CHECKPOINT-write_line_to_system_console_err-0')
        default_logger, designated_logger, default_file_handler, designated_file_handler = DebuggingHelper.get_loggers(
            name=filename)
        # print(f'CHECKPOINT-write_line_to_system_console_err-1')
        DebuggingHelper.log(
            logging_level=logging_level,
            message=message,
            logger=default_logger)
        DebuggingHelper.log(
            logging_level=logging_level,
            message=message,
            logger=designated_logger)
        default_file_handler.flush()
        designated_file_handler.flush()
        if color_code is None:
            print(message, file=sys.stderr)
        else:
            DebuggingHelper.print_stderr_in_color(message=message, color_code=color_code)

    @staticmethod
    def log_notset( \
        message: str, \
        logger):
        """
        Log
        """
        # print(f'CHECKPOINT-log_notset-0')
        logger.log(
            level=logging.NOTSET,
            msg=message)
    @staticmethod
    def log_debug( \
        message: str, \
        logger):
        """
        Log debug message.
        """
        # print(f'CHECKPOINT-log_debug-0')
        logger.debug(msg=message)
    @staticmethod
    def log_info( \
        message: str, \
        logger):
        """
        Log info message.
        """
        # print(f'CHECKPOINT-log_info-0')
        logger.info(msg=message)
    @staticmethod
    def log_warning( \
        message: str, \
        logger):
        """
        Log warning message.
        """
        # print(f'CHECKPOINT-log_warning-0')
        logger.warning(msg=message)
    @staticmethod
    def log_error( \
        message: str, \
        logger):
        """
        Log error message.
        """
        # print(f'CHECKPOINT-log_error-0')
        logger.error(msg=message)
    @staticmethod
    def log_critical( \
        message: str, \
        logger):
        """
        Log
        """
        # print(f'CHECKPOINT-log_critical-0')
        logger.critical(msg=message)

    @staticmethod
    def log(logging_level: int, message: str, logger):
        """
        Log a message with a logger and level.
        """
        switcher = {
            logging.NOTSET: DebuggingHelper.log_notset,
            logging.DEBUG: DebuggingHelper.log_debug,
            logging.INFO: DebuggingHelper.log_info,
            logging.WARNING: DebuggingHelper.log_warning,
            logging.ERROR: DebuggingHelper.log_error,
            logging.CRITICAL: DebuggingHelper.log_critical
        }
        # logging_function = switcher.get(
        #     logging_level,
        #     lambda: "Invalid logging level")
        logging_function = switcher.get(
            logging_level)
        if logging_function is None:
            raise Exception(f'CRICTIAL: illegal logging level: [{logging.getLevelName(logging_level)}:{logging_level}]')
        logging_function(message=message, logger=logger)

    @staticmethod
    def str_utf8(input_object) -> bytes:
        """
        Return an UTF8 encoded string
        """
        if input_object is None:
            return b''
        if isinstance(input_object, str):
            return input_object.encode('utf-8')
        return  str(input_object).encode('utf-8')

    @staticmethod
    def display_last_error(header_message: str)-> NoReturn:
        sys_exc_info: Any = sys.exc_info
        DebuggingHelper.write_line_to_system_console_err(
            "{}: sys_exc_info={}".format(
                header_message,
                sys_exc_info))
        # sys_last_type: Any = sys_exc_info[0]
        # sys_last_value: Any = sys_exc_info[1]
        # sys_last_traceback: Any = sys_exc_info[2]
        # DebuggingHelper.write_line_to_system_console_err(
        #     "{}: sys_last_value={}".format(
        #         header_message,
        #         sys_last_value))
        # DebuggingHelper.write_line_to_system_console_err(
        #     "{}: sys_last_type={}".format(
        #         header_message,
        #         sys_last_type))
        # DebuggingHelper.write_line_to_system_console_err(
        #     "{}: sys_last_traceback={}".format(
        #         header_message,
        #         sys_last_traceback))
