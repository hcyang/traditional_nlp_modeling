# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
This module provides some common configuration helper functions and settings.
"""

import datetime
import os
import shutil

from utility.io_helper.io_helper \
    import IoHelper

class ConfigurationHelperBase:
    """
    Base class provides some static methods for ConfigurationHelper.
    """
    # ---- NOTE-PYLINT ---- R0903: Too few public methods
    # pylint: disable=R0903

    ROOT_DIRECTORY_NAME: str = os.path.join('tmp', 'pyludispatch')

    @staticmethod
    def get_timestamp():
        """
        Get current timestamp.
        """
        return datetime.datetime.now().strftime("%Y%m%d%H%M%S")

class ConfigurationHelper:
    """
    This class contains some common configuration helper functions and settings.
    """

    BASE_STAMP = \
        ConfigurationHelperBase.get_timestamp()
    @staticmethod
    def reset_base_stamp(stamp=None):
        """
        Reset the base stamp.
        """
        if stamp is None:
            stamp = \
                ConfigurationHelperBase.get_timestamp()
        ConfigurationHelper.BASE_STAMP = stamp
        return ConfigurationHelper.BASE_STAMP

    BASE_DIRECTORY = \
        "execution"
    @staticmethod
    def reset_base_directory( \
        base_directory: str = None):
        """
        Reset the base directory.
        """
        if base_directory is not None:
            ConfigurationHelper.BASE_DIRECTORY = base_directory
        return ConfigurationHelper.BASE_DIRECTORY
    BASE_DIRECTORY_STAMPED = \
        f'{BASE_DIRECTORY}_{BASE_STAMP}'
    @staticmethod
    def reset_base_directory_stamped( \
        base_directory: str = None, \
        stamp=None):
        """
        Reset the base stamped directory.
        """
        ConfigurationHelper.reset_base_directory(
            base_directory=base_directory)
        ConfigurationHelper.reset_base_stamp(
            stamp=stamp)
        ConfigurationHelper.BASE_DIRECTORY_STAMPED = \
            f'{ConfigurationHelper.BASE_DIRECTORY}_{ConfigurationHelper.BASE_STAMP}'
        return ConfigurationHelper.BASE_DIRECTORY_STAMPED

    ROOT_DIRECTORY = \
        os.path.join(IoHelper.get_root(),
                     ConfigurationHelperBase.ROOT_DIRECTORY_NAME)
    @staticmethod
    def reset_root_directory( \
        root_directory: str = None, \
        clear_up_root_directory: bool = False):
        """
        Reset the root directory.
        """
        if root_directory is not None:
            ConfigurationHelper.ROOT_DIRECTORY = root_directory
        if clear_up_root_directory:
            shutil.rmtree(
                ConfigurationHelper.ROOT_DIRECTORY,
                ignore_errors=True)
        return ConfigurationHelper.ROOT_DIRECTORY
    ROOT_DIRECTORY_STAMPED = \
        os.path.join(ROOT_DIRECTORY,
                     BASE_DIRECTORY_STAMPED)
    @staticmethod
    def reset_root_directory_stamped( \
        root_directory: str = None, \
        base_directory: str = None, \
        stamp=None, \
        clear_up_root_directory: bool = False):
        """
        Reset the root stamped directory.
        """
        ConfigurationHelper.reset_root_directory(
            root_directory=root_directory,
            clear_up_root_directory=clear_up_root_directory)
        ConfigurationHelper.reset_base_directory_stamped(
            base_directory=base_directory,
            stamp=stamp)
        ConfigurationHelper.ROOT_DIRECTORY_STAMPED = \
            os.path.join(ConfigurationHelper.ROOT_DIRECTORY,
                         ConfigurationHelper.BASE_DIRECTORY_STAMPED)
        return ConfigurationHelper.ROOT_DIRECTORY_STAMPED

    MODEL_CACHE_DIRECTORY = \
        'model_cache'
    @staticmethod
    def reset_model_cache_directory(model_cache_directory: str = None):
        """
        Reset the model cache directory.
        """
        if model_cache_directory is not None:
            ConfigurationHelper.MODEL_CACHE_DIRECTORY = model_cache_directory
        return ConfigurationHelper.MODEL_CACHE_DIRECTORY
    MODEL_CACHE_DIRECTORY_STAMPED = \
        os.path.join(ROOT_DIRECTORY_STAMPED, MODEL_CACHE_DIRECTORY)
    @staticmethod
    def reset_model_cache_directory_stamped( \
        model_cache_directory: str = None, \
        root_directory_set_already: bool = True, \
        root_directory: str = None, \
        base_directory: str = None, \
        stamp=None, \
        clear_up_root_directory: bool = False):
        """
        Reset the root stamped directory.
        """
        ConfigurationHelper.reset_model_cache_directory(
            model_cache_directory=model_cache_directory)
        if not root_directory_set_already:
            ConfigurationHelper.reset_root_directory_stamped(
                root_directory=root_directory,
                base_directory=base_directory,
                stamp=stamp,
                clear_up_root_directory=clear_up_root_directory)
        ConfigurationHelper.MODEL_CACHE_DIRECTORY_STAMPED = \
            os.path.join(ConfigurationHelper.ROOT_DIRECTORY_STAMPED,
                         ConfigurationHelper.MODEL_CACHE_DIRECTORY)
        return ConfigurationHelper.MODEL_CACHE_DIRECTORY_STAMPED
    MODEL_OUTPUT_DIRECTORY = \
        'model_output'
    @staticmethod
    def reset_model_output_directory(model_output_directory: str = None):
        """
        Reset the model output directory.
        """
        if model_output_directory is not None:
            ConfigurationHelper.MODEL_OUTPUT_DIRECTORY = model_output_directory
        return ConfigurationHelper.MODEL_OUTPUT_DIRECTORY
    MODEL_OUTPUT_DIRECTORY_STAMPED = \
        os.path.join(ROOT_DIRECTORY_STAMPED, MODEL_OUTPUT_DIRECTORY)
    @staticmethod
    def reset_model_output_directory_stamped( \
        model_output_directory: str = None, \
        root_directory_set_already: bool = True, \
        root_directory: str = None, \
        base_directory: str = None, \
        stamp=None, \
        clear_up_root_directory: bool = False):
        """
        Reset the model output stamped directory.
        """
        ConfigurationHelper.reset_model_output_directory(
            model_output_directory=model_output_directory)
        if not root_directory_set_already:
            ConfigurationHelper.reset_root_directory_stamped(
                root_directory=root_directory,
                base_directory=base_directory,
                stamp=stamp,
                clear_up_root_directory=clear_up_root_directory)
        ConfigurationHelper.MODEL_OUTPUT_DIRECTORY_STAMPED = \
            os.path.join(ConfigurationHelper.ROOT_DIRECTORY_STAMPED,
                         ConfigurationHelper.MODEL_OUTPUT_DIRECTORY)
        return ConfigurationHelper.MODEL_OUTPUT_DIRECTORY_STAMPED
    MODEL_LOG_DIRECTORY = \
        'model_log'
    @staticmethod
    def reset_model_log_directory(model_log_directory: str = None):
        """
        Reset the model log directory.
        """
        if model_log_directory is not None:
            ConfigurationHelper.MODEL_LOG_DIRECTORY = model_log_directory
        return ConfigurationHelper.MODEL_LOG_DIRECTORY
    MODEL_LOG_DIRECTORY_STAMPED = \
        os.path.join(ROOT_DIRECTORY_STAMPED, MODEL_LOG_DIRECTORY)
    @staticmethod
    def reset_model_log_directory_stamped( \
        model_log_directory: str = None, \
        root_directory_set_already: bool = True, \
        root_directory: str = None, \
        base_directory: str = None, \
        stamp=None, \
        clear_up_root_directory: bool = False):
        """
        Reset the model log stamped directory.
        """
        ConfigurationHelper.reset_model_log_directory(
            model_log_directory=model_log_directory)
        if not root_directory_set_already:
            ConfigurationHelper.reset_root_directory_stamped(
                root_directory=root_directory,
                base_directory=base_directory,
                stamp=stamp,
                clear_up_root_directory=clear_up_root_directory)
        ConfigurationHelper.MODEL_LOG_DIRECTORY_STAMPED = \
            os.path.join(ConfigurationHelper.ROOT_DIRECTORY_STAMPED,
                         ConfigurationHelper.MODEL_LOG_DIRECTORY)
        return ConfigurationHelper.MODEL_LOG_DIRECTORY_STAMPED

    MODEL_TOKENIZER_CACHE_DIRECTORY = \
        'model_tokenizer_cache'
    @staticmethod
    def reset_model_tokenizer_cache_directory(model_tokenizer_cache_directory: str = None):
        """
        Reset the model tokenizer cache directory.
        """
        if model_tokenizer_cache_directory is not None:
            ConfigurationHelper.MODEL_TOKENIZER_CACHE_DIRECTORY = model_tokenizer_cache_directory
        return ConfigurationHelper.MODEL_TOKENIZER_CACHE_DIRECTORY
    MODEL_TOKENIZER_CACHE_DIRECTORY_STAMPED = \
        os.path.join(ROOT_DIRECTORY_STAMPED, MODEL_TOKENIZER_CACHE_DIRECTORY)
    @staticmethod
    def reset_model_tokenizer_cache_directory_stamped( \
        model_tokenizer_cache_directory: str = None, \
        root_directory_set_already: bool = True, \
        root_directory: str = None, \
        base_directory: str = None, \
        stamp=None, \
        clear_up_root_directory: bool = False):
        """
        Reset the model tokenizer cache stamped directory.
        """
        ConfigurationHelper.reset_model_tokenizer_cache_directory(
            model_tokenizer_cache_directory=model_tokenizer_cache_directory)
        if not root_directory_set_already:
            ConfigurationHelper.reset_root_directory_stamped(
                root_directory=root_directory,
                base_directory=base_directory,
                stamp=stamp,
                clear_up_root_directory=clear_up_root_directory)
        ConfigurationHelper.MODEL_TOKENIZER_CACHE_DIRECTORY_STAMPED = \
            os.path.join(ConfigurationHelper.ROOT_DIRECTORY_STAMPED,
                         ConfigurationHelper.MODEL_TOKENIZER_CACHE_DIRECTORY)
        return ConfigurationHelper.MODEL_TOKENIZER_CACHE_DIRECTORY_STAMPED
    MODEL_TOKENIZER_OUTPUT_DIRECTORY = \
        'model_tokenizer_output'
    @staticmethod
    def reset_model_tokenizer_output_directory(model_tokenizer_output_directory: str = None):
        """
        Reset the model tokenizer _output directory.
        """
        if model_tokenizer_output_directory is not None:
            ConfigurationHelper.MODEL_TOKENIZER_OUTPUT_DIRECTORY = model_tokenizer_output_directory
        return ConfigurationHelper.MODEL_TOKENIZER_OUTPUT_DIRECTORY
    MODEL_TOKENIZER_OUTPUT_DIRECTORY_STAMPED = \
        os.path.join(ROOT_DIRECTORY_STAMPED, MODEL_TOKENIZER_OUTPUT_DIRECTORY)
    @staticmethod
    def reset_model_tokenizer_output_directory_stamped( \
        model_tokenizer_output_directory: str = None, \
        root_directory_set_already: bool = True, \
        root_directory: str = None, \
        base_directory: str = None, \
        stamp=None, \
        clear_up_root_directory: bool = False):
        """
        Reset the model tokenizer output stamped directory.
        """
        ConfigurationHelper.reset_model_tokenizer_output_directory(
            model_tokenizer_output_directory=model_tokenizer_output_directory)
        if not root_directory_set_already:
            ConfigurationHelper.reset_root_directory_stamped(
                root_directory=root_directory,
                base_directory=base_directory,
                stamp=stamp,
                clear_up_root_directory=clear_up_root_directory)
        ConfigurationHelper.MODEL_TOKENIZER_OUTPUT_DIRECTORY_STAMPED = \
            os.path.join(ConfigurationHelper.ROOT_DIRECTORY_STAMPED,
                         ConfigurationHelper.MODEL_TOKENIZER_OUTPUT_DIRECTORY)
        return ConfigurationHelper.MODEL_TOKENIZER_OUTPUT_DIRECTORY_STAMPED
    MODEL_TOKENIZER_LOG_DIRECTORY = \
        'model_tokenizer_log'
    @staticmethod
    def reset_model_tokenizer_log_directory(model_tokenizer_log_directory: str = None):
        """
        Reset the model tokenizer log directory.
        """
        if model_tokenizer_log_directory is not None:
            ConfigurationHelper.MODEL_TOKENIZER_LOG_DIRECTORY = model_tokenizer_log_directory
        return ConfigurationHelper.MODEL_TOKENIZER_LOG_DIRECTORY
    MODEL_TOKENIZER_LOG_DIRECTORY_STAMPED = \
        os.path.join(ROOT_DIRECTORY_STAMPED, MODEL_TOKENIZER_LOG_DIRECTORY)
    @staticmethod
    def reset_model_tokenizer_log_directory_stamped( \
        model_tokenizer_log_directory: str = None, \
        root_directory_set_already: bool = True, \
        root_directory: str = None, \
        base_directory: str = None, \
        stamp=None, \
        clear_up_root_directory: bool = False):
        """
        Reset the model tokenizer log stamped directory.
        """
        ConfigurationHelper.reset_model_tokenizer_log_directory(
            model_tokenizer_log_directory=model_tokenizer_log_directory)
        if not root_directory_set_already:
            ConfigurationHelper.reset_root_directory_stamped(
                root_directory=root_directory,
                base_directory=base_directory,
                stamp=stamp,
                clear_up_root_directory=clear_up_root_directory)
        ConfigurationHelper.MODEL_TOKENIZER_LOG_DIRECTORY_STAMPED = \
            os.path.join(ConfigurationHelper.ROOT_DIRECTORY_STAMPED,
                         ConfigurationHelper.MODEL_TOKENIZER_LOG_DIRECTORY)
        return ConfigurationHelper.MODEL_TOKENIZER_LOG_DIRECTORY_STAMPED

    @staticmethod
    def reset_all_working_directories( \
        root_directory_set_already: bool = True, \
        root_directory: str = None, \
        base_directory: str = None, \
        stamp=None, \
        clear_up_root_directory: bool = False):
        """
        Reset all directories using new root_directory, base_directory,
        and stamp.
        """
        if not root_directory_set_already:
            ConfigurationHelper.reset_root_directory_stamped(
                root_directory=root_directory,
                base_directory=base_directory,
                stamp=stamp,
                clear_up_root_directory=clear_up_root_directory)
        ConfigurationHelper.reset_model_cache_directory_stamped(
            root_directory_set_already=True)
        ConfigurationHelper.reset_model_output_directory_stamped(
            root_directory_set_already=True)
        ConfigurationHelper.reset_model_log_directory_stamped(
            root_directory_set_already=True)
        ConfigurationHelper.reset_model_tokenizer_cache_directory_stamped(
            root_directory_set_already=True)
        ConfigurationHelper.reset_model_tokenizer_output_directory_stamped(
            root_directory_set_already=True)
        ConfigurationHelper.reset_model_tokenizer_log_directory_stamped(
            root_directory_set_already=True)
