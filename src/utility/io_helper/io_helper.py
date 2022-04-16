# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
This module provides some common IO helper functions.
"""

from typing import NoReturn
from typing import List
from typing import Tuple

import os
from os import path as ospath
import sys
import shutil
import codecs
from pathlib import Path as pathlib_path
import chardet

from utility.datatype_helper.datatype_helper \
    import DatatypeHelper
# ---- NOTE-CANNOT-IMPORT ---- from utility.debugging_helper.debugging_helper \
# ---- NOTE-CANNOT-IMPORT ----     import DebuggingHelper

class IoHelper:
    """
    This class contains some common functions for io purpose.
    """
    # ---- NOTE-PYLINT ---- R0904: Too many public methods
    # pylint: disable=R0904

    @staticmethod
    def make_parent_dirs( \
        path_with_filename: str) -> NoReturn:
        """
        Create parent directoies for a file.
        """
        path_splits: List[str] = IoHelper.path_split_all(path_with_filename)
        if len(path_splits) > 0:
            directory_splits: List[str] = path_splits[:-1]
            parent_directory: str = os.path.join(*directory_splits)
            os.makedirs(parent_directory, exist_ok=True)
    @staticmethod
    def path_split_all(path: str) -> List[str]:
        """
        Split a path into a list.
        """
        path_all_parts: List[str] = []
        while 1:
            parts: Tuple[str, str] = os.path.split(path)
            if parts[0] == path:  # sentinel for absolute paths
                path_all_parts.insert(0, parts[0])
                break
            elif parts[1] == path: # sentinel for relative paths
                path_all_parts.insert(0, parts[1])
                break
            else:
                path = parts[0]
                path_all_parts.insert(0, parts[1])
        return path_all_parts

    @staticmethod
    def codecs_open_file(filename: str, mode: str, encoding: str = 'utf-8'):
        """
        Open a file.
        """
        return codecs.open(filename=filename, mode=mode, encoding=encoding)

    @staticmethod
    def open_file(filename: str, mode: str, encoding: str = 'utf-8'):
        """
        Open a file.
        """
        if encoding is None:
            return open(file=filename, mode=mode)
        else:
            return open(file=filename, mode=mode, encoding=encoding)

    @staticmethod
    def detect_encoding(filename: str):
        """
        Detect a file's encoding.
        """
        with open(filename, 'rb') as stream:
            result = chardet.detect(stream.read())  # or readline if the file is large
            return result

    @staticmethod
    def read_all_from_file(filename: str, encoding: str = 'utf-8') -> str:
        """
        Reading all lines from a stream.
        """
        with IoHelper.open_file( \
            filename=filename, \
            mode="r", \
            encoding=encoding) as stream:
            all_of_it: str = stream.read()
            return all_of_it

    @staticmethod
    def read_lines_from_file(filename: str, encoding: str = 'utf-8') -> List[str]:
        """
        Reading all lines from a stream.
        """
        with IoHelper.open_file( \
            filename=filename, \
            mode="r", \
            encoding=encoding) as stream:
            return IoHelper.read_lines(stream)

    @staticmethod
    def read_lines(stream) -> List[str]:
        """
        Reading all lines from a stream.
        """
        lines: List[str] = []
        for _, line in enumerate(stream):
            # index, line
            # print("Line {}: {}".format(index, line.strip()))
            lines.append(line.strip())
        # ---- NOTE ---- Reading all lines from a stream can be problematic due to encoding issues.
        # lines = stream.readlines()
        return lines

    @staticmethod
    def write_string(filename: str, content: str, encoding: str = 'utf-8'):
        """
        Write a string to a file.
        """
        with IoHelper.open_file( \
            filename=filename, \
            mode='w', \
            encoding=encoding) as stream:
            stream.write(content)

    @staticmethod
    def write_strings(filename: str, lines: List[str], encoding: str = 'utf-8'):
        """
        Write a string to a file.
        """
        with IoHelper.open_file( \
            filename=filename, \
            mode='w', \
            encoding=encoding) as stream:
            stream.writelines("%s\n" % line for line in lines)

    @staticmethod
    def write_bytes(filename: str, content):
        """
        Write a string to a file.
        """
        with IoHelper.open_file( \
            filename=filename, \
            mode='w+b', \
            encoding=None) as stream:
            stream.write(content)

    @staticmethod
    def can_open_file(filename: str) -> bool:
        """
        Check if a file can be opened.
        """
        if DatatypeHelper.is_none_empty_whitespaces_or_nan(filename):
            return False
        try:
            with open(filename, 'r'):
                pass
        except FileNotFoundError:
            return False
        return True

    @staticmethod
    def isabs(path: str) -> bool:
        """
        Check if given input path  is absolute.
        """
        return os.path.isabs(path)
    @staticmethod
    def isdir(path: str) -> bool:
        """
        Check if given input path exists and is a directory not file.
        """
        return os.path.isdir(path)
    @staticmethod
    def isfile(path: str) -> bool:
        """
        Check if given input path exists and is a file not directory.
        """
        return os.path.isfile(path)
    @staticmethod
    def islink(path: str) -> bool:
        """
        Check if given input path exists and is a symbolic link.
        """
        return os.path.islink(path)
    @staticmethod
    def ismount(path: str) -> bool:
        """
        Check if given input path exists and is a mount point.
        """
        return os.path.ismount(path)

    @staticmethod
    def exists(path: str) -> bool:
        """
        Check if a path exists .
        """
        return os.path.exists(path)

    @staticmethod
    def basename(path: str) -> str:
        """
        Retrieve a file's base name.
        """
        stem: str = pathlib_path(path).stem
        # print('---- IoHelper.basename()={}'.format(stem), file=sys.stderr)
        return stem

    @staticmethod
    def get_executable() -> str:
        """
        Retrieve the executable name.
        """
        executable: str = sys.executable
        # print('---- IoHelper.get_executable()={}'.format(executable), file=sys.stderr)
        return executable

    @staticmethod
    def get_executable_path() -> str:
        """
        Retrieve the executable path.
        """
        path: str = pathlib_path(IoHelper.get_executable())
        # print('---- IoHelper.get_executable_path()={}'.format(path), file=sys.stderr)
        return path

    @staticmethod
    def get_root_path() -> str:
        """
        Retrieve the executable path.
        """
        path: str = IoHelper.get_executable_path()
        # print('---- IoHelper.get_root_path()={}'.format(path), file=sys.stderr)
        return path

    @staticmethod
    def get_root() -> str:
        """
        Retrieve the executable path.
        """
        path = IoHelper.get_root_path()
        # print('---- IoHelper.get_root()={}'.format(path), file=sys.stderr)
        return path.root

    @staticmethod
    def head(path: str) -> str:
        """
        Retrieve the head of a path.
        """
        head = ospath.split(path)[0]
        return head
    @staticmethod
    def tail(path: str) -> str:
        """
        Retrieve the tail of a path.
        """
        tail = ospath.split(path)[1]
        return tail

    @staticmethod
    def splitext_head(path: str) -> str:
        """
        Retrieve the head of a path.
        """
        head = ospath.splitext(path)[0]
        return head
    @staticmethod
    def splitext_tail(path: str) -> str:
        """
        Retrieve the tail of a path.
        """
        tail = ospath.splitext(path)[1]
        return tail

    @staticmethod
    def zip(path_to_compress: str, compression_filename: str = None):
        """
        Compress a path into a zip file.
        """
        shutil.make_archive(
            base_name=compression_filename,
            format='zip',
            root_dir=path_to_compress)

    @staticmethod
    def unzip(compression_filename: str = None, path_to_uncompress: str = None):
        """
        Compress a path into a zip file.
        """
        shutil.unpack_archive(
            filename=compression_filename,
            format='zip',
            extract_dir=path_to_uncompress)
