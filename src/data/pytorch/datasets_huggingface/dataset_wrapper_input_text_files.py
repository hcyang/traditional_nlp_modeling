# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
HuggingfaceDatasetWrapperInputTextFiles
"""

# from typing import Any
from typing import List
# from typing import Tuple
from typing import Union

import os

# import torch

import datasets

from data.pytorch.datasets_huggingface.dataset_wrapper \
    import IHuggingfaceDatasetWrapper

from data.pytorch.datasets_huggingface.huggingface_datasets_utility \
    import HuggingfaceDatasetsUtility

# from utility.string_helper.string_helper \
#     import StringHelper
# from utility.debugging_helper.debugging_helper \
#     import DebuggingHelper

class HuggingfaceDatasetWrapperInputTextFiles(IHuggingfaceDatasetWrapper):
    # ---- NOTE-PYLINT ---- C0301: Line too long
    # pylint: disable=C0301
    """
    REFERENCE:  https://www.thepythoncode.com/article/pretraining-bert-huggingface-transformers-in-python
                https://huggingface.co/docs/datasets/package_reference/loading_methods
    """

    def __init__(self, \
        huggingface_datasets_argument_input_text_files: List[str]):
        """
        Load the Huggingface Datasets 'cc_news'
        """
        self.dataset: Union[datasets.DatasetDict, datasets.Dataset, datasets.IterableDatasetDict, datasets.IterableDataset] = \
            HuggingfaceDatasetsUtility.load_dataset_by_files(
                huggingface_datasets_argument_input_text_files=huggingface_datasets_argument_input_text_files)

    def get_dataset(self) -> Union[datasets.DatasetDict, datasets.Dataset, datasets.IterableDatasetDict, datasets.IterableDataset]:
        """
        Return a wrapped HuggingFace dataset.
        """
        return self.dataset

    @staticmethod
    def scan_input_text_files_folder( \
        input_text_files_folder: str, \
        input_text_files_file_prefix: str = None) -> List[str]:
        """
        Scan through all the text files in the input folder.
        """
        text_files: List[str] = []
        for text_file in os.scandir(input_text_files_folder):
            is_file: bool = text_file.is_file()
            if is_file:
                text_file_path: str = text_file.path
                text_file_basename: str = os.path.basename(text_file_path)
                # DebuggingHelper.write_line_to_system_console_out(
                #     '==== text_file_basename: {}'.format(text_file_basename))
                if ((input_text_files_file_prefix is None) or text_file_basename.startswith(input_text_files_file_prefix)) and \
                    text_file_basename.endswith(".txt"):
                    text_files.append(text_file_path)
                    # ---- NOTE-FOR-DEBUGGING ---- DebuggingHelper.write_line_to_system_console_out(
                    # ---- NOTE-FOR-DEBUGGING ----     '==== text_file_path: {}'.format(text_file_path))
                    # ---- NOTE-CAN-BE-USED-FOR-PREPROCESSING ---- WikipediaDumpXmlProcessortextDataset.read_input_text_files_file( \
                    # ---- NOTE-CAN-BE-USED-FOR-PREPROCESSING ----     input_text_files_file=text_file_path)
        # ---- DebuggingHelper.write_line_to_system_console_out(
        # ----     '==== #text_files: {}'.format(len(text_files)))
        return text_files
