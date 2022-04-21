# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
HuggingfaceDatasetsUtility
"""

from typing import Any
from typing import List
# from typing import Tuple
from typing import Union

# ---- NOTE-PLACE-HOLDER ---- from itertools import chain

# import os
# import json

# import torch

import datasets

# from utility.string_helper.string_helper \
#     import StringHelper
# from utility.debugging_helper.debugging_helper \
#     import DebuggingHelper

class HuggingfaceDatasetsUtility:
    # ---- NOTE-PYLINT ---- C0301: Line too long
    # pylint: disable=C0301
    """
    REFERENCE:  https://www.thepythoncode.com/article/pretraining-bert-huggingface-transformers-in-python
                https://huggingface.co/docs/datasets/package_reference/loading_methods
    REFERENCE: datasets/arrow_dataset.py
        def load_dataset(
            path: str,
            name: Optional[str] = None,
            data_dir: Optional[str] = None,
            data_files: Union[Dict, List] = None,
            split: Optional[Union[str, Split]] = None,
            cache_dir: Optional[str] = None,
            features: Optional[Features] = None,
            download_config: Optional[DownloadConfig] = None,
            download_mode: Optional[GenerateMode] = None,
            ignore_verifications: bool = False,
            keep_in_memory: Optional[bool] = None,
            save_infos: bool = False,
            script_version: Optional[Union[str, Version]] = None,
            use_auth_token: Optional[Union[bool, str]] = None,
            task: Optional[Union[str, TaskTemplate]] = None,
            streaming: bool = False,
            **config_kwargs,
        ) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
            \"\"\"Load a dataset.

            This method does the following under the hood:

                1. Download and import in the library the dataset loading script from ``path`` if it's not already cached inside the library.

                    Processing scripts are small python scripts that define the citation, info and format of the dataset,
                    contain the URL to the original data files and the code to load examples from the original data files.

                    You can find some of the scripts here: https://github.com/huggingface/datasets/datasets
                    and easily upload yours to share them using the CLI ``huggingface-cli``.
                    You can find the complete list of datasets in the Datasets Hub at https://huggingface.co/datasets

                2. Run the dataset loading script which will:

                    * Download the dataset file from the original URL (see the script) if it's not already downloaded and cached.
                    * Process and cache the dataset in typed Arrow tables for caching.

                        Arrow table are arbitrarily long, typed tables which can store nested objects and be mapped to numpy/pandas/python standard types.
                        They can be directly access from drive, loaded in RAM or even streamed over the web.

                3. Return a dataset built from the requested splits in ``split`` (default: all).

            Args:

                path (:obj:`str`): Path to the dataset processing script with the dataset builder. Can be either:

                    - a local path to processing script or the directory containing the script (if the script has the same name as the directory),
                      e.g. ``'./dataset/squad'`` or ``'./dataset/squad/squad.py'``.
                    - a dataset identifier in the HuggingFace Datasets Hub (list all available datasets and ids with ``datasets.list_datasets()``)
                      e.g. ``'squad'``, ``'glue'`` or ``'openai/webtext'``.
                name (:obj:`str`, optional): Defining the name of the dataset configuration.
                data_dir (:obj:`str`, optional): Defining the data_dir of the dataset configuration.
                data_files (:obj:`str`, optional): Defining the data_files of the dataset configuration.
                split (:class:`Split` or :obj:`str`): Which split of the data to load.
                    If None, will return a `dict` with all splits (typically `datasets.Split.TRAIN` and `datasets.Split.TEST`).
                    If given, will return a single Dataset.
                    Splits can be combined and specified like in tensorflow-datasets.
                cache_dir (:obj:`str`, optional): Directory to read/write data. Defaults to "~/datasets".
                features (:class:`Features`, optional): Set the features type to use for this dataset.
                download_config (:class:`~utils.DownloadConfig`, optional): Specific download configuration parameters.
                download_mode (:class:`GenerateMode`, default ``REUSE_DATASET_IF_EXISTS``): Download/generate mode.
                ignore_verifications (:obj:`bool`, default ``False``): Ignore the verifications of the downloaded/processed dataset information (checksums/size/splits/...).
                keep_in_memory (:obj:`bool`, default ``None``): Whether to copy the dataset in-memory. If `None`, the dataset
                    will not be copied in-memory unless explicitly enabled by setting `datasets.config.IN_MEMORY_MAX_SIZE` to
                    nonzero. See more details in the :ref:`load_dataset_enhancing_performance` section.
                save_infos (:obj:`bool`, default ``False``): Save the dataset information (checksums/size/splits/...).
                script_version (:class:`~utils.Version` or :obj:`str`, optional): Version of the dataset script to load:

                    - For canonical datasets in the `huggingface/datasets` library like "squad", the default version of the module is the local version fo the lib.
                      You can specify a different version from your local version of the lib (e.g. "master" or "1.2.0") but it might cause compatibility issues.
                    - For community provided datasets like "lhoestq/squad" that have their own git repository on the Datasets Hub, the default version "main" corresponds to the "main" branch.
                      You can specify a different version that the default "main" by using a commit sha or a git tag of the dataset repository.
                use_auth_token (``str`` or ``bool``, optional): Optional string or boolean to use as Bearer token for remote files on the Datasets Hub.
                    If True, will get token from `"~/.huggingface"`.
                task (``str``): The task to prepare the dataset for during training and evaluation. Casts the dataset's :class:`Features` to standardized column names and types as detailed in :py:mod:`datasets.tasks`.
                streaming (``bool``, default ``False``): If set to True, don't download the data files. Instead, it streams the data progressively while
                    iterating on the dataset. An IterableDataset or IterableDatasetDict is returned instead in this case.

                    Note that streaming works for datasets that use data formats that support being iterated over like txt, csv, jsonl for example.
                    Json files may be downloaded completely. Also streaming from remote zip or gzip files is supported but other compressed formats
                    like rar and xz are not yet supported. The tgz format doesn't allow streaming.
                **config_kwargs: Keyword arguments to be passed to the :class:`BuilderConfig` and used in the :class:`DatasetBuilder`.

            Returns:
                :class:`Dataset` or :class:`DatasetDict`:
                    if `split` is not None: the dataset requested,
                    if `split` is None, a ``datasets.DatasetDict`` with each split.
                or :class:`IterableDataset` or :class:`IterableDatasetDict` if streaming=True:
                    if `split` is not None: the dataset requested,
                    if `split` is None, a ``datasets.streaming.IterableDatasetDict`` with each split.

            \"\"\"
    """

    @staticmethod
    def tokenize_dataset( \
        tokenizer: Any, \
        dataset: datasets.Dataset, \
        truncate_longer_samples: bool = True, \
        pad_to_max_length: bool = True, \
        max_seqequence_length: int = 384, \
        text_column_name: str = 'text', \
        preprocessing_number_workers: int = 1) -> datasets.Dataset:
        """
        Tokenize an input dataset.
        """
        # ---- NOTE ---- When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if pad_to_max_length else False
        def tokenize_function(examples):
            # ---- NOTE ---- Remove empty lines
            examples[text_column_name] = [
                line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=truncate_longer_samples,
                max_length=max_seqequence_length,
                # ---- NOTE ---- We use this option because DataCollatorForLanguageModeling (a follow-up step) is more efficient when it
                # ---- NOTE ---- receives the `special_tokens_mask`.
                return_special_tokens_mask=True)
        # ---- NOTE-PLACE-HOLDER ---- with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=preprocessing_number_workers,
            remove_columns=[text_column_name],
            # load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset line-by-line")
        return tokenized_datasets
        # ---- NOTE-PLACE-HOLDER ---- # ---- NOTE ---- $\transformers\examples\pytorch\language-modeling\run_mlm.py
        # ---- NOTE-PLACE-HOLDER ---- # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # ---- NOTE-PLACE-HOLDER ---- # max_seqequence_length.
        # ---- NOTE-PLACE-HOLDER ---- def group_texts(examples):
        # ---- NOTE-PLACE-HOLDER ----     # Concatenate all texts.
        # ---- NOTE-PLACE-HOLDER ----     concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        # ---- NOTE-PLACE-HOLDER ----     total_length = len(concatenated_examples[list(examples.keys())[0]])
        # ---- NOTE-PLACE-HOLDER ----     # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # ---- NOTE-PLACE-HOLDER ----     # customize this part to your needs.
        # ---- NOTE-PLACE-HOLDER ----     if total_length >= max_seqequence_length:
        # ---- NOTE-PLACE-HOLDER ----         total_length = (total_length // max_seqequence_length) * max_seqequence_length
        # ---- NOTE-PLACE-HOLDER ----     # Split by chunks of max_len.
        # ---- NOTE-PLACE-HOLDER ----     result = {
        # ---- NOTE-PLACE-HOLDER ----         k: [t[i : i + max_seqequence_length] for i in range(0, total_length, max_seqequence_length)]
        # ---- NOTE-PLACE-HOLDER ----         for k, t in concatenated_examples.items()
        # ---- NOTE-PLACE-HOLDER ----     }
        # ---- NOTE-PLACE-HOLDER ----     return result
        # ---- NOTE-PLACE-HOLDER ---- def encode_with_truncation(dataset: datasets.Dataset):
        # ---- NOTE-PLACE-HOLDER ----     """
        # ---- NOTE-PLACE-HOLDER ----     Mapping function to tokenize the sentences passed with truncation
        # ---- NOTE-PLACE-HOLDER ----     """
        # ---- NOTE-PLACE-HOLDER ----     return tokenizer( \
        # ---- NOTE-PLACE-HOLDER ----         dataset["text"], \
        # ---- NOTE-PLACE-HOLDER ----         truncation=True, \
        # ---- NOTE-PLACE-HOLDER ----         padding="max_length", \
        # ---- NOTE-PLACE-HOLDER ----         max_length=max_seqequence_length, \
        # ---- NOTE-PLACE-HOLDER ----         return_special_tokens_mask=True)
        # ---- NOTE-PLACE-HOLDER ---- def encode_without_truncation(dataset: datasets.Dataset):
        # ---- NOTE-PLACE-HOLDER ----     """
        # ---- NOTE-PLACE-HOLDER ----     Mapping function to tokenize the sentences passed without truncation
        # ---- NOTE-PLACE-HOLDER ----     """
        # ---- NOTE-PLACE-HOLDER ----     return tokenizer( \
        # ---- NOTE-PLACE-HOLDER ----         dataset["text"], \
        # ---- NOTE-PLACE-HOLDER ----         return_special_tokens_mask=True)
        # ---- NOTE-PLACE-HOLDER ---- if truncate_longer_samples:
        # ---- NOTE-PLACE-HOLDER ----     dataset = dataset.map(encode_with_truncation, batched=True)
        # ---- NOTE-PLACE-HOLDER ----     dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        # ---- NOTE-PLACE-HOLDER ----     dataset = dataset.map( \
        # ---- NOTE-PLACE-HOLDER ----         group_texts, \
        # ---- NOTE-PLACE-HOLDER ----         batched=True, \
        # ---- NOTE-PLACE-HOLDER ----         batch_size=2_000, \
        # ---- NOTE-PLACE-HOLDER ----         desc=f"Grouping texts in chunks of {max_seqequence_length}")
        # ---- NOTE-PLACE-HOLDER ---- else:
        # ---- NOTE-PLACE-HOLDER ----     dataset = dataset.map(encode_without_truncation, batched=True)
        # ---- NOTE-PLACE-HOLDER ----     dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
        # ---- NOTE-PLACE-HOLDER ---- return dataset

    @staticmethod
    def tokenize_dataset_text_feature( \
        tokenizer: Any, \
        dataset: datasets.Dataset, \
        truncate_longer_samples: bool = True, \
        pad_to_max_length: bool = True, \
        max_seqequence_length: int = 384, \
        text_column_name: str = 'text') -> Any:
        """
        Tokenize an input dataset's text feature.
        """
        # ---- NOTE ---- When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if pad_to_max_length else False
        text_features: List[str] = dataset[text_column_name]
        encoded: Any = None
        if truncate_longer_samples:
            encoded = tokenizer(
                text_features,
                truncation=True,
                padding=padding,
                max_length=max_seqequence_length,
                return_special_tokens_mask=True)
        else:
            encoded = tokenizer(
                text_features,
                return_special_tokens_mask=True)
        return encoded

    @staticmethod
    def load_dataset_by_name( \
        huggingface_datasets_argument_name: str, \
        huggingface_datasets_argument_cache_path: str = None, \
        huggingface_datasets_argument_split: str = 'train', \
        huggingface_datasets_argument_save_infos: bool = False, \
        huggingface_datasets_argument_ignore_verifications: bool = False) -> Union[datasets.DatasetDict, datasets.Dataset, datasets.IterableDatasetDict, datasets.IterableDataset]: # ---- NOTE-NEED-Python-3.10-FOR-ALTERNATE-TYPE-HINT ---- (datasets.DatasetDict | datasets.Dataset | datasets.IterableDatasetDict | datasets.IterableDataset):
        """
        Load the Huggingface Datasets by path in huggingface_datasets_argument_name
        """
        dataset = datasets.load_dataset(
            huggingface_datasets_argument_name,
            cache_dir=huggingface_datasets_argument_cache_path,
            split=huggingface_datasets_argument_split,
            save_infos=huggingface_datasets_argument_save_infos,
            ignore_verifications=huggingface_datasets_argument_ignore_verifications)
        return dataset

    @staticmethod
    def load_dataset_by_files( \
        huggingface_datasets_argument_input_text_files: List[str]) -> Union[datasets.DatasetDict, datasets.Dataset, datasets.IterableDatasetDict, datasets.IterableDataset]:
        """
        Create a Huggingface Datasets object using some local text files.
        """
        dataset = datasets.load_dataset(
            'text',
            data_files=huggingface_datasets_argument_input_text_files,
            split='train')
        return dataset

    @staticmethod
    def split_to_train_test_sets( \
        dataset: Union[datasets.DatasetDict, datasets.Dataset, datasets.IterableDatasetDict, datasets.IterableDataset], \
        huggingface_datasets_argument_test_portion: float = 0.1) -> datasets.DatasetDict:
        """
        Split a Huggingface datasets into train and test parts.
        REFERENCE: datasets/arrow_dataset.py
            def train_test_split(
                self,
                test_size: Union[float, int, None] = None,
                train_size: Union[float, int, None] = None,
                shuffle: bool = True,
                seed: Optional[int] = None,
                generator: Optional[np.random.Generator] = None,
                keep_in_memory: bool = False,
                load_from_cache_file: bool = True,
                train_indices_cache_file_name: Optional[str] = None,
                test_indices_cache_file_name: Optional[str] = None,
                writer_batch_size: Optional[int] = 1000,
                train_new_fingerprint: Optional[str] = None,
                test_new_fingerprint: Optional[str] = None,
            ) -> "DatasetDict":
                \"\"\"Return a dictionary (:obj:`datasets.DatsetDict`) with two random train and test subsets (`train` and `test` ``Dataset`` splits).
                Splits are created from the dataset according to `test_size`, `train_size` and `shuffle`.

                This method is similar to scikit-learn `train_test_split` with the omission of the stratified options.

                Args:
                    test_size (:obj:`numpy.random.Generator`, optional): Size of the test split
                        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
                        If int, represents the absolute number of test samples.
                        If None, the value is set to the complement of the train size.
                        If train_size is also None, it will be set to 0.25.
                    train_size (:obj:`numpy.random.Generator`, optional): Size of the train split
                        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split.
                        If int, represents the absolute number of train samples.
                        If None, the value is automatically set to the complement of the test size.
                    shuffle (:obj:`bool`, optional, default `True`): Whether or not to shuffle the data before splitting.
                    seed (:obj:`int`, optional): A seed to initialize the default BitGenerator if ``generator=None``.
                        If None, then fresh, unpredictable entropy will be pulled from the OS.
                        If an int or array_like[ints] is passed, then it will be passed to SeedSequence to derive the initial BitGenerator state.
                    generator (:obj:`numpy.random.Generator`, optional): Numpy random Generator to use to compute the permutation of the dataset rows.
                        If ``generator=None`` (default), uses np.random.default_rng (the default BitGenerator (PCG64) of NumPy).
                    keep_in_memory (:obj:`bool`, default `False`): Keep the splits indices in memory instead of writing it to a cache file.
                    load_from_cache_file (:obj:`bool`, default `True`): If a cache file storing the splits indices
                        can be identified, use it instead of recomputing.
                    train_cache_file_name (:obj:`str`, optional): Provide the name of a path for the cache file. It is used to store the
                        train split indices instead of the automatically generated cache file name.
                    test_cache_file_name (:obj:`str`, optional): Provide the name of a path for the cache file. It is used to store the
                        test split indices instead of the automatically generated cache file name.
                    writer_batch_size (:obj:`int`, default `1000`): Number of rows per write operation for the cache file writer.
                        This value is a good trade-off between memory usage during the processing, and processing speed.
                        Higher value makes the processing do fewer lookups, lower value consume less temporary memory while running `.map()`.
                    train_new_fingerprint (:obj:`str`, optional, defaults to `None`): the new fingerprint of the train set after transform.
                        If `None`, the new fingerprint is computed using a hash of the previous fingerprint, and the transform arguments
                    test_new_fingerprint (:obj:`str`, optional, defaults to `None`): the new fingerprint of the test set after transform.
                        If `None`, the new fingerprint is computed using a hash of the previous fingerprint, and the transform arguments
                \"\"\"
        """
        return dataset.train_test_split(test_size=huggingface_datasets_argument_test_portion)
