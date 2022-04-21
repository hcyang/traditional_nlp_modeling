# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
This module implements feature manager objects using a NGram + subword tokenizer.
"""

from typing import List

import json
import os
import re
import unidecode

from model.language_understanding.featurizer.\
    base_data_feature_manager_single_label_ngram_subword \
    import NGramSubwordSingleLabelDataFeatureManager
from data.manager.base_record_generator \
    import BaseRecordGenerator
from data.manager.base_data_manager \
    import BaseDataManager

from utility.string_helper.string_helper \
    import StringHelper
from utility.io_helper.io_helper \
    import IoHelper
# from utility.debugging_helper.debugging_helper \
#     import DebuggingHelper

class TokenizationNGramSubwordSingleLabelDataFeatureManagerBase(\
    NGramSubwordSingleLabelDataFeatureManager):
    """
    This class can create features on an input text through a tokenizer.
    """

    DEFAULT_SUBWORD_NGRAM_BEGIN: int = 3
    DEFAULT_SUBWORD_NGRAM_END: int = 4

    DEFAULT_NGRAM_PREFIX_CHARACTER: str = '='    # ---- '0' # ---- '^'
    DEFAULT_NGRAM_SUFFIX_CHARACTER: str = '='    # ---- '0' # ---- '$'
    DEFAULT_NGRAM_DELIMITER_CHARACTER: str = '=' # ---- '0' # ---- ' '

    LanguageTokenPunctuationDelimiters: List[str] = \
        StringHelper.LanguageTokenPunctuationDelimiters

    @staticmethod
    def prepend_append_to_punctuations( \
        input_punctuation_array: List[str] = None, \
        delimiter: str = ' ') -> List[str]:
        """
        Add space around a punctuation.
        """
        if input_punctuation_array is None:
            input_punctuation_array = \
                TokenizationNGramSubwordSingleLabelDataFeatureManagerBase.\
                    LanguageTokenPunctuationDelimiters
        output_punctuation_array = []
        for punctuation in input_punctuation_array:
            output_punctuation_array.append(delimiter + punctuation + delimiter)
        return output_punctuation_array

    def __init__(self, \
        data_manager: BaseDataManager, \
        column_index_label: int = 0, \
        column_index_weight: int = -1, \
        column_index_feature_text: int = 1, \
        column_index_feature_text_auxiliary: int = -1, \
        column_index_identity: int = -1, \
        record_generator: BaseRecordGenerator = None, \
        string_to_replace_null_label: str = '', \
        include_out_of_value_label: bool = True, \
        default_non_existent_label_id: int = 0, \
        include_out_of_value_feature: bool = False, \
        include_only_unique_feature: bool = True, \
        default_non_existent_feature_id: int = -1):
        """
        Init with a data manager object and a tokenizer object.
        """
        # ---- NOTE-PYLINT ---- R0913: Too many arguments
        # pylint: disable=R0913
        # ---- NOTE-PYLINT ---- R0914: Too many local variables
        # pylint: disable=R0914
        super(TokenizationNGramSubwordSingleLabelDataFeatureManagerBase, self).__init__( \
            data_manager=data_manager,
            column_index_label=column_index_label,
            column_index_feature_text=column_index_feature_text,
            column_index_weight=column_index_weight,
            column_index_feature_text_auxiliary=column_index_feature_text_auxiliary,
            column_index_identity=column_index_identity,
            record_generator=record_generator,
            string_to_replace_null_label=string_to_replace_null_label,
            include_out_of_value_label=include_out_of_value_label,
            default_non_existent_label_id=default_non_existent_label_id,
            include_out_of_value_feature=include_out_of_value_feature,
            include_only_unique_feature=include_only_unique_feature,
            default_non_existent_feature_id=default_non_existent_feature_id)

    def copy_core_featurization_metadata(self, other_data_feature_manager):
        # ---- NOTE-CONCRETE-OVERRIDE-FUNCTION ----
        """
        Reference the documentation in the parent class.
        For this class, the core featurization metadata are the members being copied.
        """
        super().copy_core_featurization_metadata(
            other_data_feature_manager=other_data_feature_manager)

class TokenizationNGramSubwordSingleLabelDataFeatureManager(\
    TokenizationNGramSubwordSingleLabelDataFeatureManagerBase):
    """
    This class can create features on an input text through a tokenizer.
    """
    # ---- NOTE-PYLINT ---- R0902: Too many instance attributes
    # pylint: disable=R0902

    LanguageTokenPunctuationWithSpaceDelimiters = \
        TokenizationNGramSubwordSingleLabelDataFeatureManagerBase.\
            prepend_append_to_punctuations(\
                TokenizationNGramSubwordSingleLabelDataFeatureManagerBase.\
                    LanguageTokenPunctuationDelimiters)

    def __init__(self, \
        data_manager: BaseDataManager, \
        ngram_split_separator: str = ' ', \
        ngram_subword_model_tokenizer_do_lower_case: bool = True, \
        ngram_subword_model_tokenizer_do_convert_digits: bool = True, \
        ngram_subword_model_tokenizer_do_normalization: bool = True, \
        ngram_subword_model_tokenizer_do_remove_diacritics: bool = True, \
        ngram_subword_model_tokenizer_do_include_subwords: bool = True, \
        ngram_subword_begin: int = TokenizationNGramSubwordSingleLabelDataFeatureManagerBase.\
            DEFAULT_SUBWORD_NGRAM_BEGIN, \
        ngram_subword_end: int = TokenizationNGramSubwordSingleLabelDataFeatureManagerBase.\
            DEFAULT_SUBWORD_NGRAM_END, \
        column_index_label: int = 0, \
        column_index_weight: int = -1, \
        column_index_feature_text: int = 1, \
        column_index_feature_text_auxiliary: int = -1, \
        column_index_identity: int = -1, \
        record_generator: BaseRecordGenerator = None, \
        string_to_replace_null_label: str = '', \
        include_out_of_value_label: bool = True, \
        default_non_existent_label_id: int = 0, \
        include_out_of_value_feature: bool = False, \
        include_only_unique_feature: bool = True, \
        default_non_existent_feature_id: int = -1):
        """
        Init with a data manager object and a tokenizer object.
        """
        # ---- NOTE-PYLINT ---- R0913: Too many arguments
        # pylint: disable=R0913
        # ---- NOTE-PYLINT ---- R0914: Too many local variables
        # pylint: disable=R0914
        super(TokenizationNGramSubwordSingleLabelDataFeatureManager, self).__init__( \
            data_manager=data_manager,
            column_index_label=column_index_label,
            column_index_feature_text=column_index_feature_text,
            column_index_weight=column_index_weight,
            column_index_feature_text_auxiliary=column_index_feature_text_auxiliary,
            column_index_identity=column_index_identity,
            record_generator=record_generator,
            string_to_replace_null_label=string_to_replace_null_label,
            include_out_of_value_label=include_out_of_value_label,
            default_non_existent_label_id=default_non_existent_label_id,
            include_out_of_value_feature=include_out_of_value_feature,
            include_only_unique_feature=include_only_unique_feature,
            default_non_existent_feature_id=default_non_existent_feature_id)
        self.ngram_split_separator = \
            ngram_split_separator
        self.ngram_subword_model_tokenizer_do_lower_case = \
            ngram_subword_model_tokenizer_do_lower_case
        self.ngram_subword_model_tokenizer_do_convert_digits = \
            ngram_subword_model_tokenizer_do_convert_digits
        self.ngram_subword_model_tokenizer_do_normalization = \
            ngram_subword_model_tokenizer_do_normalization
        self.ngram_subword_model_tokenizer_do_remove_diacritics = \
            ngram_subword_model_tokenizer_do_remove_diacritics
        self.ngram_subword_model_tokenizer_do_include_subwords = \
            ngram_subword_model_tokenizer_do_include_subwords
        self.ngram_subword_begin = \
            ngram_subword_begin
        self.ngram_subword_end = \
            ngram_subword_end

    def copy_core_featurization_metadata(self, other_data_feature_manager):
        # ---- NOTE-CONCRETE-OVERRIDE-FUNCTION ----
        """
        Reference the documentation in the parent class.
        For this class, the core featurization metadata are the members being copied.
        """
        super().copy_core_featurization_metadata(
            other_data_feature_manager=other_data_feature_manager)
        other_data_feature_manager.ngram_split_separator = \
            self.ngram_split_separator
        other_data_feature_manager.ngram_subword_model_tokenizer_do_lower_case = \
            self.ngram_subword_model_tokenizer_do_lower_case
        other_data_feature_manager.ngram_subword_model_tokenizer_do_convert_digits = \
            self.ngram_subword_model_tokenizer_do_convert_digits
        other_data_feature_manager.ngram_subword_model_tokenizer_do_normalization = \
            self.ngram_subword_model_tokenizer_do_normalization
        other_data_feature_manager.ngram_subword_model_tokenizer_do_remove_diacritics = \
            self.ngram_subword_model_tokenizer_do_remove_diacritics
        other_data_feature_manager.ngram_subword_model_tokenizer_do_include_subwords = \
            self.ngram_subword_model_tokenizer_do_include_subwords
        other_data_feature_manager.ngram_subword_begin = \
            self.ngram_subword_begin
        other_data_feature_manager.ngram_subword_end = \
            self.ngram_subword_end

    def serialize_featurizer(self, serialization_destination: str, dump: bool = True):
        # ---- NOTE-CONCRETE-OVERRIDE-FUNCTION ----
        """
        Serialize a featurizer.
        """
        featurization_metadata = super().serialize_featurizer(
            serialization_destination=serialization_destination,
            dump=False)
        featurization_metadata_self = {
            'ngram_split_separator': \
                self.ngram_split_separator,
            'ngram_subword_model_tokenizer_do_lower_case': \
                self.ngram_subword_model_tokenizer_do_lower_case,
            'ngram_subword_model_tokenizer_do_convert_digits': \
                self.ngram_subword_model_tokenizer_do_convert_digits,
            'ngram_subword_model_tokenizer_do_normalization': \
                self.ngram_subword_model_tokenizer_do_normalization,
            'ngram_subword_model_tokenizer_do_remove_diacritics': \
                self.ngram_subword_model_tokenizer_do_remove_diacritics,
            'ngram_subword_model_tokenizer_do_include_subwords': \
                self.ngram_subword_model_tokenizer_do_include_subwords,
            'ngram_subword_begin': \
                self.ngram_subword_begin,
            'ngram_subword_end': \
                self.ngram_subword_end
        }
        featurization_metadata.update(featurization_metadata_self)
        if dump:
            if IoHelper.isdir(serialization_destination):
                serialization_destination = os.path.join(
                    serialization_destination,
                    f'{__name__}_featurization_metadata')
            serialization_destination += '.json'
            json.dump(
                featurization_metadata,
                IoHelper.codecs_open_file(
                    filename=serialization_destination,
                    mode='w',
                    encoding='utf-8'),
                separators=(',', ':'),
                sort_keys=True,
                indent=4)
        return featurization_metadata
    def deserialize_featurizer(self, serialization_destination: str):
        # ---- NOTE-CONCRETE-OVERRIDE-FUNCTION ----
        """
        Deserialize a featurizer.
        """
        if IoHelper.isdir(serialization_destination):
            serialization_destination = os.path.join(
                serialization_destination,
                f'{__name__}_featurization_metadata')
        featurization_metadata = super().deserialize_featurizer(
            serialization_destination=serialization_destination)
        self.ngram_split_separator = \
            featurization_metadata['ngram_split_separator']
        self.ngram_subword_model_tokenizer_do_lower_case = \
            featurization_metadata['ngram_subword_model_tokenizer_do_lower_case']
        self.ngram_subword_model_tokenizer_do_convert_digits = \
            featurization_metadata['ngram_subword_model_tokenizer_do_convert_digits']
        self.ngram_subword_model_tokenizer_do_normalization = \
            featurization_metadata['ngram_subword_model_tokenizer_do_normalization']
        self.ngram_subword_model_tokenizer_do_remove_diacritics = \
            featurization_metadata['ngram_subword_model_tokenizer_do_remove_diacritics']
        self.ngram_subword_model_tokenizer_do_include_subwords = \
            featurization_metadata['ngram_subword_model_tokenizer_do_include_subwords']
        self.ngram_subword_begin = \
            featurization_metadata['ngram_subword_begin']
        self.ngram_subword_end = \
            featurization_metadata['ngram_subword_end']
        return featurization_metadata

    def create_features(self, text: str) -> List[str]:
        # ---- NOTE-CONCRETE-OVERRIDE-FUNCTION ----
        """
        Create a list of features, each is a list of strings.
        An abstract function that child classes must override.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # --------------------------------------------------------------------
        text = text.rstrip('\n\r')
        # --------------------------------------------------------------------
        if self.ngram_subword_model_tokenizer_do_lower_case:
            text_lowercase = text.lower()
            # ---- NOTE-FOR-DEBUGGING ---- DebuggingHelper.write_line_to_system_console_out(
            # ---- NOTE-FOR-DEBUGGING ----    f'text_lowercase={text_lowercase}')
            text = text_lowercase
        if self.ngram_subword_model_tokenizer_do_convert_digits:
            text = re.sub(r'\d', r'0', text)
        if self.ngram_subword_model_tokenizer_do_normalization:
            # ---- NOTE-FOR-DEBUGGING ---- DebuggingHelper.write_line_to_system_console_out(
            # ---- NOTE-FOR-DEBUGGING ----     f'ngram_subword_model_tokenizer_do_normalization-BEFORE: text=${text}$')
            text = TokenizationNGramSubwordSingleLabelDataFeatureManager.\
                normalize_with_punctuation_segmentation_tokenization(text)
            # ---- NOTE-FOR-DEBUGGING ---- DebuggingHelper.write_line_to_system_console_out(
            # ---- NOTE-FOR-DEBUGGING ----     f'ngram_subword_model_tokenizer_do_normalization-AFTER: text=${text}$')
        if self.ngram_subword_model_tokenizer_do_remove_diacritics:
            text = TokenizationNGramSubwordSingleLabelDataFeatureManager.\
                remove_diacritics(text)
        # --------------------------------------------------------------------
        feature_array = TokenizationNGramSubwordSingleLabelDataFeatureManager.\
            segment_input_to_components_with_language_token_punctuation_delimiters(
                text=text,
                ngram_split_separator=self.ngram_split_separator)
        # --------------------------------------------------------------------
        subword_ngram_features = []
        if self.ngram_subword_model_tokenizer_do_include_subwords:
            subword_ngram_features = TokenizationNGramSubwordSingleLabelDataFeatureManager.\
                extract_subword_ngrams(
                    text=text,
                    ngram_subword_begin=self.ngram_subword_begin,
                    ngram_subword_end=self.ngram_subword_end)
        feature_array.extend(subword_ngram_features)
        # --------------------------------------------------------------------
        return feature_array
        # --------------------------------------------------------------------

    @staticmethod
    def remove_punctuations(text: str) -> str:
        """
        Remove punctuations from an input text.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        for i in range(len(TokenizationNGramSubwordSingleLabelDataFeatureManagerBase.\
            LanguageTokenPunctuationDelimiters)):
            punctuation = TokenizationNGramSubwordSingleLabelDataFeatureManagerBase.\
                LanguageTokenPunctuationDelimiters[i]
            text = text.replace(punctuation, ' ')
        return text.rstrip(' ').lstrip(' ')

    @staticmethod
    def segment_input_to_components_with_language_token_punctuation_delimiters(\
        text: str, \
        ngram_split_separator: str = ' ') -> List[str]:
        """
        Segment a text into tokens.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # ---- NOTE-FOR-DEBUGGING ---- DebuggingHelper.write_line_to_system_console_out(
        # ---- NOTE-FOR-DEBUGGING ----     '---- text=$' + text + '$')
        # ---- NOTE-FOR-DEBUGGING ---- for punctuation_delimiter in TokenizationNGramSubwordSingleLabelDataFeatureManagerBase.LanguageTokenPunctuationDelimiters:
        # ---- NOTE-FOR-DEBUGGING ----     DebuggingHelper.write_line_to_system_console_out(
        # ---- NOTE-FOR-DEBUGGING ----     '---- $' + punctuation_delimiter + '$')
        # ---- NOTE-FOR-DEBUGGING ---- for punctuation_delimiter in TokenizationNGramSubwordSingleLabelDataFeatureManager.LanguageTokenPunctuationWithSpaceDelimiters:
        # ---- NOTE-FOR-DEBUGGING ----     DebuggingHelper.write_line_to_system_console_out(
        # ---- NOTE-FOR-DEBUGGING ----     '---- $' + punctuation_delimiter + '$')
        for i in range(len(TokenizationNGramSubwordSingleLabelDataFeatureManagerBase.LanguageTokenPunctuationDelimiters)):
            punctuation_delimiter = \
                TokenizationNGramSubwordSingleLabelDataFeatureManagerBase.LanguageTokenPunctuationDelimiters[i]
            punctuation_with_space_delimiter = \
                TokenizationNGramSubwordSingleLabelDataFeatureManager.LanguageTokenPunctuationWithSpaceDelimiters[i]
            text = text.replace(punctuation_delimiter, punctuation_with_space_delimiter)
        # ---- NOTE-FOR-DEBUGGING ---- DebuggingHelper.write_line_to_system_console_out(
        # ---- NOTE-FOR-DEBUGGING ----     '---- text=$' + text + '$')
        output = list(filter(None, text.split(ngram_split_separator)))
        # ---- NOTE-FOR-DEBUGGING ---- for o in output:
        # ---- NOTE-FOR-DEBUGGING ----     DebuggingHelper.write_line_to_system_console_out(
        # ---- NOTE-FOR-DEBUGGING ----     '---- $' + o + '$')
        return output

    @staticmethod
    def extract_subword_ngrams(\
        text: str, \
        ngram_subword_begin: int = TokenizationNGramSubwordSingleLabelDataFeatureManagerBase.\
            DEFAULT_SUBWORD_NGRAM_BEGIN, \
        ngram_subword_end: int = TokenizationNGramSubwordSingleLabelDataFeatureManagerBase.\
            DEFAULT_SUBWORD_NGRAM_END, \
        to_prepend_append_leading_and_trailing_character: bool = True, \
        ngram_prefix_character: int = TokenizationNGramSubwordSingleLabelDataFeatureManagerBase.\
            DEFAULT_NGRAM_PREFIX_CHARACTER, \
        ngram_suffix_character: int = TokenizationNGramSubwordSingleLabelDataFeatureManagerBase.\
            DEFAULT_NGRAM_SUFFIX_CHARACTER, \
        ngram_delimiter_character: str = TokenizationNGramSubwordSingleLabelDataFeatureManagerBase.\
            DEFAULT_NGRAM_DELIMITER_CHARACTER) -> List[str]:
        """
        Extrcat subword ngrams.
        """
        # ---- NOTE-PYLINT ---- R0913: Too many arguments
        # pylint: disable=R0913
        output_array = []
        for ngram_length in range(ngram_subword_begin, ngram_subword_end + 1):
            output_array.extend(\
                TokenizationNGramSubwordSingleLabelDataFeatureManager.extract_subword_ngram(
                    text=text,
                    ngram_length=ngram_length,
                    to_prepend_append_leading_and_trailing_character=\
                        to_prepend_append_leading_and_trailing_character,
                    ngram_prefix_character=ngram_prefix_character,
                    ngram_suffix_character=ngram_suffix_character,
                    ngram_delimiter_character=ngram_delimiter_character))
        return output_array
    @staticmethod
    def extract_subword_ngram(\
        text: str, \
        ngram_length: int = 3, \
        to_prepend_append_leading_and_trailing_character: bool = True, \
        ngram_prefix_character: int = TokenizationNGramSubwordSingleLabelDataFeatureManagerBase.\
            DEFAULT_NGRAM_PREFIX_CHARACTER, \
        ngram_suffix_character: int = TokenizationNGramSubwordSingleLabelDataFeatureManagerBase.\
            DEFAULT_NGRAM_SUFFIX_CHARACTER, \
        ngram_delimiter_character: str = TokenizationNGramSubwordSingleLabelDataFeatureManagerBase.\
            DEFAULT_NGRAM_DELIMITER_CHARACTER) -> List[str]:
        """
        Extrcat subword ngrams.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # ---- NOTE-PYLINT ---- R0913: Too many arguments
        # pylint: disable=R0913
        if to_prepend_append_leading_and_trailing_character:
            # text = text.replace(' ', ngramDelimiterCharacter)
            text = ngram_prefix_character + text.replace(' ', ngram_delimiter_character) + ngram_suffix_character
        length = len(text)
        output_array = []
        for i in range(length - ngram_length + 1):
            output_array.append(text[i : i + ngram_length])
        return output_array

    @staticmethod
    def normalize_with_punctuation_segmentation_tokenization(text: str) -> str:
        """
        Normalize an input text.
        """
        tokens = []
        in_white_space: bool = False
        for char in text:
            if not char.isspace():
                if in_white_space or not (char.isdigit() or char.isalpha()):
                    tokens.append(' ')
                    in_white_space = False
                tokens.append(char)
                # ---- NOTE-REFACTORED-TO-STATEMENT-ABOVE ---- if  char.isdigit():
                # ---- NOTE-REFACTORED-TO-STATEMENT-ABOVE ----     tokens.append(char)
                # ---- NOTE-REFACTORED-TO-STATEMENT-ABOVE ---- else:
                # ---- NOTE-REFACTORED-TO-STATEMENT-ABOVE ----     tokens.append(char)
                if not (char.isdigit() or char.isalpha()):
                    tokens.append(' ')
            else:
                in_white_space = bool(tokens)
        return "".join(tokens)

    @staticmethod
    def remove_diacritics(text: str) -> str:
        """
        Remove diacritic characters from an input text.
        """
        return unidecode.unidecode(text)
