# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
This module implements a base feature manager object.
"""

import os

from typing import List
from typing import Tuple

# from utility.io_helper.io_helper \
#     import IoHelper
from utility.debugging_helper.debugging_helper \
    import DebuggingHelper
# from utility.configuration_helper.configuration_helper \
#     import ConfigurationHelper

class BaseFeatureManager():
    """
    This class can create features on an input text.
    """

    def __init__(self):
        """
        Init with a data manager object and a tokenizer object.
        """
        pass

    def get_number_features(self) -> int:
        # ---- NOTE-ABSTRACT-FUNCTION ----
        """
        Return the number of features.
        An abstract function that child classes must override.
        """
        raise NotImplementedError()

    def serialize_featurizer(self, serialization_destination: str, dump: bool = True):
        # ---- NOTE-ABSTRACT-FUNCTION ----
        """
        Serialize a featurizer.
        """
        raise NotImplementedError()
    def deserialize_featurizer(self, serialization_destination: str):
        # ---- NOTE-ABSTRACT-FUNCTION ----
        """
        Deserialize a featurizer.
        """
        raise NotImplementedError()

    def serialize(self, serialization_destination: str, dump: bool = True):
        # ---- NOTE-CPNCREATE-BASE-FUNCTION ----
        """
        Serialize self feature manager.
        """
        return self.serialize_featurizer(
            serialization_destination=serialization_destination,
            dump=dump)
    def deserialize(self, serialization_destination: str):
        # ---- NOTE-CPNCREATE-BASE-FUNCTION ----
        """
        Deserialize self feature manager.
        """
        return self.deserialize_featurizer(
            serialization_destination=serialization_destination)

    def convert_id_to_feature(self, id: int) -> str:
        # ---- NOTE-ABSTRACT-FUNCTION ----
        """
        Convert a feature ID back to its str form.
        """
        raise NotImplementedError()
    def convert_feature_to_id(self, feature: str) -> int:
        # ---- NOTE-ABSTRACT-FUNCTION ----
        """
        Convert a feature to it ID.
        """
        raise NotImplementedError()

    def convert_ids_to_features(self, ids: List[int]) -> str:
        # ---- NOTE-ABSTRACT-FUNCTION ----
        """
        Convert a list of feature IDs back to its str form.
        """
        return [self.convert_id_to_feature(x) for x in ids]
    def convert_features_to_ids(self, features: List[str]) -> int:
        # ---- NOTE-ABSTRACT-FUNCTION ----
        """
        Convert a list of features to it ID.
        """
        return [self.convert_feature_to_id(x) for x in features]

    def create_feature_and_ids(self, text: str) -> Tuple[List[str], List[int]]:
        # ---- NOTE-ABSTRACT-FUNCTION ----
        """
        Create a list of features, each is a list of strings.
        An abstract function that child classes must override.
        """
        raise NotImplementedError()

    def create_features(self, text: str) -> List[str]:
        # ---- NOTE-ABSTRACT-FUNCTION ----
        """
        Create a list of features, each is a list of strings.
        An abstract function that child classes must override.
        """
        raise NotImplementedError()
