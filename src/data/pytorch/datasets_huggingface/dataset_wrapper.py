# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
This module implements base data manager objects
"""

from typing import Union

import datasets

class IHuggingfaceDatasetWrapper:
    """
    This class can manage data.
    """
    # ---- NOTE-PYLINT ---- R0903: Too few public methods
    # pylint: disable=R0903

    def get_dataset(self) -> Union[datasets.DatasetDict, datasets.Dataset, datasets.IterableDatasetDict, datasets.IterableDataset]:
        """
        Return a wrapped HuggingFace dataset.
        """
        raise NotImplementedError('child class should override this method')
