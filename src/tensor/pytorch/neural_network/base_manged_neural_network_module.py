# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
This module provides some ONNX helper functions.
"""

# from typing import List
# from typing import Tuple

# from tqdm import tqdm, trange

import torch
import torch.nn
import torch.onnx

class BaseMangedNeuralNetworkModule(torch.nn.Module):
    """
    BaseMangedNeuralNetworkModule
    """
    # ---- NOTE-PLACE-HOLDER ---- def __init__(self):
    # ---- NOTE-PLACE-HOLDER ----     """
    # ---- NOTE-PLACE-HOLDER ----     Init.
    # ---- NOTE-PLACE-HOLDER ----     """
    # ---- NOTE-PLACE-HOLDER ----     super(BaseMangedNeuralNetworkModule, self).__init__()

    def forward(self, intermediate_tensor: torch.Tensor) -> torch.Tensor:
        """
        forward()
        """
        raise NotImplementedError('child class should override this function')

    def prepare_batch_for_feature_forward(self, batch: any) -> any:
        """
        Prepare an input batch during model training process.
        """
        return batch
    def prepare_batch_for_label_loss_evaluation(self, batch: any) -> any:
        """
        Prepare an input batch during model training process.
        """
        return batch
    def prepare_batch_for_label_prediction_evaluation(self, batch: any) -> any:
        """
        Prepare an input batch during model predicting process.
        """
        return self.prepare_batch_for_label_loss_evaluation(batch).tolist()
