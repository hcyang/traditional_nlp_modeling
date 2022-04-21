# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
This module implements pytorch language understanding helper functions.
"""

import random
import numpy

# ---- NOTE-PYLINT ---- E0401: Unable to import 'torch' (import-error)
# pylint: disable=E0401

import torch

# from utility.debugging_helper.debugging_helper \
#     import DebuggingHelper

class PytorchLanguageUnderstandingHelper:
    """
    This class defined helper functions and data structures
    for using pytorch.
    """

    @staticmethod
    def pytorch_example_code_0_simple_function_tanh():
        """
        A Pytorch example.
        """
        class PytorchExampleTanhCell(torch.nn.Module):
            """
            A simple cell class for running Pytorch example.
            """
            # ---- NOTE-PYLINT ---- R0903: Too few public methods (1/2) (too-few-public-methods)
            # pylint: disable=R0903
            def __init__(self):
                # ---- NOTE ---- pylint complians either we call super().__init__() or not.
                # ---- NOTE-PYLINT ---- C0301: Line too long
                # pylint: disable=C0301
                # ---- NOTE-PYLINT ---- W0235: Useless super delegation in method '__init__' (useless-super-delegation)
                # pylint: disable=W0235
                super(PytorchExampleTanhCell, self).__init__()
                # ---- NOTE-PYLINT ---- C0301: Line too long
                # pylint: disable=C0301
                # ---- NOTE-PYLINT ---- W0231: __init__ method from base class 'Module' is not called (super-init-not-called)
                # pylint: disable=W0231
            def forward(self, *arguments) -> torch.Tensor:
                """
                forward function
                """
                # ---- NOTE-PYLINT ---- R0201: Method could be a function (no-self-use)
                # pylint: disable=R0201
                input_x = arguments[0]
                input_h = arguments[1]
                # ---- NOTE-PYLINT ---- E1101: Module 'torch' has no 'tanh' member (no-member)
                # pylint: disable=E1101
                new_h = torch.tanh(input_x + input_h)
                return new_h, new_h
        example_cell = PytorchExampleTanhCell()
        # ---- NOTE-PYLINT ---- E1101: Module 'torch' has no 'tanh' member (no-member)
        # pylint: disable=E1101
        input_x = torch.rand(3, 4)
        input_h = torch.rand(3, 4)
        print(example_cell(input_x, input_h))
    @staticmethod
    def pytorch_example_code_1_linear():
        """
        A Pytorch example.
        """
        class PytorchExampleLinear(torch.nn.Module):
            """
            A simple cell class for running Pytorch example.
            """
            # ---- NOTE-PYLINT ---- R0903: Too few public methods (1/2) (too-few-public-methods)
            # pylint: disable=R0903
            def __init__(self):
                # ---- NOTE ---- pylint complians either we call super().__init__() or not.
                # ---- NOTE-PYLINT ---- C0301: Line too long
                # pylint: disable=C0301
                # ---- NOTE-PYLINT ---- W0235: Useless super delegation in method '__init__' (useless-super-delegation)
                # pylint: disable=W0235
                super(PytorchExampleLinear, self).__init__()
                # ---- NOTE-PYLINT ---- C0301: Line too long
                # pylint: disable=C0301
                # ---- NOTE-PYLINT ---- W0231: __init__ method from base class 'Module' is not called (super-init-not-called)
                # pylint: disable=W0231
                self.linear = torch.nn.Linear(4, 4)
            def forward(self, *arguments) -> torch.Tensor:
                """
                forward function
                """
                # ---- NOTE-PYLINT ---- R0201: Method could be a function (no-self-use)
                # pylint: disable=R0201
                input_x = arguments[0]
                input_h = arguments[1]
                # ---- NOTE-PYLINT ---- E1101: Module 'torch' has no 'tanh' member (no-member)
                # pylint: disable=E1101
                new_h = torch.tanh(self.linear(input_x) + input_h)
                return new_h, new_h
        example_cell = PytorchExampleLinear()
        # ---- NOTE-PYLINT ---- E1101: Module 'torch' has no 'tanh' member (no-member)
        # pylint: disable=E1101
        input_x = torch.rand(3, 4)
        input_h = torch.rand(3, 4)
        print(example_cell)
        print(example_cell(input_x, input_h))
