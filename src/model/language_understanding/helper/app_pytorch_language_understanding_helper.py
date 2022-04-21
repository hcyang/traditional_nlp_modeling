# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
This module tests PytorchLanguageUnderstandingHelper functions.
"""

# ---- NOTE-PYLINT ---- E0401: Unable to import 'torch' (import-error)
# pylint: disable=E0401

import argparse
# import os

import torch
from torch.nn import CrossEntropyLoss

from model.language_understanding.helper.pytorch_language_understanding_helper \
    import PytorchLanguageUnderstandingHelper

from utility.pytorch_utility.pytorch_utility \
    import PytorchUtility

from utility.debugging_helper.debugging_helper \
    import DebuggingHelper

def example_load_pytorch_model():
    """
    example code to load a Pytorch model
    """
    # ------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_pytorch_model_filename',
        type=str,
        required=True,
        help='pytorch model filename.')
    # ------------------------------------------------------------------------
    args: argparse.Namespace = parser.parse_args()
    DebuggingHelper.write_line_to_system_console_out(
        f'args={str(args)}')
    # ------------------------------------------------------------------------
    input_pytorch_model_filename: str = args.input_pytorch_model_filename
    DebuggingHelper.write_line_to_system_console_out(
        f'input_pytorch_model_filename={input_pytorch_model_filename}')
    PytorchUtility.load_model_map_to_location(\
        input_pytorch_model_filename=input_pytorch_model_filename)
    # ------------------------------------------------------------------------

def example_function_pytorch_loss_functions():
    """
    Return CUDA memory information.
    """
    # ---- NOTE-PYLINT ---- E1101: Module 'torch' has no
    # pylint: disable=E1101
    loss_function = CrossEntropyLoss()
    predicted = torch.FloatTensor([[0, 0, 0, 1]])
    targeted = torch.LongTensor([3])
    loss = loss_function(predicted, targeted)
    DebuggingHelper.write_line_to_system_console_out(
        f'targeted={targeted}, predicted={predicted}, loss_function={loss_function}, loss={loss}')
    targeted = torch.LongTensor([0])
    loss = loss_function(predicted, targeted)
    DebuggingHelper.write_line_to_system_console_out(
        f'targeted={targeted}, predicted={predicted}, loss_function={loss_function}, loss={loss}')

def example_function_PytorchLanguageUnderstandingHelper():
    """
    The main() function can quickly test PytorchLanguageUnderstandingHelper functions.
    """
    # ---- NOTE-PYLINT ---- C0103: conform to snake_case naming style
    # pylint: disable=C0103
    gpu_configuration = PytorchUtility.get_gpu_configuration()
    DebuggingHelper.write_line_to_system_console_out(
        f'gpu configuration={str(gpu_configuration)}')
    gpu_configuration = PytorchUtility.get_gpu_configuration(
        device_gpu_cuda_local_rank=-1,
        device_gpu_disable_cuda=False)
    DebuggingHelper.write_line_to_system_console_out(
        f'gpu configuration (device_gpu_cuda_local_rank=-1, device_gpu_disable_cuda=False)='
        f'{str(gpu_configuration)}')
    gpu_configuration = PytorchUtility.get_gpu_configuration(
        device_gpu_cuda_local_rank=-1,
        device_gpu_disable_cuda=True)
    DebuggingHelper.write_line_to_system_console_out(
        f'gpu configuration (device_gpu_cuda_local_rank=-1, device_gpu_disable_cuda=True)='
        f'{str(gpu_configuration)}')
    gpu_configuration = PytorchUtility.get_gpu_configuration(
        device_gpu_cuda_local_rank=0,
        device_gpu_disable_cuda=False)
    DebuggingHelper.write_line_to_system_console_out(
        f'gpu configuration (device_gpu_cuda_local_rank=0, device_gpu_disable_cuda=False)='
        f'{str(gpu_configuration)}')
    gpu_configuration = PytorchUtility.get_gpu_configuration(
        device_gpu_cuda_local_rank=0,
        device_gpu_disable_cuda=True)
    DebuggingHelper.write_line_to_system_console_out(
        f'gpu configuration (device_gpu_cuda_local_rank=0, device_gpu_disable_cuda=True)='
        f'{str(gpu_configuration)}')
    gpu_configuration = PytorchUtility.get_gpu_configuration(
        device_gpu_cuda_local_rank=1,
        device_gpu_disable_cuda=False)
    DebuggingHelper.write_line_to_system_console_out(
        f'gpu configuration (device_gpu_cuda_local_rank=1, device_gpu_disable_cuda=False)='
        f'{str(gpu_configuration)}')
    gpu_configuration = PytorchUtility.get_gpu_configuration(
        device_gpu_cuda_local_rank=1,
        device_gpu_disable_cuda=True)
    DebuggingHelper.write_line_to_system_console_out(
        f'gpu configuration (device_gpu_cuda_local_rank=1, device_gpu_disable_cuda=True)='
        f'{str(gpu_configuration)}')
    torch_env_info = PytorchUtility.torch_utils_get_pretty_env_info()
    DebuggingHelper.write_line_to_system_console_out(
        f'torch_env_info={str(torch_env_info)}')
    torch_version = PytorchUtility.get_torch_version()
    DebuggingHelper.write_line_to_system_console_out(
        f'torch_version={str(torch_version)}')
    device_properties = PytorchUtility.get_torch_cuda_device_properties(0)
    DebuggingHelper.write_line_to_system_console_out(
        f'device_properties={str(device_properties)}')

def main():
    """
    The main() function can quickly test PytorchLanguageUnderstandingHelper functions.
    """
    example_function_PytorchLanguageUnderstandingHelper()
    example_function_pytorch_loss_functions()
    PytorchLanguageUnderstandingHelper.pytorch_example_code_0_simple_function_tanh()
    PytorchLanguageUnderstandingHelper.pytorch_example_code_1_linear()
    print("torch.__version__={0}".format(torch.__version__))
    example_load_pytorch_model()

if __name__ == '__main__':
    main()
