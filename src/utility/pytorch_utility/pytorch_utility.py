# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
This module provides some ONNX helper functions.
"""
# ---- NOTE-PYLINT ---- C0302: Too many lines in module (*/1000) (too-many-lines)
# pylint: disable=C0302

from typing import Any
from typing import Dict
from typing import List
from typing import NoReturn
from typing import Tuple
from typing import Union

# import json
# import os
import random
import numpy

import tqdm
from tqdm import tqdm
# from tqdm import trange

import torch
import torch.nn
import torch.onnx

from torch.utils.data import DataLoader
from torch.utils.collect_env import get_pretty_env_info

from sklearn.metrics import classification_report

from transformers import BertModel
# from transformers import BertForSequenceClassification

from tensor.pytorch.neural_network.base_manged_neural_network_module \
    import BaseMangedNeuralNetworkModule

# from data.manager.base_dataset_manager \
#     import BaseDatasetManager

from model.language_understanding.featurizer.base_feature_manager \
    import BaseFeatureManager

# from model.language_understanding.helper.pytorch_language_understanding_helper \
#     import PytorchLanguageUnderstandingHelper

from utility.io_helper.io_helper \
    import IoHelper
from utility.debugging_helper.debugging_helper \
    import DebuggingHelper

class PytorchUtility:
    """
    This class contains some utility functions for Pytorch tasks.
    """

    # ------------------------------------------------------------------------
    # ---- NOTE-GENERIC-PYTORCH-UTILITY-FUNCTIONS ----
    # ------------------------------------------------------------------------

    @staticmethod
    def set_all_random_number_generator_seeds( \
        random_number_generator_seed, \
        device_gpu_disable_cuda: bool = False) -> NoReturn:
        """
        Reset the randomization seed for a variety of randomizers.
        """
        PytorchUtility.set_individual_random_number_generator_seeds(
            random_number_generator_seed_random=random_number_generator_seed,
            random_number_generator_seed_random_numpy=random_number_generator_seed,
            random_number_generator_seed_random_torch_manual=random_number_generator_seed,
            random_number_generator_seed_random_torch_manual_cuda=random_number_generator_seed,
            device_gpu_disable_cuda=device_gpu_disable_cuda)
    @staticmethod
    def set_individual_random_number_generator_seeds( \
        random_number_generator_seed_random, \
        random_number_generator_seed_random_numpy, \
        random_number_generator_seed_random_torch_manual, \
        random_number_generator_seed_random_torch_manual_cuda, \
        device_gpu_disable_cuda: bool = False) -> NoReturn:
        """
        Reset the randomization seed for a variety of randomizers.
        """
        random.seed(random_number_generator_seed_random)
        numpy.random.seed(random_number_generator_seed_random_numpy)
        torch.manual_seed(random_number_generator_seed_random_torch_manual)
        if not device_gpu_disable_cuda:
            torch.cuda.manual_seed_all(random_number_generator_seed_random_torch_manual_cuda)

    @staticmethod
    def set_torch_cuda_per_process_memory_fraction( \
        fraction: float = 1, \
        device: Union[torch.device, int] = None) -> NoReturn:
        """
        set_torch_cuda_per_process_memory_fraction()
        """
        torch.cuda.set_per_process_memory_fraction(fraction, device)

    @staticmethod
    def torch_cuda_empty_cache() -> NoReturn:
        """
        Empty CUDA device cache.
        """
        torch.cuda.empty_cache()

    @staticmethod
    def torch_can_dtype_have_gradient(tensor_dtype: torch.dtype) -> bool:
        """
        Test if a dtype input can have gradient.
        REFERENCE: https://pytorch.org/docs/stable/tensor_attributes.html
        """
        return PytorchUtility.torch_is_dtype_float_complex(tensor_dtype=tensor_dtype)
    @staticmethod
    def torch_is_dtype_float_complex(tensor_dtype: torch.dtype) -> bool:
        """
        Test if a dtype input is a float or complex type.
        REFERENCE: https://pytorch.org/docs/stable/tensor_attributes.html
        """
        return PytorchUtility.torch_is_dtype_float(tensor_dtype=tensor_dtype) or \
            PytorchUtility.torch_is_dtype_complex(tensor_dtype=tensor_dtype)
    @staticmethod
    def torch_is_dtype_float(tensor_dtype: torch.dtype) -> bool:
        """
        Test if a dtype input is a float type.
        REFERENCE: https://pytorch.org/docs/stable/tensor_attributes.html
        """
        if tensor_dtype == torch.float:
            return True
        if tensor_dtype == torch.double:
            return True
        if tensor_dtype == torch.half:
            return True
        if tensor_dtype == torch.float16:
            return True
        if tensor_dtype == torch.float32:
            return True
        if tensor_dtype == torch.float64:
            return True
        if tensor_dtype == torch.bfloat16:
            return True
        return False
    @staticmethod
    def torch_is_dtype_complex(tensor_dtype: torch.dtype) -> bool:
        """
        Test if a dtype input is a complex type.
        REFERENCE: https://pytorch.org/docs/stable/tensor_attributes.html
        """
        if tensor_dtype == torch.complex32:
            return True
        if tensor_dtype == torch.complex64:
            return True
        if tensor_dtype == torch.complex128:
            return True
        return False
    @staticmethod
    def torch_is_dtype_integer(tensor_dtype: torch.dtype) -> bool:
        """
        Test if a dtype input is an integer type.
        REFERENCE: https://pytorch.org/docs/stable/tensor_attributes.html
        """
        if tensor_dtype == torch.int:
            return True
        if tensor_dtype == torch.short:
            return True
        if tensor_dtype == torch.long:
            return True
        if tensor_dtype == torch.int8:
            return True
        if tensor_dtype == torch.int16:
            return True
        if tensor_dtype == torch.int32:
            return True
        if tensor_dtype == torch.int64:
            return True
        if tensor_dtype == torch.uint8:
            return True
        return False

    # ------------------------------------------------------------------------

    @staticmethod
    def get_torch_cuda_list_gpu_processes(device=None) -> str:
        """
        Return CUDA GPU device processes.
        """
        return torch.cuda.list_gpu_processes(device)

    # ------------------------------------------------------------------------

    @staticmethod
    def get_torch_cuda_memory_stats(device=None) -> Dict[str, Any]:
        """
        Return CUDA GPU device stats.
        """
        return torch.cuda.memory_stats(device)

    @staticmethod
    def get_torch_cuda_memory_summary(device=None) -> str:
        """
        Return CUDA GPU device summary.
        """
        return torch.cuda.memory_summary(device)

    @staticmethod
    def get_torch_cuda_memory_snapshot() -> Any:
        """
        Return CUDA GPU device snapshot.
        """
        return torch.cuda.memory_snapshot()

    @staticmethod
    def get_torch_cuda_memory_allocated(device=None) -> int:
        """
        Return CUDA memory allocated.
        """
        return torch.cuda.memory_allocated(device)

    @staticmethod
    def get_torch_cuda_max_memory_allocated(device=None) -> int:
        """
        Return CUDA max memory allocated.
        """
        return torch.cuda.max_memory_allocated(device)

    @staticmethod
    def reset_torch_cuda_max_memory_allocated(device=None) -> NoReturn:
        """
        Reset CUDA max memory allocated.
        """
        torch.cuda.reset_max_memory_allocated(device)

    @staticmethod
    def get_torch_cuda_memory_reserved(device=None) -> int:
        """
        Return CUDA memory reserved.
        """
        return torch.cuda.memory_reserved(device)

    @staticmethod
    def get_torch_cuda_max_memory_reserved(device=None) -> int:
        """
        Return CUDA max memory reserved.
        """
        return torch.cuda.max_memory_reserved(device)

    @staticmethod
    def get_torch_cuda_memory_cached(device=None) -> int:
        """
        Return CUDA memory cached.
        """
        return torch.cuda.memory_cached(device)

    @staticmethod
    def get_torch_cuda_max_memory_cached(device=None) -> int:
        """
        Return CUDA max memory cached.
        """
        return torch.cuda.max_memory_cached(device)

    @staticmethod
    def reset_torch_cuda_max_memory_cached(device=None) -> NoReturn:
        """
        Reset CUDA max memory cached.
        """
        torch.cuda.reset_max_memory_cached(device)

    @staticmethod
    def reset_torch_cuda_peak_memory_stats(device=None) -> NoReturn:
        """
        Reset CUDA peak memory stats.
        """
        torch.cuda.reset_peak_memory_stats(device)

    @staticmethod
    def get_torch_cuda_memory_info(device=None) -> Any:
        """
        Return CUDA memory information.
        """
        try:
            return { \
                'device': \
                str(device), \
                'torch_cuda_get_device_capability': \
                PytorchUtility.get_torch_cuda_get_device_capability(device), \
                'torch_cuda_memory_stats': \
                PytorchUtility.get_torch_cuda_memory_stats(device), \
                'torch_cuda_memory_summary': \
                PytorchUtility.get_torch_cuda_memory_summary(device), \
                'torch_cuda_memory_snapshot': \
                PytorchUtility.get_torch_cuda_memory_snapshot(), \
                'torch_cuda_memory_allocated': \
                PytorchUtility.get_torch_cuda_memory_allocated(device), \
                'torch_cuda_max_memory_allocated': \
                PytorchUtility.get_torch_cuda_max_memory_allocated(device), \
                'torch_cuda_memory_reserved': \
                PytorchUtility.get_torch_cuda_memory_reserved(device), \
                'torch_cuda_max_memory_reserved': \
                PytorchUtility.get_torch_cuda_max_memory_reserved(device), \
                'torch_cuda_memory_cached': \
                PytorchUtility.get_torch_cuda_memory_cached(device), \
                'torch_cuda_max_memory_cached': \
                PytorchUtility.get_torch_cuda_max_memory_cached(device)}
        except ValueError: # ---- NOTE: ValueError: Expected a cuda device, but got: cpu
            pass
        return {}

    # ------------------------------------------------------------------------

    @staticmethod
    def get_torch_cuda_get_device_capability(device=None) -> Tuple[int, int]:
        """
        Return CUDA max memory allocated.
        """
        return torch.cuda.get_device_capability(device)

    @staticmethod
    def get_torch_cuda_device_properties(device=None) -> Any:
        """
        Return CUDA device properties.
        """
        return torch.cuda.get_device_properties(device)

    @staticmethod
    def torch_utils_get_pretty_env_info() -> str:
        """
        Delegate to torch.utils.get_pretty_env_info()
        """
        return get_pretty_env_info()

    @staticmethod
    def get_torch_version() -> Any:
        """
        Delegate to torch.__version__
        """
        return torch.__version__

    # ------------------------------------------------------------------------

    @staticmethod
    def get_gpu_configuration( \
        device_gpu_disable_cuda: bool = False, \
        device_gpu_disable_distributed_cuda: bool = True, \
        device_gpu_use_distributed_cuda_explicit_settings: bool = False, \
        device_gpu_cuda_local_rank: int = 0, \
        device_gpu_cuda_device_name: str = 'cuda', \
        device_gpu_cuda_distributed_process_group_backend_name: str = 'nccl', \
        device_gpu_cuda_distributed_process_group_init_method: str = 'tcp://10.1.1.20:23456', \
        device_gpu_cuda_distributed_process_group_timeout: int = 1000, \
        device_gpu_cuda_distributed_process_group_world_size: int = 4, \
        device_gpu_cuda_distributed_process_group_rank: int = 0) -> Any:
        """
        Return gpu device and configuration.
        """
        # ---- NOTE-PYLINT ---- R0913: Too many arguments
        # pylint: disable=R0913
        # ---- NOTE-PYLINT ---- R0914: Too many local variables
        # pylint: disable=R0914
        # ---- NOTE-PYLINT ---- E1101: Module 'torch' has no 'device' member
        # pylint: disable=E1101
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        torch_device_cuda_is_available = torch.cuda.is_available()
        torch_device_distributed_gpu_is_available = torch.distributed.is_available()
        torch_device_gpu_device_count = torch.cuda.device_count()
        if (device_gpu_cuda_local_rank < 0) or device_gpu_disable_cuda:
            try:
                device = torch.device(
                    device_gpu_cuda_device_name if torch_device_cuda_is_available and not device_gpu_disable_cuda else 'cpu')
            except AssertionError:
                torch_env_info = PytorchUtility.torch_utils_get_pretty_env_info()
                DebuggingHelper.write_line_to_system_console_err(
                    f'torch_env_info={str(torch_env_info)}')
                torch_version = PytorchUtility.get_torch_version()
                DebuggingHelper.write_line_to_system_console_err(
                    f'torch_version={str(torch_version)}')
                device_properties = PytorchUtility.get_torch_cuda_device_properties(0)
                DebuggingHelper.write_line_to_system_console_err(
                    f'device_properties={str(device_properties)}')
                raise
            device_number_gpus = torch_device_gpu_device_count
        else:
            if device_gpu_cuda_local_rank >= torch_device_gpu_device_count:
                device_gpu_cuda_local_rank = 0
            try:
                torch.cuda.set_device(device_gpu_cuda_local_rank)
            except AssertionError:
                torch_env_info = PytorchUtility.torch_utils_get_pretty_env_info()
                DebuggingHelper.write_line_to_system_console_err(
                    f'torch_env_info={str(torch_env_info)}')
                torch_version = PytorchUtility.get_torch_version()
                DebuggingHelper.write_line_to_system_console_err(
                    f'torch_version={str(torch_version)}')
                device_properties = PytorchUtility.get_torch_cuda_device_properties(0)
                DebuggingHelper.write_line_to_system_console_err(
                    f'device_properties={str(device_properties)}')
                raise
            device = torch.device(
                device_gpu_cuda_device_name,
                device_gpu_cuda_local_rank)
            device_number_gpus = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            if torch_device_distributed_gpu_is_available and not device_gpu_disable_distributed_cuda:
                if device_gpu_use_distributed_cuda_explicit_settings:
                    torch.distributed.init_process_group(
                        backend=device_gpu_cuda_distributed_process_group_backend_name,
                        init_method=device_gpu_cuda_distributed_process_group_init_method,
                        timeout=device_gpu_cuda_distributed_process_group_timeout,
                        world_size=device_gpu_cuda_distributed_process_group_world_size,
                        rank=device_gpu_cuda_distributed_process_group_rank)
                else:
                    torch.distributed.init_process_group(
                        backend=device_gpu_cuda_distributed_process_group_backend_name,
                        init_method=device_gpu_cuda_distributed_process_group_init_method,
                        timeout=device_gpu_cuda_distributed_process_group_timeout)
            # ---- NOTE-PYLINT ---- C0301: Line too long
            # pylint: disable=C0301
            # REFERENCE: https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization
            # pylint: enable=C0301
        return {
            'device_gpu_disable_cuda': \
                device_gpu_disable_cuda,
            'device_gpu_disable_distributed_cuda': \
                device_gpu_disable_distributed_cuda,
            'device_gpu_use_distributed_cuda_explicit_settings': \
                device_gpu_use_distributed_cuda_explicit_settings,
            'device_gpu_cuda_local_rank': \
                device_gpu_cuda_local_rank,
            'device_gpu_cuda_device_name': \
                device_gpu_cuda_device_name,
            'device_gpu_cuda_distributed_process_group_backend_name': \
                device_gpu_cuda_distributed_process_group_backend_name,
            'device_gpu_cuda_distributed_process_group_init_method': \
                device_gpu_cuda_distributed_process_group_init_method,
            'device_gpu_cuda_distributed_process_group_timeout': \
                device_gpu_cuda_distributed_process_group_timeout,
            'device_gpu_cuda_distributed_process_group_world_size': \
                device_gpu_cuda_distributed_process_group_world_size,
            'device_gpu_cuda_distributed_process_group_rank': \
                device_gpu_cuda_distributed_process_group_rank,
            'device': \
                device,
            'device_str': \
                str(device),
            'device_number_gpus': \
                device_number_gpus,
            'torch_device_cuda_is_available': \
                torch_device_cuda_is_available,
            'torch_device_distributed_gpu_is_available': \
                torch_device_distributed_gpu_is_available,
            'torch_device_gpu_device_count': \
                torch_device_gpu_device_count}

    @staticmethod
    def get_model_optimizer_parameters( \
        model, \
        no_decay_parameters: List[str] = None) -> List[Dict[str, Any]]:
        """
        get_model_optimizer_parameters()
        """
        model_named_parameters = list(model.named_parameters())
        if no_decay_parameters is None:
            no_decay_parameters = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in model_named_parameters if not any(
                        nd in n for nd in no_decay_parameters)],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in model_named_parameters if any(
                        nd in n for nd in no_decay_parameters)],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_parameters

    # ------------------------------------------------------------------------
    # ---- NOTE-GENERIC-PYTORCH-FUNCTIONS-FOR-BaseMangedNeuralNetworkModule ----
    # ------------------------------------------------------------------------

    @staticmethod
    def predict_model( \
        model: BaseMangedNeuralNetworkModule, \
        data_loader: DataLoader, \
        target_names: List[str] = None):
        """
        Predict using a neural network model.
        """
        # ---- NOTE-PYLINT ---- W0612 Unused Variable
        # pylint: disable=W0612
        if model is None:
            DebuggingHelper.throw_exception(
                'input argument, model, is None')
        # ---- DebuggingHelper.write_line_to_system_console_out(
        # ----     f'model.parameters(): {list(model.parameters())}')
        epoch_iterator = tqdm(
            data_loader,
            desc='predict_model()')
        model.eval()
        label_ids_true: List[int] = []
        label_ids_predicted: List[int] = []
        for step, batch in enumerate(epoch_iterator):
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'batch={batch}')
            # ---- NOTE-TO-DO-PLACE-TENSOR-TO-DEVICE ---- batch = tuple(t.to(device) for t in batch)
            labels = model.prepare_batch_for_label_prediction_evaluation(batch)
            features = model.prepare_batch_for_feature_forward(batch)
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'labels={labels}')
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'features={features}')
            forwarded: torch.Tensor = model.forward(features)
            # ---- NOTE-PYLINT ---- E1101: Module 'torch' has no '*' member (no-member)
            # pylint: disable=E1101
            predictions = torch.argmax(forwarded, 1)
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'forwarded={forwarded}')
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'predictions={predictions}')
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'labels={labels}')
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'forwarded.dim()={forwarded.dim()}')
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'predictions.dim()={predictions.dim()}')
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'labels.dim()={labels.dim()}')
            label_ids_true.extend(labels)
            label_ids_predicted.extend(predictions.tolist())
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'step={step}')
        classification_report_object = classification_report(
            y_true=label_ids_true,
            y_pred=label_ids_predicted,
            labels=labels,
            target_names=target_names)
        return (label_ids_true, label_ids_predicted, classification_report_object)

    @staticmethod
    def train_model( \
        model: BaseMangedNeuralNetworkModule, \
        data_loader: DataLoader, \
        epoches: int, \
        learning_rate: float = 0.01, \
        model_name_prefix: str = "model", \
        batch_logging_interval: int = 4, \
        epoch_checkpoint_interval: int = 4):
        """
        Train a neural network model.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # ---- NOTE-PYLINT ---- W0613: Unused argument
        # pylint: disable=W0613
        # ---- NOTE-PYLINT ---- R0913: Too many arguments (*/5) (too-many-arguments)
        # pylint: disable=R0913
        # ---- NOTE-PYLINT ---- R0914:Too many local variables (*/15) (too-many-locals)
        # pylint: disable=R0914
        if model is None:
            DebuggingHelper.throw_exception(
                'input argument, model, is None')
        DebuggingHelper.write_line_to_system_console_out(
            f'model.parameters() - BEFORE-TRAINING: {list(model.parameters())}')
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        loss_function = torch.nn.NLLLoss()
        for epoch in range(0, epoches):
            epoch_iterator = tqdm(
                data_loader,
                desc='train_model()')
            for step, batch in enumerate(epoch_iterator):
                model.train()
                # ---- DebuggingHelper.write_line_to_system_console_out(
                # ----     f'batch={batch}')
                # ---- NOTE-TO-DO-PLACE-TENSOR-TO-DEVICE ---- batch = tuple(t.to(device) for t in batch)
                labels = model.prepare_batch_for_label_loss_evaluation(batch)
                features = model.prepare_batch_for_feature_forward(batch)
                # ---- DebuggingHelper.write_line_to_system_console_out(
                # ----     f'labels={labels}')
                # ---- DebuggingHelper.write_line_to_system_console_out(
                # ----     f'features={features}')
                optimizer.zero_grad()
                forwarded: torch.Tensor = model.forward(features)
                # ---- DebuggingHelper.write_line_to_system_console_out(
                # ----     f'forwarded={forwarded}')
                # ---- DebuggingHelper.write_line_to_system_console_out(
                # ----     f'labels={labels}')
                # ---- DebuggingHelper.write_line_to_system_console_out(
                # ----     f'forwarded.dim()={forwarded.dim()}')
                # ---- DebuggingHelper.write_line_to_system_console_out(
                # ----     f'labels.dim()={labels.dim()}')
                loss_object = loss_function(forwarded, labels)
                DebuggingHelper.write_line_to_system_console_out(
                    f'epoch={epoch}, step={step}, loss_object={loss_object}')
                loss_object.backward()
                optimizer.step()
        DebuggingHelper.write_line_to_system_console_out(
            f'model.parameters() - AFTER-TRAINING: {list(model.parameters())}')

    # ------------------------------------------------------------------------
    # ---- NOTE-GENERIC-PYTORCH-MODEL-FUNCTIONS ----
    # ------------------------------------------------------------------------

    @staticmethod
    def train_model_per_epoch( \
        data_loader: DataLoader, \
        model: torch.nn.Module, \
        optimizer, \
        device=None, \
        device_resettled_after_operation: torch.device = None, \
        scheduler=None):
        """
        A generic training function for Pytorch models.
        """
        if model is None:
            DebuggingHelper.throw_exception( \
                'PytorchUtility.train_model_per_epoch(), model is None')
        if data_loader is None:
            DebuggingHelper.throw_exception( \
                'PytorchUtility.train_model_per_epoch(), data_loader is None')
        if optimizer is None:
            DebuggingHelper.throw_exception( \
                'PytorchUtility.train_model_per_epoch(), optimizer is None')
        model.train()
        final_loss = 0
        for data in tqdm(data_loader, total=len(data_loader)):
            if device is not None:
                for key, value in data.items():
                    data[key] = value.to(device)
            optimizer.zero_grad()
            _, loss = model(**data)
            if device_resettled_after_operation is not None:
                for key, value in data.items():
                    data[key] = value.to(device_resettled_after_operation)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            final_loss += loss.item()
        return final_loss / len(data_loader)

    @staticmethod
    def evaluate_model_per_epoch( \
        data_loader: DataLoader, \
        model: torch.nn.Module, \
        device: torch.device = None, \
        device_resettled_after_operation: torch.device = None):
        """
        A generic evaluation function for Pytorch models.
        """
        if model is None:
            DebuggingHelper.throw_exception( \
                'PytorchUtility.evaluate_model_per_epoch(), model is None')
        if data_loader is None:
            DebuggingHelper.throw_exception( \
                'PytorchUtility.evaluate_model_per_epoch(), data_loader is None')
        model.eval()
        final_loss = 0
        for data in tqdm(data_loader, total=len(data_loader)):
            if device is not None:
                for key, value in data.items():
                    data[key] = value.to(device)
            output, loss = model(**data)
            final_loss += loss.item()
            if device_resettled_after_operation is not None:
                for key, value in data.items():
                    data[key] = value.to(device_resettled_after_operation)
            # DebuggingHelper.write_line_to_system_console_out( \
            #     f'PytorchUtility.evaluate_model_per_epoch(), output={output}')
            # DebuggingHelper.write_line_to_system_console_out( \
            #     f'PytorchUtility.evaluate_model_per_epoch(), output.size()={output.size()}')
            # DebuggingHelper.write_line_to_system_console_out( \
            #     f'PytorchUtility.evaluate_model_per_epoch(), loss={loss}')
        return final_loss / len(data_loader)

    TORCH_LOSS_FUNCTION_CrossEntropyLoss = torch.nn.CrossEntropyLoss()

    @staticmethod
    def evaluate_loss_per_instance( \
        output: torch.tensor, \
        target: torch.tensor, \
        mask: torch.tensor, \
        number_labels: int) -> torch.tensor:
        """
        A generic loss evaluation function for Pytorch models.
        """
        DebuggingHelper.write_line_to_system_console_out(
            f'PytorchUtility.evaluate_loss_per_instance():'
            f' output.size()={output.size()}')
        DebuggingHelper.write_line_to_system_console_out(
            f'PytorchUtility.evaluate_loss_per_instance():'
            f' target.size()={target.size()}')
        DebuggingHelper.write_line_to_system_console_out(
            f'PytorchUtility.evaluate_loss_per_instance():'
            f' mask.size()={mask.size()}')
        DebuggingHelper.write_line_to_system_console_out(
            f'PytorchUtility.evaluate_loss_per_instance():'
            f' number_labels={number_labels}')
        # ---- NOTE ---- create a 1-D mask boolean tensor for filtering loss values.
        mask_filtering_loss: torch.tensor = mask.view(-1) == 1
        # ---- NOTE ---- conform output tensor to number of labels.
        output_viewed_in_number_labels: torch.tensor = output.view(-1, number_labels)
        # ---- NOTE ---- create a 1-D target tensor.
        target_1d: torch.tensor = target.view(-1)
        # ---- NOTE ---- filter the 1-D target tensor using the mask, the default (FALSE) is
        # ---- NOTE ---- assigned with the value of torch.nn.CrossEntropyLoss().ignore_index
        # ---- NOTE-PYLINT ---- E1102: torch.tensor is not callable (not-callable)
        # pylint: disable=E1102
        default_target_value: torch.tensor = torch.tensor( \
            PytorchUtility.TORCH_LOSS_FUNCTION_CrossEntropyLoss.ignore_index).type_as(target_1d)
        # ---- NOTE ---- select the target value based on the loss-filtering mask.
        # ---- NOTE-PYLINT ---- E1101: Module 'torch' has no '*' member (no-member)
        # pylint: disable=E1101
        revised_target_1d = torch.where(
            mask_filtering_loss,
            target_1d,
            default_target_value)
        DebuggingHelper.write_line_to_system_console_out(
            f'PytorchUtility.evaluate_loss_per_instance():'
            f' output_viewed_in_number_labels.size()={output_viewed_in_number_labels.size()}')
        DebuggingHelper.write_line_to_system_console_out(
            f'PytorchUtility.evaluate_loss_per_instance():'
            f' revised_target_1d.size()={revised_target_1d.size()}')
        loss: torch.tensor = PytorchUtility.TORCH_LOSS_FUNCTION_CrossEntropyLoss( \
            output_viewed_in_number_labels, \
            revised_target_1d)
        return loss

    # ------------------------------------------------------------------------
    # ---- NOTE-FOR-REFERENCE-SOME-PYTORCH-TRANSFORMERS-NEURAL-NETWORKS ----
    # ------------------------------------------------------------------------

    class TorchNeuralNetworkModuleEntityBertPerTokenClassificationModel(torch.nn.Module):
        """
        TorchNeuralNetworkModuleEntityBertPerTokenClassificationModel
        """
        def __init__(self, \
            number_entity_per_token_labels: int, \
            component_module_base_transformers_model_bert: BertModel, \
            component_module_base_transformers_parameter_embedding_dimension: int, \
            component_module_dropout_parameter_rate: float):
            """
            Initialize
            """
            super( \
                PytorchUtility.TorchNeuralNetworkModuleEntityBertPerTokenClassificationModel, \
                self).__init__()
            self.number_entity_per_token_labels = \
                number_entity_per_token_labels
            self.component_module_base_transformers_model_bert = \
                component_module_base_transformers_model_bert
            self.component_module_base_transformers_parameter_embedding_dimension = \
                component_module_base_transformers_parameter_embedding_dimension
            self.component_module_dropout_parameter_rate = \
                component_module_dropout_parameter_rate
            self.component_module_dropout = torch.nn.Dropout( \
                self.component_module_dropout_parameter_rate)
            self.component_module_output_linear_entity_per_token_labels = torch.nn.Linear( \
                self.component_module_base_transformers_parameter_embedding_dimension, \
                self.number_entity_per_token_labels)

        def forward( \
            self, \
            input_ids, \
            attention_mask, \
            token_type_ids, \
            target_entity_per_token_label_indexes) -> torch.tensor:
            # ---- NOTE-PYLINT ---- C0301: Line too long
            # pylint: disable=C0301
            """
            forward()
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                "target_entity_per_token_label_indexes": torch.tensor(target_entity_per_token_label_indexes, dtype=torch.long)
            }
            """
            # DebuggingHelper.write_line_to_system_console_out(
            #     f'TorchNeuralNetworkModuleEntityBertPerTokenClassificationModel.forward():'
            #     f' input_ids={input_ids}')
            DebuggingHelper.write_line_to_system_console_out(
                f'TorchNeuralNetworkModuleEntityBertPerTokenClassificationModel.forward():'
                f' input_ids.size()={input_ids.size()}')
            # DebuggingHelper.write_line_to_system_console_out(
            #     f'TorchNeuralNetworkModuleEntityBertPerTokenClassificationModel.forward():'
            #     f' attention_mask={attention_mask}')
            DebuggingHelper.write_line_to_system_console_out(
                f'TorchNeuralNetworkModuleEntityBertPerTokenClassificationModel.forward():'
                f' attention_mask.size()={attention_mask.size()}')
            # DebuggingHelper.write_line_to_system_console_out(
            #     f'TorchNeuralNetworkModuleEntityBertPerTokenClassificationModel.forward():'
            #     f' token_type_ids={token_type_ids}')
            DebuggingHelper.write_line_to_system_console_out(
                f'TorchNeuralNetworkModuleEntityBertPerTokenClassificationModel.forward():'
                f' token_type_ids.size()={token_type_ids.size()}')
            output = self.component_module_base_transformers_model_bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)
            # DebuggingHelper.write_line_to_system_console_out(
            #     f'TorchNeuralNetworkModuleEntityBertPerTokenClassificationModel:'
            #     f' output={output}')
            # DebuggingHelper.write_line_to_system_console_out(
            #     f'TorchNeuralNetworkModuleEntityBertPerTokenClassificationModel:'
            #     f' output.last_hidden_state.size()={output.last_hidden_state.size()}')
            DebuggingHelper.write_line_to_system_console_out(
                f'TorchNeuralNetworkModuleEntityBertPerTokenClassificationModel:'
                f' output.last_hidden_state.size()={output.last_hidden_state.size()}')
            output_dropout = \
                self.component_module_dropout(output.last_hidden_state)
            # DebuggingHelper.write_line_to_system_console_out(
            #     f'TorchNeuralNetworkModuleEntityBertPerTokenClassificationModel:'
            #     f' output_dropout={output_dropout}')
            DebuggingHelper.write_line_to_system_console_out(
                f'TorchNeuralNetworkModuleEntityBertPerTokenClassificationModel:'
                f' output_dropout.size()={output_dropout.size()}')
            # DebuggingHelper.write_line_to_system_console_out(
            #     f'TorchNeuralNetworkModuleEntityBertPerTokenClassificationModel:'
            #     f' output_dropout.size()={output_dropout.size()}')
            entity_per_token_labels = self.component_module_output_linear_entity_per_token_labels( \
                output_dropout)
            # DebuggingHelper.write_line_to_system_console_out(
            #     f'TorchNeuralNetworkModuleEntityBertPerTokenClassificationModel:'
            #     f' entity_per_token_labels={entity_per_token_labels}')
            DebuggingHelper.write_line_to_system_console_out(
                f'TorchNeuralNetworkModuleEntityBertPerTokenClassificationModel:'
                f' entity_per_token_labels.size()={entity_per_token_labels.size()}')
            # DebuggingHelper.write_line_to_system_console_out(
            #     f'TorchNeuralNetworkModuleEntityBertPerTokenClassificationModel:'
            #     f' target_entity_per_token_label_indexes={target_entity_per_token_label_indexes}')
            DebuggingHelper.write_line_to_system_console_out(
                f'TorchNeuralNetworkModuleEntityBertPerTokenClassificationModel:'
                f' target_entity_per_token_label_indexes.size()={target_entity_per_token_label_indexes.size()}')
            loss_entity_per_token_labels: torch.tensor = \
                PytorchUtility.evaluate_loss_per_instance( \
                    entity_per_token_labels, \
                    target_entity_per_token_label_indexes, \
                    attention_mask, \
                    self.number_entity_per_token_labels)
            loss = loss_entity_per_token_labels
            return entity_per_token_labels, loss

    # ------------------------------------------------------------------------
    # ---- NOTE-FOR-REFERENCE-SOME-PYTORCH-NEURAL-NETWORKS ----
    # ------------------------------------------------------------------------

    class TorchNeuralNetworkModuleLinear(BaseMangedNeuralNetworkModule):
        """
        TorchNeuralNetworkModuleLinear
        """
        def __init__(self, \
            linear_layer_number_input_linear_features: int, \
            linear_layer_number_output_linear_features: int, \
            has_bias: bool = True):
            super(PytorchUtility.TorchNeuralNetworkModuleLinear, self).__init__()
            self.torch_nn_linear = torch.nn.Linear( \
                linear_layer_number_input_linear_features, \
                linear_layer_number_output_linear_features, \
                has_bias)

        def forward(self, input_for_forward: torch.Tensor) -> torch.Tensor:
            """
            forward the neural network

            > ---- R-code example:
                logSoftmax <- function(input) {
                    inputExp <- exp(input)
                    inputExpSum <- sum(inputExp)
                    inputSoftmax <- inputExp / inputExpSum
                    inputLogSoftmax <- log(inputSoftmax)
                    return (inputLogSoftmax)
                }
                weights <- matrix(c(-0.4470, -0.4501, -0.0323,  0.3757,
                                    0.0561,  0.2984,  0.4758, -0.2518,
                                    -0.3531, -0.0655,  0.1988,  0.3883), ncol = 4, byrow = TRUE)
                biases <- matrix(rep(c(-0.2362, -0.2342, -0.3625), 10), ncol = 3, byrow = TRUE)
                inputs <- matrix(c(5.1000, 3.5000, 1.4000, 0.2000,
                                  4.9000, 3.0000, 1.4000, 0.2000,
                                  4.7000, 3.2000, 1.3000, 0.2000,
                                  4.6000, 3.1000, 1.5000, 0.2000,
                                  5.0000, 3.6000, 1.4000, 0.2000,
                                  5.4000, 3.9000, 1.7000, 0.4000,
                                  4.6000, 3.4000, 1.4000, 0.3000,
                                  5.0000, 3.4000, 1.5000, 0.2000,
                                  4.4000, 2.9000, 1.4000, 0.2000,
                                  4.9000, 3.1000, 1.5000, 0.1000), ncol = 4, byrow = TRUE)
                weights
                biases
                inputs
                t(weights)
                multiplied <- inputs %*% t(weights)
                multiplied
                forwarded <- multiplied + biases
                forwarded
                logSoftmaxOnForwarded <- t(apply(forwarded, 1, logSoftmax))
                logSoftmaxOnForwarded
            > ---- R-code example results:
                >                 weights
                        [,1]    [,2]    [,3]    [,4]
                [1,] -0.4470 -0.4501 -0.0323  0.3757
                [2,]  0.0561  0.2984  0.4758 -0.2518
                [3,] -0.3531 -0.0655  0.1988  0.3883
                >                 biases
                         [,1]    [,2]    [,3]
                 [1,] -0.2362 -0.2342 -0.3625
                 [2,] -0.2362 -0.2342 -0.3625
                 [3,] -0.2362 -0.2342 -0.3625
                 [4,] -0.2362 -0.2342 -0.3625
                 [5,] -0.2362 -0.2342 -0.3625
                 [6,] -0.2362 -0.2342 -0.3625
                 [7,] -0.2362 -0.2342 -0.3625
                 [8,] -0.2362 -0.2342 -0.3625
                 [9,] -0.2362 -0.2342 -0.3625
                [10,] -0.2362 -0.2342 -0.3625
                >                 inputs
                      [,1] [,2] [,3] [,4]
                 [1,]  5.1  3.5  1.4  0.2
                 [2,]  4.9  3.0  1.4  0.2
                 [3,]  4.7  3.2  1.3  0.2
                 [4,]  4.6  3.1  1.5  0.2
                 [5,]  5.0  3.6  1.4  0.2
                 [6,]  5.4  3.9  1.7  0.4
                 [7,]  4.6  3.4  1.4  0.3
                 [8,]  5.0  3.4  1.5  0.2
                 [9,]  4.4  2.9  1.4  0.2
                [10,]  4.9  3.1  1.5  0.1
                >                 t(weights)
                        [,1]    [,2]    [,3]
                [1,] -0.4470  0.0561 -0.3531
                [2,] -0.4501  0.2984 -0.0655
                [3,] -0.0323  0.4758  0.1988
                [4,]  0.3757 -0.2518  0.3883
                >                 multiplied <- inputs %*% t(weights)
                >                 multiplied
                          [,1]    [,2]     [,3]
                 [1,] -3.82513 1.94627 -1.67408
                 [2,] -3.51068 1.78585 -1.57071
                 [3,] -3.50807 1.78673 -1.53307
                 [4,] -3.42482 1.84644 -1.45145
                 [5,] -3.82544 1.97050 -1.64532
                 [6,] -4.07382 2.17484 -1.66891
                 [7,] -3.51905 1.86320 -1.45215
                 [8,] -3.73865 1.95840 -1.61234
                 [9,] -3.24217 1.72796 -1.38761
                [10,] -3.59649 1.88845 -1.59621
                >                 forwarded <- multiplied + biases
                >                 forwarded
                          [,1]    [,2]     [,3]
                 [1,] -4.06133 1.71207 -2.03658
                 [2,] -3.74688 1.55165 -1.93321
                 [3,] -3.74427 1.55253 -1.89557
                 [4,] -3.66102 1.61224 -1.81395
                 [5,] -4.06164 1.73630 -2.00782
                 [6,] -4.31002 1.94064 -2.03141
                 [7,] -3.75525 1.62900 -1.81465
                 [8,] -3.97485 1.72420 -1.97484
                 [9,] -3.47837 1.49376 -1.75011
                [10,] -3.83269 1.65425 -1.95871
                >                 logSoftmaxOnForwarded <- t(apply(forwarded, 1, logSoftmax))
                >                 logSoftmaxOnForwarded
                           [,1]        [,2]      [,3]
                 [1,] -5.799710 -0.02630953 -3.774960
                 [2,] -5.333566 -0.03503600 -3.519896
                 [3,] -5.332952 -0.03615217 -3.484252
                 [4,] -5.310206 -0.03694644 -3.463136
                 [5,] -5.824280 -0.02634026 -3.770460
                 [6,] -6.271211 -0.02055133 -3.992601
                 [7,] -5.420135 -0.03588452 -3.479535
                 [8,] -5.726759 -0.02770895 -3.726749
                 [9,] -5.017047 -0.04491696 -3.288787
                [10,] -5.517578 -0.03063821 -3.643598
            """
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinear: input_for_forward={input_for_forward}')
            # ---- DebuggingHelper.write_line_to_system_console_out( \
            # ----     f'TorchNeuralNetworkModuleLinear: self.torch_nn_linear.parameters()='
            # ----     f'{list(self.torch_nn_linear.parameters())}')
            forwarded = self.torch_nn_linear.forward(input_for_forward)
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinear: forwarded-0={forwarded}')
            # ---- NOTE-PYLINT ---- E1101: Module 'torch' has no '*' member (no-member)
            # pylint: disable=E1101
            forwarded = torch.log_softmax(forwarded, 1)
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinear: forwarded-1={forwarded}')
            return forwarded

        def prepare_batch_for_feature_forward(self, batch: any) -> any:
            """
            Prepare an input batch during model training process.
            """
            # ---- NOTE-PYLINT ---- E1101: Module 'torch' has no '*' member (no-member)
            # pylint: disable=E1101
            return torch.FloatTensor([x[1] for x in batch])
        def prepare_batch_for_label_loss_evaluation(self, batch: any) -> any:
            """
            Prepare an input batch during model training process.
            """
            # ---- NOTE-PYLINT ---- E1101: Module 'torch' has no '*' member (no-member)
            # pylint: disable=E1101
            return torch.LongTensor([x[0] for x in batch])
        def prepare_batch_for_label_prediction_evaluation(self, batch: any) -> any:
            """
            Prepare an input batch during model predicting process.
            """
            return [x[0] for x in batch]

    class TorchNeuralNetworkModuleLinearEmbedding(TorchNeuralNetworkModuleLinear):
        """
        TorchNeuralNetworkModuleLinearEmbedding contains an embedding input and a linear layer,
        then a log softmax output.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # ---- NOTE-PYLINT ---- R0913: Too many arguments (*/5) (too-many-arguments)
        # pylint: disable=R0913
        # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ---- featurizer: BaseFeatureManager,
        def __init__(self, \
            sequence_max_length: int, \
            sequence_unknown_element_id: int, \
            embedding_layer_number_embeddings: int, \
            embedding_layer_dimension: int, \
            linear_layer_number_input_linear_features: int, \
            linear_layer_number_output_linear_features: int, \
            has_bias: bool = True):
            super(PytorchUtility.TorchNeuralNetworkModuleLinearEmbedding, self).__init__( \
                linear_layer_number_input_linear_features=embedding_layer_dimension*sequence_max_length, \
                linear_layer_number_output_linear_features=linear_layer_number_output_linear_features, \
                has_bias=has_bias)
            self.sequence_max_length = sequence_max_length
            self.sequence_unknown_element_id = sequence_unknown_element_id
            self.torch_nn_embedding = torch.nn.Embedding( \
                embedding_layer_number_embeddings, \
                embedding_layer_dimension)
            # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ---- self.featurizer = featurizer

        def forward(self, input_for_forward) -> torch.Tensor:
            """
            forward the neural network
            """
            # ----------------------------------------------------------------
            number_inputs: int = len(input_for_forward)
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinearEmbedding: number_inputs={number_inputs}')
            # ----------------------------------------------------------------
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinearEmbedding: self.torch_nn_linear.parameters()={list(self.torch_nn_linear.parameters())}')
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinearEmbedding: input_for_forward={input_for_forward}')
            # ----------------------------------------------------------------
            feature_ids_array: List[List[int]] = \
                [self.normalize_input_ids_padding_or_truncating(x) for x in input_for_forward]
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinearEmbedding: feature_ids_array={feature_ids_array}')
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinearEmbedding: len(feature_ids_array)={len(feature_ids_array)}')
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinearEmbedding: [len(x) for x in feature_ids_array]={[len(x) for x in feature_ids_array]}')
            # ----------------------------------------------------------------
            # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ---- feature_and_ids_array: List[Tuple[List[str], List[int]]] = \
            # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ----     [self.featurizer.create_feature_and_ids(x) for x in input_for_forward]
            # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ---- # ----  DebuggingHelper.write_line_to_system_console_out(
            # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ---- # ----      f'TorchNeuralNetworkModuleLinearEmbedding: feature_and_ids_array={feature_and_ids_array}')
            # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ---- # ----  DebuggingHelper.write_line_to_system_console_out(
            # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ---- # ----      f'TorchNeuralNetworkModuleLinearEmbedding: [len(x) for x in feature_and_ids_array]={[len(x) for x in feature_and_ids_array]}')
            # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ---- feature_ids_array: List[List[int]] = \
            # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ----     [self.normalize_input_ids_padding_or_truncating(x[1]) for x in feature_and_ids_array]
            # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ---- # ---- DebuggingHelper.write_line_to_system_console_out(
            # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ---- # ----     f'TorchNeuralNetworkModuleLinearEmbedding: feature_ids_array={feature_ids_array}')
            # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ---- # ---- DebuggingHelper.write_line_to_system_console_out(
            # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ---- # ----     f'TorchNeuralNetworkModuleLinearEmbedding: len(feature_ids_array)={len(feature_ids_array)}')
            # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ---- # ---- DebuggingHelper.write_line_to_system_console_out(
            # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-2-works ---- # ----     f'TorchNeuralNetworkModuleLinearEmbedding: [len(x) for x in feature_ids_array]={[len(x) for x in feature_ids_array]}')
            # ----------------------------------------------------------------
            # ---- NOTE-PYLINT ---- E1101: Module 'torch' has no '*' member (no-member)
            # pylint: disable=E1101
            feature_ids_tensors: List[torch.LongTensor] = \
                [torch.LongTensor(x) for x in feature_ids_array]
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinearEmbedding: feature_ids_tensors={feature_ids_tensors}')
            # ----------------------------------------------------------------
            forwarded = [self.torch_nn_embedding.forward(y) for y in feature_ids_tensors]
            # ---- NOTE-SINGLE-ENTRY ---- forwarded = self.torch_nn_embedding.forward(feature_ids_tensors[0])
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinearEmbedding: forwarded-0={forwarded}')
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinearEmbedding: len(forwarded)-0={len(forwarded)}')
            forwarded = [y.view(-1) for y in forwarded]
            # ---- NOTE-SINGLE-ENTRY ---- forwarded = forwarded.view(-1)
            # ----------------------------------------------------------------
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinearEmbedding: forwarded-1={forwarded}')
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinearEmbedding: len(forwarded)-1={len(forwarded)}')
            forwarded = [self.torch_nn_linear.forward(y) for y in forwarded]
            # ---- NOTE-SINGLE-ENTRY ---- forwarded = self.torch_nn_linear.forward(forwarded)
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinearEmbedding: forwarded-2={forwarded}')
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinearEmbedding: len(forwarded)-2={len(forwarded)}')
            # ---- NOTE-PYLINT ---- E1101: Module 'torch' has no '*' member (no-member)
            # pylint: disable=E1101
            forwarded = [torch.log_softmax(y, 0) for y in forwarded]
            # ---- NOTE-SINGLE-ENTRY ---- forwarded = torch.log_softmax(forwarded, 0)
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinearEmbedding: forwarded-3={forwarded}')
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinearEmbedding: len(forwarded)-3={len(forwarded)}')
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinearEmbedding: forwarded.dim()-3={forwarded.dim()}')
            # ---- NOTE-PYLINT ---- E1101: Module 'torch' has no '*' member (no-member)
            # pylint: disable=E1101
            forwarded = torch.cat(forwarded)
            forwarded = forwarded.view(number_inputs, -1)
            # ---- NOTE-SINGLE-ENTRY ---- forwarded = forwarded.unsqueeze(0)
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinearEmbedding: forwarded-4={forwarded}')
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinearEmbedding: len(forwarded)-4={len(forwarded)}')
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinearEmbedding: forwarded.dim()-4={forwarded.dim()}')
            # ----------------------------------------------------------------
            return forwarded

        def prepare_batch_for_feature_forward(self, batch: any) -> any:
            """
            Prepare an input batch during model training process.
            """
            return [x[1] for x in batch]
        def prepare_batch_for_label_loss_evaluation(self, batch: any) -> any:
            """
            Prepare an input batch during model training process.
            """
            # ---- NOTE-PYLINT ---- E1101: Module 'torch' has no '*' member (no-member)
            # pylint: disable=E1101
            return torch.LongTensor([x[0] for x in batch])
        def prepare_batch_for_label_prediction_evaluation(self, batch: any) -> any:
            """
            Prepare an input batch during model predicting process.
            """
            return [x[0] for x in batch]

        def normalize_input_ids_padding_or_truncating(self, input_ids: List[int]) -> List[int]:
            """
            Normalize an input sequence to a fixed length.
            """
            length_input_ids: int = len(input_ids)
            difference_length_input_ids_to_sequence_max_length = self.sequence_max_length - length_input_ids
            if difference_length_input_ids_to_sequence_max_length < 0:
                return input_ids[:self.sequence_max_length]
            if difference_length_input_ids_to_sequence_max_length > 0:
                return input_ids + ([self.sequence_unknown_element_id] * difference_length_input_ids_to_sequence_max_length)
            return input_ids

    class TorchNeuralNetworkModuleLinearEmbeddingRaw(TorchNeuralNetworkModuleLinearEmbedding):
        """
        TorchNeuralNetworkModuleLinearEmbeddingRaw uses raw input and
        contains an embedding input and a linear layer,
        then a log softmax output.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # ---- NOTE-PYLINT ---- R0913: Too many arguments (*/5) (too-many-arguments)
        # pylint: disable=R0913
        def __init__(self, \
            featurizer: BaseFeatureManager, \
            sequence_max_length: int, \
            sequence_unknown_element_id: int, \
            embedding_layer_number_embeddings: int, \
            embedding_layer_dimension: int, \
            linear_layer_number_input_linear_features: int, \
            linear_layer_number_output_linear_features: int, \
            has_bias: bool = True):
            super(PytorchUtility.TorchNeuralNetworkModuleLinearEmbeddingRaw, self).__init__( \
                sequence_max_length=sequence_max_length, \
                sequence_unknown_element_id=sequence_unknown_element_id, \
                embedding_layer_number_embeddings=embedding_layer_number_embeddings, \
                embedding_layer_dimension=embedding_layer_dimension, \
                linear_layer_number_input_linear_features=linear_layer_number_input_linear_features, \
                linear_layer_number_output_linear_features=linear_layer_number_output_linear_features, \
                has_bias=has_bias)
            self.featurizer = featurizer

        def forward(self, input_for_forward) -> torch.Tensor:
            """
            forward the neural network
            """
            # ----------------------------------------------------------------
            number_inputs: int = len(input_for_forward)
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinearEmbeddingRaw: number_inputs={number_inputs}')
            # ----------------------------------------------------------------
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinearEmbeddingRaw: self.torch_nn_linear.parameters()={list(self.torch_nn_linear.parameters())}')
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinearEmbeddingRaw: input_for_forward={input_for_forward}')
            # ----------------------------------------------------------------
            # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-3-works ---- feature_ids_array: List[List[int]] = \
            # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-3-works ----     [self.normalize_input_ids_padding_or_truncating(x) for x in input_for_forward]
            # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-3-works ---- # ---- DebuggingHelper.write_line_to_system_console_out(
            # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-3-works ---- # ----     f'TorchNeuralNetworkModuleLinearEmbeddingRaw: feature_ids_array={feature_ids_array}')
            # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-3-works ---- # ---- DebuggingHelper.write_line_to_system_console_out(
            # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-3-works ---- # ----     f'TorchNeuralNetworkModuleLinearEmbeddingRaw: len(feature_ids_array)={len(feature_ids_array)}')
            # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-3-works ---- # ---- DebuggingHelper.write_line_to_system_console_out(
            # ---- NOTE-sparse-Embedding-tensors-fail-Pytorch-collation-test-3-works ---- # ----     f'TorchNeuralNetworkModuleLinearEmbeddingRaw: [len(x) for x in feature_ids_array]={[len(x) for x in feature_ids_array]}')
            # ----------------------------------------------------------------
            feature_and_ids_array: List[Tuple[List[str], List[int]]] = \
                [self.featurizer.create_feature_and_ids(x) for x in input_for_forward]
            # ----  DebuggingHelper.write_line_to_system_console_out(
            # ----      f'TorchNeuralNetworkModuleLinearEmbeddingRaw: feature_and_ids_array={feature_and_ids_array}')
            # ----  DebuggingHelper.write_line_to_system_console_out(
            # ----      f'TorchNeuralNetworkModuleLinearEmbeddingRaw: [len(x) for x in feature_and_ids_array]={[len(x) for x in feature_and_ids_array]}')
            feature_ids_array: List[List[int]] = \
                [self.normalize_input_ids_padding_or_truncating(x[1]) for x in feature_and_ids_array]
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinearEmbeddingRaw: feature_ids_array={feature_ids_array}')
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinearEmbeddingRaw: len(feature_ids_array)={len(feature_ids_array)}')
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinearEmbeddingRaw: [len(x) for x in feature_ids_array]={[len(x) for x in feature_ids_array]}')
            # ----------------------------------------------------------------
            # ---- NOTE-PYLINT ---- E1101: Module 'torch' has no '*' member (no-member)
            # pylint: disable=E1101
            feature_ids_tensors: List[torch.LongTensor] = \
                [torch.LongTensor(x) for x in feature_ids_array]
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinearEmbeddingRaw: feature_ids_tensors={feature_ids_tensors}')
            # ----------------------------------------------------------------
            forwarded = [self.torch_nn_embedding.forward(y) for y in feature_ids_tensors]
            # ---- NOTE-SINGLE-ENTRY ---- forwarded = self.torch_nn_embedding.forward(feature_ids_tensors[0])
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinearEmbeddingRaw: forwarded-0={forwarded}')
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinearEmbeddingRaw: len(forwarded)-0={len(forwarded)}')
            forwarded = [y.view(-1) for y in forwarded]
            # ---- NOTE-SINGLE-ENTRY ---- forwarded = forwarded.view(-1)
            # ----------------------------------------------------------------
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinearEmbeddingRaw: forwarded-1={forwarded}')
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinearEmbeddingRaw: len(forwarded)-1={len(forwarded)}')
            forwarded = [self.torch_nn_linear.forward(y) for y in forwarded]
            # ---- NOTE-SINGLE-ENTRY ---- forwarded = self.torch_nn_linear.forward(forwarded)
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinearEmbeddingRaw: forwarded-2={forwarded}')
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinearEmbeddingRaw: len(forwarded)-2={len(forwarded)}')
            # ---- NOTE-PYLINT ---- E1101: Module 'torch' has no '*' member (no-member)
            # pylint: disable=E1101
            forwarded = [torch.log_softmax(y, 0) for y in forwarded]
            # ---- NOTE-SINGLE-ENTRY ---- forwarded = torch.log_softmax(forwarded, 0)
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinearEmbeddingRaw: forwarded-3={forwarded}')
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinearEmbeddingRaw: len(forwarded)-3={len(forwarded)}')
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinearEmbeddingRaw: forwarded.dim()-3={forwarded.dim()}')
            # ---- NOTE-PYLINT ---- E1101: Module 'torch' has no '*' member (no-member)
            # pylint: disable=E1101
            forwarded = torch.cat(forwarded)
            forwarded = forwarded.view(number_inputs, -1)
            # ---- NOTE-SINGLE-ENTRY ---- forwarded = forwarded.unsqueeze(0)
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinearEmbeddingRaw: forwarded-4={forwarded}')
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinearEmbeddingRaw: len(forwarded)-4={len(forwarded)}')
            # ---- DebuggingHelper.write_line_to_system_console_out(
            # ----     f'TorchNeuralNetworkModuleLinearEmbeddingRaw: forwarded.dim()-4={forwarded.dim()}')
            # ----------------------------------------------------------------
            return forwarded

        def prepare_batch_for_feature_forward(self, batch: any) -> any:
            """
            Prepare an input batch during model training process.
            """
            return batch[1]
        def prepare_batch_for_label_loss_evaluation(self, batch: any) -> any:
            """
            Prepare an input batch during model training process.
            """
            return batch[0].view(-1)
        def prepare_batch_for_label_prediction_evaluation(self, batch: any) -> any:
            """
            Prepare an input batch during model predicting process.
            """
            return self.prepare_batch_for_label_loss_evaluation(batch).tolist()

    # ------------------------------------------------------------------------
    # ---- NOTE-TODO-TEST-THE-FUNCTION-BELOW ----
    # ------------------------------------------------------------------------

    @staticmethod
    def save_model( \
        model: BaseMangedNeuralNetworkModule, \
        output_torch_model_filename: str) -> NoReturn:
        """
        Save a model.
        """
        IoHelper.make_parent_dirs(output_torch_model_filename)
        torch.save(model, output_torch_model_filename)

    @staticmethod
    def save_model_in_string( \
        model: BaseMangedNeuralNetworkModule, \
        output_torch_model_filename: str) -> NoReturn:
        """
        Save a model.
        """
        pytorch_model_in_string: str = PytorchUtility.dump_model_to_string(model)
        IoHelper.write_string(output_torch_model_filename, pytorch_model_in_string)
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # ---- NOTE-FOR-REFERENCE ----
        # ---- NOTE-TODO: SerializeToString() does not work ---- pytorch_model_in_string: str = pytorch_model.SerializeToString()
        # ---- NOTE-TODO: SerializeToString() does not work ---- IoHelper.write_bytes(output_torch_model_json_filename, pytorch_model_in_string)
        # ---- NOTE-TODO: AttributeError: 'ModelProto' object has no attribute '__dict__' ---- json.dump(
        # ---- NOTE-TODO: AttributeError: 'ModelProto' object has no attribute '__dict__' ----     pytorch_model.__dict__,
        # ---- NOTE-TODO: AttributeError: 'ModelProto' object has no attribute '__dict__' ----     IoHelper.codecs_open_file(
        # ---- NOTE-TODO: AttributeError: 'ModelProto' object has no attribute '__dict__' ----         filename=output_torch_model_json_filename,
        # ---- NOTE-TODO: AttributeError: 'ModelProto' object has no attribute '__dict__' ----         mode='w',
        # ---- NOTE-TODO: AttributeError: 'ModelProto' object has no attribute '__dict__' ----         encoding='utf-8'),
        # ---- NOTE-TODO: AttributeError: 'ModelProto' object has no attribute '__dict__' ----     separators=(',', ':'),
        # ---- NOTE-TODO: AttributeError: 'ModelProto' object has no attribute '__dict__' ----     sort_keys=True,
        # ---- NOTE-TODO: AttributeError: 'ModelProto' object has no attribute '__dict__' ----     indent=4)
        # ---- NOTE-TODO: Object of type 'ModelProto' is not JSON serializable ---- json.dump(
        # ---- NOTE-TODO: Object of type 'ModelProto' is not JSON serializable ----     pytorch_model,
        # ---- NOTE-TODO: Object of type 'ModelProto' is not JSON serializable ----     IoHelper.codecs_open_file(
        # ---- NOTE-TODO: Object of type 'ModelProto' is not JSON serializable ----         filename=output_torch_model_json_filename,
        # ---- NOTE-TODO: Object of type 'ModelProto' is not JSON serializable ----         mode='w',
        # ---- NOTE-TODO: Object of type 'ModelProto' is not JSON serializable ----         encoding='utf-8'),
        # ---- NOTE-TODO: Object of type 'ModelProto' is not JSON serializable ----     separators=(',', ':'),
        # ---- NOTE-TODO: Object of type 'ModelProto' is not JSON serializable ----     sort_keys=True,
        # ---- NOTE-TODO: Object of type 'ModelProto' is not JSON serializable ----     indent=4)
        # ---- NOTE-TODO: Object of type 'ModelProto' is not JSON serializable ----

    @staticmethod
    def dump_model_to_string( \
        model: BaseMangedNeuralNetworkModule) -> str:
        """
        Save a model.
        """
        model_in_string: str = f'{model}'
        DebuggingHelper.write_line_to_system_console_out(
            f'model=\n{model}')
        DebuggingHelper.write_line_to_system_console_out(
            f'model_in_string=\n{model_in_string}')
        return model_in_string

    @staticmethod
    def load_model(input_pytorch_model_filename: str):
        """
        load_model()
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        model: Any = torch.load(input_pytorch_model_filename)
        # ---- NOTE-REFERENCE ---- https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_models_for_inference.html#:~:text=Saving%20and%20loading%20models%20for%20inference%20in%20PyTorch.,second%20is%20saving%20and%20loading%20the%20entire%20model.
        # ---- NOTE-REFERENCE ---- must call model.eval() to
        # ---- NOTE-REFERENCE ---- set dropout and batch normalization layers to
        # ---- NOTE-REFERENCE ---- evaluation mode before running inference.
        # ---- NOTE-REFERENCE ---- Failing to do this will yield inconsistent
        # ---- NOTE-REFERENCE ---- inference results.
        model.eval() # ---- AttributeError: 'collections.OrderedDict' object has no attribute 'eval'
        # DebuggingHelper.write_line_to_system_console_out(
        #     f'model={model}')
        return model

    @staticmethod
    def load_model_map_to_location( \
        input_pytorch_model_filename: str, \
        map_location: str = 'cpu'):
        """
        load_model_map_to_location()
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        model: Any = torch.load(input_pytorch_model_filename, map_location=map_location)
        # ---- NOTE-REFERENCE ---- https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_models_for_inference.html#:~:text=Saving%20and%20loading%20models%20for%20inference%20in%20PyTorch.,second%20is%20saving%20and%20loading%20the%20entire%20model.
        # ---- NOTE-REFERENCE ---- must call model.eval() to
        # ---- NOTE-REFERENCE ---- set dropout and batch normalization layers to
        # ---- NOTE-REFERENCE ---- evaluation mode before running inference.
        # ---- NOTE-REFERENCE ---- Failing to do this will yield inconsistent
        # ---- NOTE-REFERENCE ---- inference results.
        # ---- NOTE-TODO --------- model.eval() # ---- AttributeError: 'collections.OrderedDict' object has no attribute 'eval'
        DebuggingHelper.write_line_to_system_console_out(
            f'model={model}')
        DebuggingHelper.write_line_to_system_console_out(
            f'model.__dict__={model.__dict__}')
        return model

    @staticmethod
    def check_torch_model( \
        input_torch_model_path: str):
        """
        Replace space and punctuations from an input string.
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        # ---- NOTE: Preprocessing: load the ONNX model
        pytorch_model = torch.load(input_torch_model_path)
        # DebuggingHelper.write_line_to_system_console_out(
        #     f'The model is: {pytorch_model}')
        has_attribute_ir_version: bool = \
            hasattr(pytorch_model, "ir_version")
        DebuggingHelper.write_line_to_system_console_out(
            f'has_attribute_ir_version={has_attribute_ir_version}')
        if has_attribute_ir_version:
            model_ir_version = getattr(pytorch_model, "ir_version")
            DebuggingHelper.write_line_to_system_console_out(
                f'model_ir_version={model_ir_version}')
        has_attribute_graph: bool = \
            hasattr(pytorch_model, "graph")
        DebuggingHelper.write_line_to_system_console_out(
            f'has_attribute_graph={has_attribute_graph}')
        # if has_attribute_graph:
        #     model_graph = getattr(pytorch_model, "graph")
        #     DebuggingHelper.write_line_to_system_console_out(
        #         f'model_graph={model_graph}')
        # ---- Apply shape inference on the model
        # ---- NOTE-TODO-DOES-NOT-WORK ---- inferred_model = shape_inference.infer_shapes(pytorch_model)
        # ---- NOTE-TODO-DOES-NOT-WORK ---- DebuggingHelper.write_line_to_system_console_out(
        # ---- NOTE-TODO-DOES-NOT-WORK ----     f'The inferred_model is: {inferred_model}')
        # ---- NOTE: Check the model
        # ---- NOTE-PYLINT ---- E1101: Module 'torch' has no '*' member (no-member)
        # pylint: disable=E1101
        torch.checker.check_model(pytorch_model)
        # ---- NOTE-ON-DTE-MODELS ---- pytorch.pytorch_cpp2py_export.checker.ValidationError: Your model ir_version is higher than the checker's.
