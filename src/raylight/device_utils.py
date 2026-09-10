from __future__ import annotations

import importlib

import torch
import torch.distributed as dist


def is_xpu_available() -> bool:
    xpu = getattr(torch, "xpu", None)
    if xpu is None:
        return False
    try:
        return bool(xpu.is_available())
    except (AttributeError, ModuleNotFoundError, RuntimeError):
        return False


def get_device_type() -> str:
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass

    if is_xpu_available():
        return "xpu"

    return "cpu"


def device_module():
    device_type = get_device_type()
    if device_type == "cuda":
        return torch.cuda
    if device_type == "xpu":
        return getattr(torch, "xpu", None)
    return None


def device_count() -> int:
    module = device_module()
    if module is None or not hasattr(module, "device_count"):
        return 0
    try:
        return int(module.device_count())
    except Exception:
        return 0


def current_device_index(default: int = 0) -> int:
    module = device_module()
    if module is None or not hasattr(module, "current_device"):
        return default
    try:
        return int(module.current_device())
    except Exception:
        return default


def set_device(index: int) -> None:
    module = device_module()
    if module is None or not hasattr(module, "set_device"):
        return
    module.set_device(index)


def get_device(index: int = 0) -> torch.device:
    device_type = get_device_type()
    if device_type == "cpu":
        return torch.device("cpu")
    return torch.device(device_type, index)


def _native_xccl_available() -> bool:
    checker = getattr(dist, "is_xccl_available", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:
            return False

    checker = getattr(dist, "is_backend_available", None)
    if callable(checker):
        try:
            return bool(checker("xccl"))
        except Exception:
            return False

    return False


def _oneccl_available() -> bool:
    try:
        importlib.import_module("oneccl_bindings_for_pytorch")
    except ImportError:
        return False
    return True


def get_dist_backend() -> str:
    device_type = get_device_type()
    if device_type == "cuda":
        return "nccl"
    if device_type == "xpu":
        if _native_xccl_available():
            return "xccl"
        if _oneccl_available():
            return "ccl"
    return "gloo"


def get_visible_devices_env_var() -> str | None:
    device_type = get_device_type()
    if device_type == "cuda":
        return "CUDA_VISIBLE_DEVICES"
    if device_type == "xpu":
        return "ZE_AFFINITY_MASK"
    return None


def empty_cache() -> None:
    module = device_module()
    if module is None or not hasattr(module, "empty_cache"):
        return
    try:
        module.empty_cache()
    except Exception:
        pass


def synchronize() -> None:
    module = device_module()
    if module is None or not hasattr(module, "synchronize"):
        return
    try:
        module.synchronize()
    except Exception:
        pass


def mem_get_info():
    module = device_module()
    if module is None or not hasattr(module, "mem_get_info"):
        return None
    try:
        return module.mem_get_info()
    except Exception:
        return None


def ipc_collect() -> None:
    if get_device_type() != "cuda":
        return
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
