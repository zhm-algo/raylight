import importlib
import sys
import types

import torch

from src.raylight import device_utils


def test_device_utils_prefers_cuda_over_xpu(monkeypatch):
    class FakeXPU:
        @staticmethod
        def is_available():
            return True

    monkeypatch.setattr(device_utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(device_utils.torch, "xpu", FakeXPU(), raising=False)

    assert device_utils.get_device_type() == "cuda"
    assert device_utils.get_visible_devices_env_var() == "CUDA_VISIBLE_DEVICES"


def test_device_utils_uses_xpu_backend_and_visibility_mask(monkeypatch):
    class FakeXPU:
        @staticmethod
        def is_available():
            return True

    monkeypatch.setattr(device_utils.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(device_utils.torch, "xpu", FakeXPU(), raising=False)
    monkeypatch.setattr(device_utils.dist, "is_xccl_available", lambda: True, raising=False)

    assert device_utils.is_xpu_available() is True
    assert device_utils.get_device_type() == "xpu"
    assert device_utils.get_dist_backend() == "xccl"
    assert device_utils.get_visible_devices_env_var() == "ZE_AFFINITY_MASK"


def test_device_utils_falls_back_to_oneccl_for_xpu(monkeypatch):
    class FakeXPU:
        @staticmethod
        def is_available():
            return True

    monkeypatch.setattr(device_utils.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(device_utils.torch, "xpu", FakeXPU(), raising=False)
    monkeypatch.setattr(device_utils.dist, "is_xccl_available", lambda: False, raising=False)
    monkeypatch.setitem(sys.modules, "oneccl_bindings_for_pytorch", types.ModuleType("oneccl_bindings_for_pytorch"))

    assert device_utils.get_dist_backend() == "ccl"


def test_device_utils_falls_back_to_gloo_for_xpu(monkeypatch):
    class FakeXPU:
        @staticmethod
        def is_available():
            return True

    monkeypatch.setattr(device_utils.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(device_utils.torch, "xpu", FakeXPU(), raising=False)
    monkeypatch.setattr(device_utils.dist, "is_xccl_available", lambda: False, raising=False)
    sys.modules.pop("oneccl_bindings_for_pytorch", None)

    assert device_utils.get_dist_backend() == "gloo"


def test_controlnet_device_remap_accepts_xpu():
    module = importlib.import_module("src.raylight.distributed_worker.ray_worker_controlnet")

    class Dummy:
        load_device = torch.device("xpu", 3)

    target = torch.device("xpu", 0)
    obj = Dummy()
    module._remap_accelerator_device(obj, "load_device", target)

    assert obj.load_device == target


def test_torch_compile_falls_back_to_inductor_on_xpu(monkeypatch):
    comfy_api = types.ModuleType("comfy_api")
    comfy_api.__path__ = []
    torch_helpers = types.ModuleType("comfy_api.torch_helpers")
    calls = {}

    def fake_set_torch_compile_wrapper(**kwargs):
        calls.update(kwargs)

    torch_helpers.set_torch_compile_wrapper = fake_set_torch_compile_wrapper
    monkeypatch.setitem(sys.modules, "comfy_api", comfy_api)
    monkeypatch.setitem(sys.modules, "comfy_api.torch_helpers", torch_helpers)
    sys.modules.pop("src.raylight.comfy_extra_dist.nodes_torch_compile", None)

    module = importlib.import_module("src.raylight.comfy_extra_dist.nodes_torch_compile")
    monkeypatch.setattr(module, "get_device_type", lambda: "xpu")

    class Model:
        def clone(self, disable_dynamic=True):
            self.disable_dynamic = disable_dynamic
            return self

    model = Model()
    out = module.RayTorchCompileModel.patch.__wrapped__(module.RayTorchCompileModel(), model, "cudagraphs")

    assert out is model
    assert calls["backend"] == "inductor"
    assert model.disable_dynamic is True


def test_easycache_uses_xpu_sync_device(monkeypatch):
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    model_patcher = types.ModuleType("comfy.model_patcher")
    patcher_extension = types.ModuleType("comfy.patcher_extension")
    patcher_extension.WrappersMP = object
    comfy_extras = types.ModuleType("comfy_extras")
    comfy_extras.__path__ = []
    nodes_easycache = types.ModuleType("comfy_extras.nodes_easycache")

    class EasyCacheHolder:
        def __init__(self, *args, **kwargs):
            pass

    nodes_easycache.EasyCacheHolder = EasyCacheHolder

    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_patcher", model_patcher)
    monkeypatch.setitem(sys.modules, "comfy.patcher_extension", patcher_extension)
    monkeypatch.setitem(sys.modules, "comfy_extras", comfy_extras)
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_easycache", nodes_easycache)
    sys.modules.pop("src.raylight.comfy_extra_dist.nodes_easycache", None)

    module = importlib.import_module("src.raylight.comfy_extra_dist.nodes_easycache")
    monkeypatch.setattr(module.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(module.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(module.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(module.torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(module.torch.distributed, "get_backend", lambda: "xccl")
    monkeypatch.setattr(module, "get_device_type", lambda: "xpu")
    monkeypatch.setattr(module, "current_device_index", lambda: 2)
    monkeypatch.setattr(module, "get_device", lambda index: torch.device("xpu", index))

    mixin = module.DistributedCacheMixin()

    assert mixin._sync_device == torch.device("xpu", 2)
