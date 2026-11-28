"""Native W4A16 operators for packed INT4 and FP4 weights."""
from functools import lru_cache
import os
from pathlib import Path
from threading import Lock
from types import ModuleType

import torch


_build_lock = Lock()


@lru_cache
def _load_for_capability(major: int, minor: int) -> ModuleType:
    from torch.utils.cpp_extension import load

    if major < 8:
        raise ValueError("CUDA W4A16 requires SM80 or newer")
    arch = f"{major}{minor}"
    with _build_lock:
        previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        try:
            return load(name=f"nanovllm_cuda_int4_sm{arch}",
                        sources=[str(Path(__file__).with_name("csrc") / "cuda_int4.cu")],
                        extra_cuda_cflags=["-O3", "-lineinfo"])
        finally:
            if previous is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = previous


def load_cuda_int4(device: torch.device | None = None) -> ModuleType:
    return _load_for_capability(*torch.cuda.get_device_capability(device))


def cuda_int4_gemm(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor,
                   extension: ModuleType) -> torch.Tensor:
    """Run the native operator where supported, using the existing Triton fallback."""
    m, k = x.shape
    groups = scales.size(1)
    group = k // groups
    if (x.is_contiguous() and packed.is_contiguous() and scales.is_contiguous()
            and x.dtype in (torch.float16, torch.bfloat16) and scales.dtype == x.dtype
            and group >= 8 and 128 % group == 0 and k % groups == 0
            and x.data_ptr() % 16 == 0 and packed.data_ptr() % 4 == 0):
        out = torch.empty((m, packed.size(0)), dtype=x.dtype, device=x.device)
        extension.gemm_out(x, packed, scales, out)
        return out
    from nanovllm.layers.linear import int4_gemm
    return int4_gemm(x, packed, scales)


def cuda_fp4_gemm(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor,
                  tensor_scale: torch.Tensor | None, group: int) -> torch.Tensor:
    """Multiply BF16/FP16 activations by packed E2M1 weights."""
    x = x.contiguous()
    if x.data_ptr() % 16:
        x = x.clone()
    out = x.new_empty((x.size(0), packed.size(0)))
    extension = load_cuda_int4(x.device)
    extension.fp4_gemm_out(x, packed, scales,
                           scales if tensor_scale is None else tensor_scale, out, group)
    return out


def set_cuda_int4_expert_fusion(model: torch.nn.Module, fusion: str) -> int:
    from nanovllm.layers.moe import ExpertFFN

    if fusion not in ("none", "gate_up", "full"):
        raise ValueError(f"unknown CUDA INT4 expert fusion {fusion!r}")
    count = 0
    for expert in model.modules():
        if not isinstance(expert, ExpertFFN):
            continue
        expert._cuda_int4_fusion = None
        if fusion == "none":
            continue
        linears = (expert.gate_proj, expert.up_proj, expert.down_proj)
        if any(getattr(linear, "_cuda_int4_ext", None) is None or linear.tp_size != 1
               or linear.bias is not None or hasattr(linear, "w_deq") or hasattr(linear, "awq_scale")
               for linear in linears):
            continue
        gate_packed, gate_scales = linears[0].w_int4, linears[0].w_int4_scale
        if gate_scales.dtype not in (torch.float16, torch.bfloat16):
            continue
        for linear in linears:
            packed, scales = linear.w_int4, linear.w_int4_scale
            k = packed.size(1) * 2
            groups = scales.size(1)
            if (not groups or k % groups or k // groups < 8 or 128 % (k // groups)
                    or not packed.is_contiguous() or not scales.is_contiguous()
                    or packed.data_ptr() % 4 or scales.dtype != gate_scales.dtype
                    or packed.device != gate_packed.device or scales.device != gate_packed.device):
                break
        else:
            expert._cuda_int4_fusion = fusion == "full"
            count += 1
    return count


def install_cuda_int4(model: torch.nn.Module, expert_fusion: str = "none") -> int:
    from nanovllm.layers.linear import WeightQuantMixin

    linears = [module for module in model.modules()
               if isinstance(module, WeightQuantMixin) and module.int4]
    extensions = {}
    for module in linears:
        device = module.w_int4.device
        if device not in extensions:
            extensions[device] = load_cuda_int4(device)
        module._cuda_int4_ext = extensions[device]
    if expert_fusion != "none":
        set_cuda_int4_expert_fusion(model, expert_fusion)
    return len(linears)
