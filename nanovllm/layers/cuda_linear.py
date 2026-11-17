"""Opt-in Qwen3 CUDA projections. Packing and compilation precede graph capture."""
from __future__ import annotations
import json
import os
from pathlib import Path
from types import ModuleType
from functools import lru_cache
import torch
from torch.nn import functional as F


def load_extension(fp8: bool = False) -> ModuleType:
    from torch.utils.cpp_extension import load
    capability = torch.cuda.get_device_capability()
    if capability[0] < 8 or (fp8 and capability != (12, 0)):
        raise ValueError("CUDA linear needs SM80+; native FP8 needs SM120")
    os.environ.setdefault("CXX", "/usr/bin/g++-13")
    name = "cuda_fp8_gemm" if fp8 else "cuda_gemm"
    flags = ["-O3", "-lineinfo"]
    old_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a" if fp8 else f"{capability[0]}.{capability[1]}"
    try:
        return load(name="nanovllm_linear_" + name,
                    sources=[str(Path(__file__).parent / "csrc" / (name + ".cu"))],
                    extra_cuda_cflags=flags + ["-ccbin=/usr/bin/g++-13"],
                    extra_ldflags=["-lcublasLt" if fp8 else "-lcublas"])
    finally:
        if old_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = old_arch


@lru_cache(maxsize=1)
def load_decode_extension() -> ModuleType:
    from torch.utils.cpp_extension import load
    capability = torch.cuda.get_device_capability()
    if capability[0] < 8:
        raise ValueError("CUDA decode requires SM80+")
    os.environ.setdefault("CXX", "/usr/bin/g++-13")
    previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{capability[0]}.{capability[1]}"
    try:
        return load(name="nanovllm_cuda_decode", sources=[str(Path(__file__).parent / "csrc/cuda_decode.cu")],
                    extra_cuda_cflags=["-O3", "-lineinfo", "-ccbin=/usr/bin/g++-13"])
    finally:
        if previous is None: os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else: os.environ["TORCH_CUDA_ARCH_LIST"] = previous


class CudaLinear:
    def __init__(self, weight: torch.Tensor, backend: str, ext: ModuleType,
                 fp8_ext: ModuleType | None = None, selectors: dict | None = None,
                 decode_ext: ModuleType | None = None) -> None:
        self.weight = weight.detach()
        self.backend, self.ext, self.fp8_ext = backend, ext, fp8_ext
        self.selectors = selectors or {}
        self.decode_ext = decode_ext
        self.transposed = self.weight.T.contiguous()
        if fp8_ext is not None:
            self.scale = max(weight.abs().max().item() / 448, 1e-12)
            self.packed = (weight / self.scale).clamp(-448, 448).to(torch.float8_e4m3fn)

    def __call__(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if (x.ndim != 2 or not x.is_contiguous() or x.dtype != self.weight.dtype
                or x.data_ptr() % 16 or x.shape[1] % 8):
            return F.linear(x, self.weight, bias)
        m, k = x.shape
        n = self.weight.shape[0]
        choice = self.selectors.get(f"{m},{n},{k}", {})
        if self.backend.endswith("auto") and not choice:
            return F.linear(x, self.weight, bias)
        mode = choice.get("mode", "fp8" if self.fp8_ext is not None and m >= 32 else "gemv" if m <= 4 else "gemm")
        if self.backend.endswith("auto") and mode == "torch":
            return F.linear(x, self.weight, bias)
        out = torch.empty((m, n), device=x.device, dtype=x.dtype)
        if mode == "decode" and self.decode_ext is not None:
            self.decode_ext.decode_out(x, self.weight, out, choice.get("config", 1))
        elif mode == "fp8" and self.fp8_ext is not None and x.dtype == torch.bfloat16:
            a = torch.empty_like(x, dtype=torch.float8_e4m3fn)
            scales = torch.empty(m, device=x.device, dtype=torch.float32)
            self.fp8_ext.pack_rows_out(x, a, scales)
            self.fp8_ext.gemm_rows_out(a, self.packed, out, scales, choice.get("config", 3), self.scale)
        elif mode == "gemv":
            self.ext.linear_gemv_out(x, self.weight, out)
        else:
            self.ext.gemm_out(x, self.transposed, out, choice.get("config", 10))
        return out if bias is None else out + bias


def install_cuda_linears(model: torch.nn.Module, backend: str, tuning_path: str = "") -> int:
    from nanovllm.layers.linear import ColumnParallelLinear, RowParallelLinear
    if backend == "torch":
        return 0
    ext = load_extension()
    fp8_ext = load_extension(True) if backend.startswith("cuda_fp8") else None
    decode_ext = load_decode_extension() if backend.startswith("cuda_decode") else None
    selectors = {}
    if not tuning_path:
        dtype = next(model.parameters()).dtype
        filename = None
        if backend == "cuda_decode":
            filename = {torch.bfloat16: "cuda_decode_sm120.json",
                        torch.float16: "cuda_decode_sm120_fp16.json"}.get(dtype)
        elif backend == "cuda_fp8" and dtype == torch.bfloat16:
            filename = "cuda_fp8_sm120.json"
        path = Path(__file__).with_name(filename) if filename else None
        if path is not None and path.exists():
            bundled = json.loads(path.read_text())
            if bundled["device"] == torch.cuda.get_device_name() and bundled["backend"] == backend:
                selectors = bundled["selectors"]
    if tuning_path:
        report = json.loads(Path(tuning_path).read_text())
        if report["device"] != torch.cuda.get_device_name() or report["backend"] != backend:
            raise ValueError("CUDA linear tuning report device/backend mismatch")
        selectors = report["selectors"]
    count = 0
    for name, module in model.named_modules():
        if ".layers." in name and isinstance(module, (ColumnParallelLinear, RowParallelLinear)):
            if module.weight.dtype not in (torch.float16, torch.bfloat16):
                raise ValueError("CUDA projections require FP16/BF16 weights")
            module._cuda_linear = CudaLinear(module.weight, backend, ext, fp8_ext, selectors, decode_ext)
            count += 1
    return count
