"""CUDA decode for the DeepSeek-V2-Lite paged latent attention layout."""
from functools import lru_cache
import os
from pathlib import Path
from shutil import which
from threading import Lock
from types import ModuleType

import torch


# SM120 paired timing favors CUDA through 16 rows. At 32 rows and 1024
# cached tokens, Triton leads in the held-out comparison.
CUDA_MLA_AUTO_MAX_ROWS = 16
_build_lock = Lock()


@lru_cache
def _load_for_capability(major: int, minor: int) -> ModuleType:
    from torch.utils.cpp_extension import load

    arch = f"{major}{minor}"
    with _build_lock:
        previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        try:
            return load(name=f"nanovllm_mla_decode_sm{arch}",
                        sources=[str(Path(__file__).with_name("csrc") / "cuda_mla_decode.cu")],
                        extra_cuda_cflags=["-O3", "-lineinfo"])
        finally:
            if previous is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = previous


def load_cuda_mla_decode(device: torch.device, *, required: bool) -> ModuleType | None:
    """Prepare the CUDA kernel before serving or capturing a graph."""
    from torch.utils.cpp_extension import CUDA_HOME

    capability = torch.cuda.get_device_capability(device)
    if not required and capability != (12, 0):
        return None
    if capability[0] < 8:
        raise ValueError("CUDA MLA decode requires SM80 or newer")
    if not CUDA_HOME or not which("nvcc", path=str(Path(CUDA_HOME) / "bin")) or not which("ninja"):
        if required:
            raise RuntimeError("CUDA MLA decode needs nvcc and Ninja")
        return None
    return _load_for_capability(*capability)


def cuda_mla_decode_attention(q_abs: torch.Tensor, q_pe: torch.Tensor,
                             cache: torch.Tensor, pages: torch.Tensor,
                             lengths: torch.Tensor, softmax_scale: float,
                             c_scale: float, r_scale: float,
                             extension: ModuleType, *, variant: str = "auto") -> torch.Tensor:
    query = torch.cat((q_abs, q_pe), dim=-1)
    return extension.decode(query, cache, pages.contiguous(), lengths.contiguous(),
                            softmax_scale, c_scale, r_scale, variant)
