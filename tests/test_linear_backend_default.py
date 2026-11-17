"""Model defaults must not override explicit backends or incompatible execution."""
from types import SimpleNamespace

import pytest

from nanovllm.config import Config


def make_config(tmp_path, monkeypatch, *, hf_changes=None, **kwargs) -> Config:
    fields = dict(model_type="qwen3", hidden_size=1024, intermediate_size=3072,
                  num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
                  head_dim=128, dtype="bfloat16", max_position_embeddings=40960)
    fields.update(hf_changes or {})
    monkeypatch.setattr("nanovllm.config.AutoConfig.from_pretrained",
                        lambda _: SimpleNamespace(**fields))
    return Config(model=str(tmp_path), **kwargs)


def test_qwen_default_and_explicit_library(tmp_path, monkeypatch):
    default = make_config(tmp_path, monkeypatch)
    assert default.linear_backend == "cuda_decode"
    assert default.pd_mode_incompatibility() == "CUDA linear currently requires mixed execution"
    library = make_config(tmp_path, monkeypatch, linear_backend="torch")
    assert library.linear_backend == "torch"
    assert library.pd_mode_incompatibility() is None


@pytest.mark.parametrize("backend", ["cuda", "cuda_decode", "cuda_fp8", "cuda_decode_auto"])
def test_explicit_backend_preserved(tmp_path, monkeypatch, backend):
    config = make_config(tmp_path, monkeypatch, hf_changes={"dtype": "torch.bfloat16"},
                         linear_backend=backend,
                         cuda_linear_tuning="selectors.json" if backend.endswith("auto") else "")
    assert config.linear_backend == backend


@pytest.mark.parametrize("kwargs", [
    {"tensor_parallel_size": 2}, {"execution_mode": "pd"}, {"pd_separation": True},
    {"quantization": "int4"}, {"quantization": "fp8"},
])
def test_incompatible_default_retains_library(tmp_path, monkeypatch, kwargs):
    config = make_config(tmp_path, monkeypatch, **kwargs)
    assert config.linear_backend == "torch"


@pytest.mark.parametrize("changes", [
    {"model_type": "qwen3_moe"}, {"model_type": "deepseek_v2"},
    {"hidden_size": 2048}, {"num_hidden_layers": 32}, {"head_dim": 64},
    {"dtype": "float32"},
])
def test_other_architectures_and_dtypes_retain_library(tmp_path, monkeypatch, changes):
    assert make_config(tmp_path, monkeypatch, hf_changes=changes).linear_backend == "torch"


def test_explicit_unsupported_backend_still_rejected(tmp_path, monkeypatch):
    with pytest.raises(AssertionError, match="cannot combine with weight quantization"):
        make_config(tmp_path, monkeypatch, linear_backend="cuda_decode", quantization="int4")
    with pytest.raises(AssertionError, match="requires a frozen"):
        make_config(tmp_path, monkeypatch, linear_backend="cuda_decode_auto")
