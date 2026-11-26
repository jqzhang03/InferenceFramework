import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from benchmarks.convert_qwen_kv_heads import convert


@pytest.mark.parametrize("target", [1, 2, 8])
def test_qwen_checkpoint_conversion_preserves_group_order(tmp_path: Path, target: int) -> None:
    source, output = tmp_path / "original", tmp_path / "converted"
    source.mkdir()
    config = dict(model_type="qwen3", num_attention_heads=8, num_key_value_heads=4,
                  num_hidden_layers=1, hidden_size=16, head_dim=2)
    config_path = source / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    prefix = "model.layers.0.self_attn"
    keys = [f"{prefix}.{proj}_proj.weight" for proj in ("q", "k", "v")]
    q = torch.arange(256, dtype=torch.float32).reshape(16, 16).to(torch.bfloat16)
    k = torch.arange(128, dtype=torch.float32).reshape(8, 16).to(torch.bfloat16)
    v = (k.float() * 2 + 1).to(torch.bfloat16)
    weights = dict(zip(keys, (q, k, v)))
    save_file(weights, str(source / "model.safetensors"))

    convert(source, output, target)

    assert json.loads(config_path.read_text(encoding="utf-8")) == config
    assert json.loads((output / "config.json").read_text(encoding="utf-8")) == {
        **config, "num_key_value_heads": target,
    }
    saved = load_file(str(output / "model.safetensors"))
    torch.testing.assert_close(saved[keys[0]], q, rtol=0, atol=0)
    for key, weight in zip(keys[1:], (k, v)):
        if target == 8:
            expected = torch.stack([weight[2 * head:2 * head + 2] for head in (0, 0, 1, 1, 2, 2, 3, 3)])
        else:
            heads = weight.split(2)
            group_size = len(heads) // target
            expected = torch.stack([torch.stack(heads[start:start + group_size]).float().mean(0).to(weight.dtype)
                                    for start in range(0, len(heads), group_size)])
        torch.testing.assert_close(saved[key], expected.reshape(target * 2, 16), rtol=0, atol=0)
