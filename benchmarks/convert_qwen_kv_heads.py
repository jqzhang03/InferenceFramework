"""Derive an experimental Qwen3 checkpoint with a different number of KV heads."""

import argparse
import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def remap_kv_heads(weight: torch.Tensor, source: int, target: int, head_dim: int) -> torch.Tensor:
    if min(source, target, head_dim) <= 0:
        raise ValueError("Head counts and head dimension must be positive")
    if weight.shape[0] != source * head_dim:
        raise ValueError(f"Unexpected KV projection shape {tuple(weight.shape)}")
    heads = weight.reshape(source, head_dim, *weight.shape[1:])
    if target >= source and target % source == 0:
        mapped = heads.repeat_interleave(target // source, dim=0)
    elif source % target == 0:
        groups = heads.reshape(target, source // target, *heads.shape[1:])
        mapped = groups.float().mean(dim=1).to(weight.dtype)
    else:
        raise ValueError("The source and target must partition each other's KV groups")
    return mapped.reshape(target * head_dim, *weight.shape[1:]).contiguous()


def _write_checkpoint(source: Path, output: Path, target: int) -> None:
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3" or config.get("quantization_config"):
        raise ValueError("Expected a floating-point Qwen3 checkpoint")
    queries, original = config["num_attention_heads"], config["num_key_value_heads"]
    dim = config.get("head_dim", config["hidden_size"] // queries)
    if (min(target, original) <= 0 or queries % target or queries % original
            or max(target, original) % min(target, original)):
        raise ValueError("KV heads must divide Q heads and align with existing KV groups")
    shards = sorted(source.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"No safetensors shards in {source}")
    expected = {f"model.layers.{layer}.self_attn.{projection}_proj.weight"
                for layer in range(config["num_hidden_layers"]) for projection in ("k", "v")}
    if config.get("attention_bias"):
        expected |= {name.removesuffix("weight") + "bias" for name in expected}
    found = set()
    total_bytes = 0
    output.mkdir()
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as reader:
            weights = {}
            for name in reader.keys():
                weight = reader.get_tensor(name)
                if name in expected:
                    weight = remap_kv_heads(weight, original, target, dim)
                    found.add(name)
                total_bytes += weight.nbytes
                weights[name] = weight
            save_file(weights, str(output / shard.name), metadata=reader.metadata() or {"format": "pt"})
            del weights
    if found != expected:
        raise ValueError(f"Missing KV projections: {sorted(expected - found)}")
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json",
                 "merges.txt", "tokenizer.model", "generation_config.json", "chat_template.jinja"):
        path = source / name
        if path.is_file():
            copyfile(path, output / name)
    for path in source.glob("*.safetensors.index.json"):
        index = json.loads(path.read_text(encoding="utf-8"))
        index.setdefault("metadata", {})["total_size"] = total_bytes
        (output / path.name).write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    config["num_key_value_heads"] = target
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def convert(source: Path, output: Path, target: int) -> None:
    if output.exists():
        raise FileExistsError(f"Destination already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".qwen-kv-", dir=output.parent) as directory:
        checkpoint = Path(directory) / "checkpoint"
        _write_checkpoint(source, checkpoint, target)
        checkpoint.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kv-heads", type=int, required=True)
    args = parser.parse_args()
    convert(args.model, args.output, args.kv_heads)


if __name__ == "__main__":
    main()
