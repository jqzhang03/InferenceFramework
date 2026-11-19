"""Single-layer SpecForge EAGLE-3 adapter for the Qwen3-0.6B draft checkpoint."""

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file

from nanovllm.layers.eagle import _EagleRMSNorm


DraftKV = tuple[torch.Tensor, torch.Tensor]


class Eagle3Layer(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        h = config["hidden_size"]
        self.heads = config["num_attention_heads"]
        self.kv_heads = config["num_key_value_heads"]
        self.head_dim = config["head_dim"]
        self.vocab_size = config["vocab_size"]
        self.draft_vocab_size = config["draft_vocab_size"]
        intermediate = config["intermediate_size"]
        eps = config["rms_norm_eps"]
        self.fc = nn.Linear(3 * h, h, bias=False)
        self.input_layernorm = _EagleRMSNorm(h, eps)
        self.hidden_norm = _EagleRMSNorm(h, eps)
        self.post_attention_layernorm = _EagleRMSNorm(h, eps)
        self.norm = _EagleRMSNorm(h, eps)
        self.q_proj = nn.Linear(2 * h, self.heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(2 * h, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(2 * h, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.heads * self.head_dim, h, bias=False)
        self.gate_proj = nn.Linear(h, intermediate, bias=False)
        self.up_proj = nn.Linear(h, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, h, bias=False)
        self.lm_head = nn.Linear(h, self.draft_vocab_size, bias=False)
        theta = config.get("rope_parameters", {}).get("rope_theta", config.get("rope_theta", 10000))
        self.rope_theta = theta
        self.register_buffer("target_ids", torch.empty(self.draft_vocab_size, dtype=torch.long), persistent=False)

    @classmethod
    def load(cls, path: str) -> "Eagle3Layer":
        directory = Path(path).expanduser()
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        shape = tuple(config.get(key) for key in (
            "hidden_size", "num_hidden_layers", "vocab_size", "num_attention_heads",
            "num_key_value_heads", "head_dim", "intermediate_size", "draft_vocab_size"))
        if shape != (1024, 1, 151936, 16, 8, 128, 3072, 32000):
            raise ValueError(f"unsupported Qwen3-0.6B EAGLE-3 checkpoint dimensions: {shape}")
        layer = cls(config)
        raw = load_file(directory / "model.safetensors", device="cpu")
        params = {}
        for name, weight in raw.items():
            name = name.removeprefix("model.")
            if name in ("d2t", "t2d", "embed_tokens.weight"):
                continue
            name = name.removeprefix("midlayer.").removeprefix("layers.0.")
            name = name.removeprefix("self_attn.").removeprefix("mlp.")
            params[name] = weight
        layer.load_state_dict(params, strict=True)
        offsets = raw.get("d2t", raw.get("model.d2t"))
        if offsets is None or offsets.shape != (layer.draft_vocab_size,) or offsets.dtype != torch.int64:
            raise ValueError("EAGLE-3 checkpoint needs a valid d2t offset table")
        target_ids = offsets + torch.arange(layer.draft_vocab_size, device="cpu")
        if target_ids.min() < 0 or target_ids.max() >= layer.vocab_size or target_ids.unique().numel() != target_ids.numel():
            raise ValueError("EAGLE-3 checkpoint has invalid target token ids")
        reverse = raw.get("t2d", raw.get("model.t2d"))
        if reverse is not None and (reverse.dtype != torch.bool or reverse.shape != (layer.vocab_size,)
                                    or not torch.equal(reverse.nonzero().flatten(), target_ids.sort().values)):
            raise ValueError("EAGLE-3 d2t and t2d disagree")
        layer.target_ids.copy_(target_ids)
        return layer.eval().requires_grad_(False)

    def _rotate(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        freq = torch.arange(0, self.head_dim, 2, device=x.device, dtype=torch.float32) / self.head_dim
        angles = positions[:, None] * self.rope_theta ** (-freq)
        angles = torch.cat((angles, angles), dim=-1)
        a, b = x.chunk(2, dim=-1)
        rotated = torch.cat((-b, a), dim=-1)
        return x * angles.cos().to(x.dtype) + rotated * angles.sin().to(x.dtype)

    def forward(self, features: torch.Tensor, embeddings: torch.Tensor,
                past: DraftKV | None) -> tuple[torch.Tensor, DraftKV]:
        """Advance one or more consecutive positions, with a separate KV state per branch."""
        if features.size(-1) == 3 * self.fc.out_features:
            features = self.fc(features)
        qkv_input = torch.cat((self.input_layernorm(embeddings), self.hidden_norm(features)), dim=-1)
        b, n, _ = features.shape
        start = 0 if past is None else past[0].size(2)
        positions = torch.arange(start, start + n, device=features.device)

        def split(x: torch.Tensor, heads: int) -> torch.Tensor:
            return x.reshape(b, n, heads, self.head_dim).transpose(1, 2)

        q = self._rotate(split(self.q_proj(qkv_input), self.heads), positions)
        k = self._rotate(split(self.k_proj(qkv_input), self.kv_heads), positions)
        v = split(self.v_proj(qkv_input), self.kv_heads)
        if past is not None:
            k, v = torch.cat((past[0], k), dim=2), torch.cat((past[1], v), dim=2)
        causal = positions[:, None] >= torch.arange(start + n, device=q.device) if start and n > 1 else None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=causal,
                                             is_causal=not start and n > 1, enable_gqa=True)
        hidden = features + self.o_proj(out.transpose(1, 2).flatten(2))
        normalized = self.post_attention_layernorm(hidden)
        hidden = hidden + self.down_proj(F.silu(self.gate_proj(normalized)) * self.up_proj(normalized))
        return hidden, (k, v)

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(features))
