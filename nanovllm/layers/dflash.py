"""DFlash block drafting for full-attention Qwen3 Speculators checkpoints."""

from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from flash_attn import flash_attn_with_kvcache
from safetensors.torch import load_file

from nanovllm.layers.attention import store_kvcache
from nanovllm.layers.rotary_embedding import RotaryEmbedding, apply_rotary_emb


@torch.compile(dynamic=True)
def _draft_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    fp32 = x.float()
    return (fp32 * (fp32.square().mean(dim=-1, keepdim=True) + eps).rsqrt()).to(x.dtype) * weight


@torch.compile(dynamic=True)
def _draft_swiglu(x: torch.Tensor) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up


class DFlashRMSNorm(nn.Module):
    def __init__(self, hidden: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _draft_rms_norm(x.reshape(-1, self.weight.numel()), self.weight, self.eps).view_as(x)


@dataclass(slots=True)
class DFlashCache:
    tensors: torch.Tensor
    num_context_blocks: int

    @property
    def page_size(self) -> int:
        return self.tensors.size(3)


class DFlashDecoderLayer(nn.Module):
    def __init__(self, config: dict, *, causal: bool = False, window_size: tuple[int, int] = (-1, -1)):
        super().__init__()
        self.causal = causal
        self.window_size = window_size
        hidden = config["hidden_size"]
        intermediate = config["intermediate_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_key_value_heads"]
        self.head_dim = config["head_dim"]
        self.q_width = self.num_heads * self.head_dim
        self.kv_width = self.num_kv_heads * self.head_dim
        eps = config["rms_norm_eps"]
        self.input_layernorm = DFlashRMSNorm(hidden, eps=eps)
        self.post_attention_layernorm = DFlashRMSNorm(hidden, eps=eps)
        self.q_norm = DFlashRMSNorm(self.head_dim, eps=eps)
        self.k_norm = DFlashRMSNorm(self.head_dim, eps=eps)
        self.qkv = nn.Linear(hidden, self.q_width + 2 * self.kv_width, bias=False)
        self.o_proj = nn.Linear(self.q_width, hidden, bias=False)
        self.gate_up = nn.Linear(hidden, 2 * intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, hidden: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor],
                k_cache: torch.Tensor, v_cache: torch.Tensor,
                block_table: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch = lengths.numel()
        q, k, v = self.qkv(self.input_layernorm(hidden)).split(
            (self.q_width, self.kv_width, self.kv_width), dim=-1)
        q = apply_rotary_emb(self.q_norm(q.reshape(-1, self.num_heads, self.head_dim)), *rope)
        k = apply_rotary_emb(self.k_norm(k.reshape(-1, self.num_kv_heads, self.head_dim)), *rope)
        v = v.contiguous()
        output = flash_attn_with_kvcache(
            q.view(batch, -1, self.num_heads, self.head_dim), k_cache, v_cache,
            k.view(batch, -1, self.num_kv_heads, self.head_dim),
            v.view(batch, -1, self.num_kv_heads, self.head_dim),
            cache_seqlens=lengths, block_table=block_table, causal=self.causal,
            window_size=self.window_size,
        )
        hidden = hidden + self.o_proj(output.reshape(-1, self.q_width))
        return hidden + self.down(_draft_swiglu(self.gate_up(self.post_attention_layernorm(hidden))))


class DFlashDraft(nn.Module):
    architecture = "DFlashDraftModel"

    @staticmethod
    def _attention_settings(config: dict, layer_type: str) -> tuple[bool, tuple[int, int]]:
        arch = config["transformer_layer_config"]
        if (layer_type != "full_attention" or arch.get("is_causal") or arch.get("sliding_window")
                or config.get("sample_from_anchor") or config.get("shift_label")):
            raise ValueError("unsupported Qwen3-0.6B DFlash attention configuration")
        return False, (-1, -1)

    def __init__(self, config: dict, model_len: int, draft_len: int):
        super().__init__()
        arch = config["transformer_layer_config"]
        self.target_layers = tuple(config["aux_hidden_state_layer_ids"])
        self.block_size = config["block_size"]
        self.mask_token_id = config["mask_token_id"]
        self.num_kv_heads = arch["num_key_value_heads"]
        self.head_dim = arch["head_dim"]
        self.kv_width = self.num_kv_heads * self.head_dim
        hidden = arch["hidden_size"]
        vocab = arch["vocab_size"]
        draft_vocab = config.get("draft_vocab_size")
        if draft_vocab is None:
            draft_vocab = vocab
        self.first_draft_slot = 0 if config.get("sample_from_anchor", config.get("shift_label")) else 1
        max_draft_len = self.block_size - self.first_draft_slot
        if not 1 <= draft_len <= max_draft_len:
            raise ValueError(f"{self.architecture} max_draft_len must be in [1, {max_draft_len}]")
        layer_count = arch["num_hidden_layers"]
        layer_types = arch.get("layer_types") or ["full_attention"] * layer_count
        if (config["architectures"] != [self.architecture]
                or config["speculators_config"]["verifier"]["name_or_path"] != "Qwen/Qwen3-0.6B"
                or self.block_size > 256 or layer_count < 1 or len(layer_types) != layer_count
                or self.target_layers != tuple(sorted(set(self.target_layers)))
                or not self.target_layers or not all(1 <= i <= 27 for i in self.target_layers)
                or arch["model_type"] != "qwen3" or hidden != 1024 or vocab != 151936
                or arch["hidden_act"] != "silu" or arch.get("attention_bias")
                or arch["rope_parameters"].get("rope_type") != "default"
                or config.get("pure_draft_prefix_len", 0) != 0
                or not 0 <= self.mask_token_id < vocab or not 0 < draft_vocab <= vocab):
            raise ValueError(f"unsupported Qwen3-0.6B {self.architecture} Speculators checkpoint")
        self.layers = nn.ModuleList()
        for layer_type in layer_types:
            causal, window_size = self._attention_settings(config, layer_type)
            self.layers.append(DFlashDecoderLayer(arch, causal=causal, window_size=window_size))
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.fc = nn.Linear(len(self.target_layers) * hidden, hidden, bias=False)
        self.hidden_norm = DFlashRMSNorm(hidden, eps=arch["rms_norm_eps"])
        self.context_kv = nn.Linear(hidden, len(self.layers) * 2 * self.kv_width, bias=False)
        self.norm = DFlashRMSNorm(hidden, eps=arch["rms_norm_eps"])
        self.lm_head = nn.Linear(hidden, draft_vocab, bias=False)
        self.register_buffer("target_ids", torch.empty(draft_vocab, dtype=torch.long))
        self.rope = RotaryEmbedding(self.head_dim, self.head_dim, model_len + self.block_size,
                                    arch["rope_parameters"]["rope_theta"])

    @classmethod
    def load(cls, path: str, model_len: int, draft_len: int, *,
             target_embedding: torch.Tensor | None = None,
             target_lm_head: torch.Tensor | None = None) -> "DFlashDraft":
        directory = Path(path).expanduser()
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        with torch.device("meta"):
            model = cls(config, model_len, draft_len)
        state = load_file(directory / "model.safetensors", device="cpu")
        if "embed_tokens.weight" not in state and target_embedding is not None:
            state["embed_tokens.weight"] = target_embedding.detach()
        if ("lm_head.weight" not in state and target_lm_head is not None
                and model.lm_head.out_features == model.embed_tokens.num_embeddings):
            state["lm_head.weight"] = target_lm_head.detach()
        with torch.device("cpu"):
            model.rope.build_cache()
            target_ids = torch.arange(model.lm_head.out_features)
            if model.lm_head.out_features != model.embed_tokens.num_embeddings:
                offsets, selected = state.pop("d2t"), state.pop("t2d")
                target_ids = target_ids + offsets
                if (not torch.equal(target_ids, selected.nonzero().flatten())
                        or selected.numel() != model.embed_tokens.num_embeddings
                        or target_ids.numel() != model.lm_head.out_features):
                    raise ValueError("invalid block draft vocabulary mapping")
        state["target_ids"] = target_ids
        context_weights = []
        for i in range(len(model.layers)):
            prefix = f"layers.{i}."
            q, k, v = (state.pop(f"{prefix}self_attn.{p}_proj.weight") for p in ("q", "k", "v"))
            context_weights.extend((k, v))
            state[f"{prefix}qkv.weight"] = torch.cat((q, k, v))
            state[f"{prefix}gate_up.weight"] = torch.cat(
                [state.pop(f"{prefix}mlp.{p}_proj.weight") for p in ("gate", "up")])
            state[f"{prefix}down.weight"] = state.pop(f"{prefix}mlp.down_proj.weight")
            for name in ("q_norm", "k_norm", "o_proj"):
                state[f"{prefix}{name}.weight"] = state.pop(f"{prefix}self_attn.{name}.weight")
        state["context_kv.weight"] = torch.cat(context_weights)
        model.load_state_dict(state, strict=True, assign=True)
        return model.eval()

    def page_bytes(self, page_size: int, item_size: int) -> int:
        return 2 * len(self.layers) * page_size * self.kv_width * item_size

    def allocate_cache(self, blocks: int, scratch: int, page_size: int) -> DFlashCache:
        return DFlashCache(self.fc.weight.new_empty(
            (len(self.layers), 2, blocks + scratch, page_size, self.num_kv_heads, self.head_dim)),
            blocks)

    def store_context(self, features: torch.Tensor, positions: torch.Tensor,
                      slots: torch.Tensor, cache: DFlashCache) -> None:
        context = self.context_kv(self.hidden_norm(self.fc(features)))
        rope = self.rope.cos_sin_cache[positions].chunk(2, dim=-1)
        for layer, kv, (k_cache, v_cache) in zip(
            self.layers, context.split(2 * self.kv_width, dim=-1), cache.tensors,
        ):
            k, v = (x.reshape(-1, self.num_kv_heads, self.head_dim)
                    for x in kv.split(self.kv_width, dim=-1))
            k = apply_rotary_emb(layer.k_norm(k), *rope)
            store_kvcache(k, v, k_cache, v_cache, slots)

    def _block_hidden(self, anchors: torch.Tensor, lengths: torch.Tensor, block_table: torch.Tensor,
                      cache: DFlashCache, draft_len: int) -> torch.Tensor:
        offsets = torch.arange(self.block_size, device=anchors.device)
        positions = lengths[:, None] + offsets
        ids = anchors.new_full(positions.shape, self.mask_token_id)
        ids[:, 0] = anchors
        hidden = self.embed_tokens(ids.flatten())
        rope = self.rope.cos_sin_cache[positions.flatten()].chunk(2, dim=-1)
        for layer, (k_cache, v_cache) in zip(self.layers, cache.tensors):
            hidden = layer(hidden, rope, k_cache, v_cache, block_table, lengths)
        # Keep the trained block width for attention; only project rows we can verify.
        start = self.first_draft_slot
        return self.norm(hidden.view(anchors.numel(), self.block_size, -1)[:, start:start + draft_len])

    def forward(self, anchors: torch.Tensor, lengths: torch.Tensor,
                block_table: torch.Tensor, cache: DFlashCache, draft_len: int) -> torch.Tensor:
        """Propose a batch of continuations with one bidirectional draft pass."""
        hidden = self._block_hidden(anchors, lengths, block_table, cache, draft_len)
        return self.target_ids[self.lm_head(hidden).argmax(dim=-1)]
