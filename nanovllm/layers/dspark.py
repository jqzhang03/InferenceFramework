"""DSpark semi-autoregressive block drafting for Qwen3-0.6B."""

import torch
from torch import nn

from nanovllm.layers.dflash import DFlashCache, DFlashDraft


class MarkovHead(nn.Module):
    def __init__(self, target_vocab: int, draft_vocab: int, rank: int):
        super().__init__()
        self.markov_w1 = nn.Embedding(target_vocab, rank)
        self.markov_w2 = nn.Linear(rank, draft_vocab, bias=False)


class ConfidenceHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1)


class DSparkDraft(DFlashDraft):
    architecture = "DSparkDraftModel"
    step_ms: tuple[float, ...] = ()
    confidence_threshold = 0.3

    @staticmethod
    def _attention_settings(config: dict, layer_type: str) -> tuple[bool, tuple[int, int]]:
        arch = config["transformer_layer_config"]
        causal = arch.get("is_causal")
        if layer_type == "full_attention":
            return bool(causal), (-1, -1)
        window = arch.get("sliding_window", 0)
        if layer_type != "sliding_attention" or not window or window < config["block_size"]:
            raise ValueError("unsupported DSpark sliding window")
        if causal is None:
            causal = not config.get("sliding_window_non_causal", False)
        return causal, (window - 1, 0 if causal else -1)

    def __init__(self, config: dict, model_len: int, draft_len: int):
        super().__init__(config, model_len, draft_len)
        rank = config["markov_rank"]
        if config["markov_head_type"] != "vanilla" or rank < 1 or not config["enable_confidence_head"]:
            raise ValueError("DSpark requires a vanilla Markov head and a confidence head")
        self.markov_head = MarkovHead(self.embed_tokens.num_embeddings, self.lm_head.out_features, rank)
        self.confidence_with_markov = config["confidence_head_with_markov"]
        self.confidence_head = ConfidenceHead(self.lm_head.in_features + rank * self.confidence_with_markov)

    def _markov_proposals(self, anchors: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        base_logits = self.lm_head(hidden)
        previous = anchors
        proposals, confidence = [], []
        for state, logits in zip(hidden.unbind(1), base_logits.unbind(1)):
            previous_features = self.markov_head.markov_w1(previous)
            bias = self.markov_head.markov_w2(previous_features)
            previous = self.target_ids[(logits + bias).argmax(dim=-1)]
            confidence_features = (torch.cat((state, previous_features), dim=-1)
                                   if self.confidence_with_markov else state)
            conf = self.confidence_head.proj(confidence_features).squeeze(-1).float().sigmoid()
            proposals.append(previous)
            confidence.append(conf)
        # Float32 preserves every token ID. Copy tokens and probabilities together.
        return torch.stack((torch.stack(proposals, dim=1).float(),
                            torch.stack(confidence, dim=1)), dim=-1)

    def forward(self, anchors: torch.Tensor, lengths: torch.Tensor,
                block_table: torch.Tensor, cache: DFlashCache, draft_len: int) -> torch.Tensor:
        hidden = self._block_hidden(anchors, lengths, block_table, cache, draft_len)
        return self._markov_proposals(anchors, hidden)

    def select_prefixes(self, proposals: list[list[list[float]]], depths: list[int],
                        eos: int | None = None) -> list[list[int]]:
        """Choose a prefix per request using the profiled verifier or a survival threshold."""
        candidates = []
        for request, (row, cap) in enumerate(zip(proposals, depths)):
            survival = 1.0
            for offset, (token, probability) in enumerate(row[:cap], start=1):
                if token == eos:
                    break
                survival *= probability
                candidates.append((survival, request, offset))

        active = len(proposals)
        selected = [0] * active
        if self.step_ms and active + len(candidates) <= len(self.step_ms):
            # Stable ties keep a candidate's predecessors ahead of it.
            candidates.sort(key=lambda item: item[0], reverse=True)
            accepted = float(active)
            best = accepted / self.step_ms[active - 1]
            for rows, (survival, request, length) in enumerate(candidates, start=active + 1):
                accepted += survival
                throughput = accepted / self.step_ms[rows - 1]
                if throughput <= best:
                    break
                best = throughput
                selected[request] = length
        else:
            for survival, request, length in candidates:
                if survival >= self.confidence_threshold:
                    selected[request] = length
        return [[int(token) for token, _ in row[:length]] for row, length in zip(proposals, selected)]
