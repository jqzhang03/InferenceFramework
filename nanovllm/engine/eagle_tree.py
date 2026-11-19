"""Tree proposal and target-sampled traversal for EAGLE-2/3."""

from dataclasses import dataclass
from typing import Callable

import torch


def tree_depths(parents: list[int]) -> list[int]:
    depths = [0]
    for child, parent in enumerate(parents, 1):
        if not 0 <= parent < child:
            raise ValueError("draft parents must precede their children")
        depths.append(depths[parent] + 1)
    return depths


def tree_mask(parents: list[int]) -> list[list[bool]]:
    tree_depths(parents)
    n = len(parents) + 1
    rows = [[True] + [False] * (n - 1)]
    for child, parent in enumerate(parents, 1):
        row = rows[parent].copy()
        row[child] = True
        rows.append(row)
    return rows


def verify_tree(drafts: list[int], parents: list[int], samples: list[int],
                eos: int | None = None) -> tuple[list[int], list[int]]:
    """Draw at the root and then only at a matching child, returning visited KV rows."""
    children = {(parent, token): child for child, (parent, token) in enumerate(zip(parents, drafts), 1)}
    tokens, path = [], []
    node = 0
    while True:
        path.append(node)
        token = samples[node]
        tokens.append(token)
        child = children.get((node, token))
        if child is None or token == eos:
            return tokens, path
        node = child


def build_tree(root_logits: torch.Tensor, advance: Callable[[list[int], list[int], list[int]], torch.Tensor],
               token_ids: torch.Tensor | None, width: int, depth: int, budget: int,
               eos: int) -> tuple[list[int], list[int]]:
    """Expand the best frontier paths and prune the final tree by cumulative confidence."""
    if budget <= 0 or depth <= 0:
        return [], []
    scores = [0.0]
    tokens = [-1]
    parents = [-1]
    frontier = [0]
    logits = root_logits.reshape(1, -1)
    for level in range(depth):
        logp, local_ids = logits.float().log_softmax(-1).topk(min(width, logits.shape[-1]))
        mapped = local_ids if token_ids is None else token_ids[local_ids]
        frontier_next = []
        for parent, row_scores, row_ids in zip(frontier, logp.tolist(), mapped.tolist()):
            for score, token in zip(row_scores, row_ids):
                if token == eos:
                    continue
                frontier_next.append(len(tokens))
                scores.append(scores[parent] + score)
                parents.append(parent)
                tokens.append(token)
        if level + 1 == depth or not frontier_next:
            break
        frontier = sorted(frontier_next, key=lambda i: (-scores[i], i))[:width]
        logits = advance(frontier, [parents[i] for i in frontier], [tokens[i] for i in frontier])
    kept = sorted(sorted(range(1, len(tokens)), key=lambda i: (-scores[i], i))[:budget])
    indices = {node: i for i, node in enumerate([0, *kept])}
    return [tokens[i] for i in kept], [indices[parents[i]] for i in kept]


@dataclass(slots=True)
class TreeAttentionBatch:
    padded_rows: torch.Tensor
    flat_rows: torch.Tensor
    ancestor_mask: torch.Tensor
    prefix_lengths: torch.Tensor
    block_tables: torch.Tensor
    prefill_rows: int
    prefill_tokens: int
