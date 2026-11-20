"""Tree proposal and target-sampled traversal for EAGLE-2/3."""

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


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


def build_batched_trees(
    root_logits: torch.Tensor,
    advance: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
    token_ids: torch.Tensor | None,
    width: int,
    limits: list[tuple[int, int]],
    eos: int,
) -> list[tuple[list[int], list[int]]]:
    """Expand a padded frontier per request, keeping all selections on device."""
    batch = root_logits.size(0)
    depth = max(d for d, _ in limits)
    budget = max(n for _, n in limits)
    if not depth or not budget:
        return [([], []) for _ in limits]
    width = min(width, root_logits.size(-1))
    device = root_logits.device
    if width == 1:
        # A single frontier has no competing scores. Ancestors are the
        # preceding positions, and an EOS ends the rest of that chain.
        lengths = [min(d, n) for d, n in limits]
        chain_depth = max(lengths)
        if not chain_depth:
            return [([], []) for _ in limits]
        length = torch.tensor(lengths, device=device)[:, None]
        logits = root_logits[:, None, :]
        parent = torch.zeros((batch, 1), device=device, dtype=torch.long)
        live = torch.ones_like(parent, dtype=torch.bool)
        chain = []
        for level in range(chain_depth):
            ids = logits.topk(1, dim=-1).indices.squeeze(-1)
            tokens = ids if token_ids is None else token_ids[ids]
            live &= (tokens != eos) & (length > level)
            chain.append(tokens.masked_fill(~live, -1))
            if level + 1 < chain_depth:
                logits = advance(tokens, parent, level)
        rows = torch.cat(chain, dim=1).tolist()
        tokens = [[t for t in row if t >= 0] for row in rows]
        return [(row, list(range(len(row)))) for row in tokens]

    depths, budgets = torch.tensor(limits, device=device).unbind(-1)
    logits = root_logits[:, None, :]
    frontier = torch.zeros((batch, 1), device=device, dtype=torch.long)
    scores = torch.zeros_like(frontier, dtype=torch.float32)
    all_scores, all_tokens, all_parents = [], [], []
    next_node = 1
    for level in range(depth):
        values, tokens = logits.topk(width, dim=-1)
        if token_ids is not None:
            tokens = token_ids[tokens]
        logp = values.float() - logits.float().logsumexp(dim=-1, keepdim=True)
        allowed = (tokens != eos) & (depths[:, None, None] > level)
        candidates = (scores[..., None] + logp).masked_fill(~allowed, -torch.inf)
        candidates = candidates.flatten(1)
        candidate_tokens = tokens.flatten(1)
        all_scores.append(candidates)
        all_tokens.append(candidate_tokens)
        all_parents.append(frontier.repeat_interleave(width, dim=1))
        if level + 1 < depth:
            order = candidates.argsort(descending=True, stable=True)[:, :width]
            scores = candidates.gather(1, order)
            frontier = next_node + order
            logits = advance(candidate_tokens.gather(1, order), order // width, level)
        next_node += candidates.size(1)

    scores = torch.cat(all_scores, dim=1)
    order = scores.argsort(descending=True, stable=True)[:, :budget]
    valid = torch.isfinite(scores.gather(1, order))
    valid &= torch.arange(order.size(1), device=device) < budgets[:, None]
    # A parent always scores at least as high as its children. Stable ties
    # retain parents first. Restore insertion order for scheduler truncation.
    kept = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, order, valid)
    rank = kept.cumsum(dim=1)
    by_id = order.masked_fill(~valid, scores.size(1)).sort(dim=1).values
    present = by_id < scores.size(1)
    by_id = by_id.clamp_max(scores.size(1) - 1)
    tokens = torch.cat(all_tokens, dim=1).gather(1, by_id).masked_fill(~present, -1)
    parents = torch.cat(all_parents, dim=1).gather(1, by_id)
    parents = F.pad(rank, (1, 0)).gather(1, parents)
    rows = torch.stack((tokens, parents), dim=-1).tolist()
    return [([t for t, _ in row if t >= 0], [p for t, p in row if t >= 0]) for row in rows]


@dataclass(slots=True)
class TreeAttentionBatch:
    padded_rows: torch.Tensor
    flat_rows: torch.Tensor
    ancestor_mask: torch.Tensor
    prefix_lengths: torch.Tensor
    block_tables: torch.Tensor
    prefill_rows: int
    prefill_tokens: int
