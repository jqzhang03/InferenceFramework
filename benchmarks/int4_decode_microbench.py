"""Microbenchmark the route-major INT4 decode projection without loading the model.

Builds the DeepSeek-V2-Lite expert shapes (gate/up 1408x2048, down 2048x1408,
64 experts, 32 tokens x top_k 6 = 192 packed rows) and times:

  * the compact GEMV `expert_out_compact` over CTA fan-outs 8/16/32/64;
  * the old 5-8 / 9-16 / 17-32 MMA band launches vs the count-branched band.

Outputs of the old and new band paths are compared bitwise. This isolates the
kernel schedule from model loading, host scheduling, and GPU clocks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nanovllm.layers.cuda_int4 import load_cuda_int4


def build(counts: torch.Tensor, group: int):
    experts = counts.numel()
    rows = int(counts.sum())
    width, intermediate = 2048, 1408
    gate_w = torch.randint(-128, 128, (experts, intermediate, width // 2),
                           dtype=torch.int8, device="cuda")
    up_w = torch.randint(-128, 128, (experts, intermediate, width // 2),
                         dtype=torch.int8, device="cuda")
    down_w = torch.randint(-128, 128, (experts, width, intermediate // 2),
                           dtype=torch.int8, device="cuda")
    scale_shape = (experts, intermediate, width // group)
    gate_s = torch.rand(scale_shape, dtype=torch.bfloat16, device="cuda") / 32
    up_s = torch.rand(scale_shape, dtype=torch.bfloat16, device="cuda") / 32
    down_s = torch.rand((experts, width, intermediate // group),
                        dtype=torch.bfloat16, device="cuda") / 32
    ptrs = lambda tensor: torch.tensor(
        [tensor[i].data_ptr() for i in range(experts)],
        dtype=torch.int64, device="cuda")
    offsets = torch.cumsum(counts, 0, dtype=torch.int32) - counts
    x = torch.randn(rows + 32, width, dtype=torch.bfloat16, device="cuda")
    hidden = torch.empty(rows + 32, intermediate, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(rows + 32, width, dtype=torch.bfloat16, device="cuda")
    return {
        "experts": experts, "rows": rows, "width": width,
        "intermediate": intermediate, "group": group,
        "x": x, "hidden": hidden, "out": out, "counts": counts, "offsets": offsets,
        "gate_w": ptrs(gate_w), "gate_s": ptrs(gate_s),
        "up_w": ptrs(up_w), "up_s": ptrs(up_s),
        "down_w": ptrs(down_w), "down_s": ptrs(down_s),
    }


def time_call(fn, repeats: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--group", type=int, default=64)
    parser.add_argument("--fanouts", default="8:8,32:8,64:8,128:8,128:32,128:64,128:128")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    extension = load_cuda_int4()
    torch.manual_seed(7)

    # 192 routes over 64 experts, with explicit 5-8 / 9-16 / 17+ rows so both
    # MMA bands have real work (a plain multinomial often leaves the 9-16 band
    # nearly empty).
    counts = torch.empty(64, dtype=torch.int32, device="cuda")
    counts[:8] = torch.tensor([12, 10, 9, 7, 6, 5, 4, 4],
                              dtype=torch.int32, device="cuda")
    counts[8:16] = 3
    counts[16] = 17
    counts[17:] = 2
    assert int(counts.sum()) == 192
    data = build(counts, args.group)
    x, hidden, out = data["x"], data["hidden"], data["out"]
    counts, offsets = data["counts"], data["offsets"]
    rows = data["rows"]
    width, intermediate, group = data["width"], data["intermediate"], data["group"]
    common = (x, data["gate_w"], data["gate_s"], data["up_w"], data["up_s"],
              offsets, counts)

    def gemv(down_fanout: int, gate_fanout: int) -> None:
        extension.expert_out_compact(
            x, common[1], common[2], common[3], common[4],
            data["down_w"], data["down_s"], offsets, counts,
            hidden, out, 4, rows, group, down_fanout, gate_fanout)

    def gate_old(block: int, lo: int, hi: int) -> None:
        extension.gemm_gate_up_silu_compact(
            *common, hidden, 32, rows, group, 64, block, lo, hi)

    def gate_new() -> None:
        extension.gemm_gate_up_silu_compact(
            *common, hidden, 32, rows, group, 64, 0, 5, 16)
        extension.gemm_gate_up_silu_compact(
            *common, hidden, 32, rows, group, 64, 32, 17, 32)

    def down_old(block: int, lo: int, hi: int) -> None:
        extension.gemm_out_compact(
            hidden, data["down_w"], data["down_s"], offsets, counts,
            out, 32, rows, group, 64, block, lo, hi)

    def down_new() -> None:
        extension.gemm_out_compact(
            hidden, data["down_w"], data["down_s"], offsets, counts,
            out, 32, rows, group, 64, 0, 5, 16)
        extension.gemm_out_compact(
            hidden, data["down_w"], data["down_s"], offsets, counts,
            out, 32, rows, group, 64, 32, 17, 32)

    # Bitwise parity: old three-band vs merged 5-16 + 32.
    gate_old(8, 5, 8); gate_old(16, 9, 16); gate_old(32, 17, 32)
    reference = hidden.clone()
    gate_new()
    gate_equal = torch.equal(reference, hidden)
    down_old(8, 5, 8); down_old(16, 9, 16); down_old(32, 17, 32)
    reference = out.clone()
    down_new()
    down_equal = torch.equal(reference, out)

    payload = {"counts": counts.tolist(), "rows": rows,
               "band_gate_bitwise": gate_equal, "band_down_bitwise": down_equal,
               "gemv": {}, "bands": {}}
    for spec in args.fanouts.split(","):
        down_fanout, gate_fanout = (int(value) for value in spec.split(":"))
        samples = [time_call(lambda: gemv(down_fanout, gate_fanout), args.repeats)
                   for _ in range(3)]
        payload["gemv"][spec] = min(samples)
        print(f"gemv down={down_fanout:3d} gate={gate_fanout:3d} "
              f"{min(samples):8.3f} ms", flush=True)

    old_gate = lambda: (gate_old(8, 5, 8), gate_old(16, 9, 16), gate_old(32, 17, 32))
    old_down = lambda: (down_old(8, 5, 8), down_old(16, 9, 16), down_old(32, 17, 32))
    for name, fn in (("gate_old", old_gate), ("gate_merged", gate_new),
                     ("down_old", old_down), ("down_merged", down_new)):
        samples = [time_call(fn, args.repeats) for _ in range(3)]
        payload["bands"][name] = min(samples)
        print(f"{name:12s} {min(samples):8.3f} ms", flush=True)

    (args.root / "microbench.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
