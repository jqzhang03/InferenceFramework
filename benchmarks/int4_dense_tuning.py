"""Tile/warp sweep for the generic W4A16 MMA path behind DSV2-Lite dense linears.

The reference `w4a16_mma_kernel` handles every non-routed INT4 linear (MLA q/kv/o
projections, dense MLP, shared experts) for M > 4 rows, i.e. all prefill rows and
decode batches above 4, and the nsys ShareGPT prefill attributes 175.9 ms to it.

Variant 0 is the historical two-warp / 32-row tile; 1-6 widen the row tile (or
warp count) so one weight dequantisation feeds more rows. Every variant keeps the
per-output MMA K order, so results must be bitwise identical to variant 0.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import median

import torch

from benchmarks.deepseek_w4a16 import environment, write_json
from nanovllm.layers.cuda_int4 import load_cuda_int4

# DeepSeek-V2-Lite: hidden 2048, q_lora 1536, kv_lora 512, 16 heads,
# qk_nope 128, qk_rope 64, v 128, moe_intermediate 1408.
SHAPES = {
    "q_a_proj": (1536, 2048),
    "q_b_proj": (3072, 1536),
    "kv_a_proj": (576, 2048),
    "kv_b_proj": (4096, 512),
    "o_proj": (2048, 2048),
    "gate_up": (2816, 2048),
    "down": (2048, 1408),
}
VARIANTS = {
    -1: "production dispatch (m-based launch schedule)",
    0: "BLOCK_M=32, 64 threads (2 warps)",
    1: "BLOCK_M=64, 128 threads (4 warps)",
    2: "BLOCK_M=128, 128 threads (4 warps)",
    3: "BLOCK_M=64, 64 threads (2 warps)",
    4: "BLOCK_M=128, 256 threads (8 warps)",
    5: "BLOCK_M=32, 128 threads (4 warps)",
    6: "BLOCK_M=256, 256 threads (8 warps)",
    7: "BLOCK_M=16, 128 threads (4 warps)",
}


def time_call(call, iterations: int) -> float:
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def unpack(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    codes = packed.int()
    signed = torch.stack(((codes & 15) - 8, ((codes >> 4) & 15) - 8), dim=-1)
    k = packed.size(1) * 2
    group = k // scales.size(1)          # quantisation group size, not group count
    weights = (signed.reshape(scales.size(0), -1, group).float()
               * scales.float().unsqueeze(-1))
    return weights.reshape(scales.size(0), -1)


def build_layered(extension, tokens: int, layers: int, group: int, rng):
    """One call per (layer, linear) with distinct weights, like a real prefill."""
    calls = []
    for _ in range(layers):
        for n, k in SHAPES.values():
            packed = torch.randint(-128, 128, (n, k // 2), generator=rng,
                                   device="cuda", dtype=torch.int8)
            scales = (torch.rand((n, k // group), generator=rng,
                                 device="cuda", dtype=torch.bfloat16) / 32)
            x = torch.randn(tokens, k, generator=rng, device="cuda",
                            dtype=torch.bfloat16)
            out = torch.empty(tokens, n, device="cuda", dtype=torch.bfloat16)
            calls.append((x, packed, scales, out))
    return calls


def time_layered(extension, calls, variant: int, iterations: int) -> float:
    def run():
        for x, packed, scales, out in calls:
            if variant < 0:
                extension.gemm_out(x, packed, scales, out)
            else:
                extension.gemm_out_variant(x, packed, scales, out, variant)

    for _ in range(2):
        run()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        run()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tokens", default="32,128,512,1257,2048")
    parser.add_argument("--variants", default="0,1,2,3,4,5,6")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--group", type=int, default=64)
    parser.add_argument("--layered", default="1257",
                        help="comma-separated M values for the 27-layer distinct-weight pass")
    parser.add_argument("--layers", type=int, default=27)
    args = parser.parse_args()
    tokens = [int(value) for value in args.tokens.split(",")]
    variants = [int(value) for value in args.variants.split(",")]
    args.root.mkdir(parents=True, exist_ok=True)
    extension = load_cuda_int4()
    rng = torch.Generator(device="cuda").manual_seed(7)
    output = {"environment": environment(), "group": args.group,
              "variants": {str(k): VARIANTS[k] for k in variants},
              "shapes": {name: list(shape) for name, shape in SHAPES.items()},
              "rows": []}
    for name, (n, k) in SHAPES.items():
        packed = torch.randint(-128, 128, (n, k // 2), generator=rng,
                               device="cuda", dtype=torch.int8)
        scales = (torch.rand((n, k // args.group), generator=rng,
                             device="cuda", dtype=torch.bfloat16) / 32)
        dense = unpack(packed, scales).to(torch.bfloat16)
        for count in tokens:
            x = torch.randn(count, k, generator=rng, device="cuda",
                            dtype=torch.bfloat16)
            out = torch.empty(count, n, device="cuda", dtype=torch.bfloat16)
            expected = (x.double() @ dense.double().T).to(torch.bfloat16)
            baseline = None
            for variant in variants:
                def run(out=out, x=x, packed=packed, scales=scales, variant=variant):
                    if variant < 0:
                        extension.gemm_out(x, packed, scales, out)
                    else:
                        extension.gemm_out_variant(x, packed, scales, out, variant)
                call = run
                call()
                rel = ((out - expected).norm() / expected.norm().clamp_min(1e-12)).item()
                if rel > .01:
                    raise AssertionError(f"{name} M={count} v{variant} relL2 {rel:.4f}")
                if baseline is None:
                    baseline = out.clone()
                elif not torch.equal(out, baseline):
                    raise AssertionError(f"{name} M={count} v{variant} not bitwise")
                samples = [time_call(call, 20 if count <= 128 else 5)
                           for _ in range(args.repeats)]
                row = dict(name=name, n=n, k=k, tokens=count, variant=variant,
                           ms=median(samples), macs=count * n * k, samples=samples)
                row["tflops"] = 2 * row["macs"] / (row["ms"] * 1e9)
                output["rows"].append(row)
            line = " ".join(
                f"v{row['variant']}={row['ms']:.4f}"
                for row in output["rows"][-len(variants):])
            print(f"{name:10s} M={count:5d}  {line}", flush=True)
        write_json(args.root / "sweep.json", output)

    # Aggregate per variant at the prefill shape and the decode-32 shape.
    summary = {}
    for count in (32, 1257):
        for variant in variants:
            picked = [row for row in output["rows"]
                      if row["tokens"] == count and row["variant"] == variant]
            if not picked:
                continue
            total = sum(row["ms"] for row in picked)
            flops = sum(row["macs"] for row in picked)
            summary[f"{count}/v{variant}"] = dict(per_layer_ms=total,
                                                  layers_27_ms=total * 27,
                                                  per_layer_tflops=2 * flops / (total * 1e9))
    output["aggregate"] = summary
    write_json(args.root / "sweep.json", output)
    for key, row in summary.items():
        print(f"aggregate {key:10s} per-layer {row['per_layer_ms']:7.4f} ms "
              f"27L {row['layers_27_ms']:7.1f} ms  {row['per_layer_tflops']:6.1f} TFLOP/s")

    # Distinct weights per layer: the weights cannot stay resident in L2, which
    # is what a real 27-layer prefill sees (the per-shape loop above is warm).
    layered = {}
    for count in [int(value) for value in args.layered.split(",") if value.strip()]:
        calls = build_layered(extension, count, args.layers, args.group, rng)
        for variant in variants:
            ms = time_layered(extension, calls, variant, 3)
            layered[f"{count}/v{variant}"] = dict(ms=ms,
                                                  per_layer_ms=ms / args.layers)
            print(f"layered M={count:5d} v{variant}={ms:8.3f} ms "
                  f"({ms / args.layers:.3f} ms/layer)", flush=True)
        del calls
        torch.cuda.empty_cache()
    output["layered"] = layered
    write_json(args.root / "sweep.json", output)


if __name__ == "__main__":
    main()
