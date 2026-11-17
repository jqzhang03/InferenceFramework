"""Render cfg22 counter evidence, retaining NCU units and cache regimes."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from statistics import median


METRICS = {
    "duration_us": ("gpu__time_duration.sum", {"ns": .001, "us": 1, "ms": 1000}),
    "dram_mb": ("dram__bytes_op_read.sum", {"byte": 1e-6, "Kbyte": .001, "Mbyte": 1, "Gbyte": 1000}),
    "dram_gbs": ("dram__bytes_op_read.sum.per_second", {"byte/s": 1e-9, "Mbyte/s": .001, "Gbyte/s": 1, "Tbyte/s": 1000}),
    "dram_pct": ("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", {"%": 1}),
    "l2_hit_pct": ("lts__t_sector_hit_rate.pct", {"%": 1}),
    "occupancy_pct": ("sm__warps_active.avg.pct_of_peak_sustained_active", {"%": 1}),
    "tensor_pct": ("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed", {"%": 1}),
    "bank_load": ("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum", {"": 1}),
    "bank_store": ("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum", {"": 1}),
    "eligible": ("smsp__warps_eligible.avg.per_cycle_active", {"warp": 1, "warp/cycle": 1}),
}


def read_capture(path: Path) -> dict:
    """The raw page is a wide CSV: one units row, one kernel row."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 2:
        raise ValueError(f"expected units + single kernel: {path}")
    units, values = rows
    result = {"file": str(path), "kernel": values["Kernel Name"],
              "grid": values["Grid Size"], "block": values["Block Size"]}
    for label, (metric, factors) in METRICS.items():
        if metric not in values:
            continue
        if units[metric] not in factors:
            raise ValueError((path, metric, units[metric]))
        result[label] = float(values[metric].replace(",", "")) * factors[units[metric]]
    result["warp_stalls"] = {
        name: {"value": float(value), "unit": units[name]}
        for name, value in values.items()
        if name.startswith("smsp__average_warps_issue_stalled_") and name.endswith(".ratio")
    }
    result["resources"] = {
        name: {"value": values[name], "unit": units[name]}
        for name in ("launch__grid_size", "launch__block_size", "launch__registers_per_thread",
                     "launch__shared_mem_per_block_dynamic", "launch__occupancy_limit_shared_mem",
                     "launch__waves_per_multiprocessor", "device__attribute_l2_cache_size",
                     "device__attribute_multiprocessor_count")
        if name in values
    }
    return result


def generate(root: Path, output: Path) -> None:
    """Validate successful captures and write reproducible Markdown/JSON summaries."""
    captures = []
    for path in sorted(root.glob("*.status.json")):
        status = json.loads(path.read_text())
        if status["returncode"] != 0 or status.get("export_returncode") != 0:
            raise ValueError(f"unsuccessful capture: {path}")
        stem = path.name.removesuffix(".status.json")
        capture = read_capture(root / (stem + ".csv"))
        capture.update({key: status[key] for key in ("shape", "backend", "cache_control", "repeat", "group")})
        captures.append(capture)
    if len(captures) != 38:
        raise ValueError(f"expected 36 primary and two controls, got {len(captures)}")
    groups = []
    for n in (4096, 6144):
        for cache in ("none", "all"):
            for backend in ("cfg22", "alt_a", "torch"):
                rows = [r for r in captures if r["group"] == "primary" and r["shape"] == [8, n, 1024]
                        and r["backend"] == backend and r["cache_control"] == cache]
                if {r["repeat"] for r in rows} != {0, 1, 2}:
                    raise ValueError((n, cache, backend, "missing repeat"))
                groups.append(dict(n=n, cache=cache, backend=backend,
                                   **{key: median(r[key] for r in rows) for key in METRICS}))
    timing = json.loads((root / "timing_amortized.json").read_text())
    timings = []
    for shape in ([8, 4096, 1024], [8, 6144, 1024], [16, 4096, 1024], [8, 8192, 1024]):
        for regime in ("resident", "rotating28"):
            rows = [r for r in timing["rows"] if r["shape"] == shape and r["regime"] == regime]
            if {r["seed"] for r in rows} != {17, 42, 99}:
                raise ValueError((shape, regime, "missing seed"))
            timings.append(dict(shape=shape, regime=regime,
                                **{key: median(r["median_us"][key] for r in rows)
                                   for key in ("cfg22", "alt_a", "torch")}))
    model = []
    for name in ("layer14_qkv", "layer14_gate_up"):
        folder = root / "model_context"
        status = json.loads((folder / (name + ".status.json")).read_text())
        if status["returncode"] != 0:
            raise ValueError(name)
        row = read_capture(folder / (name + ".csv"))
        if "16, 128, 32, 16, 32, 2, 0>" not in row["kernel"]:
            raise ValueError(f"model capture is not original cfg22: {row['kernel']}")
        model.append(dict(name=name, **row))
    summary = dict(captures=captures, medians=groups, timing=timings, model_context=model)
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = ["# cfg22 Nsight Compute 瓶颈诊断（2026-11-17 续测）", "",
        "结论：BF16、B=8 的 Qwen3-0.6B 实际解码中，cfg22 的首要限制是从显存读取权重。"
        "单权重驻留 L2 的微基准则暴露了小 grid、有限的延迟隐藏能力和 A 共享内存 bank-conflict。"
        "消除冲突能改善部分热缓存测量，但没有改善轮换权重时延，因此不能把 bank-conflict 当作整个模型的主要瓶颈。", "",
        "## 测量范围和方法", "",
        "- RTX 5060 Ti / SM120，36 SM、32 MiB L2；PyTorch 2.8.0+cu128、CUDA 12.8、Nsight Compute 2026.3.1。",
        "- 真实投影尺寸采用 `X[M,K] × W[N,K]^T`：QKV=(8,4096,1024)，gate/up=(8,6144,1024)。",
        "- 原 cfg22：BM/BN/BK=16/128/32，WM/WN=16/32，两级 cp.async、128 threads/CTA、4 warps/CTA。"
        "每线程 56 registers、动态共享内存 18,432 B、local memory=0；共享内存限制每 SM 最多 5 CTA，理论 occupancy 41.67%。",
        "- 原算子、只更改 A 共享 swizzle 的 diagnostic variant=14、PyTorch `F.linear` 三方对照。"
        "variant=14 不是普通 cfg14；普通 cfg22 和线上 selector 保持不变。调库保持 PyTorch 当前策略，"
        "`allow_bf16_reduced_precision_reduction=True`，不宣称其内部归约顺序与自研相同。",
        "- 36 份主采集=2 尺寸×3 实现×2 缓存策略×3 次独立采集；另有 2 份 grid/有效行数控制。"
        "原始 kernel replay 每份 35 passes，`clock-control=none`。`cache-control=none` 在同一权重预热后采集；"
        "`all` 每 pass 刷新缓存。它们是受控对照，不直接代表端到端耗时。",
        "- 实际模型额外 2 份 application replay，每份 3 passes，TP=1/B=8/eager/128 输入 token，"
        "预热完整生成后，prefill 在采集外，首个 decode forward 由 profiler start/stop 限定。"
        "过滤 `gemm_kernel`，跳过 28/29 次匹配，分别选第 14 层（0-based）的 QKV/gate-up。"
        "保持前序模型计算和自然缓存状态，无模块 Event hooks；每个位置仅一份，不能估计跨层分布。",
        "- 独立非 NCU 时延：3 个种子×9 轮随机实现顺序，表中为每种子中位数再取中位数。"
        "resident 使用 32 节点 CUDA Graph 分摊 launch 开销；rotating28 每个图轮换 28 份权重。"
        "权重转换、分配、编译均不计时。早期单节点 pilot `timing.json` 不用于结论，使用 `timing_amortized.json`。", "",
        "## 实际模型：自然缓存状态", "",
        "| 位置 | NCU kernel μs | DRAM read MB | DRAM 峰值% | L2 hit% | Tensor active% | achieved occupancy% | shared load conflicts |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in model:
        lines.append(f"| {r['name']} | {r['duration_us']:.3f} | {r['dram_mb']:.3f} | {r['dram_pct']:.2f} | {r['l2_hit_pct']:.2f} | {r['tensor_pct']:.2f} | {r['occupancy_pct']:.2f} | {r['bank_load']:,.0f} |")
    lines += ["", "QKV/gate-up 的 BF16 权重分别为 8/12 MiB；28 层仅这两类权重就有 560 MiB，"
              "远大于 32 MiB L2。实际模型低 L2 命中与受控冷缓存相符，不能用同一权重反复运行的热缓存结果代替。"
              "DRAM 峰值百分比是 NCU 的 sustained peak 口径，受当前显存频率与平台状态影响。", "",
              "## 独立时延（μs，非 NCU）", "",
              "| M,N,K | 权重模式 | cfg22 | A swizzle 对照 | PyTorch |", "|---|---|---:|---:|---:|"]
    for r in timings:
        lines.append(f"| {','.join(map(str,r['shape']))} | {r['regime']} | {r['cfg22']:.3f} | {r['alt_a']:.3f} | {r['torch']:.3f} |")
    lines += ["", "resident 下 QKV 的 A swizzle 改善约 5.8%，gate/up 约 1.2%；rotating28 下二者基本持平或略慢。"
              "cfg22 在本次 rotating28 对照中已略快于调库；这是指定尺寸/缓存工作集的算子结果，"
              "不是模型吞吐提升结论。计数器采集时延受 replay/采样影响，性能比较优先看上表。", "",
              "## 主计数器（每组 3 份的中位数）", "",
              "`none` 为预热同一权重，`all` 为逐 pass 刷新。下表包含 NCU 时延，仅用于对应计数器场景。", "",
              "| N | 缓存 | 实现 | μs | DRAM GB/s | DRAM% | L2 hit% | Tensor% | occupancy% | eligible warp/cycle | load conflicts |",
              "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in groups:
        lines.append(f"| {r['n']} | {r['cache']} | {r['backend']} | {r['duration_us']:.3f} | {r['dram_gbs']:.2f} | {r['dram_pct']:.2f} | {r['l2_hit_pct']:.2f} | {r['tensor_pct']:.2f} | {r['occupancy_pct']:.2f} | {r['eligible']:.3f} | {r['bank_load']:,.0f} |")
    lines += ["", "### 1. 真实解码优先解决权重流量和带宽", "",
        "冷缓存 cfg22 的 QKV/gate-up 读取约 8.45/12.65 MB，达到约 377/386 GB/s、工具峰值的 86%/87%。"
        "M=8 时仅按权重计算的算术强度约 8 FLOP/byte；权重带宽对应的有效算力约 3 TFLOP/s，"
        "远低于 Tensor Core 峰值。Tensor active 低不代表此时应该优先增加 MMA 指令。"
        "消除 A 冲突后读流量和冷缓存时延基本不变，是反驳‘bank-conflict 主导实际解码’的直接对照。", "",
        "### 2. 热缓存还有 grid/延迟隐藏限制", "",
        "QKV 仅 32 CTA，小于 36 SM；gate/up 48 CTA，只有少量 SM 能同时分到第二个 CTA。"
        "每 CTA 仅 4 warp，与 5 CTA/SM 的资源上限不同，实际 grid 无法填满该上限。"
        "achieved occupancy 约 8.3%/11.2%，eligible warp/cycle 约 0.19/0.26。"
        "不能只凭 occupancy 低就认定因果：这里同时有 grid 计算、N 扩大控制和热/冷对照作为依据。", "",
        "| 控制尺寸 M,N,K | CTA | NCU μs | occupancy% | Tensor% | L2 hit% |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in captures:
        if r["group"] == "control":
            lines.append(f"| {','.join(map(str,r['shape']))} | {r['resources']['launch__grid_size']['value']} | {r['duration_us']:.3f} | {r['occupancy_pct']:.2f} | {r['tensor_pct']:.2f} | {r['l2_hit_pct']:.2f} |")
    lines += ["", "M=8→16 仍是同一 M16 物理 MMA tile；有效行翻倍但工作量/CTA 数基本不变，"
              "独立 resident 时延只从 13.287→13.818 μs。N=4096→8192 变为 64 CTA，"
              "热缓存 occupancy≈14.8%、Tensor active≈37.2%，独立时延仅到 14.776 μs；"
              "rotating28 则接近按权重大小翻倍。控制点各一份 NCU，非 NCU 时延仍有三个种子。", "",
              "### 3. A bank-conflict 确实存在，但优先级取决于缓存状态", "",
              "BK32 原 A 行 swizzle 使用 `r`；诊断改为 `r/2`，保持拷贝和 ldmatrix 地址一致。"
              "前 8 行的 16-byte bank group 从 `[0,5,2,7,0,5,2,7]` 变为 `[0,4,1,5,2,6,3,7]`。"
              "热缓存 QKV shared-load conflicts 32,768→0，gate/up 49,153→约 7，资源配置相同。"
              "计数是冲突事件数，不等于耗时百分比；残余 gate/up 冲突未完全消失。", "",
              "热缓存 warp-state 中位数如下，保留原始 ratio（CSV 单位 `inst`，按发射指令归一化的等待周期），"
              "不是 kernel 时间占比，也不能与其它类相加推导 wall time。", "",
              "| N | 实现 | short_scoreboard | wait | barrier | math_pipe_throttle | long_scoreboard |",
              "|---:|---|---:|---:|---:|---:|---:|"]
    for n in (4096, 6144):
        for backend in ("cfg22", "alt_a"):
            rows = [r for r in captures if r["group"] == "primary" and r["shape"][1] == n
                    and r["cache_control"] == "none" and r["backend"] == backend]
            vals = []
            for name in ("short_scoreboard", "wait", "barrier", "math_pipe_throttle", "long_scoreboard"):
                key = f"smsp__average_warps_issue_stalled_{name}_per_issue_active.ratio"
                vals.append(median(r["warp_stalls"][key]["value"] for r in rows))
            lines.append(f"| {n} | {backend} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    lines += ["", "short_scoreboard 支持检查 shared/ldmatrix 依赖；wait 也可能来自固定延迟的算术依赖。"
              "本次不将采样 stall ratio 独立归因为某条 SASS 指令。A swizzle 对照比单看 stall 名称更有因果解释力。", "",
              "## 下一步优化顺序", "",
              "1. 用 rotating28 和完整模型作为选型门槛。当前冷缓存已接近带宽上限，继续微调 shared 布局的收益有限。"
              "要进一步大幅提速，优先评估减少权重字节量（低精度需独立精度评估）、增加请求批量摊销权重读取，"
              "或能实际命中缓存的融合；这些尚未在本轮实现/验证。",
              "2. 若目标是热缓存/更小投影，测试更窄 BN 或受控 Split-K 增加 CTA，"
              "同时计入重复加载、FP32 partial 与 finish 开销，不能仅以 occupancy 提升判断成功。",
              "3. 保留 variant=14 作为 A swizzle 诊断候选，仅在独立复测和完整模型都胜出后再更改线上配置。", "",
              "## 正确性、复现与报告", "",
              "trace 用 FP64 参考验证 BF16 relative L2<0.006；timing 的两个主尺寸和两个控制尺寸均覆盖三个种子。"
              "`tests/test_cuda_cfg22_diag.py` 检查 FP16/BF16 四个尺寸（含 M/N/K 边界）、"
              "原 cfg22 与 variant14 bitwise 相等，以及更改输入后的 CUDA Graph replay。"
              "本次 GPU 回归测试 2 passed（FP16/BF16 各一项）；此项不建立模型质量结论。", "",
              "```bash",
              "bash benchmarks/run_qwen_cuda.sh --cfg22-ncu --mode capture --output results/cfg22_NEW",
              "bash benchmarks/run_qwen_cuda.sh --cfg22-ncu --mode timing --output results/cfg22_NEW",
              "bash benchmarks/run_qwen_cuda.sh --cfg22-ncu --mode model-capture --output results/cfg22_NEW",
              "bash benchmarks/run_qwen_cuda.sh --cfg22-pytest",
              "python -m benchmarks.cfg22_ncu_report --root results/cfg22_ncu_20261116",
              "```", "",
              "原始目录保留 20261116 命名，续测在 2026-11-17 完成；`--resume` 可保留成功采集后续跑。"
              "NCU 2026.3.1 输出的是压缩 `.ncu-repz`，请使用兼容版本的 **Nsight Compute UI** 打开；"
              "它与 Nsight Systems 的 `.nsys-rep` 属于不同工具。", ""]
    for name, path in (("实际模型 QKV", root / "model_context/layer14_qkv.ncu-repz"),
                       ("实际模型 gate/up", root / "model_context/layer14_gate_up.ncu-repz"),
                       ("隔离 cfg22 QKV 冷缓存", root / "primary_cfg22_8x4096x1024_all_r0.ncu-repz"),
                       ("隔离 cfg22 QKV 热缓存", root / "primary_cfg22_8x4096x1024_none_r0.ncu-repz")):
        # Generated Markdown lives one level below the repository root.
        target = Path(os.path.relpath(path, output.parent)).as_posix()
        lines.append(f"- [{name}]({target})")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output}; {len(captures)} isolated + {len(model)} model captures")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/cfg22_ncu_20261116"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/_cfg22_ncu_20261117.md"))
    args = parser.parse_args()
    generate(args.root, args.output)
