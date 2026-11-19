"""Summarize matched DeepSeek W4A16 timing, memory and raw Nsight captures."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import re
import sqlite3
from statistics import median

from benchmarks.cuda_operator_profile_report import METRICS, union_ns


MIB = 2 ** 20
INT4_KERNELS = ("gemm_int4_kernel", "w4a16_gemv_kernel", "w4a16_mma_kernel")


def driver_memory(db: sqlite3.Connection, pid: int, begin: int, end: int) -> dict:
    rows = db.execute("""
        SELECT start, bytes, memoryOperationType FROM CUDA_GPU_MEMORY_USAGE_EVENTS
        WHERE globalPid=? AND memKind=2 AND start<=? ORDER BY start
    """, (pid, end))
    live = peak = before = count = 0
    for tick, size, op in rows:
        live += size if op == 0 else -size
        if tick < begin:
            before = peak = live
        else:
            count += 1
            peak = max(peak, live)
    return dict(before_bytes=before, peak_bytes=peak, after_bytes=live, event_count=count)


def analyze_nsys(path: Path) -> list[dict]:
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        strings = dict(db.execute("SELECT id, value FROM StringIds"))
        ranges = list(db.execute("""
            SELECT text, start, end, globalTid FROM NVTX_EVENTS
            WHERE end IS NOT NULL AND
                (text LIKE 'model_capture/%' OR text LIKE 'step/%') ORDER BY start
        """))
        kernels = defaultdict(list)
        for pid, correlation, begin, end, name in db.execute("""
            SELECT globalPid, correlationId, start, end, shortName
            FROM CUPTI_ACTIVITY_KIND_KERNEL
        """):
            kernels[pid, correlation].append((begin, end, strings[name]))
        result = []
        for label, begin, end, tid in ranges:
            if label.startswith("model_capture/"):
                model = dict(backend=label.split("/")[-1], wall_ms=(end - begin) / 1e6,
                             steps=[], start=begin, end=end)
                model["driver_memory"] = driver_memory(db, tid & ~0xFFFFFF, begin, end)
                result.append(model)
                continue
            model = result[-1]
            if begin < model["start"] or end > model["end"]:
                raise ValueError(f"{label} falls outside its model-capture range")
            calls = db.execute("""
                SELECT DISTINCT correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME
                WHERE globalTid=? AND start>=? AND end<=?
            """, (tid, begin, end))
            spans = []
            kernel_ns, kernel_counts = Counter(), Counter()
            for (correlation,) in calls:
                for start, stop, name in kernels.get((tid & ~0xFFFFFF, correlation), ()):
                    spans.append((start, stop))
                    kernel_ns[name] += stop - start
                    kernel_counts[name] += 1
            int4_ns = sum(kernel_ns.get(name, 0) for name in INT4_KERNELS)
            int4_calls = sum(kernel_counts.get(name, 0) for name in INT4_KERNELS)
            if not int4_calls:
                raise ValueError(f"no INT4 launches matched in {path.name} {label}")
            model["steps"].append(dict(
                kind=("prefill" if label == "step/0" else "decode"),
                wall_ms=(end - begin) / 1e6, gpu_kernel_union_ms=union_ns(spans) / 1e6,
                int4_ms=int4_ns / 1e6, int4_calls=int4_calls,
                kernel_calls=len(spans), top_kernel_ms={k: v / 1e6 for k, v in kernel_ns.most_common(12)},
                int4_launches={name: kernel_counts[name] for name in INT4_KERNELS if kernel_counts[name]},
            ))
    if not result or any(len(item["steps"]) != 4 for item in result):
        raise ValueError(f"incomplete timeline in {path}")
    return result


def read_ncu(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    units, *launches = rows
    labels = re.findall(r"^\d+ \w+ (\w+) M=", path.with_suffix(".log").read_text(), re.M)
    cases = labels[:8]
    if (len(launches) != 8 or len(cases) != 8 or
            any(labels[i:i + 8] != cases for i in range(0, len(labels), 8))):
        raise ValueError(f"incomplete NCU case attribution in {path}")
    metrics = dict(METRICS)
    metrics["dram_total_gbs"] = ("dram__bytes.sum.per_second", METRICS["dram_gbs"][1])
    for kind in ("allocated", "static", "dynamic"):
        metrics[f"shared_{kind}_kb"] = (f"launch__shared_mem_per_block_{kind}",
                                         {"byte/block": .001, "Kbyte/block": 1})
    result = []
    for index, (case, launch) in enumerate(zip(cases, launches)):
        if int(launch["ID"]) != index or not any(name in launch["Kernel Name"] for name in INT4_KERNELS):
            raise ValueError(f"unexpected launch in {path}: {launch['Kernel Name']}")
        row = dict(case=case, kernel=launch["Kernel Name"])
        for label, (key, factors) in metrics.items():
            value = launch.get(key)
            if value not in (None, "", "n/a"):
                row[label] = float(value.replace(",", "")) * factors[units[key]]
        result.append(row)
    return result


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def model_table(timings: list[dict]) -> tuple[list[str], dict]:
    lines = ["| 后端 | TTFT p50 ms | TPOT p50 ms | 总耗时 ms | Prefill ms | Decode step ms | 稳态已分配 MiB | 峰值已分配 MiB | 峰值保留 MiB |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    summary = {}
    for backend in ("triton", "cuda"):
        rows = [row for row in timings if row["backend"] == backend]
        values = [median(row[key] for row in rows) for key in ("ttft_ms", "tpot_ms", "wall_ms")]
        values += [median(row["steps"][0]["ms"] for row in rows),
                   median(step["ms"] for row in rows for step in row["steps"][1:])]
        allocated = [median(row["memory_after"][key] for row in rows) / MIB
                     for key in ("allocated", "peak_allocated", "peak_reserved")]
        summary[backend] = dict(ttft_ms=values[0], tpot_ms=values[1], wall_ms=values[2],
                                prefill_ms=values[3], decode_ms=values[4],
                                allocated_mib=allocated[0], peak_allocated_mib=allocated[1],
                                peak_reserved_mib=allocated[2])
        lines.append(f"| {backend} | " + " | ".join(f"{v:.2f}" for v in values + allocated) + " |")
    return lines, summary


def report(root: Path, output: Path) -> None:
    baseline = load(root / "model_baseline.json")
    pair = load(root / "model_pair.json")
    smoke = load(root / "cuda_smoke.json")
    ops = load(root / "operators.json")["cases"]
    timelines = {mode: analyze_nsys(root / f"{mode}.sqlite")
                 for mode in ("triton_baseline", "model_pair")}
    counters = {path.stem: read_ncu(path) for path in sorted((root / "ncu").glob("*.csv"))
                if path.stem in ("triton_none", "triton_all", "cuda_none", "cuda_all")}
    if len(counters) != 4:
        raise ValueError("expected four backend/cache combinations")
    live = {path.stem: read_ncu(path) for path in sorted((root / "ncu").glob("live_*.csv"))}
    if live and set(live) != {"live_triton", "live_cuda"}:
        raise ValueError("expected both in-model NCU captures")
    if live and [row["case"] for row in live["live_triton"]] != [row["case"] for row in live["live_cuda"]]:
        raise ValueError("in-model NCU capture order differs")
    baseline_trace, = timelines["triton_baseline"]
    before, after = timelines["model_pair"]
    if [x["backend"] for x in (baseline_trace, before, after)] != ["triton", "triton", "cuda"]:
        raise ValueError("incorrect capture ordering")
    if [s["int4_calls"] for s in before["steps"]] != [s["int4_calls"] for s in after["steps"]]:
        raise ValueError("paired model INT4 launch counts differ")
    timings, medians = model_table(pair["timings"])
    pairs = defaultdict(dict)
    for row in pair["timings"]:
        pairs[row["repeat"]][row["backend"]] = row
    delta = [(values["cuda"]["wall_ms"] / values["triton"]["wall_ms"] - 1) * 100
             for values in pairs.values()]
    weight = smoke["int4_weight_bytes"]
    lines = ["# DeepSeek-V2-Lite BF16 / INT4 W4A16 对照", "",
             f"{pair['environment']['gpu']}，PyTorch {pair['environment']['torch']}，CUDA {pair['environment']['cuda']}。"
             "TP=1，INT4 group 64，流式加载，eager，逐专家执行。随机种子 42，"
             "4 个同时提交的请求各输入 64 个固定合成 token、输出 4 个 token，采样温度 0.8。"
             "实测为 1 次 256-row prefill 和 3 次 4-row decode。为固定批次关闭自适应配额与多步 decode。", "",
             "先运行现有 Triton 框架的 Systems 采集，保存真实激活及其打包权重，再采集 Triton Compute 计数器。"
             "算子对照复用快照中的 BF16 输入、INT4 字节和 scale。整模配对在同一实例切换后端，共享权重和 MLA KV 池，"
             "回放同一组 token 与 Triton 路由结果，保留两次实际的 router 和 sampler 计算。每轮交替执行顺序。", "",
             "## 整模延迟与显存", "",
             *timings, "",
             f"{len(delta)} 轮固定路径配对中，CUDA 快了 {sum(d < 0 for d in delta)} 轮。"
             f"总耗时变化的配对中位数 {median(delta):+.1f}%，"
             f"单轮范围 {min(delta):+.1f}% 至 {max(delta):+.1f}%。",
             f"配对 MLA KV 池 {pair['kv_blocks']} 块（{pair['kv_bytes'] / MIB:.2f} MiB）。"
             f"首次基线采集的 KV 池 {baseline['kv_blocks']} 块（{baseline['kv_bytes'] / MIB:.2f} MiB），"
             "由启动时空闲显存自动确定。表中延迟来自无 profiler 的配对运行，显存来自 PyTorch 计数器。", "",
             f"模型 INT4 字节 {weight['packed'] / MIB:.2f} MiB，scale {weight['scales'] / MIB:.2f} MiB，"
             f"MLA 保留的 dense 副本 {weight['dense_copy'] / MIB:.2f} MiB。两后端复用这些缓冲区。", "",
             "## Nsight Systems", "",
             "通过 CUDA API correlation ID 与进程 ID 归属 GPU kernel。GPU active 为 kernel 时间并集。"
             "驱动显存为跟踪进程的设备 allocation/free 事件账本。Systems 内的 wall time 含追踪开销。", "",
             "| 采集 | 阶段 | NVTX wall ms | GPU active ms | INT4 核时间 ms | INT4 launches |",
             "|---|---|---:|---:|---:|---:|"]
    for tag, capture in (("初始 Triton", baseline_trace), ("配对 Triton", before), ("配对 CUDA", after)):
        for kind in ("prefill", "decode"):
            steps = [s for s in capture["steps"] if s["kind"] == kind]
            lines.append(f"| {tag} | {kind} ×{len(steps)} | " +
                         " | ".join(f"{sum(s[k] for s in steps):.2f}" for k in
                                    ("wall_ms", "gpu_kernel_union_ms", "int4_ms")) +
                         f" | {sum(s['int4_calls'] for s in steps)} |")
    for tag, capture in (("初始 Triton", baseline_trace), ("配对 Triton", before), ("配对 CUDA", after)):
        info = capture["driver_memory"]
        lines += ["", f"{tag} 的驱动设备 allocation 账本为 "
                  f"{info['before_bytes'] / MIB:.2f} → {info['peak_bytes'] / MIB:.2f} MiB"
                  f"（范围内 {info['event_count']} 次 allocation/free）。"]
    comp = pair["diagnostics"]["logit_comparison"]
    diag = pair["diagnostics"]
    adapters = diag["cuda"]["cuda_adapters"]
    same_sampled = sum(a == b for left, right in zip(diag["triton"]["sampled_tokens"],
                                                    diag["cuda"]["sampled_tokens"])
                       for a, b in zip(left, right))
    lines += ["", "## 单矩阵对照", "",
              "CUDA Graph 重放同一份权重与激活，32 次调用/图，分别报告多轮 CUDA Event 计时中位数。"
              "真实形状来自采样到的单个投影。带 m 后缀的扫点切取或重复真实 prefill 激活行。"
              "在 FP64 GEMM 前按对应的 BF16 或 FP16 精度舍入反量化权重。单矩阵显存用 eager 独立测试。", "",
              "| 投影 | M × N × K | Triton μs | CUDA μs | CUDA / Triton | CUDA 对 Triton 相对 L2 | packed + scale MiB | 额外峰值 MiB（T/C） |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, case in ops.items():
        triton, cuda = (median(case["graph_us"][key]) for key in ("triton", "cuda"))
        weight = (case["packed_bytes"] + case["scale_bytes"]) / MIB
        extras = [case["memory"][key]["extra_peak_bytes"] / MIB for key in ("triton", "cuda")]
        lines.append(f"| {name} | {case['m']} × {case['n']} × {case['k']} | {triton:.3f} | {cuda:.3f} | "
                     f"{cuda / triton:.3f} | {case['errors']['cuda_vs_triton']['relative_l2']:.3g} | "
                     f"{weight:.3f} | {extras[0]:.3f} / {extras[1]:.3f} |")
    max_l2 = max(case["errors"]["cuda_vs_fp64"]["relative_l2"] for case in ops.values())
    max_triton_l2 = max(case["errors"]["triton_vs_fp64"]["relative_l2"] for case in ops.values())
    wins = sum(median(case["graph_us"]["cuda"]) < median(case["graph_us"]["triton"]) for case in ops.values())
    extra_bytes = {backend: max(case["memory"][backend]["extra_peak_bytes"] for case in ops.values())
                   for backend in ("triton", "cuda")}
    lines += ["", f"CUDA 在 {wins}/{len(ops)} 个形状或输入算例中更快。"
              f"在输入、打包权重、scale 和输出之外，分配峰值增量最高为 Triton {extra_bytes['triton']} B、"
              f"CUDA {extra_bytes['cuda']} B，为 PyTorch 分配对齐量级。"
              f"相对 FP64 参考最大 L2 误差为 CUDA {max_l2:.3g}、Triton {max_triton_l2:.3g}。"
              f"CUDA 安装了 {adapters} 个投影。固定 token、各自自然路由时，整模 logits 相对 L2 {comp['relative_l2']:.3g}，"
              f"最大绝对差 {comp['max_abs']:.4g}，argmax {comp['argmax_matches']}/16 一致。"
              f"CUDA 有 {diag['cuda']['matching_route_indices']}/{diag['cuda']['route_calls']} 次 router 调用的专家索引完整匹配。"
              f"另以同一随机种子自然采样，逐位置输出 {same_sampled}/16 一致。", "",
              "## Nsight Compute", "",
              "同一批真实投影快照，各后端分别使用预热缓存（none）与逐 replay 清缓存（all）。GPU 时钟保持系统设置。"
              "下表展示 all 组计数器，两个缓存组的原始 CSV、单位和原生报告均保留。", "",
              "| 投影 | 后端 | DRAM% | L2 hit% | Tensor active% | Occupancy% | 静态+动态 Shared KB/CTA | 加载/写入 bank conflicts |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for left, right in zip(counters["triton_all"], counters["cuda_all"]):
        if left["case"] != right["case"]:
            raise ValueError("Triton/CUDA NCU case order differs")
        for backend, row in (("triton", left), ("cuda", right)):
            values = [f"{row[key]:.2f}" if key in row else "—" for key in
                      ("dram_pct", "l2_hit_pct", "tensor_pct", "occupancy_pct")]
            shared = row["shared_static_kb"] + row["shared_dynamic_kb"]
            lines.append(f"| {row['case']} | {backend} | " + " | ".join(values) +
                         f" | {shared:.2f} | {row.get('bank_load', 0):.0f} / {row.get('bank_store', 0):.0f} |")
    triton_q = next(row for row in counters["triton_all"] if row["case"] == "q_prefill")
    cuda_q = next(row for row in counters["cuda_all"] if row["case"] == "q_prefill")
    lines += ["", f"Q prefill 的共享内存加载冲突由 {triton_q['bank_load']:.0f} 降至 {cuda_q['bank_load']:.0f}，"
              f"写入冲突由 {triton_q['bank_store']:.0f} 升至 {cuda_q['bank_store']:.0f}，"
              f"Tensor active 由 {triton_q['tensor_pct']:.2f}% 升至 {cuda_q['tensor_pct']:.2f}%。"
              "单矩阵延迟取独立 Graph 测量，NCU replay 用于诊断硬件计数器。"]
    if live:
        lines += ["", f"另在固定 token 与路由的完整模型 forward 中，各采集了 {len(live['live_cuda'])} 个选中投影。"
                  "整模在两次运行中按各自算法产生隐藏状态。相同激活字节的比较见上面的快照回放。",
                  "", "| 实际模型投影 | Triton launch | CUDA launch | Triton / CUDA Occupancy% |",
                  "|---|---|---|---:|"]
        for left, right in zip(live["live_triton"], live["live_cuda"]):
            cuda_kind = next(name for name in INT4_KERNELS if name in right["kernel"])
            lines.append(f"| {left['case']} | gemm_int4_kernel | {cuda_kind} | "
                         f"{left['occupancy_pct']:.2f} / {right['occupancy_pct']:.2f} |")
    lines += ["", "## 启用与复现", "",
              "CUDA W4A16 通过 `LLM(..., quantization=\"int4\", int4_group_size=64, "
              "streaming_load=True, enforce_eager=True, moe_int4_backend=\"loop\", "
              "int4_kernel_backend=\"cuda\")` 启用。"
              "默认选用 Triton。CUDA 对小 M 使用向量化 SIMT，对其余 M 使用 Tensor Core MMA。"
              "反量化仅在 kernel 内进行，FP32 累加后输出 BF16 或 FP16。"
              f"配置入口实测安装 {smoke['adapters']} 个投影并完成 4 请求各 4 token 的生成。", "", "```bash",
              "out=profiles/deepseek_w4a16_repro",
              "bash benchmarks/run_deepseek_w4a16.sh baseline $out",
              "bash benchmarks/run_deepseek_w4a16.sh ncu $out triton",
              "bash benchmarks/run_deepseek_w4a16.sh pytest $out -q",
              "bash benchmarks/run_deepseek_w4a16.sh operators $out",
              "bash benchmarks/run_deepseek_w4a16.sh pair $out",
              "bash benchmarks/run_deepseek_w4a16.sh nsys $out",
              "bash benchmarks/run_deepseek_w4a16.sh ncu $out cuda",
              "bash benchmarks/run_deepseek_w4a16.sh smoke $out --backend cuda",
              "bash benchmarks/run_deepseek_w4a16.sh ncu_model $out",
              "python3 -m benchmarks.deepseek_w4a16_report $out",
              "```", ""]
    links = (("完整汇总", "summary.json"), ("配对端到端原始数值", "model_pair.json"),
             ("算子原始数值", "operators.json"), ("初始 Systems", "triton_baseline.nsys-rep"),
             ("配对 Systems", "model_pair.nsys-rep"), ("Compute 原始报告", "ncu"),
             ("CUDA 配置入口生成验证", "cuda_smoke.json"))
    lines += [" · ".join(f"[{name}]({Path(os.path.relpath(root / path, output.parent)).as_posix()})"
                         for name, path in links), ""]
    output.write_text("\n".join(lines), encoding="utf-8")
    (root / "summary.json").write_text(json.dumps(dict(model=medians, nsys=timelines, ncu=counters,
                                                     ncu_in_model=live),
                                                indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report(args.root, args.output or args.root / "REPORT.md")


if __name__ == "__main__":
    main()
