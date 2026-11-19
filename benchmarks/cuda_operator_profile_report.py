"""Analyze custom-only NVTX windows and multi-kernel NCU exports, without CUDA."""
from __future__ import annotations

import argparse
from collections import Counter
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import sqlite3
import statistics


CUSTOM_NAMES = ("gemm_kernel", "linear_gemv_kernel", "decode_mma_kernel", "decode_simt_kernel", "decode_warp_gemv_kernel",
                "finish_kernel", "fp8_kernel", "pack_rows_kernel", "fallback_kernel")
METRICS = {
    "us": ("gpu__time_duration.sum", {"ns": .001, "us": 1, "ms": 1000, "s": 1e6}),
    "dram_mb": ("dram__bytes_op_read.sum", {"byte": 1e-6, "Kbyte": .001, "Mbyte": 1, "Gbyte": 1000}),
    "dram_gbs": ("dram__bytes_op_read.sum.per_second", {"byte/s": 1e-9, "Kbyte/s": 1e-6, "Mbyte/s": .001, "Gbyte/s": 1, "Tbyte/s": 1000}),
    "dram_pct": ("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", {"%": 1}),
    "tensor_pct": ("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed", {"%": 1}),
    "occupancy_pct": ("sm__warps_active.avg.pct_of_peak_sustained_active", {"%": 1}),
    "l2_hit_pct": ("lts__t_sector_hit_rate.pct", {"%": 1}),
    "eligible": ("smsp__warps_eligible.avg.per_cycle_active", {"warp": 1}),
    "bank_load": ("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum", {"": 1}),
    "bank_store": ("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum", {"": 1}),
    "grid": ("launch__grid_size", {"": 1}),
    "registers": ("launch__registers_per_thread", {"register/thread": 1}),
}


def kernel_kind(name: str) -> str:
    if not any(token in name for token in CUSTOM_NAMES):
        raise ValueError(f"non-custom kernel in measured window: {name}")
    for token in ("pack_rows_kernel", "finish_kernel", "linear_gemv_kernel", "decode_warp_gemv_kernel", "decode_mma_kernel",
                  "decode_simt_kernel", "fp8_kernel", "gemm_kernel", "fallback_kernel"):
        if token in name:
            return token
    raise ValueError(name)


def read_ncu(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2 or rows[0].get("Kernel Name"):
        raise ValueError(f"expected NCU raw wide CSV with units row: {path}")
    units = rows[0]
    result = []
    for values in rows[1:]:
        name = values["Kernel Name"]
        row = dict(name=name, kind=kernel_kind(name), raw_metrics=values, raw_units=units)
        for label, (metric, factors) in METRICS.items():
            value = values.get(metric)
            if value in (None, "", "n/a"):
                continue
            if units[metric] not in factors:
                raise ValueError((path, metric, units[metric]))
            row[label] = float(value.replace(",", "")) * factors[units[metric]]
        result.append(row)
    return result


def union_ns(intervals: list[tuple[int, int]]) -> int:
    end = -1
    total = 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def analyze_nsys(path: Path) -> list[dict]:
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    strings = {r["id"]: r["value"] for r in db.execute("SELECT * FROM StringIds")}
    ranges = []
    for r in db.execute("SELECT * FROM NVTX_EVENTS WHERE end IS NOT NULL ORDER BY start"):
        label = r["text"] or strings.get(r["textId"], "")
        if label.startswith("op_capture/"):
            ranges.append((label, r["start"], r["end"], r["globalTid"]))
    if not ranges:
        raise ValueError("no op_capture NVTX windows in Nsight Systems export")
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    by_correlation = defaultdict(list)
    for source in db.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        by_correlation[source["correlationId"]].append(dict(source))
    result = []
    for label, begin, end, tid in ranges:
        _, case, regime, execution = label.split("/")
        # CPU NVTX and GPU timestamp alignment can move a boundary kernel outside
        # the NVTX interval. Attribute by the launching API, including graph nodes.
        api_rows = list(db.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start>=? AND end<=? AND globalTid=?", (begin, end, tid)))
        correlations = {r["correlationId"] for r in api_rows}
        kernels = []
        for correlation in correlations:
            for r in by_correlation.get(correlation, []):
                if r["globalPid"] != (tid & ~0xffffff):
                    continue
                name = strings.get(r.get("demangledName"), strings.get(r.get("shortName"), "unknown"))
                kernels.append(dict(start=r["start"], end=r["end"], name=name,
                                    kind=kernel_kind(name), correlation_id=correlation))
        kernels.sort(key=lambda r: r["start"])
        if not kernels:
            raise ValueError(f"no GPU kernels in {label}")
        spans = [(r["start"], r["end"]) for r in kernels]
        kernel_sum = sum(stop-start for start, stop in spans)
        by_kind = Counter()
        for r in kernels:
            by_kind[r["kind"]] += (r["end"]-r["start"])/1000
        apis = Counter()
        for r in api_rows:
            apis[strings.get(r["nameId"], "unknown")] += 1
        copy_count = 0
        for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
            if table in tables:
                copy_count += sum(r[0] in correlations for r in db.execute(f"SELECT correlationId FROM {table}"))
        span = max(r["end"] for r in kernels)-min(r["start"] for r in kernels)
        busy = union_ns(spans)
        operations = 3*(32 if regime == "resident" else 28)
        result.append(dict(case=case, regime=regime, execution=execution,
                           window_ms=(end-begin)/1e6, kernel_calls=len(kernels), operations=operations,
                           kernel_us_per_operation=kernel_sum/1000/operations,
                           gpu_span_ms=span/1e6, between_kernel_idle_ms=(span-busy)/1e6,
                           active_pct_in_gpu_span=busy/span*100, copies_or_memsets=copy_count,
                           kernel_time_by_kind_us=dict(by_kind), runtime_api_calls=dict(apis),
                           kernels=kernels))
    meta_path = path.with_suffix(".meta.json")
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())
        for r in result:
            meta, = [m for m in metadata["rows"] if m["case"]["id"] == r["case"] and m["regime"] == r["regime"]]
            choice = meta["choice"]
            stages = 2 if choice["mode"] == "fp8" or (choice["mode"] == "decode" and choice["config"] in (6, 7, 8, 9, 10, 12, 13, 23, 31, 32, 33, 34, 35)) else 1
            if r["operations"] != meta["invocations"] or r["kernel_calls"] != stages*meta["invocations"]:
                raise ValueError(f"missing/extra kernels in {r['case']} {r['regime']}: {r['kernel_calls']}")
    db.close()
    return result


def link(path: Path, report: Path) -> str:
    return Path(os.path.relpath(path, report.parent)).as_posix()


def generate(root: Path) -> None:
    from benchmarks.cuda_operator_profile import write_json
    timing = json.loads((root / "timing.json").read_text())
    timelines = []
    for execution in ("eager", "graph"):
        rows = analyze_nsys(root / f"nsys_{execution}.sqlite")
        write_json(root / f"nsys_{execution}.analysis.json", rows)
        timelines += rows
    raw_counters = [json.loads(p.read_text()) for p in sorted((root / "ncu").glob("*.analysis.json"))]
    if not raw_counters:
        raise ValueError("no NCU captures")
    manifest = json.loads((root / "manifest.json").read_text())
    expected = {c["id"] for c in manifest["cases"]}
    if {r["case"]["id"] for r in timing["rows"]} != expected:
        raise ValueError("incomplete timing cases")
    for execution in ("eager", "graph"):
        windows = [(r["case"], r["regime"]) for r in timelines if r["execution"] == execution]
        if set(windows) != {(c, regime) for c in expected for regime in ("resident", "rotating28")} or len(windows) != 2*len(expected):
            raise ValueError(f"incomplete or duplicate {execution} trace windows")
    cases = {r["case"]["id"]: r["case"] for r in timing["rows"]}
    timings = []
    for case in cases:
        for regime in ("resident", "rotating28"):
            rows = [r for r in timing["rows"] if r["case"]["id"] == case and r["regime"] == regime]
            if {r["seed"] for r in rows} != {17, 42, 99} or len(rows) != 3:
                raise ValueError(f"incomplete independent timings: {case} {regime}")
            medians = [r["median_us"] for r in rows]
            m, n, k = cases[case]["shape"]
            us = statistics.median(medians)
            timings.append(dict(case=case, regime=regime, us=us, min_seed_us=min(medians),
                                max_seed_us=max(medians), effective_tflops=2*m*n*k/us/1e6,
                                choice=rows[0]["choice"]))
    summaries = []
    for case in sorted({r["case"]["id"] for r in raw_counters}):
        for cache in ("none", "all"):
            captures = [r for r in raw_counters if r["case"]["id"] == case and r["cache"] == cache]
            if not captures:
                raise ValueError(f"missing counter cache regime: {case} {cache}")
            if {r["repeat"] for r in captures} != set(range(manifest["counter_repeats"])) or len(captures) != manifest["counter_repeats"]:
                raise ValueError(f"incomplete or duplicate counters: {case} {cache}")
            names = {tuple(k["name"] for k in r["kernels"]) for r in captures}
            if len(names) != 1:
                raise ValueError(f"kernel sequence changed: {case} {cache}")
            for i, name in enumerate(next(iter(names))):
                rows = [r["kernels"][i] for r in captures]
                values = {key: statistics.median(r[key] for r in rows) for key in METRICS if all(key in r for r in rows)}
                summaries.append(dict(case=case, cache=cache, repeats=len(captures), name=name,
                                      kind=rows[0]["kind"], **values))
    data = dict(timing=timings, nsys=timelines, ncu=summaries,
                attribution="CUDA API correlation ID + process/thread; complete stage counts checked")
    write_json(root / "summary.json", data)
    report = root / "REPORT.md"
    lines = ["# 自研 CUDA 算子独立性能与 Nsight 诊断", "",
             f"GPU：{timing['environment']['device']}；PyTorch {timing['environment']['torch']}；CUDA {timing['environment']['cuda']}。",
             f"本次覆盖 {len(cases)} 个算例、{len(timing['rows'])} 组种子/工作集计时、2 份 Nsight Systems 时间线和 {len(raw_counters)} 份 Nsight Compute 原生报告。", "",
             "## 测量边界", "",
             "- 正式计时和 `op_capture/...` NVTX 范围只执行原生自研 CUDA kernel；不加载模型、不调度请求、不测 cuBLAS 或 Triton 性能。参考 FP64 运算、随机初始化、编译和权重转换仅在范围外用于准备/正确性检查。完整 nsys 文件仍保留这些范围外活动。",
             "- Qwen BF16 在设备匹配时使用当前 cuda_decode 冻结配置；FP16 使用现有未匹配 selector 时的默认分派。DeepSeek 仅测试真实矩阵尺寸的独立 FP16/BF16 kernel，不代表已接入 INT4 MoE 推理。FP8 是 pack_rows + native GEMM 的实验路径。",
             "- resident：同一权重、32 次算子/图；rotating28：28 份独立权重各一次/图。后者是工作集控制，不模拟 MoE 路由或整个模型。3 个种子、每个9轮、按种子中位数再取中位数；输入输出预分配。Decode Split-K 内部 partial 分配与实际 API 一致，Graph 捕获后复用。",
             "- 正确性检查首尾权重及首尾最多8行，检查全部输出有限性；FP16/BF16 relative L2 门限为 .001/.006。FP8 分别记录量化误差与解码后输入的运算误差，不建立模型质量结论。",
             "- NCU 同时采集操作的所有 kernel（包括 Split-K finish、FP8 pack），不只取第一个。cache-control=none 在同权重预热后采集，all 每 pass 清缓存；clock-control=none。保留原始 CSV 单位，warp-state ratio 不是耗时百分比。", "",
             "## 独立计时（非 profiler）", "",
             "| 算例 | M,N,K | 路径/cfg | resident μs | rotating28 μs | rotating28 有效 TFLOP/s |",
             "|---|---|---|---:|---:|---:|"]
    for case, detail in cases.items():
        rr = next(r for r in timings if r["case"] == case and r["regime"] == "resident")
        ss = next(r for r in timings if r["case"] == case and r["regime"] == "rotating28")
        choice = rr["choice"]
        lines.append(f"| {case} | {','.join(map(str,detail['shape']))} | {choice['mode']}/{choice['config']} | {rr['us']:.3f} | {ss['us']:.3f} | {ss['effective_tflops']:.3f} |")
    lines += ["", "有效 TFLOP/s 按逻辑 2MNK 计算，不计 padding 或 Split-K 额外运算；FP8 时延包含激活量化。没有调库对照，因此本报告不声称相对 cuBLAS 的效率。", "",
              "## Nsight Systems：启动间隙与执行方式", "",
              "按范围内同线程 CUDA API 的 correlation ID 归属 GPU kernel（包括 Graph 节点），并核对完整调用次数；不按 GPU 时间是否完全落在 CPU NVTX 范围内过滤，避免时间对齐误差漏计。表中 idle 仅为首个 kernel 开始至最后一个结束之间无 GPU kernel 的区间；活动比例不是 occupancy。NVTX wall time 含同步等待，不能称为纯 CPU 计算。", "",
              "| 算例 | 权重 | 执行 | kernel 数 | kernel μs/操作 | 首尾间 idle ms | GPU span 活动% |",
              "|---|---|---|---:|---:|---:|---:|"]
    for r in timelines:
        lines.append(f"| {r['case']} | {r['regime']} | {r['execution']} | {r['kernel_calls']} | {r['kernel_us_per_operation']:.3f} | {r['between_kernel_idle_ms']:.3f} | {r['active_pct_in_gpu_span']:.1f} |")
    lines += ["", "FP8 pack、Split-K finish 的时间和 API 次数保留于 summary.json；eager/graph 的差异用于判断启动间隙，正式时延仍以上面的无 profiler 计时为准。", "",
              "## Nsight Compute：每个 kernel 的计数器中位数", "",
              "none=同权重预热，all=逐 pass 清缓存。μs 是 NCU replay 时间，不能当独立 benchmark；各 kernel 的缓存状态也不等于连续流水执行。", "",
              "| 算例 | 缓存 | kernel | 次数 | NCU μs | DRAM GB/s | DRAM% | L2 hit% | Tensor% | occupancy% | eligible | load conflicts | grid | regs |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(r, key, places=2):
        return f"{r[key]:.{places}f}" if key in r else "—"
    for r in summaries:
        lines.append(f"| {r['case']} | {r['cache']} | {r['kind']} | {r['repeats']} | " +
                     " | ".join(fmt(r, key, places) for key, places in
                                (("us",3),("dram_gbs",2),("dram_pct",2),("l2_hit_pct",2),("tensor_pct",2),
                                 ("occupancy_pct",2),("eligible",3),("bank_load",0),("grid",0),("registers",0)))+" |")
    lines += ["", "## 下一步优化方向（按证据分类，不把单一指标当因果）", ""]
    for r in summaries:
        if r["cache"] != "all" or r["kind"] in ("pack_rows_kernel", "finish_kernel"):
            continue
        notes = []
        if r.get("dram_pct",0) >= 70:
            notes.append("冷缓存 DRAM 吞吐较高：优先减少权重字节/重复读取、提高批内复用；shared 布局是否有效需看 rotating28")
        if r.get("grid",1e9) < timing["environment"]["sm_count"]*2 and r.get("eligible",1) < .5:
            notes.append("grid 与 eligible warp 较少：排查并行度、依赖隐藏，尝试调整 tile/Split-K，同时计入 finish 与重复读取")
        if not notes:
            notes.append("结合 Tensor 活动、原始 warp-state 和实际 grid 检查流水与归约，不能只看 occupancy")
        if r.get("bank_load",0) > 1000:
            notes.append("存在 shared-load conflicts：用相同 tile 的布局对照验证，不能把冲突次数直接换算成耗时")
        lines.append(f"- **{r['case']} / {r['kind']}**："+"；".join(notes)+"。")
    fp8 = [r for r in timelines if "fp8" in r["case"] and r["execution"] == "graph"]
    for r in fp8:
        by_kind = r["kernel_time_by_kind_us"]
        fraction = by_kind.get("pack_rows_kernel",0)/sum(by_kind.values())*100
        lines.append(f"- **{r['case']} / {r['regime']}**：FP8 pack 占本操作 kernel 时间 {fraction:.1f}%；优化 GEMM 前同时评估量化 launch 和低行数归约开销。")
    lines += ["", "这些方向是下一轮对照实验的候选，不是本轮已经证明的优化收益。FP16/BF16、warm/cold、eager/graph 必须分开判断。", "",
              "## 原生报告与复现", "",
              f"- [Eager 时间线]({link(root/'nsys_eager.nsys-rep',report)})",
              f"- [CUDA Graph 时间线]({link(root/'nsys_graph.nsys-rep',report)})",
              f"- [汇总数据]({link(root/'summary.json',report)})；每份 NCU 的 .ncu-repz/.ncu-rep、CSV、日志、准确度及命令在 ncu/。", "",
              "使用 Nsight Systems 打开 .nsys-rep，过滤 `op_capture/`，按算例/权重模式查看；使用兼容的 Nsight Compute UI 打开 ncu/ 中原生报告。初始化参考核不属于正式测量。", "",
              "```bash",
              "bash benchmarks/run_qwen_cuda.sh --operator-profile --output profiles/cuda_operator_NEW",
              "# 分阶段：--mode timing / nsys / ncu / report；已有成功采集可 --resume",
              "# 缩小范围：--suite qwen --dtypes bf16 --no-fp8",
              "# 单算例：--case qwen_bf16_m8_down",
              "```", ""]
    report.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"wrote {report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    generate(parser.parse_args().root)
