"""Paired native-CUDA tile/pipeline tuning, independent validation and NCU."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys

from benchmarks.cuda_operator_profile import Workload, environment, make_cases, resolve_choice, run_logged, write_json

MODULE = "benchmarks.cuda_pipeline_study"
ROOT = Path(__file__).resolve().parents[1]


def cases() -> list[dict]:
    return [c for c in make_cases("all", ["bf16", "fp16"], True)
            if c["dtype"] == "fp8" or c["shape"][0] == 8
            or (c["model"] == "qwen" and c["shape"][0] == 512 and c["role"] == "gate_up")]


def label(choice: dict) -> str:
    return f"{choice['mode']}_{choice['config']}" + ("_nosw" if choice.get("no_swizzle") else "")


def candidates(case: dict, baseline: dict) -> list[dict]:
    if case["dtype"] == "fp8":
        configs = [dict(mode="fp8", config=c) for c in range(34)]
    else:
        configs = [dict(mode="gemm", config=c) for c in range(36)]
        if case["shape"][0] <= 32:
            configs += [dict(mode="decode", config=c) for c in range(38) if c != 23]
            configs += [dict(mode="gemv", config=0)]
    return list({label(c): c for c in [baseline, *configs]}.values())


def paired(case: dict, choices: list[dict], regime: str, seed: int, rounds: int) -> list[dict]:
    import torch
    workloads, graphs, checks = {}, {}, {}
    for choice in choices:
        key = label(choice)
        w = Workload(case, regime, seed, choice)
        checks[key] = w.validate()
        w.run(); w.run(); torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            w.run()
        graph.replay(); torch.cuda.synchronize()
        workloads[key], graphs[key] = w, graph
    samples = {key: [] for key in graphs}
    rng = random.Random(seed + 731)
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(rounds):
        order = list(graphs)
        rng.shuffle(order)
        for key in order:
            begin.record(); graphs[key].replay(); end.record(); end.synchronize()
            samples[key].append(begin.elapsed_time(end)*1000/workloads[key].operations)
    result = [dict(choice=choice, label=label(choice), regime=regime, seed=seed,
                   us=statistics.median(samples[label(choice)]), samples_us=samples[label(choice)],
                   accuracy=checks[label(choice)]) for choice in choices]
    del graphs, workloads, graph, w
    gc.collect()
    return result


def snapshot(args: argparse.Namespace) -> dict:
    path = args.output / "snapshot.json"
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (ROOT / "nanovllm/layers/csrc").glob("cuda_*.cu")}
    if path.exists():
        data = json.loads(path.read_text())
        if data["cuda_sha256"] != hashes:
            raise ValueError("CUDA sources changed; preserve run and use a fresh output")
        return data
    env = environment()
    data = dict(environment=env, cuda_sha256=hashes,
                baselines={c["id"]: resolve_choice(c, env["device"]) for c in cases()},
                bundled_bf16=json.loads((ROOT / "nanovllm/layers/cuda_decode_sm120.json").read_text()))
    write_json(path, data)
    return data


def screen(args: argparse.Namespace) -> None:
    state = snapshot(args)
    path = args.output / "screen.json"
    data = json.loads(path.read_text()) if path.exists() else dict(rows=[])
    done = {r["case"]["id"] for r in data["rows"]}
    for case in cases():
        if case["id"] in done:
            continue
        baseline = state["baselines"][case["id"]]
        resident = paired(case, candidates(case, baseline), "resident", 17, 7)
        best = sorted(resident, key=lambda r: r["us"])
        shortlist = [baseline, *[r["choice"] for r in best[:3]]]
        for mode in ("gemm", "decode"):
            family = [r for r in best if r["choice"]["mode"] == mode]
            if family:
                shortlist.append(family[0]["choice"])
        new_decode = [r for r in best if r["choice"]["mode"] == "decode" and r["choice"]["config"] >= 24]
        if new_decode:
            shortlist.append(new_decode[0]["choice"])
        shortlist = list({label(c): c for c in shortlist}.values())
        streaming = paired(case, shortlist, "rotating28", 17, 9)
        winner = min(streaming, key=lambda r: r["us"])["choice"]
        row = dict(case=case, baseline=baseline, winner=winner, resident=resident, streaming=streaming)
        data["rows"].append(row)
        write_json(path, data)
        base_us = next(r["us"] for r in streaming if r["choice"] == baseline)
        win_us = min(r["us"] for r in streaming)
        print("SCREEN", case["id"], label(baseline), "->", label(winner), round(base_us,3), round(win_us,3), flush=True)


def validate(args: argparse.Namespace) -> None:
    data = json.loads((args.output / "screen.json").read_text())
    path = args.output / "validation.json"
    result = json.loads(path.read_text()) if path.exists() else dict(rows=[])
    done = {r["case"]["id"] for r in result["rows"]}
    for row in data["rows"]:
        case = row["case"]
        if case["id"] in done:
            continue
        choices = list({label(c): c for c in [row["baseline"], row["winner"]]}.values())
        measurements = []
        for seed in (42, 99, 123):
            regimes = ["resident", "rotating28"]
            random.Random(seed).shuffle(regimes)
            for regime in regimes:
                measurements += paired(case, choices, regime, seed, 11)
        ratios = {}
        for regime in ("resident", "rotating28"):
            ratios[regime] = []
            for seed in (42, 99, 123):
                rows = [r for r in measurements if r["regime"] == regime and r["seed"] == seed]
                times = {r["label"]: r["us"] for r in rows}
                ratios[regime].append(times[label(row["winner"])] / times[label(row["baseline"])])
        accept = (row["baseline"] != row["winner"] and statistics.median(ratios["rotating28"]) <= .97
                  and max(ratios["rotating28"]) <= 1.02 and max(ratios["resident"]) <= 1.05)
        result["rows"].append(dict(case=case, baseline=row["baseline"], winner=row["winner"],
                                   accepted=accept, ratios=ratios, measurements=measurements))
        write_json(path, result)
        print("VALIDATE", case["id"], label(row["winner"]), "accepted", accept,
              "stream ratios", [round(r,3) for r in ratios["rotating28"]], flush=True)
    freeze(args)


def freeze(args: argparse.Namespace) -> None:
    state = snapshot(args)
    result = json.loads((args.output / "validation.json").read_text())
    for dtype in ("bf16", "fp16", "fp8"):
        backend = "cuda_fp8" if dtype == "fp8" else "cuda_decode"
        base = dict(state["bundled_bf16"]["selectors"]) if dtype == "bf16" else {}
        selected = dict(base)
        for row in result["rows"]:
            c = row["case"]
            if c["model"] != "qwen" or c["dtype"] != dtype:
                continue
            key = ",".join(map(str, c["shape"]))
            base[key] = row["baseline"]
            selected[key] = row["winner"] if row["accepted"] else row["baseline"]
        common = dict(device=state["environment"]["device"], backend=backend, dtype=dtype,
                      selection_policy="seed17 screen; held-out42/99/123; rotating median >=3% gain; no >2% rotating or >5% resident regression")
        write_json(args.output / f"baseline_{dtype}.json", dict(common, selectors=base))
        write_json(args.output / f"deployment_{dtype}.json", dict(common, selectors=selected))


def probe(args: argparse.Namespace) -> None:
    import torch
    case = next(c for c in cases() if c["id"] == args.case)
    choice = json.loads(args.choice)
    work = Workload(case, "resident", 42, choice)
    accuracy = work.validate()
    for _ in range(10):
        work.functions[0]()
    torch.cuda.synchronize()
    write_json(args.meta, dict(case=case, choice=choice, accuracy=accuracy))
    torch.cuda.profiler.start()
    work.functions[0]()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


def counters(args: argparse.Namespace) -> None:
    from benchmarks.cuda_gemm_profile import select_ncu
    from benchmarks.cuda_operator_profile_report import read_ncu
    tool = select_ncu()
    folder = args.output / "ncu"
    folder.mkdir(exist_ok=True)
    validation = json.loads((args.output / "validation.json").read_text())
    both = {"qwen_bf16_m8_o", "qwen_bf16_m512_gate_up", "qwen_fp8_m128_gate_up"}
    cold = {"qwen_fp16_m8_down", "deepseek_bf16_m8_expert_gate_up"}
    targets = []
    for row in ([] if args.mode == "ablate-ncu" else validation["rows"]):
        case = row["case"]
        if case["id"] not in both | cold:
            continue
        for name in ("baseline", "winner"):
            for cache in (("none", "all") if case["id"] in both else ("all",)):
                targets.append((case, name, row[name], cache))
        if case["id"] == "qwen_bf16_m8_o" and row["winner"]["mode"] == "decode":
            for cache in ("none", "all"):
                targets.append((case, "winner_nosw", dict(row["winner"], no_swizzle=True), cache))
    if args.mode == "ablate-ncu":
        o_case = next(c for c in cases() if c["id"] == "qwen_bf16_m8_o")
        prefill = next(c for c in cases() if c["id"] == "qwen_bf16_m512_gate_up")
        for config in (1, 26, 28, 36):
            targets.append((o_case, f"controlled_cfg{config}", dict(mode="decode", config=config), "none"))
        targets.append((o_case, "controlled_cfg26_nosw", dict(mode="decode", config=26, no_swizzle=True), "none"))
        for cache in ("none", "all"):
            targets.append((prefill, "controlled_alt_a", dict(mode="gemm", config=28), cache))
        for case_id in ("qwen_fp8_m128_qkv", "qwen_fp16_m8_gate_up"):
            row = next(r for r in validation["rows"] if r["case"]["id"] == case_id)
            for variant in ("baseline", "winner"):
                targets.append((row["case"], variant, row[variant], "all"))
    for case, name, choice, cache in targets:
        for rep in range(3):
            prefix = folder / f"{case['id']}_{name}_{cache}_r{rep}"
            native = list(folder.glob(prefix.name + ".ncu-rep*"))
            if not native:
                command = [tool, "--target-processes", "all", "--profile-from-start", "off",
                           "--replay-mode", "kernel", "--cache-control", cache, "--clock-control", "none"]
                for section in ("LaunchStats", "Occupancy", "SpeedOfLight", "SchedulerStats", "WarpStateStats",
                                "MemoryWorkloadAnalysis", "MemoryWorkloadAnalysis_Tables", "ComputeWorkloadAnalysis"):
                    command += ["--section", section]
                command += ["--metrics", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
                            "-o", str(prefix), sys.executable, "-m", MODULE, "--mode", "probe", "--case", case["id"],
                            "--choice", json.dumps(choice), "--meta", str(prefix)+".meta.json", "--output", str(args.output)]
                print("NCU", prefix.name, flush=True)
                run_logged(command, prefix)
                native = list(folder.glob(prefix.name + ".ncu-rep*"))
            if len(native) != 1:
                raise ValueError(f"missing native report: {prefix}")
            status = json.loads(prefix.with_suffix(".status.json").read_text())
            if status["returncode"]:
                raise ValueError(f"failed capture: {prefix}")
            csv_path = prefix.with_suffix(".csv")
            if not csv_path.exists():
                run = subprocess.run([tool, "--import", str(native[0]), "--page", "raw", "--csv"], capture_output=True, text=True, check=True)
                csv_path.write_text(run.stdout, encoding="utf-8")
            write_json(prefix.with_suffix(".analysis.json"), dict(case=case, variant=name, choice=choice, cache=cache,
                       repeat=rep, kernels=read_ncu(csv_path), native=str(native[0])))


def ablate(args: argparse.Namespace) -> None:
    """Hold geometry fixed while changing warp work, stages, or shared layout."""
    specs = {
        "qwen_bf16_m8_o": [dict(mode="decode", config=c) for c in (1, 5, 26, 28, 36)] +
                             [dict(mode="decode", config=26, no_swizzle=True)],
        "qwen_bf16_m512_gate_up": [dict(mode="gemm", config=c) for c in (10, 0, 5, 28, 29)],
        "qwen_fp8_m128_gate_up": [dict(mode="fp8", config=c) for c in (3, 14, 16, 28, 29, 30)],
    }
    output = args.output / "ablation.json"
    if output.exists():
        raise ValueError("preserve existing ablation timings")
    rows = []
    for case in cases():
        if case["id"] not in specs:
            continue
        for seed in (42, 99, 123):
            for regime in ("resident", "rotating28"):
                values = paired(case, specs[case["id"]], regime, seed, 11)
                rows += [dict(case=case, **r) for r in values]
                write_json(output, dict(rows=rows))
        print("ABLATE", case["id"], flush=True)


def report(args: argparse.Namespace) -> None:
    result = json.loads((args.output / "validation.json").read_text())
    paths = sorted((args.output / "ncu").glob("*.analysis.json"))
    captures = [json.loads(p.read_text()) for p in paths]
    summary = []
    groups = sorted({(r["case"]["id"], r["variant"], r["cache"]) for r in captures})
    for case, variant, cache in groups:
        runs = [r for r in captures if (r["case"]["id"], r["variant"], r["cache"]) == (case, variant, cache)]
        if len(runs) != 3:
            raise ValueError(f"incomplete counter repeats {case} {variant} {cache}")
        for i, kernel in enumerate(runs[0]["kernels"]):
            keys = ("us", "dram_pct", "tensor_pct", "occupancy_pct", "eligible", "bank_load", "bank_store", "grid", "registers")
            summary.append(dict(case=case, variant=variant, cache=cache, choice=runs[0]["choice"],
                                kind=kernel["kind"], **{key: statistics.median(r["kernels"][i][key] for r in runs) for key in keys}))
    ablation_path = args.output / "ablation.json"
    ablations = json.loads(ablation_path.read_text())["rows"] if ablation_path.exists() else []
    write_json(args.output / "summary.json", dict(validation=result["rows"], ncu=summary, ablations=ablations))
    lines = ["# CUDA tile / pipeline paired study", "", "Independent graph timings; seed17 selection, held-out42/99/123; preallocated outputs; resident32 / rotating28 operations. No cuBLAS timing or model-throughput claim.", "",
             "| case | baseline | candidate | resident old/new us | rotating old/new us | rotating change % | accepted |",
             "|---|---|---|---:|---:|---:|---|"]
    for row in result["rows"]:
        times = {}
        for regime in ("resident", "rotating28"):
            times[regime] = [statistics.median(r["us"] for r in row["measurements"] if r["regime"] == regime and r["choice"] == choice)
                             for choice in (row["baseline"], row["winner"])]
        lines.append(f"| {row['case']['id']} | {label(row['baseline'])} | {label(row['winner'])} | " +
                     " | ".join(f"{a:.3f}/{b:.3f}" for a,b in times.values()) +
                     f" | {(statistics.median(row['ratios']['rotating28'])-1)*100:.2f} | {row['accepted']} |")
    lines += ["", "NCU replay durations below are diagnostic, not benchmark gains. all=flushed per pass, none=warmed without flush. Every operation retains all stages (including finish / FP8 pack). Raw units and native .ncu-repz stay in ncu/.", "",
              "| case | variant | cache | kernel | NCU us | DRAM% | Tensor% | occupancy% | eligible | load conflicts | grid | registers |",
              "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['case']} | {r['variant']} | {r['cache']} | {r['kind']} | " +
                     " | ".join(f"{r[k]:.3f}" for k in ("us", "dram_pct", "tensor_pct", "occupancy_pct", "eligible", "bank_load", "grid", "registers")) + " |")
    if ablations:
        lines += ["", "## Controlled ablations (three-seed unprofiled timings)", "",
                  "| case | choice | resident us | rotating28 us |", "|---|---|---:|---:|"]
        for case, choice in sorted({(r["case"]["id"], r["label"]) for r in ablations}):
            values = [statistics.median(r["us"] for r in ablations if r["case"]["id"] == case and r["label"] == choice and r["regime"] == regime)
                      for regime in ("resident", "rotating28")]
            lines.append(f"| {case} | {choice} | {values[0]:.3f} | {values[1]:.3f} |")
    (args.output / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("REPORT", args.output / "REPORT.md", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "screen", "validate", "ncu", "ablate", "ablate-ncu", "controls", "probe", "report"), default="all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case")
    parser.add_argument("--choice")
    parser.add_argument("--meta", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode in ("all", "screen"):
        screen(args)
    if args.mode in ("all", "validate"):
        validate(args)
    if args.mode in ("all", "ncu", "ablate-ncu"):
        counters(args)
    if args.mode in ("all", "controls"):
        if not (args.output / "ablation.json").exists():
            ablate(args)
        original_mode = args.mode
        args.mode = "ablate-ncu"
        counters(args)
        args.mode = original_mode
    if args.mode == "ablate":
        ablate(args)
    if args.mode == "probe":
        probe(args)
    if args.mode in ("all", "controls", "report"):
        report(args)


if __name__ == "__main__":
    main()
