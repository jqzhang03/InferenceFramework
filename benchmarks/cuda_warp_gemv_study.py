"""M1-4 warp-owned GEMV: paired graph timings, held-out seeds and NCU controls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

from benchmarks.cuda_operator_profile import Workload, environment, resolve_choice, run_logged, write_json
from benchmarks.cuda_pipeline_study import label, paired

ROOT = Path(__file__).resolve().parents[1]
MODULE = "benchmarks.cuda_warp_gemv_study"


def cases() -> list[dict]:
    shapes = {"qkv": (4096, 1024), "o": (1024, 2048),
              "gate_up": (6144, 1024), "down": (1024, 3072)}
    return [dict(id=f"qwen_{dtype}_m{m}_{role}", model="qwen", role=role,
                 dtype=dtype, shape=[m, n, k])
            for dtype in ("bf16", "fp16") for m in range(1, 5)
            for role, (n, k) in shapes.items()]


def snapshot(output: Path) -> dict:
    path = output / "snapshot.json"
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (ROOT / "nanovllm/layers/csrc").glob("cuda_*.cu")}
    if path.exists():
        data = json.loads(path.read_text())
        if data["cuda_sha256"] != hashes:
            raise ValueError("CUDA source changed; preserve results and use a fresh output")
        return data
    env = environment()
    data = dict(environment=env, cuda_sha256=hashes,
                baselines={c["id"]: resolve_choice(c, env["device"]) for c in cases()})
    write_json(path, data)
    return data


def shared_choice(case: dict) -> dict:
    return dict(mode="decode", config=14 if case["shape"][0] == 1 else 17)


def screen(output: Path) -> None:
    state = snapshot(output)
    path = output / "screen.json"
    result = json.loads(path.read_text()) if path.exists() else dict(rows=[])
    done = {r["case"]["id"] for r in result["rows"]}
    for case in cases():
        if case["id"] in done:
            continue
        base = state["baselines"][case["id"]]
        choices = [base, dict(mode="gemv", config=0), shared_choice(case)]
        choices += [dict(mode="decode", config=c) for c in range(38, 50)]
        choices = list({label(c): c for c in choices}.values())
        resident = paired(case, choices, "resident", 17, 9)
        new = sorted((r for r in resident if r["choice"]["mode"] == "decode"
                      and r["choice"]["config"] >= 38), key=lambda r: r["us"])
        shortlist = [base, dict(mode="gemv", config=0), shared_choice(case)] + [r["choice"] for r in new[:3]]
        shortlist = list({label(c): c for c in shortlist}.values())
        rotating = paired(case, shortlist, "rotating28", 17, 11)
        winner = min((r for r in rotating if r["choice"]["mode"] == "decode"
                      and r["choice"]["config"] >= 38), key=lambda r: r["us"])["choice"]
        result["rows"].append(dict(case=case, baseline=base, winner=winner,
                                   resident=resident, rotating=rotating))
        write_json(path, result)
        print("SCREEN", case["id"], label(winner), flush=True)


def validate(output: Path) -> None:
    snapshot(output)
    path = output / "validation.json"
    result = json.loads(path.read_text()) if path.exists() else dict(rows=[])
    done = {r["case"]["id"] for r in result["rows"]}
    for row in json.loads((output / "screen.json").read_text())["rows"]:
        case = row["case"]
        if case["id"] in done:
            continue
        choices = [row["baseline"], row["winner"], dict(mode="gemv", config=0), shared_choice(case)]
        choices = list({label(c): c for c in choices}.values())
        measurements = []
        for seed in (42, 99, 123):
            regimes = ("resident", "rotating28") if seed == 99 else ("rotating28", "resident")
            for regime in regimes:
                measurements += paired(case, choices, regime, seed, 13)
        ratios = {}
        for regime in ("resident", "rotating28"):
            ratios[regime] = []
            for seed in (42, 99, 123):
                times = {r["label"]: r["us"] for r in measurements if r["seed"] == seed and r["regime"] == regime}
                ratios[regime].append(times[label(row["winner"])]/times[label(row["baseline"])])
        accepted = (statistics.median(ratios["rotating28"]) <= .97
                    and max(ratios["rotating28"]) <= 1.02 and max(ratios["resident"]) <= 1.05)
        result["rows"].append(dict(case=case, baseline=row["baseline"], winner=row["winner"],
                                   ratios=ratios, accepted=accepted, measurements=measurements))
        write_json(path, result)
        print("VALID", case["id"], accepted, round(statistics.median(ratios["rotating28"]), 4), flush=True)
    for dtype in ("bf16", "fp16"):
        filename = "cuda_decode_sm120.json" if dtype == "bf16" else "cuda_decode_sm120_fp16.json"
        bundled = json.loads((ROOT / "nanovllm/layers" / filename).read_text())
        selectors = dict(bundled["selectors"])
        for r in result["rows"]:
            if r["accepted"] and r["case"]["dtype"] == dtype:
                selectors[",".join(map(str, r["case"]["shape"]))] = r["winner"]
        write_json(output / f"deployment_{dtype}.json", dict(device=snapshot(output)["environment"]["device"],
                   backend="cuda_decode", dtype=dtype, selectors=selectors))


def probe(args: argparse.Namespace) -> None:
    import torch
    case = next(c for c in cases() if c["id"] == args.case)
    work = Workload(case, "resident", 42, json.loads(args.choice))
    check = work.validate()
    for _ in range(10):
        work.functions[0]()
    torch.cuda.synchronize()
    write_json(args.meta, dict(case=case, choice=work.choice, accuracy=check))
    torch.cuda.profiler.start()
    work.functions[0]()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


def counters(output: Path) -> None:
    from benchmarks.cuda_gemm_profile import select_ncu
    from benchmarks.cuda_operator_profile_report import read_ncu
    snapshot(output)
    tool = select_ncu()
    folder = output / "ncu"
    folder.mkdir(exist_ok=True)
    rows = json.loads((output / "validation.json").read_text())["rows"]
    for row in rows:
        if row["case"]["id"] not in ("qwen_bf16_m1_down", "qwen_bf16_m4_down"):
            continue
        for variant, choice in (("shared", shared_choice(row["case"])), ("warp", row["winner"])):
            for rep in range(3):
                prefix = folder / f"{row['case']['id']}_{variant}_r{rep}"
                native = list(folder.glob(prefix.name + ".ncu-rep*"))
                if not native:
                    command = [tool, "--profile-from-start", "off", "--replay-mode", "kernel",
                               "--cache-control", "none", "--clock-control", "none"]
                    for section in ("LaunchStats", "Occupancy", "SpeedOfLight", "SchedulerStats",
                                    "WarpStateStats", "MemoryWorkloadAnalysis", "MemoryWorkloadAnalysis_Tables"):
                        command += ["--section", section]
                    command += ["--metrics", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
                                "-o", str(prefix), sys.executable, "-m", MODULE, "--mode", "probe",
                                "--output", str(output), "--case", row["case"]["id"],
                                "--choice", json.dumps(choice), "--meta", str(prefix)+".meta.json"]
                    print("NCU", prefix.name, flush=True)
                    run_logged(command, prefix)
                    native = list(folder.glob(prefix.name + ".ncu-rep*"))
                if len(native) != 1 or json.loads(prefix.with_suffix(".status.json").read_text())["returncode"]:
                    raise ValueError(f"failed native capture {prefix}")
                csv_path = prefix.with_suffix(".csv")
                if not csv_path.exists():
                    run = subprocess.run([tool, "--import", str(native[0]), "--page", "raw", "--csv"],
                                         capture_output=True, text=True, check=True)
                    csv_path.write_text(run.stdout, encoding="utf-8")
                write_json(prefix.with_suffix(".analysis.json"), dict(case=row["case"], variant=variant,
                           choice=choice, repeat=rep, native=str(native[0]), kernels=read_ncu(csv_path)))


def report(output: Path) -> None:
    rows = json.loads((output / "validation.json").read_text())["rows"]
    lines = ["# Warp-owned GEMV M1-4", "", "Seed17 selection; held-out42/99/123, 13 paired graph rounds; resident32 / rotating28. Outputs preallocated. No cuBLAS or model-throughput claim.", "",
             "Acceptance: rotating median >=3% gain, every rotating regression <=2%, every resident regression <=5%. NCU replay times are diagnostic only.", "",
             "| case | baseline | warp candidate | resident old/new us | rotating old/new us | rotating GEMV/shared us | paired rotating change % | accepted |",
             "|---|---|---|---:|---:|---:|---:|---|"]
    for row in rows:
        times = [[statistics.median(r["us"] for r in row["measurements"] if r["regime"] == regime and r["choice"] == choice)
                  for choice in (row["baseline"], row["winner"])] for regime in ("resident", "rotating28")]
        controls = [statistics.median(r["us"] for r in row["measurements"] if r["regime"] == "rotating28" and r["choice"] == c)
                    for c in (dict(mode="gemv", config=0), shared_choice(row["case"]))]
        lines.append(f"| {row['case']['id']} | {label(row['baseline'])} | {label(row['winner'])} | " +
                     " | ".join(f"{a:.3f}/{b:.3f}" for a,b in times) +
                     f" | {controls[0]:.3f}/{controls[1]:.3f} | {(statistics.median(row['ratios']['rotating28'])-1)*100:.2f} | {row['accepted']} |")
    captures = [json.loads(p.read_text()) for p in sorted((output / "ncu").glob("*.analysis.json"))]
    if captures:
        lines += ["", "## NCU controls (three captures per condition; warm cache)", "",
                  "Shared static/dynamic allocations exclude driver-reserved shared memory. Barrier ratio is stalled warps per issue-active cycle, not a time percentage.", "",
                  "| case | variant | registers/thread | static shared | dynamic shared | load conflicts | barrier ratio |",
                  "|---|---|---:|---|---|---:|---:|"]
        for case, variant in sorted({(r["case"]["id"], r["variant"]) for r in captures}):
            runs = [r for r in captures if (r["case"]["id"], r["variant"]) == (case, variant)]
            if len(runs) != 3 or any(len(r["kernels"]) != 1 for r in runs):
                raise ValueError(f"incomplete NCU control {case}/{variant}")
            kernels = [r["kernels"][0] for r in runs]
            raw, units = kernels[0]["raw_metrics"], kernels[0]["raw_units"]
            alloc = [raw[k] + " " + units[k] for k in ("launch__shared_mem_per_block_static", "launch__shared_mem_per_block_dynamic")]
            barrier = statistics.median(float(k["raw_metrics"]["smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio"].replace(",", "")) for k in kernels)
            lines.append(f"| {case} | {variant} | {statistics.median(k['registers'] for k in kernels):.0f} | {alloc[0]} | {alloc[1]} | {statistics.median(k['bank_load'] for k in kernels):.0f} | {barrier:.6f} |")
    (output / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("REPORT", output / "REPORT.md", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "screen", "validate", "ncu", "probe", "report"), default="all")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case")
    parser.add_argument("--choice")
    parser.add_argument("--meta", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode in ("all", "screen"):
        screen(args.output)
    if args.mode in ("all", "validate"):
        validate(args.output)
    if args.mode in ("all", "ncu"):
        counters(args.output)
    if args.mode == "probe":
        probe(args)
    if args.mode in ("all", "report"):
        report(args.output)


if __name__ == "__main__":
    main()
