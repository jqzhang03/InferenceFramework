"""Counter and SASS verification for the frozen native FP8 GEMM benchmark."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def main() -> None:
    from benchmarks.cuda_gemm_profile import select_ncu
    from benchmarks.cuda_gemm_counter_study import read_ncu_csv

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=Path("results/cuda_fp8_gemm_grouped_20261116.json"))
    parser.add_argument("--output-dir",type=Path,default=Path("results/cuda_fp8_grouped_profile_20261116"))
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    tool=select_ncu()
    data=json.loads(args.input.read_text())
    captures=[]
    for row in data["results"]:
        m,n,k=row["shape_mnk"]
        if m!=n or m not in (1024,4096) or k!=m: continue
        for backend in ("cuda","cublas"):
            prefix=args.output_dir/f"{backend}_{m}"
            command=[tool,"--profile-from-start","off","--launch-count","1","--clock-control","none","--cache-control","none"]
            for section in ("LaunchStats","Occupancy","SpeedOfLight","SchedulerStats","WarpStateStats","MemoryWorkloadAnalysis","ComputeWorkloadAnalysis"):
                command += ["--section",section]
            command += ["--force-overwrite","-o",str(prefix),sys.executable,"-m","benchmarks.cuda_fp8_gemm_bench",
                        "--trace","--backend",backend,"--shapes",f"{m},{n},{k}","--tuning-report",str(args.input)]
            run=subprocess.run(command,capture_output=True,text=True)
            Path(str(prefix)+".log").write_text(run.stdout+run.stderr,encoding="utf-8")
            if run.returncode: raise RuntimeError(f"FP8 counter capture failed: {prefix}.log")
            report=max(args.output_dir.glob(prefix.name+".ncu-rep*"),key=lambda p:p.stat().st_mtime_ns)
            exported=subprocess.run([tool,"--import",str(report),"--page","raw","--csv"],capture_output=True,text=True,check=True)
            csv=Path(str(prefix)+".csv");csv.write_text(exported.stdout,encoding="utf-8")
            captures.append({"shape_mnk":[m,n,k],"backend":backend,"command":command,"returncode":run.returncode,
                             "report":str(report),"csv":str(csv),**read_ncu_csv(csv)})
            (args.output_dir/"counters.json").write_text(json.dumps(captures,indent=2)+"\n",encoding="utf-8")
            print(f"FP8 counters collected: {backend} {m}",flush=True)
    from benchmarks.cuda_fp8_gemm_bench import load_extension
    ext=load_extension()
    sass=subprocess.run(["cuobjdump","--dump-sass",ext.__file__],capture_output=True,text=True,check=True).stdout
    (args.output_dir/"sass.txt").write_text(sass,encoding="utf-8")
    instructions={kind:[line.strip() for line in sass.splitlines() if re.search(r"\b"+kind+r"\b",line)]
                  for kind in ("QMMA","LDSM","LDGSTS")}
    if not instructions["QMMA"] or not any("E4M3.E4M3" in s for s in instructions["QMMA"]):
        raise RuntimeError("native FP8 QMMA not found in disassembly")
    summary={"extension":ext.__file__,"instruction_counts":{kind:len(lines) for kind,lines in instructions.items()},
             "examples":{kind:lines[:3] for kind,lines in instructions.items()}}
    (args.output_dir/"sass_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
