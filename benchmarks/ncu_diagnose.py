"""Read-only inventory and standalone Nsight Compute counter probe on WSL."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import subprocess
from pathlib import Path


def inventory() -> dict:
    result={"uid":os.getuid(),"uname":os.uname(),
            "environment":{key:os.environ.get(key) for key in ("CUDA_HOME","LD_LIBRARY_PATH","LD_PRELOAD","PATH")}}
    for key,command in {
        "nvidia_smi":["nvidia-smi"],
        "ncu":["/usr/local/cuda/bin/ncu","--version"],
        "ldconfig":["ldconfig","-p"],
    }.items():
        run=subprocess.run(command,capture_output=True,text=True)
        out=run.stdout+run.stderr
        result[key]=out if key!="ldconfig" else "\n".join(line for line in out.splitlines() if any(s in line for s in ("cuda","nvidia","nvperf")))
    cuda=ctypes.CDLL("libcuda.so.1")
    result["cuInit"]=cuda.cuInit(0)
    version=ctypes.c_int()
    result["cuDriverGetVersion_status"]=cuda.cuDriverGetVersion(ctypes.byref(version))
    result["cuDriverGetVersion"]=version.value
    result["cuda_maps"]=[line for line in Path("/proc/self/maps").read_text().splitlines() if "libcuda" in line]
    result["debugger_batches"]=[str(p) for p in Path("/usr/local/cuda").rglob("*DebuggerInterface*")]
    result["wsl_driver_files"]=[str(p) for p in Path("/usr/lib/wsl").glob("drivers/*/*") if re.search(r"(perf|dbg|debug|cuda)",p.name,re.I)]
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=["inventory","probe","loader"],default="inventory")
    parser.add_argument("--ncu",help="explicit Linux CLI; default uses newest installed ncu")
    parser.add_argument("--output-dir",type=Path,default=Path("results/ncu_diagnose"))
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.mode=="inventory":
        result=inventory()
        (args.output_dir/"inventory.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(result,indent=2),flush=True)
        return
    binary=args.output_dir/"ncu_probe"
    compiler=Path(os.environ.get("NCU_PROBE_CXX","/usr/bin/g++-13"))
    subprocess.run([str(Path(os.environ["CUDA_HOME"])/"bin/nvcc"),"-ccbin",str(compiler),"-O2","-std=c++17","-arch=sm_120",
                    str(Path(__file__).with_suffix(".cu").with_name("ncu_probe.cu")),"-o",str(binary)],check=True)
    native=subprocess.run([str(binary)],capture_output=True,text=True)
    print(native.stdout+native.stderr,flush=True)
    if native.returncode:
        raise SystemExit(native.returncode)
    from benchmarks.cuda_gemm_profile import select_ncu
    tool=select_ncu(args.ncu)
    print(f"Selected ncu: {tool}",flush=True)
    command=[tool,"--clock-control","none","--cache-control","none","--launch-count","1",
             "--section","LaunchStats","--section","SpeedOfLight","--force-overwrite",
             "-o",str(args.output_dir/"probe_report"),str(binary)]
    env=os.environ.copy()
    if args.mode=="loader":
        env["LD_DEBUG"]="libs,files"
        env["LD_DEBUG_OUTPUT"]=str((args.output_dir/"ld_debug").resolve())
    run=subprocess.run(command,capture_output=True,text=True,env=env)
    (args.output_dir/f"{args.mode}.log").write_text(run.stdout+run.stderr,encoding="utf-8")
    print(run.stdout+run.stderr,flush=True)
    print(f"ncu exit={run.returncode}",flush=True)
    if args.mode=="loader":
        for path in sorted(args.output_dir.glob("ld_debug.*")):
            lines=path.read_text(errors="replace").splitlines()
            selected=[line for line in lines if re.search(r"(libcuda|libnvperf|libnvidia|not found|error:)",line,re.I)]
            (args.output_dir/(path.name+".filtered")).write_text("\n".join(selected)+"\n",encoding="utf-8")
            print(path.name,"\n"+"\n".join(selected[-70:]),flush=True)
    raise SystemExit(run.returncode)


if __name__=="__main__":
    main()
