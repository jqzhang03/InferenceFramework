# Nsight Compute counter recovery — 2026-11-15

Hardware-counter collection is working on this RTX 5060 Ti / SM120 / WSL2 machine.
Changing the profiler version fixed the initialization failure; no driver,
registry, debugger-interface, or counter-permission settings were changed.

## Failure isolation

- `/usr/local/cuda/bin/ncu` resolves to Nsight Compute 2025.1.1. The standalone
  `ncu_probe.cu` succeeds without profiling, but this profiler fails before counter
  collection with `Failed to initialize the profiler: LibraryNotLoaded`, exit 9.
  This reproduces the failure independently of PyTorch, cuBLAS, and the GEMM kernel.
- Loader tracing shows the WSL bridge and its Windows-provided driver library are
  loaded. The existence of another Linux `libcuda` installation does not establish
  that it caused this failure; no driver packages were removed.
- The existing Windows Nsight Compute installation includes a Linux CLI at
  `/mnt/d/NsightCompute/target/linux-desktop-glibc_2_11_3-x64/ncu`, version
  **2026.3.1.0 (build 38829034)**. Using it on the same probe implementation and
  unchanged driver environment succeeds in **9 passes**, exit 0, with output
  validation passing and a `.ncu-repz` report produced.

The controlled tool substitution identifies an incompatibility in the old profiler
and current driver combination. It does not identify NVIDIA's internal failing
ABI or initialization check. There is no evidence here that counter access needed
to be enabled. This recovery does not validate the separately failing old Compute
Sanitizer installation.

## Real GEMM validation

All four custom-kernel cases completed collection and raw CSV export with exit 0:

| dtype | M=N=K | config | replay passes | active warp occupancy (%) | Tensor pipe active / elapsed (%) |
|---|---:|---:|---:|---:|---:|
| FP16 | 1024 | 10 | 21 | 22.93 | 70.54 |
| FP16 | 4096 | 17 | 22 | 32.60 | 93.04 |
| BF16 | 1024 | 10 | 21 | 22.71 | 69.76 |
| BF16 | 4096 | 17 | 21 | 32.60 | 95.66 |

Collected sections: LaunchStats, Occupancy, SpeedOfLight, SchedulerStats,
WarpStateStats, MemoryWorkloadAnalysis, and ComputeWorkloadAnalysis. The table
uses `sm__warps_active.avg.pct_of_peak_sustained_active` and
`sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` from the CSV.
Warp-stall and memory-workload metrics are also present. Occupancy here is
measured, unlike the static CUDA occupancy API limits in the original report.

These are single profiled launches after ten warmups, with profiler cache and
clock control disabled. They demonstrate working counters and lower measured
occupancy/Tensor activity in the small shape; they are not repeated performance
results or a causal percentage decomposition of the cuBLAS gap. No cuBLAS counter
comparison was collected in this recovery validation. The original benchmark and
attribution datasets remain unchanged.

## Project fix and reproduction

`cuda_gemm_profile.py` discovers installed Linux ncu versions and selects the
highest version instead of relying on the old CUDA-toolkit PATH entry. Explicit
`--ncu` takes priority over `NCU_BIN`, which takes priority over auto-discovery.
An invalid override raises an error. Reports are placed in a version directory;
logs, command/status JSON, raw CSV, and compressed reports are retained. Collection
and CSV export failures propagate a nonzero exit code.

```bash
bash benchmarks/run_cuda_gemm.sh --profile --mode tools
bash benchmarks/run_cuda_gemm.sh --ncu-diagnose --mode probe
bash benchmarks/run_cuda_gemm.sh --profile --mode ncu --dtype fp16 --size 1024 --config 10
bash benchmarks/run_cuda_gemm.sh --profile --mode ncu --dtype bf16 --size 4096 --config 17

# Reproduce the old failure without overwriting the successful probe report:
bash benchmarks/run_cuda_gemm.sh --ncu-diagnose --mode loader \
  --ncu /opt/nvidia/nsight-compute/2025.1.1/ncu \
  --output-dir results/ncu_diagnose/old2025
```

The probe compiler defaults to `/usr/bin/g++-13` with CUDA 12.8, avoiding the
incompatible GCC 14 host-math declarations encountered during setup. Override
`NCU_PROBE_CXX`, `CUDA_HOME`, and `GEMM_PYTHON` as needed on other machines.

Local ignored evidence:

- `results/ncu_diagnose/inventory.json`, `loader.log`, and `ld_debug.*.filtered`:
  original environment and failing library initialization.
- `results/ncu_diagnose/ncu2026/probe.log` and `probe_report.ncu-repz`:
  successful independent probe.
- `results/cuda_gemm_profile/ncu_2026_3_1/ncu_cuda_{fp16,bf16}_{1024,4096}_cfg*.{log,status.json,csv,ncu-repz}`:
  real GEMM collection and export evidence.

The four pure-Python selector tests cover version ordering, override precedence,
invalid override failure, and missing version diagnostics. Run them with
`python -m pytest tests/test_ncu_selection.py -q` in the WSL project environment.
