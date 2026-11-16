#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
GEMM_PYTHON="${GEMM_PYTHON:-/home/zjq/miniconda3/envs/nano-vllm/bin/python}"
export PATH="$CUDA_HOME/bin:$(dirname "$GEMM_PYTHON"):$PATH"
export MAX_JOBS="${MAX_JOBS:-2}"
if [[ "${1:-}" == "--validate" ]]; then
    for seed in 42 17 7; do
        "$GEMM_PYTHON" -m benchmarks.cuda_gemm_bench \
            --tuning-report results/cuda_gemm_20261115_tune.json \
            --seed "$seed" --iterations 100 --repeats 7 \
            --output "results/cuda_gemm_20261115_seed${seed}.json"
    done
elif [[ "${1:-}" == "--fp8-ablation" ]]; then
    exec "$GEMM_PYTHON" -m benchmarks.cuda_fp8_gemm_ablation
elif [[ "${1:-}" == "--fp8-pytest" ]]; then
    exec "$GEMM_PYTHON" -m pytest tests/test_cuda_fp8_gemm.py -q
elif [[ "${1:-}" == "--fp8-profile" ]]; then
    shift
    exec "$GEMM_PYTHON" -m benchmarks.cuda_fp8_gemm_profile "$@"
elif [[ "${1:-}" == "--fp8" ]]; then
    shift
    exec "$GEMM_PYTHON" -m benchmarks.cuda_fp8_gemm_bench "$@"
elif [[ "${1:-}" == "--counter-study" ]]; then
    shift
    exec "$GEMM_PYTHON" -m benchmarks.cuda_gemm_counter_study "$@"
elif [[ "${1:-}" == "--ncu-diagnose" ]]; then
    shift
    exec "$GEMM_PYTHON" -m benchmarks.ncu_diagnose "$@"
elif [[ "${1:-}" == "--profile" ]]; then
    shift
    exec "$GEMM_PYTHON" -m benchmarks.cuda_gemm_profile "$@"
elif [[ "${1:-}" == "--pytest" ]]; then
    exec "$GEMM_PYTHON" -m pytest tests/test_cuda_gemm.py -q
elif [[ "${1:-}" == "--sanitize" ]]; then
    exec "$CUDA_HOME/bin/compute-sanitizer" --error-exitcode 1 --tool "${2:-memcheck}" \
        "$GEMM_PYTHON" -m benchmarks.cuda_gemm_bench --check-only
else
    exec "$GEMM_PYTHON" -m benchmarks.cuda_gemm_bench "$@"
fi
