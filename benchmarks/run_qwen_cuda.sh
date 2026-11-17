#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:/home/zjq/miniconda3/envs/nano-vllm/bin:$PATH"
export CXX=/usr/bin/g++-13
export MAX_JOBS=2
if [[ "${1:-}" == "--pipeline-study" ]]; then
    shift
    exec python -m benchmarks.cuda_pipeline_study "$@"
fi
if [[ "${1:-}" == "--pipeline-pytest" ]]; then
    exec python -m pytest tests/test_cuda_pipeline.py tests/test_cuda_decode.py tests/test_cuda_linear.py -q
fi
if [[ "${1:-}" == "--operator-pytest" ]]; then
    exec python -m pytest tests/test_cuda_operator_profile.py -q
fi
if [[ "${1:-}" == "--operator-profile" ]]; then
    shift
    exec python -m benchmarks.cuda_operator_profile "$@"
fi
if [[ "${1:-}" == "--config-pytest" ]]; then
    exec python -m pytest tests/test_linear_backend_default.py -q
fi
if [[ "${1:-}" == "--cfg22-report" ]]; then
    shift
    exec python -m benchmarks.cfg22_ncu_report "$@"
fi
if [[ "${1:-}" == "--cfg22-pytest" ]]; then
    exec python -m pytest tests/test_cuda_cfg22_diag.py -q
fi
if [[ "${1:-}" == "--cfg22-ncu" ]]; then
    shift
    exec python -m benchmarks.cfg22_ncu "$@"
fi
if [[ "${1:-}" == "--full-nsys-report" ]]; then
    shift
    exec python -m benchmarks.qwen_decode_full_report "$@"
fi
if [[ "${1:-}" == "--decode-report" ]]; then
    shift
    exec python -m benchmarks.cuda_decode_report "$@"
fi
if [[ "${1:-}" == "--decode-study" ]]; then
    shift
    exec python -m benchmarks.cuda_decode_study "$@"
fi
if [[ "${1:-}" == "--decode-pytest" ]]; then
    exec python -m pytest tests/test_cuda_decode.py tests/test_cuda_linear.py -q
fi
if [[ "${1:-}" == "--pytest" ]]; then
    exec python -m pytest tests/test_cuda_linear.py -q
fi
if [[ "${1:-}" == "--report" ]]; then
    exec python -m benchmarks.qwen_cuda_linear_report
fi
if [[ "${1:-}" == "--nsys" ]]; then
    shift
    exec python -m benchmarks.qwen_cuda_nsys "$@"
fi
if [[ "${1:-}" == "--nsys-report" ]]; then
    shift
    exec python -m benchmarks.qwen_cuda_nsys_report "$@"
fi
exec python -m benchmarks.qwen_cuda_linear "$@"
