#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:/home/zjq/miniconda3/envs/nano-vllm/bin:$PATH"
export CXX="${CXX:-/usr/bin/g++-13}"
export MAX_JOBS="${MAX_JOBS:-2}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PWD/profiles/.torch_extensions_w4a16}"
stage="${1:?stage required}"
root="${2:?output directory required}"
shift 2
mkdir -p "$root"
nsys="${NSYS_BIN:-/usr/local/bin/nsys}"
case "$stage" in
    baseline|nsys)
        if [[ "$stage" == baseline ]]; then
            prefix="$root/triton_baseline"
        else
            prefix="$root/model_pair"
        fi
        "$nsys" profile --trace=cuda,nvtx,osrt,cublas --sample=none --cpuctxsw=none \
            --cuda-memory-usage=true -o "$prefix" python -m benchmarks.deepseek_w4a16 \
            "$stage" --root "$root" --repeats 1 --trace "$@" 2>&1 | tee "$prefix.log"
        "$nsys" export --type=sqlite -o "$prefix.sqlite" "$prefix.nsys-rep" \
            > "$prefix.export.log" 2>&1
        ;;
    ncu)
        backend="${1:?triton or cuda required}"
        shift
        ncu="$(python -c 'from benchmarks.cuda_gemm_profile import select_ncu; print(select_ncu())')"
        mkdir -p "$root/ncu"
        cases=(q_decode q_prefill dense_down_decode dense_down_prefill expert_gate_decode
               expert_gate_prefill expert_down_decode expert_down_prefill)
        for cache in none all; do
            prefix="$root/ncu/${backend}_${cache}"
            "$ncu" --target-processes all --profile-from-start off --replay-mode kernel \
                --clock-control none --cache-control "$cache" \
                --section LaunchStats --section Occupancy --section SpeedOfLight \
                --section MemoryWorkloadAnalysis --section SchedulerStats \
                --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
                -o "$prefix" python -m benchmarks.deepseek_w4a16 ncu --root "$root" \
                --backend "$backend" --cases "${cases[@]}" "$@" 2>&1 | tee "$prefix.log"
            reports=("$prefix".ncu-rep*)
            [[ ${#reports[@]} == 1 && -f "${reports[0]}" ]]
            "$ncu" --import "${reports[0]}" --page raw --csv > "$prefix.csv"
        done
        ;;
    ncu_model)
        ncu="$(python -c 'from benchmarks.cuda_gemm_profile import select_ncu; print(select_ncu())')"
        mkdir -p "$root/ncu"
        for backend in triton cuda; do
            prefix="$root/ncu/live_$backend"
            "$ncu" --target-processes all --profile-from-start off --replay-mode application \
                --clock-control none --cache-control none --nvtx --nvtx-include 'w4a16_projection/' \
                --kernel-name-base function --kernel-name 'regex:^(gemm_int4_kernel|w4a16_gemv_kernel|w4a16_mma_kernel)$' \
                --section LaunchStats --metrics sm__warps_active.avg.pct_of_peak_sustained_active \
                -o "$prefix" \
                python -m benchmarks.deepseek_w4a16 ncu_model --root "$root" --backend "$backend" \
                --cases q_prefill dense_down_prefill expert_gate_prefill expert_down_prefill \
                        q_decode dense_down_decode expert_gate_decode expert_down_decode "$@" 2>&1 | tee "$prefix.log"
            reports=("$prefix".ncu-rep*)
            [[ ${#reports[@]} == 1 && -f "${reports[0]}" ]]
            "$ncu" --import "${reports[0]}" --page raw --csv > "$prefix.csv"
        done
        ;;
    fusion)
        python -m benchmarks.deepseek_w4a16_fusion --root "$root" "$@" 2>&1 | tee "$root/fusion.log"
        ;;
    fusion_nsys)
        prefix="$root/fusion"
        "$nsys" profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none -o "$prefix" \
            python -m benchmarks.deepseek_w4a16_fusion --root "$root" --trace "$@" 2>&1 | tee "$prefix.nsys.log"
        "$nsys" export --type=sqlite -o "$prefix.sqlite" "$prefix.nsys-rep" > "$prefix.export.log" 2>&1
        ;;
    bottleneck)
        python -m benchmarks.deepseek_bottleneck --root "$root" "$@" 2>&1 | tee "$root/timing.log"
        ;;
    bottleneck_nsys)
        prefix="$root/bottleneck"
        "$nsys" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none -o "$prefix" \
            python -m benchmarks.deepseek_bottleneck --root "$root" --trace "$@" 2>&1 | tee "$prefix.nsys.log"
        "$nsys" export --type=sqlite -o "$prefix.sqlite" "$prefix.nsys-rep" > "$prefix.export.log" 2>&1
        ;;
    pytest)
        python -m pytest tests/test_cuda_int4.py tests/test_moe_int4_segment.py "$@" 2>&1 | tee "$root/pytest.log"
        ;;
    *)
        python -m benchmarks.deepseek_w4a16 "$stage" --root "$root" "$@" 2>&1 | tee "$root/$stage.log"
        ;;
esac
