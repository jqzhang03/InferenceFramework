"""Render the serial Qwen CUDA integration experiments, without merging old runs."""
from pathlib import Path
import json
import statistics
import torch


def main() -> None:
    root=Path('results/qwen_cuda_controlled_20261116')
    names=['PyTorch','CUDA initial','CUDA tuned','BF16 hybrid','FP8 hybrid']
    reports=[[json.loads((root/f'v{v}_rep{r}.json').read_text()) for r in range(2)] for v in range(5)]
    lines=['# Qwen3-0.6B CUDA projection integration — 2026-11-16','',
           'RTX 5060 Ti / SM120, BF16 model, TP=1, eager prefill and CUDA Graph decode. '
           '112 decoder projections are installed (QKV/O and MLP); vocabulary head and FlashAttention are unchanged.','',
           'Latency/SLO-aware prefill quotas and TPOT adaptation are disabled; fixed 2048-token budget fits each prefill in one forward. '
           'Earlier adaptive-scheduler attempts remain under results/qwen_cuda_inference_20261116 and are not pooled here. '
           'Their first-chunk logit diagnostic compared different positions and must not be used as precision evidence.','',
           'Three synthetic token workloads, fixed 32-token outputs, five distinct prompts per condition, '
           'two fresh-process rounds in opposite backend orders. One warmup per condition. '
           'Initialization/compilation/packing are excluded; dynamic FP8 activation quantization and allocation are included. '
           'This is offline generation, not HTTP/SLO goodput or real-text quality evaluation.','',
           'Reported time is the mean of the two per-round medians (five complete generate() wall times per round). Speedup is library time / backend time. '
           'Round-to-round variation is retained below; there is no claim of a universal speedup.','',
           '| Batch / input tokens | PyTorch ms | CUDA initial ms | CUDA tuned ms | BF16 hybrid ms | FP8 hybrid ms |',
           '|---|---:|---:|---:|---:|---:|']
    summary=[]
    for c in range(3):
        case=reports[0][0]['cases'][c]
        med=[statistics.mean([r['cases'][c]['median_s'] for r in reports[v]])*1000 for v in range(5)]
        lines.append(f"| {case['batch']} / {case['input_length']} | "+' | '.join(f'{t:.2f}' for t in med)+' |')
        summary.append(dict(batch=case['batch'],length=case['input_length'],ms=med,speedups=[med[0]/t for t in med]))
    lines+=['','## Repeat variation and token parity','',
            '| Backend | Batch / input | Round medians (ms) | Whole request token parity against paired PyTorch |',
            '|---|---|---|---|']
    for v in range(1,5):
        for c in range(3):
            times=[];same=total=0
            for r in range(2):
                case=reports[v][r]['cases'][c];times.append(case['median_s']*1000)
                for actual,baseline in zip(case['tokens'],reports[0][r]['cases'][c]['tokens']):
                    for a,b in zip(actual,baseline):same+=a==b;total+=1
            lines.append(f"| {names[v]} | {case['batch']} / {case['input_length']} | {times[0]:.2f}, {times[1]:.2f} | {same}/{total} |")
    lines+=['','## Fixed-input first-position logit diagnostic','',
            'Two synthetic prompts (128/1024 tokens), one output position each; collected after timing. '
            'These are numerical checks, not perplexity/task-quality evidence.','',
            '| Backend | Input length | Relative L2 vs BF16 library | Top-1 equal |','|---|---:|---:|---|']
    base=torch.load(root/'v0_rep1.pt',weights_only=True)
    quality=[]
    for v in range(1,5):
        actual=torch.load(root/f'v{v}_rep1.pt',weights_only=True)
        for length,a,b in zip((128,1024),actual,base):
            error=((a.double()-b.double()).norm()/b.double().norm()).item()
            equal=torch.equal(a.argmax(-1),b.argmax(-1))
            quality.append(dict(backend=names[v],length=length,relative_l2=error,top1_equal=equal))
            lines.append(f'| {names[v]} | {length} | {error:.6g} | {equal} |')
    followup=[json.loads((root/f'v5_rep{r}.json').read_text()) for r in range(2)]
    followup_ms=[statistics.mean([r['cases'][c]['median_s'] for r in followup])*1000 for c in range(3)]
    lines+=['','## Follow-up: FP8 prefill only','',
            'A separate two-process screen replaces FP8 selections below 32 rows with the BF16 hybrid selections. '
            'It was run after the primary comparisons, not in the reverse-order pairing; it does not establish a stable improvement.','',
            '| Batch / input | Mean of round medians (ms) | Round medians (ms) |',
            '|---|---:|---|']
    for c in range(3):
        case=followup[0]['cases'][c]
        times=[r['cases'][c]['median_s']*1000 for r in followup]
        lines.append(f"| {case['batch']} / {case['input_length']} | {followup_ms[c]:.2f} | {times[0]:.2f}, {times[1]:.2f} |")
    lines+=['','## Optimizations and limits','',
            '- Vectorized row-major GEMV avoids the large MMA tile for small batches. Runtime GEMM adds 16/32-row tiles (configs 21–23); original standalone IDs stay unchanged.',
            '- Offline shape tuning freezes tile/warp/stage choices. BF16 hybrid keeps the library on slower or unmeasured shapes; FP8 hybrid additionally considers native FP8 MMA.',
            '- FP8 packs weights once, quantizes BF16 activations per row in a CUDA kernel, and applies row scales in the native FP8 MMA epilogue. This replaces multi-operation runtime quantization.',
            '- The initial CUDA variant already includes GEMV, but uses cfg10 for GEMM. The tuned variant includes the new small tiles. Thus the initial/tuned comparison isolates configuration selection, not the entire GEMV contribution.',
            '- Raw tuning uses repeated warm-cache matrices and is distinct from model inference. FP8/BF16 compare different arithmetic precisions. Some BF16 generation token sequences also diverge; do not describe these paths as bitwise equivalent.',
            '- Original BF16 weights remain resident; transposed copies and optional FP8 copies increase memory consumption. These are not weight-memory compression measurements.',
            '- Defaults remain PyTorch. Unsupported layouts fall back; FP8 requires BF16 Qwen3 on SM120; TP>1, PD and combination with existing quantization are rejected. Mixed-batch CUDA Graph capture is unverified; decode replay is tested.',
            '- GPU tests: 3 passed (FP16/BF16 accuracy + changing-input graph replay; FP8 zero row, per-row scale epilogue and graph replay). Tuning additionally checks every candidate and freezes selections before held-out BF16 activation checks.','',
            '## Reproduce','',
            '```bash',
            'bash benchmarks/run_qwen_cuda.sh --backend cuda_fp8_auto --output results/qwen_cuda_tune.json',
            'bash benchmarks/run_qwen_cuda.sh --backend cuda --output results/qwen_cuda_bf16_tune.json',
            'bash benchmarks/run_qwen_cuda.sh --pytest',
            'bash benchmarks/run_qwen_cuda.sh --mode suite --output results/qwen_cuda_controlled_20261116',
            'bash benchmarks/run_qwen_cuda.sh --mode prefill-suite --output results/qwen_cuda_controlled_20261116',
            'python -m benchmarks.qwen_cuda_linear_report','```','',
            'Raw reports and frozen selectors are local ignored artifacts under results/. Historical standalone results remain separate.']
    Path('benchmarks/_qwen_cuda_linear_20261116.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(summary=summary,quality=quality),indent=2))


if __name__=='__main__':main()
