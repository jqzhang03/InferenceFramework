import os
from math import isfinite
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str # 模型所在目录
    linear_backend: str | None = None # None: model default (Qwen3-0.6B -> cuda_decode); explicit torch/cuda variants override
    cuda_linear_tuning: str = "" # frozen per-shape selector JSON
    max_num_batched_tokens: int = 16384 # 一次批处理的最大token数
    max_num_seqs: int = 512 # 最多同时处理序列数
    max_model_len: int = 4096 # 最大上下文长度
    gpu_memory_utilization: float = 0.9 # GPU内存利用率
    tensor_parallel_size: int = 1 # 张量并行使用的GPU数
    execution_mode: str = "auto" # 执行模式：auto（单卡mixed，多卡可自动PD）| mixed | pd
    pd_separation: bool | None = None # 兼容旧接口：True=PD、False=mixed；建议使用 execution_mode
    prefill_device: int = 0 # PD分离时运行Prefill的CUDA设备序号
    decode_device: int = 1 # PD分离时运行Decode的CUDA设备序号
    enforce_eager: bool = False # 允许使用框架自己的推理策略。
    # True：不使用图优化，优点(兼容性更好、调试方便、某些环境下稳定)，缺点(降低推理速度)
    hf_config: AutoConfig | None = None # hugging face模型配置对象
    eos: int = -1 # EOS的token id
    kvcache_block_size: int = 256 # 在PagedAttention中，一个KV缓存块(页)的大小，在vllm生产环境下一般是16，必须保持为16的倍数
    num_kvcache_blocks: int = -1 # KV Cache块的数量，-1表示GPU根据显存大小、模型大小、block size等自动计算
    kv_cache_dtype: str = "auto" # KV缓存数据类型："auto"（模型dtype，默认）或 "fp8_e4m3"（FP8 E4M3量化，容量翻倍，decode用自研Triton内核）
    kv_calibration_path: str = "" # FP8 KV校准token IDs JSON；为空时使用内置随机token校准
    kv_fp8_scale_margin: float = 1.1 # 校准最大值对应E4M3量化上限的安全因子；<1会主动裁剪
    kv_swap: bool = True # KV swap 抢占：KV块不足时把序列的KV拷到CPU内存并释放GPU块，恢复时直接换回（bit-exact，免重新prefill）。支持TP=1的MHA/MLA与auto/fp8 KV；TP>1仍回退recompute
    kv_swap_space_gb: float = 2.0 # KV swap 的 CPU 缓冲空间上限（GB，vLLM swap_space 同款）。换出缓冲累计超限时回落 recompute 抢占，防止 CPU RAM 耗尽（本机 WSL 内存有限，0.6B 单 KV 块 28MB）
    latency_aware_scheduling: bool = True # 混合批次为等待prefill预留预算，并在候选窗口内优先预计prefill较短的请求
    cache_affinity_admission: bool = True # Top-W准入：优先复用更多前缀KV的请求
    prefix_feature_cache: bool = True # 缓存generation标记的Top-W前缀特征；False用于反复解析消融
    admission_window: int = 16 # latency/cache admission候选窗口 W
    aging_fairness: bool = True # 等待超过阈值后优先最老请求，防止长请求饿死
    aging_timeout_ms: float = 2000.0 # aging提升阈值
    prefill_reserve_tokens: int = 256 # 混合调度至少为一个等待prefill保留的token预算
    slo_aware_scheduling: bool = True # 按请求TTFT目标、队列压力和实测prefill速度调整配额
    default_ttft_slo_ms: float | None = 500.0 # 请求未单独设置时采用的TTFT目标；None表示不设默认目标
    tpot_aware_scheduling: bool = True # 按活动请求TPOT slack动态降低prefill配额、优先decode
    default_tpot_slo_ms: float | None = None # 请求未单独设置时使用的TPOT目标；None表示不设默认目标
    tpot_decode_ms_fallback: float = 20.0 # 尚无该请求decode样本时用于TPOT slack估算的回退值
    tpot_prefill_throttle_margin: float = 0.5 # TPOT 压力对 prefill 配额的收缩死区：目标超出 50% 以内视为噪声，不压缩 prefill。0=旧行为（任何超出都饱和到 min_budget，会把 prefill 饿成 256-token 碎片步、反过来又抬高 TPOT，形成自锁）
    tpot_throttle_max_waiting: int = 16 # 等待队列达到该深度时**停止**用 TPOT 压力压缩 prefill：此时瓶颈是 prefill 吞吐（新请求拿不到首 token），继续压 prefill 会自锁（实测 128 请求/16 req/s 下 prefill 被压到 256 token/步、TTFT p50 43.7s）；0=不设门限（旧行为）
    max_prefill_chunk_tokens: int = 4096 # 自适应调度单步prefill预算上限
    queue_depth_for_full_prefill: int = 16 # 等待队列达到该长度时进入最高队列压力
    multi_step_decode: bool = True # 纯decode时连续复用调度批次；prefill/spec/mixed路径仍单步
    decode_lookahead: bool = True # 在纯 decode burst 中提前准备下一轮的静态输入
    max_decode_steps: int = 4 # 每个decode调度窗口最多执行的模型前向轮数（含首轮）
    decode_burst_yield: bool = True # burst 提前让出：后续轮开始前检查prefill队列，有等待请求就不再复用余下轮次（False=消融，跑满max_decode_steps再回外循环，会推迟新请求的TTFT；PD模式另有独立prefill队列让出）
    decode_burst_yield_on_arrival: bool = True # burst 期间新请求到达（服务端信号/trace 回调）时也提前收尾：mixed 模式下请求只在两次 step 之间提交，等待队列检查抓不到"正在到达"的请求，需要这个外部压力信号；False=消融
    recompute_aware_preemption: bool = True # 按估算swap往返成本与cache-aware recompute成本选择抢占方式
    preempt_prefill_tokens_per_second: float = 10000.0 # 尚无在线样本时的recompute估算回退值
    preempt_kv_transfer_gbps: float = 12.0 # 尚无在线样本时的KV swap带宽估算回退值
    rolling_cache: bool = False # SWA 滚动缓冲：Mistral 使用统一环池；Gemma-2 交替 local/global 层使用环池+full 双池。两者支持 auto/fp8_e4m3 KV、权重量化、KV swap 和 decode/mixed CUDA Graph；sparse24 权重量化保留但 CUDA Graph 自动回退 eager。rolling+Medusa/EAGLE 与 Gemma-2 split+speculative 暂不支持。滚动模型不参与前缀缓存发布/消费（窗口内容过期，重复 prompt 有代价，见 block_manager.py 头注）；仅对具有 sliding-window 语义的 Mistral/Gemma-2 生效。
    num_ring_kvcache_blocks: int = 0  # split 模式：环池块数（runner 分配后写回，scheduler 建 BM 用）
    num_full_kvcache_blocks: int = 0  # split 模式：full 池块数（同上）
    quantization: str = "none" # "none" | "w8a8" | "int4" | "nvfp4" | "mxfp4" | "awq" | "sparse24" | "fp8"
    nvfp4_backend: str = "software" # Triton W4A16 | CUDA W4A16 | cuBLASLt W4A4
    awq_scales_path: str = "" # AWQ激活感知缩放文件（.pt，benchmarks/awq_calibrate.py真实文本校准产出）；为空时用随机token内联校准
    quantize_lm_head: bool = False # 是否量化LM head（默认不量化——与w8a8一致：logits由lm_head点积直接决定，量化它精度损失最大，见INTERVIEW.md §10.3.6）
    int4_dense_path: bool = True # int4双路径模式（默认开）：大M prefill/decode 与小N层走 w_deq 稠密反量化（cuBLAS，收掉大M亏损与TTFT回归），小M大N层走int4内核；代价是显存 1.73GB（比fp16的1.50还大）。False=纯int4（0.85GB，大batch慢），见INTERVIEW.md §10.3.6。streaming 模式下自动强制 False（w_deq 全尺寸副本与按层加载目的冲突；MLA kv_b 例外保留反量化副本）
    int4_group_size: int = 128 # int4 量化组大小（K 维，须整除各线性层 K）。DeepSeek-V2-Lite 的 dense 中间维 10944 = 64×171 不能被 128 整除 → 该模型需 64（精度代价小，官方社区 int4 同样受此约束）
    moe_int4_backend: str = "loop" # 纯 INT4 MoE 专家：loop（逐专家）| segment（无 padding 分段 Triton；仅 TP=1、非 AWQ、无 w_deq）| compact（decode/prefill ≤4096 行走 GPU 路由 + expert 2D grid 的 CUDA W4A16，1-4 行 GEMV、5-16 行按 count 分支的 MMA8/16、17+ 行 MMA32 chunk；>4096 才走每层一次 host 打包；仅 TP=1、非 AWQ、无 w_deq）
    moe_int4_fuse_gate_up: bool = True # segment INT4 时将 gate/up 投影合并为一次 GEMM launch；False 用于融合消融
    moe_int4_fuse_gate_up_silu: bool = True # compact INT4：5+ 行 gate/up MMA 的 epilogue 直接算 silu(gate)*up，省掉 gate/up 写回与 elementwise；False 用于位级/性能消融（1-4 行 GEMV 本来就已融合）
    moe_int4_cta_fanout: int = 0 # compact decode 每个 expert 展开的 CTA 上限；0=按 T 自动（T<=4:8，5<=T<=8:128，T>8:64），>0 显式覆盖
    streaming_load: bool = False # 按层流式加载+即时量化（16GB 卡跑 7B+ 的前提，见 INTERVIEW.md §1.7）：模型在 meta 设备构造（0显存）→ loader 逐 decoder layer 物化→加载→立即量化→释放 fp16。显式开启；或当估算 fp16 权重超过空闲显存 45% 且启用了权重量化时自动开启（7B+ 必触发）。限制：int4 强制纯 int4（无 w_deq，MLA kv_b 例外保留反量化副本 ~1%）；w8a8 无 SmoothQuant 校准（需全模型前向）；awq 仅支持预生成 awq_scales_path（内联校准需全 fp16 模型）
    speculative: str = "none" # 投机解码："none" | "ngram"（n-gram/prompt-lookup草稿，无模型零显存，见INTERVIEW.md §10.3.5）| "medusa"（Medusa多头，需medusa_path）| "eagle"（EAGLE-1草稿层：无RoPE transformer层 + 共享LM head 自回归草稿，需eagle_path）
    ngram_window: int = 4 # n-gram窗口上限（vLLM --ngram-prompt-lookup-max 默认同款）
    ngram_min_window: int = 1 # n-gram窗口下限（先长后短回退，vLLM --ngram-prompt-lookup-min 默认同款）
    max_draft_len: int = 4 # 每步最大草稿数γ（vLLM --num-speculative-tokens 常用值）
    medusa_path: str = "" # Medusa头权重文件（.pt，benchmarks/medusa_train.py训练产出）；speculative="medusa"时必须
    eagle_path: str = "" # local .pt for EAGLE-1/2, checkpoint directory for EAGLE-3
    dflash_path: str = "" # local Qwen3-0.6B DFlash Speculators checkpoint directory
    dspark_path: str = "" # local Qwen3-0.6B DSpark Speculators checkpoint directory
    dspark_confidence_threshold: float = 0.3 # prefix survival cutoff when step costs are unavailable
    dspark_step_ms: tuple[float, ...] = () # measured verifier costs indexed by row count minus one
    eagle_tree_top_k: int = 2
    eagle_tree_max_nodes: int = 8 # candidate nodes, excluding the already emitted root token
    eagle_tree_policy: str = "fixed" # fixed | adaptive (EAGLE-2/3)
    medusa_hidden: int = 256 # Medusa头隐藏维（输出层256×vocab是主要参数，控制总规模）
    prefix_cache_max_free_blocks: int = 0 # 空闲前缀缓存块上限；0表示不设上限，分配压力下按LRU回收
    mixed_cudagraph: bool = True # 普通混合Prefill/Decode批次使用按形状惰性捕获的CUDA Graph
    mixed_cudagraph_max_graphs: int = 4 # 混合图形状缓存上限，超限淘汰最久未使用图
    mixed_cudagraph_max_tokens: int = 4096 # 超过该批次token数时走eager，限制图捕获峰值显存

    int4_kernel_backend: str | None = None # None selects the model default for packed W4A16 GEMM
    cuda_int4_expert_fusion: str = "none" # none | gate_up | full (cooperative launch)
    mla_decode_backend: str = "auto" # auto | cuda | triton

    def __post_init__(self):
        assert self.mla_decode_backend in ("auto", "cuda", "triton"), \
            f"unknown MLA decode backend: {self.mla_decode_backend!r}"
        assert self.linear_backend in (None, "torch", "cuda", "cuda_auto", "cuda_fp8", "cuda_fp8_auto", "cuda_decode", "cuda_decode_auto")
        if self.linear_backend is not None and self.linear_backend.endswith("auto"):
            assert self.cuda_linear_tuning, "auto CUDA linear requires a frozen cuda_linear_tuning report"
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.kv_swap_space_gb >= 0, "kv_swap_space_gb 必须非负"
        assert self.prefix_cache_max_free_blocks >= 0, \
            "prefix_cache_max_free_blocks 必须非负"
        assert self.mixed_cudagraph_max_graphs >= 1, \
            "mixed_cudagraph_max_graphs 必须至少为1"
        assert self.mixed_cudagraph_max_tokens >= 1, \
            "mixed_cudagraph_max_tokens 必须至少为1"
        assert isfinite(self.kv_fp8_scale_margin) and self.kv_fp8_scale_margin > 0, \
            "kv_fp8_scale_margin 必须是有限正数"
        if self.kv_calibration_path:
            assert self.kv_cache_dtype == "fp8_e4m3", \
                "kv_calibration_path 仅适用于 kv_cache_dtype='fp8_e4m3'"
            assert os.path.isfile(self.kv_calibration_path), \
                f"FP8 KV校准文件不存在: {self.kv_calibration_path}"
        assert self.admission_window >= 1, "admission_window 必须至少为1"
        assert self.aging_timeout_ms >= 0, "aging_timeout_ms 必须非负"
        assert self.prefill_reserve_tokens >= 1, "prefill_reserve_tokens 必须至少为1"
        assert self.default_ttft_slo_ms is None or (isfinite(self.default_ttft_slo_ms)
                                                    and self.default_ttft_slo_ms > 0), \
            "default_ttft_slo_ms 必须为正数或 None"
        assert self.default_tpot_slo_ms is None or (isfinite(self.default_tpot_slo_ms)
                                                    and self.default_tpot_slo_ms > 0), \
            "default_tpot_slo_ms 必须为正数或 None"
        assert isfinite(self.tpot_decode_ms_fallback) and self.tpot_decode_ms_fallback > 0, \
            "tpot_decode_ms_fallback 必须为有限正数"
        assert 1 <= self.max_decode_steps <= 16, \
            "max_decode_steps 必须在1到16之间"
        assert self.max_prefill_chunk_tokens >= 1, \
            "max_prefill_chunk_tokens 必须至少为1"
        assert self.queue_depth_for_full_prefill >= 1, \
            "queue_depth_for_full_prefill 必须至少为1"
        assert self.preempt_prefill_tokens_per_second > 0, \
            "preempt_prefill_tokens_per_second 必须为正数"
        assert self.preempt_kv_transfer_gbps > 0, "preempt_kv_transfer_gbps 必须为正数"
        assert self.execution_mode in ("auto", "mixed", "pd"), \
            f"unknown execution_mode: {self.execution_mode!r}"
        if self.pd_separation is not None:
            legacy_mode = "pd" if self.pd_separation else "mixed"
            if self.execution_mode == "auto":
                self.execution_mode = legacy_mode
            else:
                assert self.execution_mode == legacy_mode, \
                    "pd_separation 与 execution_mode 指定了不同的执行模式"
        if self.execution_mode == "pd":
            assert self.tensor_parallel_size == 1, \
                "当前 PD 分离版本每个角色使用一张 GPU，tensor_parallel_size 必须为 1"
            assert self.prefill_device >= 0 and self.decode_device >= 0, \
                "prefill_device/decode_device 必须是非负 CUDA 设备序号"
            assert self.prefill_device != self.decode_device, \
                "PD 分离需要为 Prefill 和 Decode 指定不同 GPU"
            assert self.speculative == "none", \
                "当前 PD 分离版本暂不支持投机解码"
            assert not self.rolling_cache, \
                "当前 PD 分离版本暂不支持 rolling_cache"
            assert self.kv_cache_dtype == "auto", \
                "当前 PD KV 传输暂要求 kv_cache_dtype='auto'"
        assert self.speculative in ("none", "ngram", "medusa", "eagle", "eagle2", "eagle3", "dflash", "dspark"), \
            f"unknown speculative: {self.speculative}"
        assert self.quantization in ("none", "w8a8", "int4", "nvfp4", "mxfp4", "awq", "sparse24", "fp8"), \
            f"unknown quantization: {self.quantization}"
        assert self.moe_int4_backend in ("loop", "segment", "compact"), \
            f"unknown MoE INT4 backend: {self.moe_int4_backend!r}"
        assert 0 <= self.moe_int4_cta_fanout <= 256, \
            "moe_int4_cta_fanout must be in [0, 256] (0=auto)"
        assert self.int4_kernel_backend in (None, "triton", "cuda"), \
            f"unknown INT4 kernel backend: {self.int4_kernel_backend!r}"
        assert self.cuda_int4_expert_fusion in ("none", "gate_up", "full"), \
            f"unknown CUDA INT4 expert fusion: {self.cuda_int4_expert_fusion!r}"
        assert self.nvfp4_backend in ("software", "cuda", "cublaslt"), \
            f"unknown NVFP4 backend: {self.nvfp4_backend}"
        if self.nvfp4_backend != "software":
            assert self.quantization == "nvfp4", \
                f"nvfp4_backend={self.nvfp4_backend!r} requires quantization='nvfp4'"
        if self.speculative == "medusa":
            assert self.medusa_path, "speculative=medusa requires medusa_path"
        if self.speculative in ("eagle", "eagle2", "eagle3"):
            assert self.eagle_path, f"speculative={self.speculative} requires eagle_path"
        if self.speculative in ("dflash", "dspark"):
            path_field = f"{self.speculative}_path"
            path = getattr(self, path_field)
            assert path, f"speculative={self.speculative} requires {path_field}"
            directory = os.path.expanduser(path)
            setattr(self, path_field, directory)
            for filename in ("config.json", "model.safetensors"):
                path = os.path.join(directory, filename)
                if not os.path.isfile(path):
                    raise FileNotFoundError(path)
        if self.speculative == "dspark":
            assert 0 <= self.dspark_confidence_threshold <= 1, "invalid DSpark confidence threshold"
            assert all(isfinite(value) and value > 0 for value in self.dspark_step_ms), \
                "DSpark step costs must be finite and positive"
        self.hf_config = AutoConfig.from_pretrained(self.model)
        if self.speculative in ("eagle2", "eagle3", "dflash", "dspark"):
            hf = self.hf_config
            shape = tuple(getattr(hf, key, None) for key in (
                "model_type", "hidden_size", "num_hidden_layers", "intermediate_size",
                "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size"))
            assert shape == ("qwen3", 1024, 28, 3072, 16, 8, 128, 151936), \
                f"{self.speculative} currently requires Qwen3-0.6B"
            assert str(hf.dtype) in ("torch.bfloat16", "bfloat16") and self.tensor_parallel_size == 1 \
                and self.quantization == "none" and self.kv_cache_dtype == "auto" and not self.rolling_cache, \
                f"{self.speculative} requires single-GPU BF16 with unquantized paged KV and weights"
            self.enforce_eager = True
        if self.speculative in ("eagle2", "eagle3"):
            assert 1 <= self.eagle_tree_top_k <= 8 and 1 <= self.max_draft_len <= 8 \
                and 1 <= self.eagle_tree_max_nodes <= 32, "invalid EAGLE tree budget"
            assert self.eagle_tree_policy in ("fixed", "adaptive"), "invalid EAGLE tree policy"
        if self.int4_kernel_backend is None:
            self.int4_kernel_backend = self._default_int4_kernel_backend()
        if self.int4_kernel_backend == "cuda":
            assert self.quantization == "int4" and self.moe_int4_backend in ("loop", "compact"), \
                "CUDA W4A16 requires quantization='int4' and moe_int4_backend in ('loop','compact')"
        if self.moe_int4_backend == "compact":
            assert self.int4_kernel_backend == "cuda" and self.tensor_parallel_size == 1, \
                "compact INT4 MoE requires the CUDA W4A16 backend and TP=1"
        if self.cuda_int4_expert_fusion != "none":
            assert self.int4_kernel_backend == "cuda" and self.tensor_parallel_size == 1, \
                "CUDA INT4 expert fusion requires the CUDA W4A16 backend and TP=1"
        if self.linear_backend is None:
            self.linear_backend = self._default_linear_backend()
        if self.linear_backend != "torch":
            assert self.hf_config.model_type == "qwen3" and self.tensor_parallel_size == 1, "CUDA linear currently requires Qwen3 TP=1"
            assert self.execution_mode != "pd", "CUDA linear currently requires mixed execution"
            assert self.quantization == "none", "CUDA linear cannot combine with weight quantization"
            if self.linear_backend.startswith("cuda_fp8"):
                assert str(self.hf_config.dtype) == "torch.bfloat16", "native FP8 linear requires BF16 model"
        self.hf_config.moe_int4_backend = self.moe_int4_backend
        self.hf_config.moe_int4_fuse_gate_up = self.moe_int4_fuse_gate_up
        self.hf_config.moe_int4_fuse_gate_up_silu = self.moe_int4_fuse_gate_up_silu
        self.hf_config.moe_int4_cta_fanout = self.moe_int4_cta_fanout
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)

    def _default_linear_backend(self) -> str:
        """Select the Qwen3-0.6B default by architecture, independent of directory name."""
        hf = self.hf_config
        shape = tuple(getattr(hf, key, None) for key in (
            "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim"))
        compatible = (self.tensor_parallel_size == 1 and self.execution_mode != "pd"
                      and self.quantization == "none")
        dtype = str(getattr(hf, "dtype", None))
        if (getattr(hf, "model_type", None) == "qwen3"
                and shape == (1024, 3072, 28, 16, 8, 128) and compatible
                and dtype in ("torch.float16", "torch.bfloat16", "float16", "bfloat16")):
            return "cuda_decode"
        return "torch"

    def _default_int4_kernel_backend(self) -> str:
        hf = self.hf_config
        shape = tuple(getattr(hf, key, None) for key in (
            "hidden_size", "intermediate_size", "moe_intermediate_size",
            "num_hidden_layers", "n_routed_experts"))
        if (getattr(hf, "model_type", None) == "deepseek_v2"
                and shape == (2048, 10944, 1408, 27, 64)
                and str(getattr(hf, "dtype", None)) in ("torch.float16", "torch.bfloat16", "float16", "bfloat16")
                and self.quantization == "int4" and self.int4_group_size == 64
                and self.moe_int4_backend in ("loop", "compact")
                and self.tensor_parallel_size == 1
                and self.execution_mode != "pd"):
            return "cuda"
        return "triton"

    def pd_mode_incompatibility(self) -> str | None:
        """Return why PD mode cannot run with this configuration, if applicable."""
        if self.linear_backend != "torch":
            return "CUDA linear currently requires mixed execution"
        if self.tensor_parallel_size != 1:
            return "PD separation requires tensor_parallel_size=1"
        if self.speculative != "none":
            return "PD separation does not support speculative decoding"
        if self.rolling_cache:
            return "PD separation does not support rolling_cache"
        if self.kv_cache_dtype != "auto":
            return "PD KV transfer currently requires kv_cache_dtype='auto'"
        return None
