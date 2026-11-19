# DeepSeek-V2-Lite CUDA W4A16 expert 融合

RTX 5060 Ti，BF16 激活，原有 INT4 group 64，TP=1。三条路径都复用同一个已加载模型的打包权重和 scale。单 expert 宽度为 2048，中间维度为 1408。

- `none` 分开执行 gate、up、SiLU、乘法和 down。
- `gate_up` 在一次 kernel 中计算 gate、up 和 SiLU 乘法，随后启动 down。
- `full` 用一次 cooperative launch 执行完整 expert。所有 CTA 写完中间激活后做网格同步，随后各 CTA 执行 down 的输出列。

融合覆盖每个 expert 收到 1–4 行的情况。两种融合保留 gate、up 和 SiLU 的 BF16/FP16 舍入点。空 expert 和超过 4 行的 expert 都按常规路径执行。

## 单 expert

取模型中实际记录的 BF16 激活行。两种工作集分别反复执行一个 expert 和轮换 16 个真实 expert。表中为独立 CUDA Event / Graph 计时，12 轮覆盖两遍全部执行顺序。

| 工作集 | 每 expert 行数 | 分开 µs | 两次启动 µs | 单次启动 µs |
|---|---:|---:|---:|---:|
| 单 expert | 1 | 19.16 | 16.87 | 19.01 |
| 单 expert | 2 | 31.73 | 23.56 | 21.20 |
| 单 expert | 3 | 36.63 | 25.63 | 31.01 |
| 单 expert | 4 | 38.07 | 26.02 | 29.06 |
| 轮换 16 个 expert | 1 | 28.86 | 27.84 | 31.30 |
| 轮换 16 个 expert | 2 | 48.54 | 35.89 | 34.34 |
| 轮换 16 个 expert | 3 | 52.17 | 36.70 | 40.53 |
| 轮换 16 个 expert | 4 | 51.12 | 36.36 | 39.60 |

两次启动在 8 个算例中的降幅为 3.5%–31.6%。单次启动在轮换工作集的 M=1 算例增加了 8.4%。两种融合分别与分开执行逐位一致，共比较 68 组 expert 输出。

## 整模

同一引擎与 376 块 MLA KV 池，eager 执行。每轮 4 个请求各输入 64 token、输出 4 token，固定 token 和路由，保留 router 与 sampler 计算。12 轮配对在 profiler 外执行，六种执行顺序各出现两次。

| 指标 | 分开 | 两次启动 | 单次启动 |
|---|---:|---:|---:|
| 首 token 延迟 p50 | 1,057.29 ms | 1,073.73 ms | 1,165.17 ms |
| 每输出 token 延迟 p50 | 844.79 ms | 736.67 ms | 549.23 ms |
| 全批次墙钟中位数 | 3,538.88 ms | 3,270.17 ms | 3,034.68 ms |
| 配对总耗时变动中位数 | — | +1.8% | −12.6% |
| 配对 TPOT 变动中位数 | — | +1.3% | −30.0% |
| 总耗时更短轮数 | — | 6/12 | 8/12 |

相比分开执行，两次启动的配对总耗时变动范围为 −46.7% 至 +74.4%，单次启动为 −41.5% 至 +56.0%。两种融合的整模 logits 与分开执行逐位一致，同种子自然采样的 16 个 token 全部一致。

## Nsight Systems 启动次数

单 expert 的 M1–4 在三条路径分别观测到 5、2、1 次 kernel 启动。Decode 的 1,378 次有效 expert 调用均在 M1–4。Prefill 共有 1,513 次有效调用，其中 398 次落在融合范围，时间线中的融合 kernel 次数与这两个数字一致。

| 指标 | 分开 | 两次启动 | 单次启动 |
|---|---:|---:|---:|
| Decode expert FFN kernel 数 | 6,890 | 2,756 | 1,378 |
| Decode 全模型 kernel 数 | 71,605 | 67,471 | 66,093 |
| Prefill 全模型 kernel 数 | 34,273 | 33,079 | 32,681 |
| Decode GPU kernel 活动时间 | 117.38 ms | 110.81 ms | 113.23 ms |

GPU kernel 活动时间为三步内各 kernel 时间区间的并集之和。全模型 kernel 数包含路由、索引、MLA、采样等操作。

## 运行

用 `int4_kernel_backend="cuda"` 和 `cuda_int4_expert_fusion="gate_up"` 或 `"full"` 启用相应选项。完整模型实验沿用 `int4_group_size=64`、`streaming_load=True` 和 `enforce_eager=True`。

```bash
out=profiles/deepseek_w4a16_fusion_repro
bash benchmarks/run_deepseek_w4a16.sh pytest "$out" -q
bash benchmarks/run_deepseek_w4a16.sh fusion "$out"
bash benchmarks/run_deepseek_w4a16.sh fusion_nsys "$out"
python3 -m benchmarks.deepseek_w4a16_fusion_trace "$out/fusion.sqlite"
```

[单 expert 原始数据](../profiles/deepseek_w4a16_fusion_20261118/operators.json) · [整模原始数据](../profiles/deepseek_w4a16_fusion_20261118/model.json) · [Nsight Systems 原始报告](../profiles/deepseek_w4a16_fusion_20261118/fusion.nsys-rep) · [28 项测试结果](../profiles/deepseek_w4a16_fusion_20261118/pytest.log)
