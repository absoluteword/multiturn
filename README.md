# Multi-Turn Benchmark

多轮并发 LLM 推理基准测试工具。模拟多个客户端进行多轮对话，测量 TTFT、ITL、延迟、吞吐量以及每轮 KV Cache 命中率。

## 目录结构

```
multi_turn/
├── bench_multiturn.py                  # 多轮并发基准测试主程序
├── process_multiturn_result_to_csv.py  # TXT 日志 → CSV 结构化数据
├── plot_mul.py                         # TTFT/ITL/CHR 逐轮柱状+折线图
├── plot_round_tput.py                  # 逐轮 Total Throughput 折线对比图
├── estimate_chr.py                     # 估算多轮 KV Cache 命中率
├── estimate_concurrency.py             # 估算 GPU 显存支持的最大并发数
├── bench_data/                         # 输出目录
│   ├── *.txt                           # 原始日志
│   ├── *.csv                           # 转换后的数据
│   └── *.png                           # 图表
├── AGENTS.md
└── README.md
```

## 依赖

- Python 3.x
- `sglang` package（提供 `sglang.bench_serving`、`sglang.benchmark.*`、`sglang.test.kits.cache_hit_kit`）
- `aiohttp`（vLLM / OpenAI-compatible 模式必需）
- `requests`、`numpy`、`tqdm`、`pandas`、`matplotlib`

---

## 使用流程

### 1. 运行基准测试

#### SGLang 原生 API（默认）

```bash
python bench_multiturn.py \
    --model-path /data/models/GLM-5-NVFP4/ \
    --dataset-path /data/models/ShareGPT_V3_unfiltered_cleaned_split.json \
    --ready-queue-policy random
```

#### vLLM / OpenAI-Compatible API

连接 vLLM 等 OpenAI-compatible 服务端点时，需显式指定 `--api-format openai`。

> **关键点：** 若希望 vLLM 返回真实的 `cached_tokens`，启动 vLLM 服务时须添加 `--enable-prompt-tokens-details` 参数。否则脚本会基于客户端已知的 `prompt_len - sub_q_len` 自动估算 `cache_hit_rate`，其值与 `predicted_chr` 基本一致。

```bash
python bench_multiturn.py \
    --api-format openai \
    --num-clients 8 \
    --max-parallel 8 \
    --request-length 1700 \
    --first-round-input-length 16000 \
    --output-length 500 \
    --num-rounds 32 \
    --request-rate 10 \
    --port 9000 \
    --model-path /data/models/GLM-5-NVFP4/ \
    --served-model-name GLM-5 \
    --dataset-path /data/models/ShareGPT_V3_unfiltered_cleaned_split.json \
    --disable-auto-run \
    --disable-random-sample \
    --enable-round-barrier \
    --range-ratio 0.8 \
    --output-dir 6000D_GLM5_tp8_vllm
```

> **关键点：**
> - `--api-format openai` 切换到 OpenAI-compatible `/v1/chat/completions` 端点，内部使用自定义 `async_request_vllm_chat_completions` 处理 SSE 流式响应。
> - 脚本会自动跳过 vLLM 不支持的 `/flush_cache` 和 SGLang `/generate` 心跳请求。
> - 由于 vLLM 的 OpenAI API 通常**不返回** `usage.prompt_tokens_details.cached_tokens`，脚本会基于客户端已知的 `prompt_len - sub_q_len` 自动估算 `cache_hit_rate`，其值与 `predicted_chr` 基本一致。若 vLLM 启动时添加了 `--enable-prompt-tokens-details`，则可获取真实的 `cached_tokens`。
> - **如果启动 vLLM 时用了 `--served-model-name` 自定义模型名，请务必通过 `--served-model-name` 参数传入**，否则脚本默认使用 `--model-path` 的 basename，可能报 404 `model does not exist`。

#### 完整参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model-path` | str | /data/models/GLM-5-NVFP4/ | 模型路径（HuggingFace 格式） |
| `--num-clients` | int | 32 | 并发客户端数 |
| `--max-parallel` | int | 16 | 最大并发请求数 |
| `--request-length` | int | 1700 | 每轮请求长度（后续轮次的输入 token 数） |
| `--first-round-input-length` | int | 16000 | 第一轮输入长度。若为 0 则使用 --request-length |
| `--output-length` | int | 500 | 输出长度（max_tokens） |
| `--num-rounds` | int | 32 | 每客户端轮次 |
| `--min-rounds` | int | 0 | 每客户端最小轮次（0 = 使用 --num-rounds） |
| `--max-rounds` | int | 0 | 每客户端最大轮次（0 = 使用 --num-rounds） |
| `--request-rate` | float | 10.0 | 请求速率 (req/s) |
| `--distribution` | str | poisson | 请求间隔分布：poisson 或 uniform |
| `--sub-question-input-length` | int | 0 | 子问题输入长度（0 = 使用 --request-length） |
| `--host` | str | localhost | 服务器地址 |
| `--port` | int | 8000 | 服务器端口 |
| `--dataset-path` | str | (内置路径) | ShareGPT 数据集路径 |
| `--range-ratio` | float | 0.8 | 长度变化比例（1.0 = 无变化，0.5 = ±50%） |
| `--seed` | int | 42 | 随机种子 |
| `--api-format` | str | sglang | API 格式：`sglang`（原生 /generate）或 `openai`（/v1/chat/completions） |
| `--served-model-name` | str | "" | 服务端注册的模型名（空则使用 --model-path 的 basename） |
| `--lora-path` | str | "" | LoRA 适配器路径（仅支持单个） |
| `--ready-queue-policy` | str | random | 就绪队列弹出策略：random 或 fifo |
| `--tag` | str | "" | 本次运行的标签（写入 JSONL 日志） |
| `--log-file` | str | performance_metrics.jsonl | JSONL 日志文件名 |
| `--output-dir` | str | . | 输出目录 |

**固定开启的选项：**
- `--disable-auto-run`：禁用自动多速率测试
- `--disable-random-sample`：禁用随机采样
- `--enable-round-barrier`：启用轮次屏障（只有第 i-1 轮全部完成才发送第 i 轮请求）

**输出文件命名：**
```
{output_dir}/bench_data/multi_turn_{model_name}_input{request_length}_output{output_length}_clients{num_clients}_concurrency{max_parallel}.txt
```

**计算公式：**

| 指标 | 公式 |
|------|------|
| `cache_hit_rate` | `sum(cached_tokens) / sum(prompt_len)` |
| `hit_tokens` | `estimated_cached_tokens` |
| `miss_tokens` | `max(0, prompt_len - estimated_cached_tokens)` |
| `total_tp` | `(sum(miss_tokens) + sum(generated_len)) / duration` |
| `input_tp` | `sum(prompt_len) / duration` |
| `output_tp` | `sum(generated_len) / duration` |
| `predicted_chr`（每轮） | Round 0 = 0；Round i = `(prompt_len - sub_q_len) / prompt_len` |
| `ITL`（每 token） | 若 `gen_len > 1 && latency > ttft`：`(latency - ttft) / (gen_len - 1)`；否则使用服务端返回的 ITL |

> **注意：** vLLM (openai) 模式下，`estimated_cached_tokens` 仅在服务端返回值 > 0 时采用，否则为 0。因此实际 `cache_hit_rate` 可能显示为 0，此时应参考 `predicted_chr`。

---

### 2. 转换为 CSV

```bash
python process_multiturn_result_to_csv.py ./bench_data
python process_multiturn_result_to_csv.py ./bench_data ./output_csv  # 指定输出目录
```

将 txt 日志文件转换为同名 csv 文件。从文件名提取 `input_tokens`、`output_tokens`、`concurrency`；从日志内容提取所有指标。

**CSV 列顺序：**

| # | 列名 | 来源 | 单位/说明 |
|---|------|------|-----------|
| 1 | `filename` | 文件名 | |
| 2 | `input_tokens` | 文件名正则 | |
| 3 | `output_tokens` | 文件名正则 | |
| 4 | `concurrency` | 文件名正则 | |
| 5 | `clients` | Round 0 行提取 | |
| 6 | `avg_ttft` | Average TTFT | ms |
| 7 | `p99_ttft` | P99 TTFT | ms |
| 8 | `median_ttft` | Median TTFT | ms |
| 9 | `avg_itl` | Average ITL | ms |
| 10 | `p99_itl` | P99 ITL | ms |
| 11 | `median_itl` | Median ITL | ms |
| 12 | `input_tp` | Input token throughput | tokens/s |
| 13 | `output_tp` | Output token throughput | tokens/s |
| 14 | `total_tp` | Total token throughput (miss+out) | tokens/s |
| 15 | `cache_hit_rate` | Cache Hit Rate | 百分比（整数） |
| 16 | `total_hit_tokens` | Total Hit Tokens | |
| 17 | `total_miss_tokens` | Total Miss Tokens | |

**逐轮动态列**（每轮 N 添加以下 9 列）：

| 列名 | 说明 | 单位/说明 |
|------|------|-----------|
| `round_N_ttft` | 该轮平均 TTFT | ms |
| `round_N_itl` | 该轮平均 ITL | ms |
| `round_N_chr` | 该轮实际 Cache Hit Rate | 百分比（整数） |
| `round_N_predicted_chr` | 该轮预测 Cache Hit Rate | 百分比（整数） |
| `round_N_hit_tokens` | 该轮命中 token 数 | |
| `round_N_miss_tokens` | 该轮未命中 token 数 | |
| `round_N_output_tp` | 该轮输出吞吐 | tokens/s |
| `round_N_total_tp` | 该轮总吞吐 (miss+out) | tokens/s |
| `round_N_duration` | 该轮持续时间 | 秒（整数） |

> **注意：** `cache_hit_rate` 和 `round_N_chr` 存储的是百分比值（整数），`ttft`/`itl` 为毫秒，`duration` 为秒（int 截断）。

---

### 3. 生成图表

#### plot_mul.py — TTFT/ITL/CHR 逐轮图表

```bash
# 处理整个目录（生成所有 csv 对应的 png）
python plot_mul.py ./bench_data

# 处理单个 csv 文件
python plot_mul.py ./bench_data/result.csv --model-name "GLM-5-NVFP4"

# 处理多个 csv 文件
python plot_mul.py ./bench_data/file1.csv ./bench_data/file2.csv

# 指定输出目录
python plot_mul.py ./bench_data --model-name "GLM-5-NVFP4" --output-dir ./plots

# 隐藏 ITL 轴
python plot_mul.py ./bench_data --no-itl
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `csv_input`（必需，可多个） | CSV 文件路径或目录 |
| `--model-name` | 图片标题的模型名称（默认从文件名提取） |
| `--compare` | 开启对比模式 |
| `--titles` | 对比模式的标题，逗号分隔（配合 --compare 使用） |
| `--output-dir` | 输出目录（默认 csv 所在目录） |
| `--no-itl` | 隐藏 ITL 轴和数据 |

**单文件模式输出：**
- TTFT 柱状图（左 Y 轴）+ ITL 折线图（右 Y 轴）
- 图下方数据表格：Round / TTFT / CHR% / ITL（含平均值列）
- 输出文件：`{csv_stem}.png`

**对比模式输出：**
- TTFT 折线图（左 Y 轴，不同线型+颜色）+ ITL 分组柱状图（右 Y 轴）
- 自动取最小公共轮次数
- 输出文件：`TTFT-ITL-multi-turn-perf.png`

```bash
# 对比模式示例
python plot_mul.py csv1.csv csv2.csv --compare --titles "DeepSeek-V4-Flash,GLM-5-NVFP4"
```

#### plot_round_tput.py — 逐轮 Total Throughput 折线对比图

```bash
# 两个 CSV 对比
python plot_round_tput.py \
    csv1.csv csv2.csv \
    --titles "Model1,Model2" \
    --title-prefix "exp1_" \
    --output-dir ./bench_data \
    --output-filename comparison_round_tput.png

# 单个文件
python plot_round_tput.py ./bench_data/result.csv --titles "ModelA" --title-prefix "exp1_"
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `csv_files`（必需，可多个） | CSV 文件路径 |
| `--titles`（必需） | 图例名称，逗号分隔，数量需与 CSV 文件一致 |
| `--title-prefix` | 图片标题前缀，最终标题为 `{prefix}round_vs_total_tput`（默认空） |
| `--output-dir` | 输出目录（默认第一个 CSV 所在目录） |
| `--output-filename` | 输出 PNG 文件名（默认自动根据前缀生成） |

**图表特性：**
- 多条折线按不同颜色/线型区分
- 横坐标轮次从 **1** 开始
- 图例位于右上角
- 图片下方带数据表格：每轮 Total Tput + 平均值，字体大小自适应列数
- 输出文件：默认 `{title_prefix}round_vs_total_tput.png`

---

### 4. 对比图表

```bash
# 对比模式：多个 CSV 绘制在同一张图上
python plot_mul.py csv1.csv csv2.csv csv3.csv --compare --titles "Model1,Model2,Model3"

# 示例
python plot_mul.py \
    "./bench_data/file1.csv" \
    "./bench_data/file2.csv" \
    --compare --titles "TP8,TP16"
```

输出文件：`TTFT-ITL-multi-turn-perf.png`

**对比图特性：**
- TTFT：折线图（左侧 Y 轴）
- ITL：柱状图（右侧 Y 轴）
- 自动取最小公共轮次数
- 不同数据集用不同颜色和线型区分

---

## 工具脚本

### estimate_chr.py — 估算多轮 KV Cache 命中率

根据输入/输出长度预测每轮的 cache hit rate。

```bash
python estimate_chr.py \
    --first-input-len 16000 \
    --subsequent-input-len 1700 \
    --output-len 500 \
    --num-rounds 32
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `--first-input-len`（必需） | 第一轮输入长度 (tokens) |
| `--subsequent-input-len`（必需） | 后续轮次输入长度 (tokens) |
| `--output-len`（必需） | 每轮输出长度 (tokens) |
| `--num-rounds`（必需） | 总轮次数 |

**估算逻辑：**

| 轮次 | 累计已缓存 tokens | 本轮 prompt 总长 | 命中率 |
|------|-------------------|------------------|--------|
| Round 0 | 0 | `first_input_len` | 0% |
| Round i (i≥1) | `first_input_len + (i-1) × (output_len + subsequent_input_len)` | `accumulated_before + output_len + subsequent_input_len` | `accumulated_before / total_prompt` |

**输出示例：**
```
  Round    Total Prompt     Cached       New  Cache Hit Rate
----------------------------------------------------------------
     0          16000         0      16000          0.0000
     1          22200     16500       1700          0.7432
     2          23900     22200       1700          0.9290
```

### estimate_concurrency.py — 估算 GPU 显存支持的最大并发数

根据模型架构和 GPU 配置估算最大并发数。

```bash
# 列出支持的模型
python estimate_concurrency.py --list-models

# 估算并发数
python estimate_concurrency.py \
    --model minimax_m2.5 \
    --gpu-memory 80 \
    --tp-size 8 \
    --context-length 16384 \
    --weight-dtype fp8 \
    --kv-cache-dtype fp8

# 指定额外的固定开销
python estimate_concurrency.py \
    --model glm5 \
    --gpu-memory 80 \
    --tp-size 8 \
    --overhead-fixed-gb 2.0 \
    --overhead-rate 0.85
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | 必需 | 模型名称（见支持列表） |
| `--total-params` | None | 覆盖模型总参数量（十亿） |
| `--gpu-memory` | 80 | 单卡显存 (GB) |
| `--tp-size` | 8 | Tensor Parallel 并行数 |
| `--context-length` | 16384 | 上下文长度 |
| `--weight-dtype` | fp8 | 权重数据类型 (fp16/bf16/fp8/int8/int4/nvfp4) |
| `--kv-cache-dtype` | fp8 | KV 缓存数据类型 |
| `--mem-fraction-static` | 0.85 | 静态显存比例 |
| `--overhead-rate` | 0.90 | KV 缓存可用比例 |
| `--overhead-fixed-gb` | 0.0 | 每卡固定扣除 (GB) |
| `--kv-cache-bytes-override` | None | 覆盖 KV 缓存池总大小 (bytes) |
| `--bytes-per-token-override` | None | 覆盖每 token KV 缓存字节数 |
| `--list-models` | False | 列出支持的模型并退出 |

**估算逻辑：**

1. **模型权重显存**：`per_gpu_weight_gb = total_params × dtype_bytes / (1024³) / tp_size`
2. **每卡 KV 缓存池**：`per_gpu_kv = gpu_mem × mem_fraction - per_gpu_weight`
3. **扣除固定开销后有效 KV**：`effective_kv = (per_gpu_kv - overhead_fixed) × overhead_rate`
4. **每 token KV 字节数**（取决于注意力机制）：
   - MLA 模型：`num_layers × (kv_lora_rank + qk_rope_head_dim [+ index_head_dim])`
   - 标准 MHA/GQA：`2 × num_kv_heads × head_dim × num_gqa_layers`
   - Gated DeltaNet：由固定线性注意力状态决定，非逐 token 增长
   - V4 (CSA+HCA)：`v4_slope_bytes × seq_len + v4_fixed_kb × 1024`
5. **每序列 KV 大小**：`single_token_bytes × context_length`（部分模型含固定开销）
6. **最大并发数**：`int(kv_cache_budget / kv_per_sequence)`

**注意：** MLA 模型的 KV 缓存不在 TP 间分片，每卡持有完整副本；非 MLA 模型的 KV 缓存在 TP 间分布。

**支持模型列表：**

| Key | 模型全称 | 注意力机制 | 总参数 | 激活参数 | 层数 |
|-----|----------|-----------|--------|----------|------|
| `deepseekv4_pro` | DeepSeek-V4-Pro | CSA + HCA | 1600B | 49B | 61 |
| `deepseekv4_flash` | DeepSeek-V4-Flash | CSA + HCA | 284B | 13B | 43 |
| `step3.7_flash` | Step-3.7-Flash | GQA (full + sliding window) | 196B | 11B | 45 |
| `qwen3.5_35b_a3b` | Qwen3.5-35B-A3B | Gated DeltaNet + GA | 35B | 3B | 40 |
| `qwen3.5_397b_a17b` | Qwen3.5-397B-A17B | Gated DeltaNet + GA | 397B | 17B | 60 |
| `qwen3.5_122b_a10b` | Qwen3.5-122B-A10B | Gated DeltaNet + GA | 122B | 10B | 48 |
| `minimax_m2.5` | MiniMax-M2.5 | Standard MHA | 230B | 10B | 62 |
| `minimax_m2` | MiniMax-M2 | Standard MHA | 230B | 10B | 62 |
| `kimi_k2.6` | Kimi-K2.6 | MLA | 1000B | 32B | 61 |
| `kimi_k2.5` | Kimi-K2.5 | MLA | 1000B | 32B | 61 |
| `glm5.1` | GLM-5.1 | MLA + DSA | 744B | 40B | 78 |
| `glm5` | GLM-5 | MLA + DSA | 744B | 40B | 78 |

---

## 一键执行

```bash
# SGLang 原生 API
python bench_multiturn.py \
    --model-path /data/models/GLM-5-NVFP4/ \
    --dataset-path /data/models/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-clients 32 --max-parallel 16 \
    --request-length 1700 --first-round-input-length 16000 \
    --output-length 500 --num-rounds 32 \
    --disable-auto-run --disable-random-sample --enable-round-barrier \
    --ready-queue-policy random

# 转换并绘图
python process_multiturn_result_to_csv.py ./bench_data
python plot_mul.py ./bench_data --model-name "GLM-5-NVFP4"
python plot_round_tput.py ./bench_data/*.csv --titles "GLM-5-NVFP4" --title-prefix "glm5_"
```

```bash
# vLLM / OpenAI-Compatible API
python bench_multiturn.py \
    --api-format openai \
    --model-path /data/models/GLM-5-NVFP4/ \
    --served-model-name GLM-5 \
    --num-clients 8 --max-parallel 8 \
    --request-length 1700 --first-round-input-length 16000 \
    --output-length 500 --num-rounds 32 \
    --port 9000 \
    --disable-auto-run --disable-random-sample --enable-round-barrier \
    --range-ratio 0.8 \
    --output-dir 6000D_GLM5_tp8_vllm
```
