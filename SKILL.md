---
name: multi-turn-bench
description: Multi-turn LLM inference benchmark toolchain — run bench, convert TXT→CSV, generate plots, estimate CHR/concurrency. Fully self-contained guide for any agent to operate the toolchain.
---

# Multi-Turn Benchmark Skill

You are operating the multi-turn LLM inference benchmark toolchain. This skill provides everything you need to run benchmarks, process results, and generate visualizations.

## Toolchain Location

All scripts reside in `D:\自动化推理测试\bench_llm_infer\multi_turn\multi_turn\`. All commands below assume this as the working directory.

## Scripts Overview

| Script | Purpose |
|--------|---------|
| `bench_multiturn.py` | Multi-turn concurrent benchmark against an LLM serving endpoint |
| `process_multiturn_result_to_csv.py` | Convert TXT log files to structured CSV |
| `plot_mul.py` | TTFT/ITL/CHR per-round bar+line charts (single or comparison) |
| `plot_round_tput.py` | Per-round total throughput line charts (single or comparison) |
| `estimate_chr.py` | Estimate per-round KV Cache hit rate from input/output lengths |
| `estimate_concurrency.py` | Estimate max concurrency from GPU memory and model architecture |

## Dependencies

- Python 3.x
- `sglang` (provides `sglang.bench_serving`, `sglang.benchmark.datasets.random`, `sglang.test.kits.cache_hit_kit`)
- `aiohttp` (required for vLLM / OpenAI-compatible mode)
- `requests`, `numpy`, `tqdm`, `pandas`, `matplotlib`

---

## Workflow

The standard workflow is: **Benchmark → CSV Convert → Plot**.

```
1. Start inference server (SGLang / vLLM, default port 8000)
2. Run bench_multiturn.py → produces .txt log in {output_dir}/bench_data/
3. Run process_multiturn_result_to_csv.py → converts .txt to .csv
4. Run plot_mul.py / plot_round_tput.py → generates .png charts
```

---

## Module 1 — Benchmark (`bench_multiturn.py`)

### When to Use

Run this when you need to measure multi-turn concurrent LLM inference performance: TTFT, ITL, throughput, and KV Cache hit rate per round.

### Required Parameters

| Parameter | Prompt to User | Default |
|-----------|---------------|---------|
| `--model-path` | "Model path (HuggingFace format)?" | `/data/models/GLM-5-NVFP4/` |
| `--dataset-path` | "Dataset path (ShareGPT JSON)?" | `/data/models/ShareGPT_V3_unfiltered_cleaned_split.json` |
| `--api-format` | "API format: sglang or openai?" | `sglang` |
| `--host` | "Server host?" | `localhost` |
| `--port` | "Server port?" | `8000` |

### Optional Parameters (use defaults unless user specifies)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-clients` | 32 | Concurrent clients |
| `--max-parallel` | 16 | Max parallel requests |
| `--request-length` | 1700 | Per-round input tokens (subsequent rounds) |
| `--first-round-input-length` | 16000 | First-round input tokens (0 = use request-length) |
| `--output-length` | 500 | Output tokens per round |
| `--num-rounds` | 32 | Rounds per client |
| `--min-rounds` | 0 | Min rounds per client (0 = use num-rounds) |
| `--max-rounds` | 0 | Max rounds per client (0 = use num-rounds) |
| `--request-rate` | 10.0 | Request rate (req/s) |
| `--distribution` | poisson | Interval distribution: poisson or uniform |
| `--sub-question-input-length` | 0 | Sub-question input length (0 = use request-length) |
| `--range-ratio` | 0.8 | Length variation (1.0 = none, 0.5 = ±50%) |
| `--seed` | 42 | Random seed |
| `--served-model-name` | "" | Server-side model name (empty = basename of model-path) |
| `--ready-queue-policy` | random | Ready queue policy: random or fifo |
| `--lora-path` | "" | LoRA adapter path |
| `--output-dir` | . | Output directory |
| `--tag` | "" | Run tag (written to JSONL) |
| `--log-file` | performance_metrics.jsonl | JSONL log filename |

### Fixed Flags (always included)

- `--disable-auto-run` — disable automatic multi-rate testing
- `--disable-random-sample` — disable random sampling
- `--enable-round-barrier` — only send round i after all round i-1 complete

### Execution Commands

**SGLang native API:**
```bash
python bench_multiturn.py \
    --model-path {model_path} \
    --dataset-path {dataset_path} \
    --api-format sglang \
    --host {host} --port {port} \
    --num-clients {num_clients} --max-parallel {max_parallel} \
    --request-length {req_len} --first-round-input-length {first_len} \
    --output-length {out_len} --num-rounds {num_rounds} \
    --request-rate {rate} --distribution {dist} \
    --range-ratio {ratio} --seed {seed} \
    --ready-queue-policy {policy} \
    --disable-auto-run --disable-random-sample --enable-round-barrier \
    --output-dir {output_dir}
```

**vLLM / OpenAI-compatible API:**
```bash
python bench_multiturn.py \
    --api-format openai \
    --model-path {model_path} \
    --served-model-name {served_name} \
    --dataset-path {dataset_path} \
    --host {host} --port {port} \
    --num-clients {num_clients} --max-parallel {max_parallel} \
    --request-length {req_len} --first-round-input-length {first_len} \
    --output-length {out_len} --num-rounds {num_rounds} \
    --request-rate {rate} \
    --range-ratio {ratio} --seed {seed} \
    --ready-queue-policy {policy} \
    --disable-auto-run --disable-random-sample --enable-round-barrier \
    --output-dir {output_dir}
```

### Output

- TXT log: `{output_dir}/bench_data/multi_turn_{model}_input{req_len}_output{out_len}_clients{n}_concurrency{max_par}.txt`
- JSONL log: same filename with `.jsonl` extension

### Key Metrics (computed in script)

| Metric | Formula |
|--------|---------|
| `cache_hit_rate` | `sum(cached_tokens) / sum(prompt_len)` |
| `miss_tokens` | `max(0, prompt_len - estimated_cached_tokens)` |
| `total_tp` | `(sum(miss_tokens) + sum(generated_len)) / duration` |
| `predicted_chr` (per round) | Round 0 = 0; Round i = `(prompt_len - sub_q_len) / prompt_len` |
| `ITL` (per token) | If `gen_len > 1 && latency > ttft`: `(latency - ttft) / (gen_len - 1)` |

### vLLM Important Notes

- vLLM server usually does **not** return `cached_tokens`. The script auto-estimates `cache_hit_rate` using `predicted_chr`.
- To get real `cached_tokens`, start vLLM with `--enable-prompt-tokens-details` flag (this is a **vLLM server** flag, NOT a bench script flag).
- The script auto-detects if `prompt_tokens_details` is missing and warns after 5 responses.
- Always set `--served-model-name` if vLLM was started with a custom model name, otherwise the script uses `--model-path` basename which may cause 404 errors.
- The script auto-skips `/flush_cache` and heartbeat requests for vLLM (not supported by OpenAI API).

---

## Module 2 — CSV Convert (`process_multiturn_result_to_csv.py`)

### When to Use

After running benchmarks, convert the TXT log files to structured CSV for plotting or analysis.

### Required Parameters

| Parameter | Prompt to User | Default |
|-----------|---------------|---------|
| `input_dir` | "Input directory containing TXT log files?" | (must specify) |

### Optional Parameters

| Parameter | Prompt to User | Default |
|-----------|---------------|---------|
| `output_dir` | "Output directory for CSV files? (leave blank = same as input)" | same as input |

### Execution

```bash
# Same output dir
python process_multiturn_result_to_csv.py {input_dir}

# Custom output dir
python process_multiturn_result_to_csv.py {input_dir} {output_dir}
```

### CSV Structure

**Fixed columns:** `filename`, `input_tokens`, `output_tokens`, `concurrency`, `clients`, `avg_ttft`, `p99_ttft`, `median_ttft`, `avg_itl`, `p99_itl`, `median_itl`, `input_tp`, `output_tp`, `total_tp`, `cache_hit_rate`, `total_hit_tokens`, `total_miss_tokens`

**Per-round dynamic columns** (9 columns per round N):
`round_N_ttft`, `round_N_itl`, `round_N_chr`, `round_N_predicted_chr`, `round_N_hit_tokens`, `round_N_miss_tokens`, `round_N_output_tp`, `round_N_total_tp`, `round_N_duration`

**Units:** `ttft`/`itl` = milliseconds, `cache_hit_rate`/`chr` = integer percentage, `duration` = seconds (int truncated)

---

## Module 3 — Plot (`plot_mul.py` / `plot_round_tput.py`)

### When to Use

Generate visualization charts from CSV data. Two chart types available.

### Plot Type Selection

Ask user which chart type they want:

| # | Type | Script | Description |
|---|------|--------|-------------|
| A | **TTFT/ITL/CHR chart** | `plot_mul.py` | Per-round bar+line chart (single or comparison) |
| B | **Round throughput chart** | `plot_round_tput.py` | Per-round total throughput line chart (single or comparison) |
| C | **Both** | both | Generate both chart types |

### Plot Type A — `plot_mul.py`

#### Single File Mode

| Parameter | Prompt | Default |
|-----------|--------|---------|
| `csv_path` | "CSV file path or directory?" | (must specify) |
| `--model-name` | "Model name for chart title? (leave blank = auto from filename)" | auto |
| `--no-itl` | "Hide ITL axis?" | False |
| `--output-dir` | "Output directory? (leave blank = CSV directory)" | CSV directory |

```bash
python plot_mul.py {csv_path} --model-name "{model_name}"
```

**Output:** `{csv_stem}.png` — TTFT bar chart (left Y) + ITL line chart (right Y) + data table below

#### Comparison Mode

| Parameter | Prompt | Default |
|-----------|--------|---------|
| `csv_files` | "CSV file paths (space-separated)?" | (must specify, ≥2) |
| `--titles` | "Legend titles (comma-separated, one per CSV)?" | (must specify) |
| `--no-itl` | "Hide ITL axis?" | False |
| `--output-dir` | "Output directory?" | first CSV directory |

```bash
python plot_mul.py {csv1} {csv2} --compare --titles "{title1},{title2}"
```

**Output:** `TTFT-ITL-multi-turn-perf.png` — TTFT line chart (left Y) + ITL grouped bar chart (right Y), auto-aligned to minimum common rounds

### Plot Type B — `plot_round_tput.py`

| Parameter | Prompt | Default |
|-----------|--------|---------|
| `csv_files` | "CSV file paths (space-separated)?" | (must specify) |
| `--titles` | "Legend titles (comma-separated)?" | (must specify) |
| `--title-prefix` | "Title prefix? (final title = {prefix}round_vs_total_tput)" | "" |
| `--output-dir` | "Output directory?" | first CSV directory |
| `--output-filename` | "Output filename? (leave blank = auto)" | auto |

```bash
python plot_round_tput.py {csv1} {csv2} --titles "{title1},{title2}" --title-prefix "{prefix}"
```

**Output:** `{prefix}round_vs_total_tput.png` — multi-line chart with data table below, X-axis rounds start from 1

---

## Module 4 — Estimate CHR (`estimate_chr.py`)

### When to Use

Predict per-round KV Cache hit rate based on input/output token lengths, without running an actual benchmark.

### Required Parameters

| Parameter | Prompt | Default |
|-----------|--------|---------|
| `--first-input-len` | "First-round input length (tokens)?" | (must specify) |
| `--subsequent-input-len` | "Subsequent round input length (tokens)?" | (must specify) |
| `--output-len` | "Per-round output length (tokens)?" | (must specify) |
| `--num-rounds` | "Total number of rounds?" | (must specify) |

### Execution

```bash
python estimate_chr.py \
    --first-input-len 16000 \
    --subsequent-input-len 1700 \
    --output-len 500 \
    --num-rounds 32
```

### Estimation Logic

- Round 0: `cached = 0`, `hit_rate = 0%`
- Round i (i≥1): `cached = first_input_len + (i-1) × (output_len + subsequent_input_len)`, `total_prompt = cached + output_len + subsequent_input_len`, `hit_rate = cached / total_prompt`

---

## Module 5 — Estimate Concurrency (`estimate_concurrency.py`)

### When to Use

Estimate the maximum number of concurrent sequences a GPU cluster can support, based on model architecture and memory constraints.

### Quick Commands

```bash
# List all supported models
python estimate_concurrency.py --list-models

# Estimate concurrency
python estimate_concurrency.py --model minimax_m2.5 --gpu-memory 80 --tp-size 8

# With fixed overhead
python estimate_concurrency.py --model glm5 --gpu-memory 80 --tp-size 8 --overhead-fixed-gb 2.0 --overhead-rate 0.85
```

### Required Parameters

| Parameter | Prompt | Default |
|-----------|--------|---------|
| `--model` | "Model name? (use --list-models to see available)" | (must specify) |

### Optional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--total-params` | None | Override total params (billions) |
| `--gpu-memory` | 80 | GPU memory per card (GB) |
| `--tp-size` | 8 | Tensor parallelism size |
| `--context-length` | 16384 | Context length |
| `--weight-dtype` | fp8 | Weight dtype (fp16/bf16/fp8/int8/int4/nvfp4) |
| `--kv-cache-dtype` | fp8 | KV cache dtype |
| `--mem-fraction-static` | 0.85 | Static memory fraction |
| `--overhead-rate` | 0.90 | KV cache usable fraction |
| `--overhead-fixed-gb` | 0.0 | Fixed deduction per GPU (GB) |
| `--kv-cache-bytes-override` | None | Override KV cache pool size (bytes) |
| `--bytes-per-token-override` | None | Override bytes per token |

### Supported Models

| Key | Model | Attention | Total Params | Active Params | Layers |
|-----|-------|-----------|-------------|---------------|--------|
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

### Estimation Logic

1. Model weight per GPU: `total_params × dtype_bytes / (1024³) / tp_size`
2. KV pool per GPU: `gpu_mem × mem_fraction - per_gpu_weight`
3. Effective KV: `(per_gpu_kv - overhead_fixed) × overhead_rate`
4. Bytes per token depends on attention mechanism:
   - **MLA**: `num_layers × (kv_lora_rank + qk_rope_head_dim [+ index_head_dim])`
   - **MHA/GQA**: `2 × num_kv_heads × head_dim × num_gqa_layers`
   - **Gated DeltaNet**: fixed linear attention state (not per-token)
   - **V4 (CSA+HCA)**: `v4_slope_bytes × seq_len + v4_fixed_kb × 1024`
5. Per-sequence KV: `single_token_bytes × context_length` (some models add fixed overhead)
6. Max concurrency: `int(kv_cache_budget / kv_per_sequence)`

**Important:** MLA models do NOT shard KV across TP — each GPU holds a full copy. Non-MLA models shard KV across TP.

---

## End-to-End Example

When a user wants to run a full benchmark pipeline:

```
Step 1: Collect benchmark parameters from user
Step 2: Run bench_multiturn.py
Step 3: Run process_multiturn_result_to_csv.py on the output directory
Step 4: Run plot_mul.py and/or plot_round_tput.py on the CSV files
```

**SGLang full pipeline:**
```bash
python bench_multiturn.py \
    --model-path /data/models/GLM-5-NVFP4/ \
    --dataset-path /data/models/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-clients 32 --max-parallel 16 \
    --request-length 1700 --first-round-input-length 16000 \
    --output-length 500 --num-rounds 32 \
    --disable-auto-run --disable-random-sample --enable-round-barrier \
    --ready-queue-policy random

python process_multiturn_result_to_csv.py ./bench_data
python plot_mul.py ./bench_data --model-name "GLM-5-NVFP4"
python plot_round_tput.py ./bench_data/*.csv --titles "GLM-5-NVFP4" --title-prefix "glm5_"
```

**vLLM full pipeline:**
```bash
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

python process_multiturn_result_to_csv.py ./6000D_GLM5_tp8_vllm/bench_data
python plot_mul.py ./6000D_GLM5_tp8_vllm/bench_data --model-name "GLM-5-NVFP4"
python plot_round_tput.py ./6000D_GLM5_tp8_vllm/bench_data/*.csv --titles "GLM-5-NVFP4" --title-prefix "vllm_glm5_"
```

---

## Important Notes

- Working directory for all commands: `D:\自动化推理测试\bench_llm_infer\multi_turn\multi_turn\`
- `bench_multiturn.py` requires `sglang` package; vLLM mode additionally requires `aiohttp`
- In vLLM (openai) mode, `cache_hit_rate` may show 0 if vLLM server lacks `--enable-prompt-tokens-details`. Use `predicted_chr` as reference.
- CSV stores `cache_hit_rate` and `round_X_chr` as integer percentages; `ttft`/`itl` in milliseconds
- `total_tp` = `(sum(miss_tokens) + sum(generated_len)) / duration` — only counts miss + output tokens
- Plot scripts use `matplotlib.use('Agg')` for headless environments. Font: WenQuanYi Zen Hei → SimHei → DejaVu Sans fallback
- If running all modules sequentially (benchmark → CSV → plot), collect parameters for each module upfront, then execute in order
