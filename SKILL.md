---
name: multi-turn-bench
description: Multi-turn LLM inference benchmark toolchain — run bench, convert TXT→CSV, or generate plots. Guides user through parameter collection per module.
---

# Multi-Turn Benchmark Skill

You are operating the multi-turn LLM inference benchmark toolchain located at `D:\自动化推理测试\bench_llm_infer\multi_turn\multi_turn\`.

The toolchain has **3 modules**. You MUST first ask the user which module(s) they want to use, then collect the required parameters before executing.

## Module Selection

Ask the user to choose one or more modules:

| # | Module | Script(s) | Purpose |
|---|--------|-----------|---------|
| 1 | **Benchmark** | `bench_multiturn.py` | Run multi-turn concurrent benchmark against an LLM serving endpoint |
| 2 | **CSV Convert** | `process_multiturn_result_to_csv.py` | Convert TXT log files to structured CSV |
| 3 | **Plot** | `plot_mul.py`, `plot_round_tput.py` | Generate TTFT/ITL/CHR charts and round-throughput line charts |

---

## Module 1 — Benchmark (`bench_multiturn.py`)

### Required Parameters

Collect these from the user (show defaults in parentheses):

| Parameter | Prompt | Default |
|-----------|--------|---------|
| `--model-path` | "Model path (HuggingFace format)?" | `/data/models/GLM-5-NVFP4/` |
| `--dataset-path` | "Dataset path (ShareGPT JSON)?" | `/data/models/ShareGPT_V3_unfiltered_cleaned_split.json` |
| `--api-format` | "API format: sglang or openai?" | `sglang` |
| `--host` | "Server host?" | `localhost` |
| `--port` | "Server port?" | `8000` |

### Optional Parameters

Only ask about these if the user wants to customize; otherwise use defaults:

| Parameter | Prompt | Default |
|-----------|--------|---------|
| `--num-clients` | "Number of concurrent clients?" | `32` |
| `--max-parallel` | "Max parallel requests?" | `16` |
| `--request-length` | "Per-round input length (tokens)?" | `1700` |
| `--first-round-input-length` | "First-round input length (tokens)? (0 = use request-length)" | `16000` |
| `--output-length` | "Output length (tokens)?" | `500` |
| `--num-rounds` | "Rounds per client?" | `32` |
| `--min-rounds` | "Min rounds per client? (0 = use num-rounds)" | `0` |
| `--max-rounds` | "Max rounds per client? (0 = use num-rounds)" | `0` |
| `--request-rate` | "Request rate (req/s)?" | `10.0` |
| `--distribution` | "Request interval distribution: poisson or uniform?" | `poisson` |
| `--sub-question-input-length` | "Sub-question input length? (0 = use request-length)" | `0` |
| `--range-ratio` | "Length variation ratio? (1.0 = none, 0.5 = ±50%)" | `0.8` |
| `--seed` | "Random seed?" | `42` |
| `--served-model-name` | "Served model name? (empty = basename of model-path)" | `""` |
| `--ready-queue-policy` | "Ready queue policy: random or fifo?" | `random` |
| `--lora-path` | "LoRA adapter path? (empty = none)" | `""` |
| `--output-dir` | "Output directory?" | `.` |
| `--tag` | "Run tag?" | `""` |

### Fixed Flags (always included)

- `--disable-auto-run`
- `--disable-random-sample`
- `--enable-round-barrier`

### Execution

Build and run the command. Working directory is `D:\自动化推理测试\bench_llm_infer\multi_turn\multi_turn\`.

**SGLang example:**
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

**vLLM / OpenAI example:**
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

**Output:** `{output_dir}/bench_data/multi_turn_{model}_input{req_len}_output{out_len}_clients{n}_concurrency{max_par}.txt`

---

## Module 2 — CSV Convert (`process_multiturn_result_to_csv.py`)

### Required Parameters

| Parameter | Prompt | Default |
|-----------|--------|---------|
| `input_dir` | "Input directory containing TXT log files?" | (no default, must specify) |

### Optional Parameters

| Parameter | Prompt | Default |
|-----------|--------|---------|
| `output_dir` | "Output directory for CSV files? (leave blank = same as input)" | same as input |

### Execution

```bash
python process_multiturn_result_to_csv.py {input_dir} {output_dir}
```

If `output_dir` is not provided (user left blank), omit it from the command.

---

## Module 3 — Plot (`plot_mul.py` / `plot_round_tput.py`)

First ask the user which plot type they want:

| # | Plot Type | Script | Description |
|---|-----------|--------|-------------|
| A | **TTFT/ITL/CHR chart** | `plot_mul.py` | Per-round bar+line chart (single or comparison) |
| B | **Round throughput chart** | `plot_round_tput.py` | Per-round total throughput line chart (single or comparison) |
| C | **Both** | both | Generate both chart types |

### Plot Type A — `plot_mul.py`

#### Single file mode

| Parameter | Prompt | Default |
|-----------|--------|---------|
| `csv_path` | "CSV file path or directory?" | (must specify) |
| `--model-name` | "Model name for chart title? (leave blank = auto from filename)" | auto |
| `--no-itl` | "Hide ITL axis?" | `False` |
| `--output-dir` | "Output directory? (leave blank = CSV directory)" | CSV directory |

```bash
python plot_mul.py {csv_path} --model-name "{model_name}"
```

#### Comparison mode

| Parameter | Prompt | Default |
|-----------|--------|---------|
| `csv_files` | "CSV file paths (comma-separated)?" | (must specify, ≥2) |
| `--titles` | "Legend titles (comma-separated, one per CSV)?" | (must specify) |
| `--no-itl` | "Hide ITL axis?" | `False` |
| `--output-dir` | "Output directory?" | first CSV directory |

```bash
python plot_mul.py {csv1} {csv2} --compare --titles "{title1},{title2}"
```

**Output:** Single mode → `{csv_stem}.png`; Comparison → `TTFT-ITL-multi-turn-perf.png`

### Plot Type B — `plot_round_tput.py`

| Parameter | Prompt | Default |
|-----------|--------|---------|
| `csv_files` | "CSV file paths (space-separated)?" | (must specify) |
| `--titles` | "Legend titles (comma-separated, one per CSV)?" | (must specify) |
| `--title-prefix` | "Title prefix? (final title = {prefix}round_vs_total_tput)" | `""` |
| `--output-dir` | "Output directory?" | first CSV directory |
| `--output-filename` | "Output filename? (leave blank = auto)" | auto |

```bash
python plot_round_tput.py {csv1} {csv2} --titles "{title1},{title2}" --title-prefix "{prefix}"
```

**Output:** `{prefix}round_vs_total_tput.png` or custom filename.

---

## Important Notes

- Working directory for all commands: `D:\自动化推理测试\bench_llm_infer\multi_turn\multi_turn\`
- The `bench_multiturn.py` script requires `sglang` package and optionally `aiohttp` for vLLM mode.
- In vLLM (openai) mode, the server usually does not return `cached_tokens`, so `cache_hit_rate` will show as 0. Use `predicted_chr` as reference.
- CSV stores `cache_hit_rate` and `round_X_chr` as integer percentages; `ttft`/`itl` in milliseconds.
- `total_tp` = `(sum(miss_tokens) + sum(generated_len)) / duration` — only counts miss + output tokens.
- Plot scripts use `matplotlib.use('Agg')` for headless environments. Font falls back to SimHei/DejaVu Sans on Windows.
- If the user wants to run all 3 modules sequentially (benchmark → CSV → plot), collect parameters for each module upfront, then execute in order.
