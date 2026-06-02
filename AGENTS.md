# Multi-Turn Benchmark

## 依赖

- Python 3.x
- `sglang` package (for `sglang.bench_serving`, `sglang.benchmark.*`, `sglang.test.kits.cache_hit_kit`)
- `aiohttp` (vLLM / OpenAI-compatible 模式必需)
- `requests`, `numpy`, `tqdm`, `pandas`, `matplotlib`

## 工作流

```bash
# 1. 启动推理服务器（SGLang / vLLM，端口默认 8000）
# 2. 运行基准测试
python bench_multiturn.py --model-path /data/models/GLM-5-NVFP4/ --dataset-path /data/models/ShareGPT_V3_unfiltered_cleaned_split.json --ready-queue-policy random

# 3. TXT → CSV
python process_multiturn_result_to_csv.py ./bench_data

# 4. 生成图表
python plot_mul.py ./bench_data --model-name "ModelName"

# 对比模式
python plot_mul.py csv1.csv csv2.csv --compare --titles "Model1,Model2"

# 逐轮 Total Throughput 折线图
python plot_round_tput.py csv1.csv csv2.csv --titles "Model1,Model2" --title-prefix "exp1_"
```

## 脚本概览

| 脚本 | 用途 |
|------|------|
| `bench_multiturn.py` | 多轮并发基准测试主程序 |
| `process_multiturn_result_to_csv.py` | TXT 日志 → CSV 结构化数据 |
| `plot_mul.py` | TTFT/ITL/CHR 逐轮柱状+折线图（单文件 / 对比） |
| `plot_round_tput.py` | 逐轮 Total Throughput 折线对比图 |
| `estimate_chr.py` | 估算多轮 KV Cache 命中率 |
| `estimate_concurrency.py` | 估算 GPU 显存支持的最大并发数 |

## 关键参数

| 脚本 | 重要参数 |
|------|----------|
| `bench_multiturn.py` | `--host/--port`, `--num-clients/--max-parallel`, `--api-format sglang/openai`, `--served-model-name`, `--output-dir` |
| `process_multiturn_result_to_csv.py` | `input_dir`（必需）, `output_dir`（可选，默认同输入） |
| `plot_mul.py` | `--compare`, `--no-itl`, `--output-dir`, `--titles` |
| `plot_round_tput.py` | `--titles`（必需）, `--title-prefix`, `--output-dir`, `--output-filename` |
| `estimate_chr.py` | `--first-input-len`, `--subsequent-input-len`, `--output-len`, `--num-rounds` |
| `estimate_concurrency.py` | `--model`, `--gpu-memory`, `--tp-size`, `--list-models` |

## 工具脚本

```bash
# 估算 KV Cache 命中率
python estimate_chr.py --first-input-len 16000 --subsequent-input-len 1700 --output-len 500 --num-rounds 32

# 估算最大并发数
python estimate_concurrency.py --model minimax_m2.5 --gpu-memory 80 --tp-size 8
python estimate_concurrency.py --list-models  # 查看支持的模型
```

## 注意事项

- `plot_mul.py` / `plot_round_tput.py` 必须使用 `matplotlib.use('Agg')`（无头环境），字体优先使用 WenQuanYi Zen Hei，Windows 下回退到 SimHei / DejaVu Sans
- `bench_multiturn.py` 默认 `--disable-auto-run`（禁用自动多速率测试）、`--disable-random-sample`（禁用随机采样）、`--enable-round-barrier`（启用轮次屏障）
- vLLM 模式下服务器通常不返回 `cached_tokens`，脚本会自动以 `predicted_chr` 估算 cache hit rate
- CSV 中 `cache_hit_rate` 和 `round_X_chr` 存储的是百分比值（整数），`ttft`/`itl` 单位为毫秒
- 全局 `total_tp` 定义为 `(sum(miss_tokens) + sum(generated_len)) / duration`，仅计入 miss + output
- 输出文件自动保存到 `{output_dir}/bench_data/`
- 对比图输出为 `TTFT-ITL-multi-turn-perf.png`
