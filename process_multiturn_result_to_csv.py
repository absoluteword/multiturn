import os
import re
import csv
import argparse
from pathlib import Path
from collections import defaultdict

def parse_filename(filename):
    """从文件名解析 input, output, clients, concurrency"""
    input_val = None
    output_val = None
    concurrency_val = None

    input_match = re.search(r'input(\d+)', filename)
    output_match = re.search(r'output(\d+)', filename)
    concurrency_match = re.search(r'concurrency(\d+)', filename)

    if input_match:
        input_val = int(input_match.group(1))
    if output_match:
        output_val = int(output_match.group(1))
    if concurrency_match:
        concurrency_val = int(concurrency_match.group(1))

    return input_val, output_val, concurrency_val

def parse_log_content(content):
    """解析所有指标：全局指标 + 逐轮指标"""
    result = {}

    # 0. Clients 数量（从 Round 0 中提取）
    clients_match = re.search(r'Round 0:.*?\((\d+) requests?, (\d+) clients?\)', content)
    if clients_match:
        result['clients'] = int(clients_match.group(2))
    else:
        result['clients'] = None

    # 1. TTFT
    ttft_avg = re.search(r'Average TTFT:\s*([\d.]+)', content)
    ttft_p99 = re.search(r'P99 TTFT:\s*([\d.]+)', content)
    ttft_median = re.search(r'Median TTFT:\s*([\d.]+)', content)
    result['avg_ttft'] = int(float(ttft_avg.group(1)) * 1000) if ttft_avg else None
    result['p99_ttft'] = int(float(ttft_p99.group(1)) * 1000) if ttft_p99 else None
    result['median_ttft'] = int(float(ttft_median.group(1)) * 1000) if ttft_median else None

    # 2. ITL
    itl_avg = re.search(r'Average ITL:\s*([\d.]+)', content)
    itl_p99 = re.search(r'P99 ITL:\s*([\d.]+)', content)
    itl_median = re.search(r'Median ITL:\s*([\d.]+)', content)
    result['avg_itl'] = int(float(itl_avg.group(1)) * 1000) if itl_avg else None
    result['p99_itl'] = int(float(itl_p99.group(1)) * 1000) if itl_p99 else None
    result['median_itl'] = int(float(itl_median.group(1)) * 1000) if itl_median else None

    # 3. Throughput
    in_tp = re.search(r'Input token throughput:\s*([\d.]+)', content)
    out_tp = re.search(r'Output token throughput:\s*([\d.]+)', content)
    total_tp = re.search(r'Total token throughput \(miss\+out\):\s*([\d.]+)', content)
    result['input_tp'] = int(float(in_tp.group(1))) if in_tp else None
    result['output_tp'] = int(float(out_tp.group(1))) if out_tp else None
    result['total_tp'] = int(float(total_tp.group(1))) if total_tp else None

    # 4. 全局 Cache Hit Rate & Token 统计
    chr = re.search(r'Cache Hit Rate:\s*([\d.]+)', content)
    result['cache_hit_rate'] = int(float(chr.group(1)) * 100) if chr else None

    hit_tokens = re.search(r'Total Hit Tokens:\s*(\d+)', content)
    miss_tokens = re.search(r'Total Miss Tokens:\s*(\d+)', content)
    result['total_hit_tokens'] = int(hit_tokens.group(1)) if hit_tokens else None
    result['total_miss_tokens'] = int(miss_tokens.group(1)) if miss_tokens else None

    # 5. 逐轮指标（Round 0~N）
    round_pattern = re.compile(
        r'Round (\d+): Average TTFT = ([\d.]+)s, '
        r'Average ITL = ([\d.]+)s, '
        r'P90 ITL = ([\d.]+)s, '
        r'P99 ITL = ([\d.]+)s, '
        r'Median ITL = ([\d.]+)s, '
        r'Max ITL = ([\d.]+)s, '
        r'Cache Hit Rate = ([\d.]+) \(predicted: ([\d.]+)\), '
        r'Hit Tokens = (\d+), '
        r'Miss Tokens = (\d+), '
        r'Output Tput = ([\d.]+) tok/s, '
        r'Total Tput = ([\d.]+) tok/s, '
        r'Duration = ([\d.]+)s'
    )
    rounds = round_pattern.findall(content)
    for rnd, ttft, itl, _, _, _, _, chr_val, pred_chr, hit_toks, miss_toks, out_tp, total_tp, duration in rounds:
        result[f'round_{rnd}_ttft'] = int(float(ttft) * 1000)
        result[f'round_{rnd}_itl'] = int(float(itl) * 1000)
        result[f'round_{rnd}_chr'] = int(float(chr_val) * 100)
        result[f'round_{rnd}_predicted_chr'] = int(float(pred_chr) * 100)
        result[f'round_{rnd}_hit_tokens'] = int(hit_toks)
        result[f'round_{rnd}_miss_tokens'] = int(miss_toks)
        result[f'round_{rnd}_output_tp'] = int(float(out_tp))
        result[f'round_{rnd}_total_tp'] = int(float(total_tp))
        result[f'round_{rnd}_duration'] = int(float(duration))

    return result

def process_single_txt(txt_file, csv_file):
    """将单个txt文件转换为csv文件"""
    with open(txt_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    input_val, output_val, concur_val = parse_filename(txt_file.name)
    parsed = parse_log_content(content)
    
    row = {
        "filename": txt_file.name,
        "input_tokens": input_val,
        "output_tokens": output_val,
        "concurrency": concur_val,
        **parsed
    }
    
    all_rounds = set()
    for key in row:
        m = re.match(r'round_(\d+)_', key)
        if m:
            all_rounds.add(int(m.group(1)))
    
    headers = [
        "filename",
        "input_tokens",
        "output_tokens",
        "concurrency",
        "clients",
        "avg_ttft",
        "p99_ttft",
        "median_ttft",
        "avg_itl",
        "p99_itl",
        "median_itl",
        "input_tp",
        "output_tp",
        "total_tp",
        "cache_hit_rate",
        "total_hit_tokens",
        "total_miss_tokens",
    ]
    for rnd in sorted(all_rounds):
        headers.append(f"round_{rnd}_ttft")
        headers.append(f"round_{rnd}_itl")
        headers.append(f"round_{rnd}_chr")
        headers.append(f"round_{rnd}_predicted_chr")
        headers.append(f"round_{rnd}_hit_tokens")
        headers.append(f"round_{rnd}_miss_tokens")
        headers.append(f"round_{rnd}_output_tp")
        headers.append(f"round_{rnd}_total_tp")
        headers.append(f"round_{rnd}_duration")
    
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerow(row)
    
    return True

def process_log_dir(input_dir, output_dir=None):
    """遍历文件夹下的所有txt文件，转换为同名csv文件"""
    if output_dir is None:
        output_dir = input_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    txt_files = list(Path(input_dir).glob("*.txt"))
    if not txt_files:
        print("未找到日志文件")
        return
    
    processed_count = 0
    for txt_file in txt_files:
        csv_filename = txt_file.stem + ".csv"
        csv_file = os.path.join(output_dir, csv_filename)
        
        try:
            process_single_txt(txt_file, csv_file)
            print(f"[OK] {csv_filename}")
            processed_count += 1
        except Exception as e:
            print(f"[FAIL] {txt_file.name} failed: {e}")
    
    print(f"\n处理完成！共转换 {processed_count} 个文件")
    print(f"输出目录：{output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量解析SGLang日志 → CSV")
    parser.add_argument("input_dir", help="日志文件夹路径")
    parser.add_argument("output_dir", nargs="?", default=None, help="输出CSV文件夹路径（默认与输入相同）")
    args = parser.parse_args()
    process_log_dir(args.input_dir, args.output_dir)