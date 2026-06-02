import matplotlib
matplotlib.use('Agg')  # 无头环境

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

matplotlib.rcParams['font.family'] = 'WenQuanYi Zen Hei'
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def get_round_total_tp_columns(df):
    """提取 CSV 中 round_N_total_tp 列并按轮次排序"""
    cols = [c for c in df.columns if c.startswith('round_') and c.endswith('_total_tp')]
    cols.sort(key=lambda x: int(x.split('_')[1]))
    return cols


def plot_round_vs_total_tp(csv_files, titles, title_prefix, output_dir, output_filename):
    if len(csv_files) != len(titles):
        print('Error: csv_files and titles must have the same length')
        sys.exit(1)

    all_data = []
    used_titles = []
    min_rounds = float('inf')

    for csv_file, title in zip(csv_files, titles):
        df = pd.read_csv(csv_file)
        total_tp_cols = get_round_total_tp_columns(df)
        if not total_tp_cols:
            print(f"Warning: no round total_tp columns found in {csv_file}, skipping")
            continue
        n_rounds = len(total_tp_cols)
        min_rounds = min(min_rounds, n_rounds)
        vals = df.iloc[0][total_tp_cols].values.astype(float)
        all_data.append({
            'csv': csv_file,
            'total_tp': vals,
        })
        used_titles.append(title)

    if not all_data:
        print('Error: No valid total_tp data found in provided CSV files')
        sys.exit(1)

    # 统一轮次数为最少轮次
    for d in all_data:
        if len(d['total_tp']) > min_rounds:
            d['total_tp'] = d['total_tp'][:min_rounds]

    rounds = np.arange(1, min_rounds + 1)  # 从1开始

    fig_width = max(14, min_rounds * 0.6)
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.2)
    fig = plt.figure(figsize=(fig_width, 10))

    ax1 = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis('off')

    colors = plt.cm.tab10.colors
    line_styles = ['-', '--', '-.', ':']

    for i, (data, title) in enumerate(zip(all_data, used_titles)):
        color = colors[i % len(colors)]
        ls = line_styles[i % len(line_styles)]
        ax1.plot(
            rounds,
            data['total_tp'],
            color=color,
            linestyle=ls,
            marker='o',
            linewidth=2,
            markersize=6,
            label=title,
            zorder=3,
        )

    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Total Throughput (tokens/s)', fontsize=12)
    title_text = f'{title_prefix}round_vs_total_tput'
    ax1.set_title(title_text, fontsize=14, fontweight='bold', pad=10)

    label_interval = max(1, min_rounds // 15)
    ax1.set_xticks(rounds[::label_interval])
    ax1.set_xticklabels([str(r) for r in rounds[::label_interval]], fontsize=10)
    ax1.set_xlim(0.5, min_rounds + 0.5)

    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False)

    # ===================== 下方表格 =====================
    n_cols = min_rounds + 1
    n_rows = len(used_titles)

    col_labels = [f'Round {r}' for r in rounds] + ['Avg']
    cell_text = []
    for data in all_data:
        row = [f'{v:.0f}' for v in data['total_tp']]
        avg = np.mean(data['total_tp'])
        row.append(f'{avg:.0f}')
        cell_text.append(row)

    table = ax_table.table(
        cellText=cell_text,
        rowLabels=used_titles,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1],
    )

    table.auto_set_font_size(False)
    font_size = max(5, min(10, 120 // n_cols))
    table.set_fontsize(font_size)
    table.scale(1, 1.8)

    # 统一遍历设置样式，避免硬编码索引导致的 KeyError
    for key, cell in table.get_celld().items():
        row, col = key
        if row == -1 and col == -1:
            # 左上角表头格
            cell.set_facecolor('#2F5597')
            cell.get_text().set_color('white')
            cell.get_text().set_fontweight('bold')
        elif row == -1:
            # 列标题
            cell.set_facecolor('#4472C4')
            cell.get_text().set_color('white')
            cell.get_text().set_fontweight('bold')
        elif col == -1:
            # 行标题
            cell.set_facecolor('#4472C4')
            cell.get_text().set_color('white')
            cell.get_text().set_fontweight('bold')
        else:
            # 数据格
            if col == n_cols - 1:
                cell.set_facecolor('#D6E4F0')
            else:
                cell.set_facecolor('#f5f5f5')
            cell.get_text().set_fontweight('bold')

    if output_filename is None:
        safe_prefix = title_prefix.replace(' ', '_').replace('/', '_')
        output_filename = f'{safe_prefix}round_vs_total_tput.png'
    else:
        if not output_filename.lower().endswith('.png'):
            output_filename += '.png'

    save_path = Path(output_dir) / output_filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Plot round vs total throughput from multi-turn CSV results'
    )
    parser.add_argument(
        'csv_files',
        nargs='+',
        help='One or more CSV files generated by process_multiturn_result_to_csv.py',
    )
    parser.add_argument(
        '--titles',
        required=True,
        help='Comma-separated legend titles, e.g. "Model1,Model2"',
    )
    parser.add_argument(
        '--title-prefix',
        default='',
        help='Prefix for plot title (default: empty). Final title = {prefix}round_vs_total_tput',
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Output directory for the PNG (default: directory of the first CSV)',
    )
    parser.add_argument(
        '--output-filename',
        default=None,
        help='Output PNG filename (default: auto-generated from prefix)',
    )
    args = parser.parse_args()

    csv_files = [Path(f) for f in args.csv_files]
    for f in csv_files:
        if not f.exists():
            print(f'Error: file not found: {f}')
            sys.exit(1)

    titles = [t.strip() for t in args.titles.split(',')]
    if len(titles) != len(csv_files):
        print(f'Error: Number of titles ({len(titles)}) != number of CSV files ({len(csv_files)})')
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else csv_files[0].parent
    os.makedirs(output_dir, exist_ok=True)

    plot_round_vs_total_tp(
        csv_files=csv_files,
        titles=titles,
        title_prefix=args.title_prefix,
        output_dir=output_dir,
        output_filename=args.output_filename,
    )


if __name__ == '__main__':
    main()
