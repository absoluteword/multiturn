import matplotlib
matplotlib.use('Agg')  # 服务器无桌面必须加这行

# 强制指定中文黑体（Ubuntu 自带，绝对可用）
matplotlib.rcParams['font.family'] = 'WenQuanYi Zen Hei'
matplotlib.rcParams['axes.unicode_minus'] = False  # 负号正常

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import argparse
import os
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
#plt.rcParams['font.family'] = 'sans-serif'
#plt.rcParams['font.sans-serif'] = [
#    'WenQuanYi Zen Hei',  # Linux 自带中文黑体
#    'SimHei',             # Windows 黑体
#    'Microsoft YaHei',    # 微软雅黑
#    'Arial'
#]
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示

def get_round_columns(df, suffix):
    cols = [c for c in df.columns if c.startswith('round_') and c.endswith('_' + suffix) and '_predicted_' not in c]
    cols.sort(key=lambda x: int(x.split('_')[1]))
    return cols


def plot_concurrency(df, conc_group, conc_val, input_tokens, output_tokens, title_text, output_dir, csv_stem, no_itl=False):
    round_ttft_cols = get_round_columns(conc_group, 'ttft')
    round_itl_cols = get_round_columns(conc_group, 'itl')
    round_chr_cols = get_round_columns(conc_group, 'chr')

    if not round_ttft_cols:
        print(f"  No round data for concurrency {conc_val}, skipping")
        return

    n_rounds = len(round_ttft_cols)
    rounds = list(range(n_rounds))

    ttft_vals = conc_group[round_ttft_cols].iloc[0].values.astype(float)
    chr_vals = conc_group[round_chr_cols].iloc[0].values.astype(float)

    avg_ttft = float(conc_group['avg_ttft'].iloc[0])
    cache_hit_rate = float(conc_group['cache_hit_rate'].iloc[0])

    if not no_itl and round_itl_cols:
        itl_vals = conc_group[round_itl_cols].iloc[0].values.astype(float)
        avg_itl = float(conc_group['avg_itl'].iloc[0])

    fig_width = max(14, n_rounds * 0.5)

    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.15)
    fig = plt.figure(figsize=(fig_width, 10))

    ax1 = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis('off')

    bar_width = 0.5
    x = np.arange(n_rounds)
    bars = ax1.bar(x, ttft_vals, bar_width, color='#4472C4', label='TTFT', zorder=3)

    for i, (bar, v) in enumerate(zip(bars, ttft_vals)):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015 * max(ttft_vals),
                 f'{int(round(v))}', ha='center', va='bottom', fontsize=6, fontweight='bold')

    if not no_itl and round_itl_cols:
        ax2 = ax1.twinx()
        ax2.plot(x, itl_vals, 'o-', color='#ED7D31', linewidth=2, markersize=6, label='ITL', zorder=4)
        for i, (xi, yi) in enumerate(zip(x, itl_vals)):
            offset_y = 12 if i % 2 == 0 else -20
            ax2.annotate(f'{int(round(yi))}', (xi, yi), textcoords="offset points", xytext=(0, offset_y),
                         ha='center', fontsize=6, color='#ED7D31', fontweight='bold')
        ax2.set_ylabel('ITL (ms)', fontsize=12)

    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('TTFT (ms)', fontsize=12)

    title = f'{title_text} | input={input_tokens} output={output_tokens}'
    ax1.set_title(title, fontsize=14, fontweight='bold', pad=10)

    label_interval = max(1, n_rounds // 15)
    ax1.set_xticks(x[::label_interval])
    ax1.set_xticklabels([str(r) for r in rounds[::label_interval]], fontsize=8)
    ax1.set_xlim(-0.6, n_rounds - 0.4)

    lines1, labels1 = ax1.get_legend_handles_labels()
    if not no_itl and round_itl_cols:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        ax1.legend(lines1, labels1, loc='upper left')

    ax1.grid(axis='y', alpha=0.3)
    ax1.spines[['top']].set_visible(False)

    round_labels = [f'{r}' for r in rounds] + ['Avg']
    ttft_row = [f'{int(round(v))}' for v in ttft_vals] + [f'{avg_ttft:.0f}']
    chr_row = [f'{v:.0f}' for v in chr_vals] + [f'{cache_hit_rate:.0f}']

    if no_itl or not round_itl_cols:
        table_data = [round_labels, ttft_row, chr_row]
        row_labels = ['Round', 'TTFT', 'CHR%']
        n_rows = 3
    else:
        itl_row = [f'{int(round(v))}' for v in itl_vals] + [f'{avg_itl:.0f}']
        table_data = [round_labels, ttft_row, chr_row, itl_row]
        row_labels = ['Round', 'TTFT', 'CHR%', 'ITL']
        n_rows = 4

    n_cols = n_rounds + 1

    table = ax_table.table(cellText=table_data,
                           rowLabels=row_labels,
                           colLabels=None,
                           cellLoc='center', loc='center', bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    font_size = max(5, min(9, 50 // n_rounds))
    table.set_fontsize(font_size)
    table.scale(1, 1.8)

    for r in range(n_rows):
        for c in range(n_cols):
            cell = table[(r, c)]
            if c == n_cols - 1:
                cell.set_facecolor('#D6E4F0')
            else:
                cell.set_facecolor('#f5f5f5')
            cell.get_text().set_fontweight('bold')

    for r in range(n_rows):
        table[(r, -1)].set_facecolor('#4472C4')
        table[(r, -1)].get_text().set_color('white')
        table[(r, -1)].get_text().set_fontweight('bold')

    png_stem = csv_stem
    save_path = output_dir / f'{png_stem}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_multi_csv_comparison(csv_files, titles, output_dir, no_itl=False):
    """绘制多个CSV的对比图：TTFT折线 + ITL柱状图（双Y轴）"""
    if len(csv_files) != len(titles):
        print("Error: csv_files and titles must have the same length")
        return
    
    colors = plt.cm.tab10.colors
    line_styles = ['-', '--', '-.', ':']
    
    all_data = []
    min_rounds = float('inf')
    
    for csv_file, title in zip(csv_files, titles):
        df = pd.read_csv(csv_file)
        row = df.iloc[0]
        
        round_ttft_cols = get_round_columns(df, 'ttft')
        round_itl_cols = get_round_columns(df, 'itl')
        
        n_rounds = len(round_ttft_cols)
        min_rounds = min(min_rounds, n_rounds)
        
        ttft_vals = row[round_ttft_cols].values.astype(float)
        if not no_itl and round_itl_cols:
            itl_vals = row[round_itl_cols].values.astype(float)
        
        if n_rounds > min_rounds:
            ttft_vals = ttft_vals[:min_rounds]
            if not no_itl and round_itl_cols:
                itl_vals = itl_vals[:min_rounds]
        
        data_entry = {
            'title': title,
            'ttft': ttft_vals,
        }
        if not no_itl and round_itl_cols:
            data_entry['itl'] = itl_vals
        
        all_data.append(data_entry)
    
    if min_rounds == float('inf'):
        print("Error: No valid data found")
        return
    
    x = np.arange(min_rounds)
    
    fig_width = max(14, min_rounds * 0.4)
    fig, ax1 = plt.subplots(figsize=(fig_width, 8))
    
    if not no_itl:
        ax2 = ax1.twinx()
        bar_width = 0.15
        n_bars = len(csv_files)
        offset_base = -(n_bars - 1) * bar_width / 2
        
        for i, data in enumerate(all_data):
            if 'itl' in data:
                offset = offset_base + i * bar_width
                ax2.bar(x + offset, data['itl'], bar_width * 0.9, 
                        color=colors[i % len(colors)], alpha=0.7, label=f'{data["title"]} ITL', zorder=3)
    
    for i, data in enumerate(all_data):
        ls = line_styles[i % len(line_styles)]
        label = data['title']
        ax1.plot(x, data['ttft'], 'o-', color=colors[i % len(colors)], 
                 linewidth=2, markersize=6, linestyle=ls, label=f'{label} TTFT', zorder=4)
    
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('TTFT (ms)', fontsize=12)
    if not no_itl:
        ax2.set_ylabel('ITL (ms)', fontsize=12)
    ax1.set_title('Multi-turn Performance Comparison', fontsize=14, fontweight='bold', pad=15)
    
    label_interval = max(1, min_rounds // 15)
    ax1.set_xticks(x[::label_interval])
    ax1.set_xticklabels([str(r) for r in range(0, min_rounds, label_interval)], fontsize=10)
    ax1.set_xlim(-0.6, min_rounds - 0.4)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    if not no_itl:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9, ncol=2)
    else:
        ax1.legend(lines1, labels1, loc='upper left', fontsize=9)
    
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines[['top']].set_visible(False)
    
    plt.tight_layout()
    
    save_path = output_dir / 'TTFT-ITL-multi-turn-perf.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_single_csv(csv_file, model_name=None, output_dir=None, no_itl=False):
    """为单个csv文件生成图表"""
    csv_path = Path(csv_file)
    csv_stem = csv_path.stem
    df = pd.read_csv(csv_path)
    
    if model_name is None:
        model_name = csv_stem.split('_input')[0]
    
    out_dir = Path(output_dir) if output_dir else csv_path.parent
    
    for _, row in df.iterrows():
        input_tokens = int(row['input_tokens'])
        output_tokens = int(row['output_tokens'])
        conc_val = row['concurrency']

        conc_df = pd.DataFrame([row])
        plot_concurrency(df, conc_df, conc_val, input_tokens, output_tokens, model_name,
                         out_dir, csv_stem, no_itl=no_itl)


def main():
    parser = argparse.ArgumentParser(description='Multi-turn performance plotting')
    parser.add_argument('csv_input', type=str, nargs='+', help='CSV file(s) or directory containing CSV files')
    parser.add_argument('--model-name', type=str, default=None,
                        help='Model name for plot title (default: extracted from filename)')
    parser.add_argument('--compare', action='store_true', default=False,
                        help='Enable multi-CSV comparison mode')
    parser.add_argument('--titles', type=str, default=None,
                        help='Comma-separated titles for comparison mode (e.g., "Model1,Model2,Model3")')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for saved plots (default: same dir as CSV)')
    parser.add_argument('--no-itl', action='store_true', default=False,
                        help='Hide ITL axis and data in plots')
    args = parser.parse_args()

    if args.compare:
        if not args.titles:
            print("Error: --titles is required when --compare is enabled")
            return
        
        csv_files = [Path(f) for f in args.csv_input]
        if len(csv_files) < 2:
            print("Error: --compare mode requires at least 2 CSV files")
            return
        
        titles = [t.strip() for t in args.titles.split(',')]
        if len(csv_files) != len(titles):
            print(f"Error: Number of CSV files ({len(csv_files)}) != number of titles ({len(titles)})")
            return
        
        output_dir = Path(args.output_dir) if args.output_dir else csv_files[0].parent
        print(f"Generating comparison plot for {len(csv_files)} files...")
        plot_multi_csv_comparison(csv_files, titles, output_dir, no_itl=args.no_itl)
    else:
        if len(args.csv_input) == 1:
            csv_input = Path(args.csv_input[0])
            if csv_input.is_file():
                csv_files = [csv_input]
                output_dir = Path(args.output_dir) if args.output_dir else csv_input.parent
            elif csv_input.is_dir():
                csv_files = list(csv_input.glob("*.csv"))
                output_dir = Path(args.output_dir) if args.output_dir else csv_input
            else:
                print(f"Error: {csv_input} is not a valid file or directory")
                return
        else:
            csv_files = [Path(f) for f in args.csv_input]
            output_dir = Path(args.output_dir) if args.output_dir else csv_files[0].parent

        if not csv_files:
            print(f"No CSV files found")
            return

        print(f"Found {len(csv_files)} CSV files, generating plots...")
        for csv_file in csv_files:
            print(f"Processing: {csv_file.name}")
            try:
                plot_single_csv(csv_file, args.model_name, output_dir=output_dir, no_itl=args.no_itl)
            except Exception as e:
                print(f"  Failed to plot {csv_file.name}: {e}")

        print(f"\nAll plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
