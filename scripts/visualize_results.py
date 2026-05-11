"""Visualization script for VoteKV evaluation results

Creates publication-ready plots for needle evaluation, disagreement analysis, etc.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def load_needle_results(jsonl_path: Path) -> pd.DataFrame:
    """Load needle-in-haystack results from JSONL file"""
    results = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return pd.DataFrame(results)


def load_disagreement_results(json_path: Path) -> Dict:
    """Load disagreement analysis results"""
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_needle_heatmap(df: pd.DataFrame, output_dir: Path):
    """Plot accuracy heatmap by method and depth
    
    Creates a heatmap showing accuracy for each method at different depths.
    """
    methods = df['method'].unique()
    depths = sorted(df['depth'].unique())
    context_lengths = sorted(df['context_len'].unique())
    budget_ratios = sorted(df['budget_ratio'].unique())
    
    # For each context length and budget ratio
    for ctx_len in context_lengths:
        for budget in budget_ratios:
            subset = df[(df['context_len'] == ctx_len) & (df['budget_ratio'] == budget)]
            
            if len(subset) == 0:
                continue
            
            # Create accuracy matrix
            accuracy_matrix = []
            method_labels = []
            
            for method in methods:
                method_data = subset[subset['method'] == method]
                accuracies = []
                
                for depth in depths:
                    depth_data = method_data[method_data['depth'] == depth]
                    if len(depth_data) > 0:
                        acc = depth_data['correct_contains'].mean() * 100
                        accuracies.append(acc)
                    else:
                        accuracies.append(np.nan)
                
                if not all(np.isnan(accuracies)):
                    accuracy_matrix.append(accuracies)
                    method_labels.append(method)
            
            # Plot heatmap
            fig, ax = plt.subplots(figsize=(10, 6))
            
            im = ax.imshow(accuracy_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
            
            # Set ticks and labels
            ax.set_xticks(range(len(depths)))
            ax.set_xticklabels([f"{d:.1f}" for d in depths])
            ax.set_yticks(range(len(method_labels)))
            ax.set_yticklabels(method_labels)
            
            # Add text annotations
            for i in range(len(method_labels)):
                for j in range(len(depths)):
                    if not np.isnan(accuracy_matrix[i][j]):
                        text = ax.text(j, i, f'{accuracy_matrix[i][j]:.1f}',
                                     ha="center", va="center", color="black", fontsize=10)
            
            ax.set_xlabel('Needle Depth', fontsize=14)
            ax.set_ylabel('Method', fontsize=14)
            ax.set_title(f'Needle Accuracy (%) - Context: {ctx_len}, Budget: {budget:.2f}', 
                        fontsize=16)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Accuracy (%)', fontsize=12)
            
            plt.tight_layout()
            
            output_file = output_dir / f'needle_heatmap_ctx{ctx_len}_budget{budget:.2f}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved: {output_file}")


# Canonical method order + per-method styling (used by tradeoff and Figure 1).
_METHOD_ORDER = [
    'full_cache', 'gqa_mean', 'gqa_max', 'gqa_vote', 'gqa_rank_vote', 'gqa_vote_rescue',
]
_METHOD_COLORS = {
    'full_cache':      '#95a5a6',
    'gqa_mean':        '#3498db',
    'gqa_max':         '#e67e22',
    'gqa_vote':        '#9b59b6',
    'gqa_rank_vote':   '#1abc9c',
    'gqa_vote_rescue': '#e74c3c',
}
_METHOD_MARKERS = {
    'full_cache':      'o',
    'gqa_mean':        's',
    'gqa_max':         '^',
    'gqa_vote':        'D',
    'gqa_rank_vote':   'v',
    'gqa_vote_rescue': '*',
}


def _ordered_methods(present):
    """Return present methods in canonical order, unknown ones appended."""
    known = [m for m in _METHOD_ORDER if m in present]
    extra = [m for m in present if m not in _METHOD_ORDER]
    return known + extra


def plot_compression_accuracy_tradeoff(df: pd.DataFrame, output_dir: Path):
    """Plot accuracy vs compression ratio as small multiples (one panel per method).

    Each panel shows the mean accuracy across all depths at every compression
    ratio that appeared in the sweep, with std error bars to expose variance.
    Methods no longer overlap, and we use ALL depth measurements rather than
    just the middle one — so the curves have meaningful uncertainty.

    The X axis range is auto-fit from data with a 10% headroom on top, but
    never tighter than [0.5, 60] — that keeps `full_cache` (ratio=1.0) visible
    and prevents a sparse quick-run from squishing the curve into one corner.
    """
    context_lengths = sorted(df['context_len'].unique())
    x_min = 0.5
    data_max = float(df['compression_ratio'].max())
    x_max = max(60.0, data_max * 1.10)

    # `full_cache` has compression_ratio == 1.0 exactly (no compression), so it
    # always lands at a single x. Other methods are bucketed by `budget_ratio`
    # (a clean float we control), then the compression_ratio is averaged
    # WITHIN that bucket across depths — otherwise tiny variations in
    # tokenised prompt length produce 5 near-identical x's that pandas treats
    # as separate groups, and errorbar's line connects them into a fake "tail".
    group_key = 'budget_ratio' if 'budget_ratio' in df.columns else 'compression_ratio'

    for ctx_len in context_lengths:
        ctx_df = df[df['context_len'] == ctx_len]
        methods = _ordered_methods(list(ctx_df['method'].unique()))
        n = len(methods)
        cols = 3 if n >= 3 else max(n, 1)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(
            rows, cols,
            figsize=(4.2 * cols, 3.4 * rows),
            sharex=True, sharey=True,
            squeeze=False,
        )
        axes_flat = axes.flatten()

        # Shared x range across the whole grid for fair side-by-side comparison.
        # We use a fixed ceiling (default 60) so that quick/full/wide runs are
        # plotted on the same scale and you can stack figures from different
        # models without the x axis subtly shifting.
        x_lo, x_hi = x_min, x_max

        for idx, method in enumerate(methods):
            ax = axes_flat[idx]
            mdf = ctx_df[ctx_df['method'] == method]

            # Aggregate over depths: one row per (method, budget_ratio).
            grouped = (
                mdf.groupby(group_key)
                .agg(
                    accuracy_mean=('correct_contains', 'mean'),
                    accuracy_std=('correct_contains', 'std'),
                    compression_ratio=('compression_ratio', 'mean'),
                    n=('correct_contains', 'count'),
                )
                .reset_index()
                .sort_values('compression_ratio')
            )
            grouped['mean_pct'] = grouped['accuracy_mean'] * 100
            grouped['std_pct'] = grouped['accuracy_std'].fillna(0) * 100

            color = _METHOD_COLORS.get(method, 'gray')
            marker = _METHOD_MARKERS.get(method, 'o')

            ax.errorbar(
                grouped['compression_ratio'],
                grouped['mean_pct'],
                yerr=grouped['std_pct'],
                marker=marker,
                linewidth=2,
                markersize=10,
                color=color,
                ecolor=color,
                capsize=4,
                alpha=0.95,
            )

            # 100% reference (full cache theoretical ceiling).
            ax.axhline(y=100, color='gray', linestyle=':', linewidth=1, alpha=0.5)

            # Annotation: overall accuracy across all sweep cells for this method.
            overall = mdf['correct_contains'].mean() * 100
            ax.text(
                0.04, 0.06, f'avg = {overall:.1f}%',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.85, edgecolor=color, linewidth=1),
            )

            ax.set_title(method, fontweight='bold', fontsize=12, color=color)
            ax.set_ylim(-3, 108)
            ax.set_xlim(x_lo, x_hi)
            ax.grid(True, alpha=0.3)

        # Hide any unused subplots.
        for idx in range(n, len(axes_flat)):
            axes_flat[idx].axis('off')

        # Common axis labels.
        fig.supxlabel('Compression Ratio', fontsize=13)
        fig.supylabel('Accuracy (%)  (mean ± std over depths)', fontsize=13)
        fig.suptitle(
            f'Accuracy vs Compression Ratio  —  context = {ctx_len} tokens',
            fontsize=14, fontweight='bold', y=1.00,
        )

        plt.tight_layout()

        output_file = output_dir / f'compression_accuracy_ctx{ctx_len}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {output_file}")


def plot_latency_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot latency breakdown by method"""
    # Average across all runs
    latency_data = df.groupby('method').agg({
        'prefill_sec': 'mean',
        'compression_sec': 'mean',
        'decode_sec': 'mean',
        'total_sec': 'mean'
    }).reset_index()
    
    methods = latency_data['method'].tolist()
    prefill = latency_data['prefill_sec'].tolist()
    compression = latency_data['compression_sec'].tolist()
    decode = latency_data['decode_sec'].tolist()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(methods))
    width = 0.6
    
    p1 = ax.bar(x, prefill, width, label='Prefill', color='#3498db')
    p2 = ax.bar(x, compression, width, bottom=prefill, label='Compression', color='#e74c3c')
    p3 = ax.bar(x, decode, width, 
               bottom=np.array(prefill) + np.array(compression), 
               label='Decode', color='#2ecc71')
    
    ax.set_ylabel('Time (seconds)', fontsize=14)
    ax.set_title('Latency Breakdown by Method', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add total time on top
    for i, (m, t) in enumerate(zip(methods, latency_data['total_sec'])):
        ax.text(i, t + 0.05, f'{t:.2f}s', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    output_file = output_dir / 'latency_breakdown.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_file}")


def plot_disagreement_by_layer(disagreement_data: Dict, output_dir: Path):
    """Plot disagreement scores across layers"""
    per_layer_data = disagreement_data['per_layer_per_kv_head']

    # Accept BOTH formats:
    #   * flat   = [ {layer, kv_head, disagreement, avg_jaccard}, ... ]   (new, canonical)
    #   * nested = [ [ {...}, {...} ],  [...],  ... ]                     (old JSONs)
    if per_layer_data and isinstance(per_layer_data[0], list):
        flat = [d for sub in per_layer_data for d in sub]
    else:
        flat = per_layer_data

    layer_disagreements = {}
    for item in flat:
        layer = item['layer']
        if layer not in layer_disagreements:
            layer_disagreements[layer] = []
        layer_disagreements[layer].append(item['disagreement'])
    
    layers = sorted(layer_disagreements.keys())
    avg_disagreements = [np.mean(layer_disagreements[l]) for l in layers]
    std_disagreements = [np.std(layer_disagreements[l]) for l in layers]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(layers, avg_disagreements, marker='o', linewidth=2, markersize=6, color='#e74c3c')
    ax.fill_between(layers, 
                     np.array(avg_disagreements) - np.array(std_disagreements),
                     np.array(avg_disagreements) + np.array(std_disagreements),
                     alpha=0.3, color='#e74c3c')
    
    ax.set_xlabel('Layer', fontsize=14)
    ax.set_ylabel('Disagreement (1 - Jaccard)', fontsize=14)
    ax.set_title('Query Head Disagreement Across Layers', fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line for overall average
    avg_overall = disagreement_data['average_disagreement']
    ax.axhline(y=avg_overall, color='gray', linestyle='--', linewidth=2, 
              label=f'Overall Avg: {avg_overall:.3f}')
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    
    output_file = output_dir / 'disagreement_by_layer.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_file}")


def plot_vote_histogram(histogram_path: Path, output_dir: Path):
    """Plot vote distribution histogram"""
    with open(histogram_path, 'r') as f:
        data = json.load(f)
    
    # Aggregate histograms across all layers/heads
    total_histogram = {}
    for item in data['vote_histograms']:
        for vote_count, num_tokens in item['histogram'].items():
            vote_count = int(vote_count)
            total_histogram[vote_count] = total_histogram.get(vote_count, 0) + num_tokens
    
    if not total_histogram:
        print("No vote histogram data found")
        return
    
    votes = sorted(total_histogram.keys())
    counts = [total_histogram[v] for v in votes]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71']
    bars = ax.bar(votes, counts, color=[colors[min(v-1, 3)] for v in votes], 
                  edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Number of Votes', fontsize=14)
    ax.set_ylabel('Number of Tokens', fontsize=14)
    ax.set_title('Vote Distribution for Selected Tokens', fontsize=16)
    ax.set_xticks(votes)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(count)}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add percentage labels
    total = sum(counts)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        pct = 100 * count / total
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
               f'{pct:.1f}%',
               ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    
    plt.tight_layout()
    
    method_name = histogram_path.stem.replace('vote_histogram_', '')
    output_file = output_dir / f'vote_histogram_{method_name}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_file}")


def plot_summary_comparison(df: pd.DataFrame, output_dir: Path):
    """Create a comprehensive comparison plot"""
    # Calculate summary statistics per method
    summary = df.groupby('method').agg({
        'correct_contains': 'mean',
        'compression_ratio': 'mean',
        'total_sec': 'mean',
        'peak_memory_gb': 'mean'
    }).reset_index()
    
    summary['accuracy'] = summary['correct_contains'] * 100
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    methods = summary['method'].tolist()
    colors = plt.cm.tab10(range(len(methods)))
    
    # 1. Accuracy
    ax = axes[0, 0]
    bars = ax.bar(methods, summary['accuracy'], color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Average Accuracy', fontsize=14, fontweight='bold')
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, acc in zip(bars, summary['accuracy']):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
               f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # 2. Compression Ratio
    ax = axes[0, 1]
    bars = ax.bar(methods, summary['compression_ratio'], color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Compression Ratio', fontsize=12)
    ax.set_title('Average Compression Ratio', fontsize=14, fontweight='bold')
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, ratio in zip(bars, summary['compression_ratio']):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
               f'{ratio:.1f}x', ha='center', va='bottom', fontsize=10)
    
    # 3. Latency
    ax = axes[1, 0]
    bars = ax.bar(methods, summary['total_sec'], color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Average Total Time', fontsize=14, fontweight='bold')
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, time in zip(bars, summary['total_sec']):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
               f'{time:.2f}s', ha='center', va='bottom', fontsize=10)
    
    # 4. Memory
    ax = axes[1, 1]
    bars = ax.bar(methods, summary['peak_memory_gb'], color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Memory (GB)', fontsize=12)
    ax.set_title('Peak GPU Memory', fontsize=14, fontweight='bold')
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, mem in zip(bars, summary['peak_memory_gb']):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
               f'{mem:.1f}GB', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('VoteKV Method Comparison', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = output_dir / 'summary_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_file}")


def _safe(label: str, fn, *args, **kwargs):
    """Run a plotting function and report failures without aborting the rest."""
    print(f"\n{label} ...")
    try:
        fn(*args, **kwargs)
    except FileNotFoundError as e:
        print(f"  SKIPPED ({label}): {e}")
    except Exception as e:
        print(f"  FAILED  ({label}): {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Visualize VoteKV evaluation results")
    parser.add_argument("--needle_results", type=str, help="Path to needle JSONL file")
    parser.add_argument("--disagreement_results", type=str, help="Path to disagreement JSON file")
    parser.add_argument("--vote_histogram", type=str, help="Path to vote histogram JSON file")
    parser.add_argument("--output_dir", type=str, default="outputs/plots", help="Output directory for plots")
    parser.add_argument("--plots", nargs="+",
                       choices=['all', 'heatmap', 'tradeoff', 'latency', 'disagreement', 'histogram', 'summary'],
                       default=['all'], help="Which plots to generate")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_all = 'all' in args.plots
    
    # Needle evaluation plots
    if args.needle_results and Path(args.needle_results).exists():
        df = load_needle_results(Path(args.needle_results))
        print(f"\nLoaded {len(df)} needle evaluation results")
        
        if plot_all or 'heatmap' in args.plots:
            _safe("needle accuracy heatmaps", plot_needle_heatmap, df, output_dir)
        if plot_all or 'tradeoff' in args.plots:
            _safe("compression-accuracy tradeoff", plot_compression_accuracy_tradeoff, df, output_dir)
        if plot_all or 'latency' in args.plots:
            _safe("latency comparison", plot_latency_comparison, df, output_dir)
        if plot_all or 'summary' in args.plots:
            _safe("summary comparison", plot_summary_comparison, df, output_dir)
    elif args.needle_results:
        print(f"\nSKIPPED needle plots: {args.needle_results} not found")
    
    # Disagreement analysis plots
    if args.disagreement_results and Path(args.disagreement_results).exists():
        disagreement_data = load_disagreement_results(Path(args.disagreement_results))
        print(f"\nLoaded disagreement analysis results")
        if plot_all or 'disagreement' in args.plots:
            _safe("disagreement by layer", plot_disagreement_by_layer, disagreement_data, output_dir)
    elif args.disagreement_results:
        print(f"\nSKIPPED disagreement plot: {args.disagreement_results} not found")
    
    # Vote histogram plots
    if args.vote_histogram and Path(args.vote_histogram).exists():
        if plot_all or 'histogram' in args.plots:
            _safe("vote histogram", plot_vote_histogram, Path(args.vote_histogram), output_dir)
    elif args.vote_histogram:
        print(f"\nSKIPPED vote histogram plot: {args.vote_histogram} not found")

    print(f"\n[done] plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
