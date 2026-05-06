"""Generate publication-ready figures for VoteKV paper

Combines multiple visualizations into comprehensive figures for paper submission.
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
from matplotlib.gridspec import GridSpec

# Publication style settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13


def create_figure_1_main_results(needle_path: Path, output_dir: Path):
    """Figure 1: Main needle evaluation results
    
    Layout:
    - (a) Heatmap: accuracy by method and depth
    - (b) Compression-accuracy tradeoff
    - (c) Vote histogram
    """
    with open(needle_path, 'r') as f:
        results = [json.loads(line) for line in f]
    df = pd.DataFrame(results)
    
    # Filter to one context length and budget for clarity
    ctx_len = 4096
    budget = 0.08
    subset = df[(df['context_len'] == ctx_len) & (df['budget_ratio'] == budget)]
    
    fig = plt.figure(figsize=(14, 4.5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)
    
    # (a) Accuracy heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    
    methods = ['full_cache', 'gqa_mean', 'gqa_max', 'gqa_vote', 'gqa_vote_rescue']
    depths = sorted(subset['depth'].unique())
    
    accuracy_matrix = []
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
        accuracy_matrix.append(accuracies)
    
    im = ax1.imshow(accuracy_matrix, cmap='RdYlGn', aspect='auto', vmin=50, vmax=100)
    ax1.set_xticks(range(len(depths)))
    ax1.set_xticklabels([f"{d:.1f}" for d in depths])
    ax1.set_yticks(range(len(methods)))
    ax1.set_yticklabels(methods)
    
    for i in range(len(methods)):
        for j in range(len(depths)):
            if not np.isnan(accuracy_matrix[i][j]):
                ax1.text(j, i, f'{accuracy_matrix[i][j]:.0f}',
                        ha="center", va="center", color="black", fontsize=8)
    
    ax1.set_xlabel('Needle Depth')
    ax1.set_ylabel('Method')
    ax1.set_title('(a) Retrieval Accuracy (%)')
    
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Accuracy (%)', fontsize=9)
    
    # (b) Compression-accuracy tradeoff
    ax2 = fig.add_subplot(gs[0, 1])
    
    middle_depth = depths[len(depths) // 2]
    tradeoff_data = df[(df['context_len'] == ctx_len) & (df['depth'] == middle_depth)]
    
    colors = {'full_cache': '#95a5a6', 'gqa_mean': '#3498db', 'gqa_max': '#e67e22',
              'gqa_vote': '#9b59b6', 'gqa_vote_rescue': '#e74c3c'}
    markers = {'full_cache': 'o', 'gqa_mean': 's', 'gqa_max': '^',
               'gqa_vote': 'D', 'gqa_vote_rescue': '*'}
    
    for method in methods:
        method_data = tradeoff_data[tradeoff_data['method'] == method]
        if len(method_data) > 0:
            method_data = method_data.sort_values('compression_ratio')
            ax2.plot(method_data['compression_ratio'], 
                    method_data['correct_contains'] * 100,
                    marker=markers.get(method, 'o'), label=method, linewidth=2, 
                    markersize=8, color=colors.get(method, 'gray'))
    
    ax2.set_xlabel('Compression Ratio')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'(b) Accuracy vs Compression (depth={middle_depth:.1f})')
    ax2.legend(loc='lower left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # (c) Method comparison bar chart
    ax3 = fig.add_subplot(gs[0, 2])
    
    summary = subset.groupby('method')['correct_contains'].mean() * 100
    summary = summary.reindex(methods)
    
    bars = ax3.barh(range(len(methods)), summary.values, 
                    color=[colors.get(m, 'gray') for m in methods],
                    edgecolor='black', linewidth=1)
    
    ax3.set_yticks(range(len(methods)))
    ax3.set_yticklabels(methods)
    ax3.set_xlabel('Average Accuracy (%)')
    ax3.set_title('(c) Overall Performance')
    ax3.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, val) in enumerate(zip(bars, summary.values)):
        ax3.text(val + 1, i, f'{val:.1f}%', va='center', fontsize=9)
    
    plt.suptitle(f'VoteKV Main Results (context={ctx_len}, budget={budget})', 
                fontweight='bold', y=0.98)
    
    output_file = output_dir / 'figure_1_main_results.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(output_dir / 'figure_1_main_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created Figure 1: {output_file}")


def create_figure_2_disagreement(disagreement_path: Path, output_dir: Path):
    """Figure 2: Query head disagreement analysis"""
    with open(disagreement_path, 'r') as f:
        data = json.load(f)
    
    fig = plt.figure(figsize=(12, 4))
    gs = GridSpec(1, 2, figure=fig, wspace=0.3)
    
    # (a) Disagreement by layer
    ax1 = fig.add_subplot(gs[0, 0])
    
    per_layer_data = data['per_layer_per_kv_head']
    layer_disagreements = {}
    for item in per_layer_data:
        layer = item['layer']
        if layer not in layer_disagreements:
            layer_disagreements[layer] = []
        layer_disagreements[layer].append(item['disagreement'])
    
    layers = sorted(layer_disagreements.keys())
    avg_disagreements = [np.mean(layer_disagreements[l]) for l in layers]
    std_disagreements = [np.std(layer_disagreements[l]) for l in layers]
    
    ax1.plot(layers, avg_disagreements, marker='o', linewidth=2, 
            markersize=5, color='#e74c3c', label='Per-layer average')
    ax1.fill_between(layers, 
                     np.array(avg_disagreements) - np.array(std_disagreements),
                     np.array(avg_disagreements) + np.array(std_disagreements),
                     alpha=0.3, color='#e74c3c')
    
    avg_overall = data['average_disagreement']
    ax1.axhline(y=avg_overall, color='gray', linestyle='--', linewidth=2, 
               label=f'Overall: {avg_overall:.3f}')
    
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Disagreement (1 - Jaccard)')
    ax1.set_title('(a) Query Head Disagreement Across Layers')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # (b) Jaccard similarity distribution
    ax2 = fig.add_subplot(gs[0, 1])
    
    jaccards = [item['avg_jaccard'] for item in per_layer_data]
    
    ax2.hist(jaccards, bins=20, color='#3498db', edgecolor='black', alpha=0.7)
    ax2.axvline(x=np.mean(jaccards), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(jaccards):.3f}')
    
    ax2.set_xlabel('Jaccard Similarity')
    ax2.set_ylabel('Frequency (# of GQA groups)')
    ax2.set_title('(b) Distribution of Pairwise Jaccard Similarity')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Query Head Disagreement Analysis', fontweight='bold', y=0.98)
    
    output_file = output_dir / 'figure_2_disagreement.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(output_dir / 'figure_2_disagreement.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created Figure 2: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Create paper figures")
    parser.add_argument("--needle_results", type=str, required=True)
    parser.add_argument("--disagreement_results", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/paper_figures")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating publication-ready figures...\n")
    
    create_figure_1_main_results(Path(args.needle_results), output_dir)
    create_figure_2_disagreement(Path(args.disagreement_results), output_dir)
    
    print(f"\n✓ All figures saved to: {output_dir}")
    print("  - PDF format for papers")
    print("  - PNG format for presentations")


if __name__ == "__main__":
    main()
