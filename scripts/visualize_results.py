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


def plot_compression_accuracy_tradeoff(df: pd.DataFrame, output_dir: Path):
    """Plot accuracy vs compression ratio tradeoff"""
    context_lengths = sorted(df['context_len'].unique())
    
    for ctx_len in context_lengths:
        subset = df[df['context_len'] == ctx_len]
        
        # Filter to middle depth (most challenging)
        depths = sorted(subset['depth'].unique())
        middle_depth = depths[len(depths) // 2]
        subset = subset[subset['depth'] == middle_depth]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = subset['method'].unique()
        colors = plt.cm.tab10(range(len(methods)))
        
        for idx, method in enumerate(methods):
            method_data = subset[subset['method'] == method].copy()
            method_data = method_data.sort_values('compression_ratio')
            
            ax.plot(method_data['compression_ratio'], 
                   method_data['correct_contains'] * 100,
                   marker='o', label=method, linewidth=2, markersize=8,
                   color=colors[idx])
        
        ax.set_xlabel('Compression Ratio', fontsize=14)
        ax.set_ylabel('Accuracy (%)', fontsize=14)
        ax.set_title(f'Accuracy vs Compression Ratio\nContext: {ctx_len}, Depth: {middle_depth:.1f}', 
                    fontsize=16)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
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
    
    # Organize by layer
    layer_disagreements = {}
    for item in per_layer_data:
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
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_all = 'all' in args.plots
    
    # Needle evaluation plots
    if args.needle_results:
        df = load_needle_results(Path(args.needle_results))
        print(f"\nLoaded {len(df)} needle evaluation results")
        
        if plot_all or 'heatmap' in args.plots:
            print("\nGenerating needle accuracy heatmaps...")
            plot_needle_heatmap(df, output_dir)
        
        if plot_all or 'tradeoff' in args.plots:
            print("\nGenerating compression-accuracy tradeoff plots...")
            plot_compression_accuracy_tradeoff(df, output_dir)
        
        if plot_all or 'latency' in args.plots:
            print("\nGenerating latency comparison...")
            plot_latency_comparison(df, output_dir)
        
        if plot_all or 'summary' in args.plots:
            print("\nGenerating summary comparison...")
            plot_summary_comparison(df, output_dir)
    
    # Disagreement analysis plots
    if args.disagreement_results:
        disagreement_data = load_disagreement_results(Path(args.disagreement_results))
        print(f"\nLoaded disagreement analysis results")
        
        if plot_all or 'disagreement' in args.plots:
            print("\nGenerating disagreement by layer plot...")
            plot_disagreement_by_layer(disagreement_data, output_dir)
    
    # Vote histogram plots
    if args.vote_histogram:
        if plot_all or 'histogram' in args.plots:
            print("\nGenerating vote histogram...")
            plot_vote_histogram(Path(args.vote_histogram), output_dir)
    
    print(f"\n✓ All plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
