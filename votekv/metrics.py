"""Metrics and analysis utilities"""

import torch
from typing import Dict, List, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_compression_ratio(original_len: int, compressed_len: int) -> float:
    """Compute compression ratio
    
    Args:
        original_len: Original sequence length
        compressed_len: Compressed sequence length
        
    Returns:
        Compression ratio (higher = more compression)
    """
    if compressed_len == 0:
        return float('inf')
    return original_len / compressed_len


def compute_gqa_disagreement(
    scores: torch.Tensor,
    model_config: Dict[str, int],
    topk: int = 128,
) -> Dict:
    """Compute disagreement analysis for GQA groups
    
    Measures how much query heads within the same GQA group disagree
    on which tokens are important.
    
    Args:
        scores: [num_layers, num_attention_heads, seq_len]
        model_config: Dict with GQA info
        topk: Top-k tokens to consider per head
        
    Returns:
        Dictionary with disagreement statistics
    """
    from .gqa_utils import group_scores_by_gqa
    
    num_layers, num_attention_heads, seq_len = scores.shape
    num_kv_heads = model_config["num_key_value_heads"]
    group_size = model_config["group_size"]
    
    group_scores = group_scores_by_gqa(scores, num_attention_heads, num_kv_heads)
    
    results = {
        "per_layer_per_kv_head": [],
        "average_disagreement": 0.0,
        "average_jaccard": 0.0,
    }
    
    all_disagreements = []
    all_jaccards = []
    
    topk = min(topk, seq_len)
    
    for layer_idx in range(num_layers):
        layer_results = []
        
        for kv_head in range(num_kv_heads):
            scores_group = group_scores[layer_idx, kv_head]  # [group_size, seq_len]
            
            # Get top-k for each head
            topk_sets = []
            for head_in_group in range(group_size):
                _, top_indices = torch.topk(scores_group[head_in_group], topk, largest=True)
                topk_sets.append(set(top_indices.cpu().tolist()))
            
            # Compute pairwise Jaccard similarity
            jaccards = []
            for i in range(group_size):
                for j in range(i + 1, group_size):
                    intersection = len(topk_sets[i] & topk_sets[j])
                    union = len(topk_sets[i] | topk_sets[j])
                    jaccard = intersection / union if union > 0 else 0.0
                    jaccards.append(jaccard)
            
            avg_jaccard = np.mean(jaccards) if jaccards else 0.0
            disagreement = 1.0 - avg_jaccard
            
            layer_results.append({
                "layer": layer_idx,
                "kv_head": kv_head,
                "avg_jaccard": avg_jaccard,
                "disagreement": disagreement,
            })
            
            all_disagreements.append(disagreement)
            all_jaccards.append(avg_jaccard)
        
        results["per_layer_per_kv_head"].append(layer_results)
    
    results["average_disagreement"] = float(np.mean(all_disagreements))
    results["average_jaccard"] = float(np.mean(all_jaccards))
    
    return results


def compute_vote_histogram(
    scores: torch.Tensor,
    mask: torch.Tensor,
    model_config: Dict[str, int],
    topk: int = 128,
) -> Dict:
    """Compute vote histogram: how many tokens received 1, 2, 3, ... votes
    
    Args:
        scores: [num_layers, num_attention_heads, seq_len]
        mask: [num_layers, num_key_value_heads, seq_len]
        model_config: Dict with GQA info
        topk: Top-k tokens per head
        
    Returns:
        Dictionary with vote histogram
    """
    from .gqa_utils import group_scores_by_gqa
    
    num_layers, num_attention_heads, seq_len = scores.shape
    num_kv_heads = model_config["num_key_value_heads"]
    group_size = model_config["group_size"]
    
    group_scores = group_scores_by_gqa(scores, num_attention_heads, num_kv_heads)
    
    vote_histograms = []
    topk = min(topk, seq_len)
    
    for layer_idx in range(num_layers):
        for kv_head in range(num_kv_heads):
            scores_group = group_scores[layer_idx, kv_head]
            selected_mask = mask[layer_idx, kv_head]
            vote_counts = torch.zeros(seq_len, dtype=torch.int32, device=scores.device)

            for head_in_group in range(group_size):
                _, top_indices = torch.topk(scores_group[head_in_group], topk, largest=True)
                vote_counts[top_indices] += 1
            
            # Get vote counts for selected tokens
            selected_indices = torch.where(selected_mask)[0]
            selected_votes = vote_counts[selected_indices]
            
            # Create histogram
            histogram = {}
            for v in range(group_size + 1):
                count = (selected_votes == v).sum().item()
                if count > 0:
                    histogram[v] = count
            
            vote_histograms.append({
                "layer": layer_idx,
                "kv_head": kv_head,
                "histogram": histogram,
            })
    
    return {"vote_histograms": vote_histograms}


def exact_match(pred: str, gold: str) -> bool:
    """Check if prediction exactly matches gold answer
    
    Args:
        pred: Predicted answer
        gold: Gold answer
        
    Returns:
        True if exact match (case-insensitive, stripped)
    """
    return pred.strip().lower() == gold.strip().lower()


def contains_answer(pred: str, gold: str) -> bool:
    """Check if prediction contains gold answer
    
    Args:
        pred: Predicted text
        gold: Gold answer
        
    Returns:
        True if gold is substring of pred
    """
    return gold.strip().lower() in pred.strip().lower()
