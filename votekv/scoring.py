"""Attention score computation from observation window"""

import torch
from typing import Tuple


def compute_snapkv_scores_from_attentions(
    attentions: Tuple[torch.Tensor, ...],
    observation_window: int,
) -> torch.Tensor:
    """Compute importance scores from attention in observation window
    
    Following SnapKV methodology: use attention from the last `observation_window`
    tokens during prefill to determine which earlier tokens are important.
    
    Args:
        attentions: Tuple of length num_layers
            Each element shape: [batch, num_query_heads, seq_len, seq_len]
        observation_window: Number of recent tokens to use as observers
        
    Returns:
        scores: [num_layers, num_query_heads, seq_len]
            Importance score for each token as sum of attention from observation window
    """
    num_layers = len(attentions)
    batch_size, num_heads, seq_len, _ = attentions[0].shape
    
    assert batch_size == 1, f"Only batch_size=1 supported for MVP, got {batch_size}"
    
    all_scores = []
    
    for layer_idx in range(num_layers):
        attn = attentions[layer_idx]  # [batch, num_heads, seq_len, seq_len]

        # Observation window: last `observation_window` tokens as queries.
        obs_start = max(0, seq_len - observation_window)
        obs_end = seq_len

        # Sum attention from observation window to all tokens.
        # attn[:, :, obs_start:obs_end, :] -> [batch, num_heads, obs_window, seq_len]
        # Cast to float32 before reduction: attention weights are typically
        # bfloat16 (eager attention with bf16 weights), and accumulating
        # `observation_window` values in bf16 loses enough precision to make
        # topk unstable for closely-ranked tokens.
        scores = attn[:, :, obs_start:obs_end, :].float().sum(dim=2)

        all_scores.append(scores)
    
    # Stack all layers: [num_layers, batch, num_heads, seq_len]
    all_scores = torch.stack(all_scores, dim=0)
    
    # Remove batch dimension (batch=1)
    all_scores = all_scores.squeeze(1)  # [num_layers, num_heads, seq_len]
    
    return all_scores


def get_always_keep_indices(seq_len: int, sink_tokens: int, recent_tokens: int) -> list:
    """Get indices of tokens that must always be kept
    
    Args:
        seq_len: Total sequence length
        sink_tokens: Number of initial tokens to always keep
        recent_tokens: Number of final tokens to always keep
        
    Returns:
        List of sorted unique indices to always keep
    """
    # Sink tokens: first few tokens
    sink = list(range(min(sink_tokens, seq_len)))
    
    # Recent tokens: last few tokens
    recent_start = max(0, seq_len - recent_tokens)
    recent = list(range(recent_start, seq_len))
    
    # Combine and deduplicate
    return sorted(set(sink + recent))
