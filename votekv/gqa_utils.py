"""GQA (Grouped-Query Attention) utilities"""

import torch
from typing import Dict, List


def get_gqa_info(model) -> Dict[str, int]:
    """Extract GQA configuration from model
    
    Args:
        model: HuggingFace model with GQA config
        
    Returns:
        Dictionary with num_attention_heads, num_key_value_heads, group_size
    """
    config = model.config
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    group_size = num_attention_heads // num_key_value_heads
    
    assert num_attention_heads % num_key_value_heads == 0, (
        f"num_attention_heads ({num_attention_heads}) must be divisible by "
        f"num_key_value_heads ({num_key_value_heads})"
    )
    
    return {
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "group_size": group_size,
    }


def query_head_to_kv_head(query_head_id: int, group_size: int) -> int:
    """Map query head ID to its corresponding KV head ID
    
    Args:
        query_head_id: Index of query head (0 to num_attention_heads-1)
        group_size: Number of query heads per KV head
        
    Returns:
        Index of KV head (0 to num_key_value_heads-1)
    """
    return query_head_id // group_size


def build_gqa_groups(num_attention_heads: int, num_key_value_heads: int) -> Dict[int, List[int]]:
    """Build mapping from KV head to query heads in that group
    
    For Mistral with 32 attention heads and 8 KV heads:
    - KV head 0 -> Q heads [0,1,2,3]
    - KV head 1 -> Q heads [4,5,6,7]
    - ...
    - KV head 7 -> Q heads [28,29,30,31]
    
    Args:
        num_attention_heads: Total number of query heads
        num_key_value_heads: Number of KV heads
        
    Returns:
        Dictionary mapping kv_head_id -> list of query_head_ids
    """
    group_size = num_attention_heads // num_key_value_heads
    groups = {}
    
    for kv_head in range(num_key_value_heads):
        start = kv_head * group_size
        end = start + group_size
        groups[kv_head] = list(range(start, end))
    
    return groups


def group_scores_by_gqa(
    scores: torch.Tensor,
    num_attention_heads: int,
    num_key_value_heads: int
) -> torch.Tensor:
    """Reshape scores from per-query-head to per-KV-head grouped format.

    Assumes the HF convention that consecutive query heads share a KV head:
    Q[0..g-1] -> KV[0], Q[g..2g-1] -> KV[1], ... This holds for Mistral, Llama,
    and Qwen GQA models. If a future model uses an interleaved layout, this
    function must be updated.

    Args:
        scores: [num_layers, num_attention_heads, seq_len]
        num_attention_heads: Total query heads
        num_key_value_heads: Total KV heads

    Returns:
        group_scores: [num_layers, num_kv_heads, group_size, seq_len]
    """
    group_size = num_attention_heads // num_key_value_heads
    num_layers, n_heads, seq_len = scores.shape

    assert n_heads == num_attention_heads, (
        f"Expected {num_attention_heads} heads, got {n_heads}"
    )

    # Use reshape (not view) to tolerate non-contiguous score tensors that may
    # arrive from torch.stack + squeeze in the scoring path.
    return scores.reshape(num_layers, num_key_value_heads, group_size, seq_len)
