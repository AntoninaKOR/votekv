"""KV-cache compression utilities"""

import torch
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


def convert_kv_head_mask_to_layer_indices(
    mask: torch.Tensor,
    scores: torch.Tensor,
    max_budget: int,
) -> List[torch.Tensor]:
    """Convert per-KV-head masks to layer-shared selected indices
    
    For MVP, we use layer-shared compression: take union of selected tokens
    across all KV heads in a layer. If union exceeds budget, rank tokens by
    how many KV heads selected them.
    
    Args:
        mask: [num_layers, num_kv_heads, seq_len] bool tensor
        scores: [num_layers, num_query_heads, seq_len] for tie-breaking
        max_budget: Maximum budget per layer
        
    Returns:
        List of tensors, one per layer, containing sorted selected indices
    """
    num_layers, num_kv_heads, seq_len = mask.shape
    selected_indices_per_layer = []
    
    for layer_idx in range(num_layers):
        layer_mask = mask[layer_idx]  # [num_kv_heads, seq_len]
        
        # Take union: token selected if ANY KV head selected it
        union_mask = layer_mask.any(dim=0)  # [seq_len]
        selected = torch.where(union_mask)[0]
        
        if len(selected) <= max_budget:
            # Within budget
            selected_sorted = torch.sort(selected).values
            selected_indices_per_layer.append(selected_sorted)
        else:
            # Exceeds budget: rank by importance
            importance = layer_mask.float().sum(dim=0)  # How many KV heads selected each token
            
            # Tie-break with average score across query heads
            avg_score = scores[layer_idx].mean(dim=0)  # [seq_len]
            
            composite = importance * 1e6 + avg_score
            selected_scores = composite[selected]
            
            # Take top max_budget
            _, top_indices = torch.topk(selected_scores, max_budget, largest=True)
            selected_final = selected[top_indices]
            selected_sorted = torch.sort(selected_final).values
            
            selected_indices_per_layer.append(selected_sorted)
            
            logger.warning(
                f"Layer {layer_idx}: union ({len(selected)}) exceeds budget ({max_budget}), "
                f"trimmed to {len(selected_sorted)}"
            )
    
    return selected_indices_per_layer


def compress_past_key_values_layer_shared(
    past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    selected_indices_per_layer: List[torch.Tensor],
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    """Compress past_key_values using layer-shared selected indices
    
    Args:
        past_key_values: Tuple of (key, value) per layer
            key/value shape: [batch, num_kv_heads, seq_len, head_dim]
        selected_indices_per_layer: List of sorted indices per layer
        
    Returns:
        Compressed past_key_values with same structure
    """
    compressed = []
    
    for layer_idx, (k, v) in enumerate(past_key_values):
        idx = selected_indices_per_layer[layer_idx].to(k.device)
        
        # k/v shape: [batch, num_kv_heads, seq_len, head_dim]
        # Index along seq_len dimension (dim=2)
        k_new = k.index_select(dim=2, index=idx)
        v_new = v.index_select(dim=2, index=idx)
        
        compressed.append((k_new, v_new))
    
    logger.info(
        f"Compressed cache: layer 0 from {k.shape[2]} to {k_new.shape[2]} tokens"
    )
    
    return tuple(compressed)


def get_cache_info(past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]) -> dict:
    """Get information about cache structure
    
    Args:
        past_key_values: Tuple of (key, value) per layer
        
    Returns:
        Dictionary with cache statistics
    """
    if not past_key_values:
        return {}
    
    num_layers = len(past_key_values)
    k, v = past_key_values[0]
    batch_size, num_kv_heads, seq_len, head_dim = k.shape
    
    total_elements = sum(
        k.numel() + v.numel() for k, v in past_key_values
    )
    
    return {
        "num_layers": num_layers,
        "batch_size": batch_size,
        "num_kv_heads": num_kv_heads,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "total_elements": total_elements,
        "memory_mb": total_elements * k.element_size() / (1024 ** 2),
    }
