"""KV-cache compression utilities"""

import torch
from typing import Tuple, List, Union
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

    # Make sure mask and scores live on the same device before mixing them in
    if mask.device != scores.device:
        mask = mask.to(scores.device)

    for layer_idx in range(num_layers):
        layer_mask = mask[layer_idx]  # [num_kv_heads, seq_len]

        # Take union: token selected if ANY KV head selected it.
        union_mask = layer_mask.any(dim=0)  # [seq_len]
        selected = torch.where(union_mask)[0]

        if len(selected) <= max_budget:
            selected_sorted = torch.sort(selected).values
            selected_indices_per_layer.append(selected_sorted)
        else:
            # Exceeds budget: rank by importance (how many KV heads agreed),
            # tie-break with average score across query heads.
            importance = layer_mask.float().sum(dim=0)
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
    past_key_values,
    selected_indices_per_layer: List[torch.Tensor],
):
    """Compress cache for transformers 5.8.0+ DynamicCache
    
    Args:
        past_key_values: DynamicCache with .layers[i].keys/.values
        selected_indices_per_layer: List of sorted indices per layer
        
    Returns:
        Compressed DynamicCache
    """
    from transformers.cache_utils import DynamicCache, DynamicLayer
    
    # transformers 5.8.0: DynamicCache.layers[i].keys/values
    layers = past_key_values.layers
    compressed_layers = []
    
    for layer_idx, layer in enumerate(layers):
        k = layer.keys
        v = layer.values
        idx = selected_indices_per_layer[layer_idx].to(k.device)

        # KV-cache index_select requires monotonically increasing indices to keep
        # RoPE / position ordering consistent with the original prompt positions.
        if idx.numel() > 1:
            assert torch.all(idx[1:] >= idx[:-1]), (
                f"Layer {layer_idx}: selected indices must be sorted ascending"
            )

        k_new = k.index_select(dim=2, index=idx)
        v_new = v.index_select(dim=2, index=idx)
        
        # Create new DynamicLayer
        new_layer = DynamicLayer()
        new_layer.keys = k_new
        new_layer.values = v_new
        new_layer.dtype = layer.dtype
        new_layer.device = layer.device
        new_layer.is_initialized = True
        
        compressed_layers.append(new_layer)
    
    logger.info(
        f"Compressed cache: layer 0 from {layers[0].keys.shape[2]} to {compressed_layers[0].keys.shape[2]} tokens"
    )
    
    # Create new DynamicCache
    compressed_cache = DynamicCache()
    compressed_cache.layers = compressed_layers
    
    return compressed_cache


def get_cache_info(past_key_values) -> dict:
    """Get cache info for transformers 5.8.0+ DynamicCache
    
    Args:
        past_key_values: DynamicCache with .layers[i].keys/.values
        
    Returns:
        Dictionary with cache statistics
    """
    if not past_key_values:
        return {}
    
    # transformers 5.8.0: DynamicCache.layers[i].keys/values
    layers = past_key_values.layers
    num_layers = len(layers)
    
    k = layers[0].keys
    v = layers[0].values
    
    total_elements = sum(
        layer.keys.numel() + layer.values.numel() for layer in layers
    )
    
    batch_size, num_kv_heads, seq_len, head_dim = k.shape

    return {
        "num_layers": num_layers,
        "batch_size": batch_size,
        "num_kv_heads": num_kv_heads,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "total_elements": total_elements,
        "memory_mb": total_elements * k.element_size() / (1024 ** 2),
    }
