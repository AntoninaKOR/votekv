"""KV token selection methods"""

import torch
from typing import Dict, Tuple
from .config import VoteKVConfig
from .gqa_utils import group_scores_by_gqa
from .scoring import get_always_keep_indices
import logging

logger = logging.getLogger(__name__)


def select_tokens(
    scores: torch.Tensor,
    method: str,
    config: VoteKVConfig,
    model_config: Dict[str, int],
) -> torch.Tensor:
    """Select tokens to keep using specified method
    
    Args:
        scores: [num_layers, num_attention_heads, seq_len]
        method: Selection method name
        config: VoteKV configuration
        model_config: Dict with num_attention_heads, num_key_value_heads, group_size
        
    Returns:
        mask: [num_layers, num_key_value_heads, seq_len] bool tensor
    """
    num_layers, num_attention_heads, seq_len = scores.shape
    num_key_value_heads = model_config["num_key_value_heads"]
    
    if method == "full_cache":
        return select_full_cache(num_layers, num_key_value_heads, seq_len)
    elif method == "gqa_mean":
        return select_gqa_mean(scores, config, model_config)
    elif method == "gqa_max":
        return select_gqa_max(scores, config, model_config)
    elif method == "gqa_vote":
        return select_gqa_vote(scores, config, model_config)
    elif method == "gqa_vote_rescue":
        return select_gqa_vote_rescue(scores, config, model_config)
    elif method == "gqa_rank_vote":
        return select_gqa_rank_vote(scores, config, model_config)
    else:
        raise ValueError(f"Unknown method: {method}")


def select_full_cache(num_layers: int, num_kv_heads: int, seq_len: int) -> torch.Tensor:
    """Full cache: keep all tokens"""
    return torch.ones(num_layers, num_kv_heads, seq_len, dtype=torch.bool)


def select_gqa_mean(
    scores: torch.Tensor,
    config: VoteKVConfig,
    model_config: Dict[str, int],
) -> torch.Tensor:
    """GQA-Mean: Average scores across query heads in each GQA group
    
    Inspired by Ada-KV aggregation strategy, but with uniform budget allocation.
    Original Ada-KV includes adaptive per-layer budgets, which is not implemented here.
    
    Args:
        scores: [num_layers, num_attention_heads, seq_len]
        config: VoteKV configuration
        model_config: GQA info
        
    Returns:
        mask: [num_layers, num_key_value_heads, seq_len]
    """
    num_layers, num_attention_heads, seq_len = scores.shape
    num_kv_heads = model_config["num_key_value_heads"]
    
    # Group scores by GQA groups
    group_scores = group_scores_by_gqa(scores, num_attention_heads, num_kv_heads)
    # [num_layers, num_kv_heads, group_size, seq_len]
    
    budget = config.resolve_budget(seq_len)
    always_keep = get_always_keep_indices(seq_len, config.sink_tokens, config.recent_tokens)
    
    mask = torch.zeros(num_layers, num_kv_heads, seq_len, dtype=torch.bool)
    
    for layer_idx in range(num_layers):
        for kv_head in range(num_kv_heads):
            # Mean score across query heads in this group
            mean_score = group_scores[layer_idx, kv_head].mean(dim=0)  # [seq_len]
            
            # Select top tokens
            selected = _select_top_tokens_with_budget(
                mean_score, budget, always_keep, seq_len
            )
            
            mask[layer_idx, kv_head, selected] = True
    
    return mask


def select_gqa_max(
    scores: torch.Tensor,
    config: VoteKVConfig,
    model_config: Dict[str, int],
) -> torch.Tensor:
    """GQA-Max: Maximum score across query heads in each GQA group
    
    Inspired by R-KV max aggregation for redundancy handling.
    Simplified baseline focusing on max aggregation strategy.
    
    Args:
        scores: [num_layers, num_attention_heads, seq_len]
        config: VoteKV configuration
        model_config: GQA info
        
    Returns:
        mask: [num_layers, num_key_value_heads, seq_len]
    """
    num_layers, num_attention_heads, seq_len = scores.shape
    num_kv_heads = model_config["num_key_value_heads"]
    
    group_scores = group_scores_by_gqa(scores, num_attention_heads, num_kv_heads)
    budget = config.resolve_budget(seq_len)
    always_keep = get_always_keep_indices(seq_len, config.sink_tokens, config.recent_tokens)
    
    mask = torch.zeros(num_layers, num_kv_heads, seq_len, dtype=torch.bool)
    
    for layer_idx in range(num_layers):
        for kv_head in range(num_kv_heads):
            # Max score across query heads in this group
            max_score = group_scores[layer_idx, kv_head].max(dim=0).values  # [seq_len]
            
            selected = _select_top_tokens_with_budget(
                max_score, budget, always_keep, seq_len
            )
            
            mask[layer_idx, kv_head, selected] = True
    
    return mask


def select_gqa_vote(
    scores: torch.Tensor,
    config: VoteKVConfig,
    model_config: Dict[str, int],
) -> torch.Tensor:
    """GQA-Vote: Voting-based selection within each GQA group
    
    Each query head votes for its top-k tokens. Tokens with most votes are selected.
    
    Args:
        scores: [num_layers, num_attention_heads, seq_len]
        config: VoteKV configuration
        model_config: GQA info
        
    Returns:
        mask: [num_layers, num_key_value_heads, seq_len]
    """
    num_layers, num_attention_heads, seq_len = scores.shape
    num_kv_heads = model_config["num_key_value_heads"]
    group_size = model_config["group_size"]
    
    group_scores = group_scores_by_gqa(scores, num_attention_heads, num_kv_heads)
    budget = config.resolve_budget(seq_len)
    always_keep = get_always_keep_indices(seq_len, config.sink_tokens, config.recent_tokens)
    vote_topk = min(config.vote_topk, seq_len - len(always_keep))
    
    mask = torch.zeros(num_layers, num_kv_heads, seq_len, dtype=torch.bool)
    
    for layer_idx in range(num_layers):
        for kv_head in range(num_kv_heads):
            scores_group = group_scores[layer_idx, kv_head]  # [group_size, seq_len]
            
            # Voting
            vote_counts = torch.zeros(seq_len, dtype=torch.float32, device=scores.device)
            sum_scores = scores_group.sum(dim=0)  # For tie-breaking
            
            # Get candidates (exclude always_keep)
            candidates_mask = torch.ones(seq_len, dtype=torch.bool, device=scores.device)
            candidates_mask[always_keep] = False
            candidate_indices = torch.where(candidates_mask)[0]
            
            # Each head votes for top-k tokens
            for head_in_group in range(group_size):
                head_scores = scores_group[head_in_group]
                candidate_scores = head_scores[candidate_indices]
                
                if len(candidate_indices) > 0 and vote_topk > 0:
                    k = min(vote_topk, len(candidate_indices))
                    _, top_indices = torch.topk(candidate_scores, k, largest=True)
                    voted_tokens = candidate_indices[top_indices]
                    vote_counts[voted_tokens] += 1
            
            # Select top tokens by vote count, tie-break with sum_scores
            # Composite score: vote_count * large_number + normalized_sum_score
            composite = vote_counts * 1e6 + sum_scores
            
            selected = _select_top_tokens_with_budget(
                composite, budget, always_keep, seq_len
            )
            
            mask[layer_idx, kv_head, selected] = True
    
    return mask


def select_gqa_vote_rescue(
    scores: torch.Tensor,
    config: VoteKVConfig,
    model_config: Dict[str, int],
) -> torch.Tensor:
    """GQA-Vote with Rescue: Voting + head-specific rescue tokens
    
    First select consensus tokens via voting, then add rescue tokens from each head
    to preserve head-specific signals.
    
    Args:
        scores: [num_layers, num_attention_heads, seq_len]
        config: VoteKV configuration
        model_config: GQA info
        
    Returns:
        mask: [num_layers, num_key_value_heads, seq_len]
    """
    num_layers, num_attention_heads, seq_len = scores.shape
    num_kv_heads = model_config["num_key_value_heads"]
    group_size = model_config["group_size"]
    
    group_scores = group_scores_by_gqa(scores, num_attention_heads, num_kv_heads)
    budget = config.resolve_budget(seq_len)
    always_keep = get_always_keep_indices(seq_len, config.sink_tokens, config.recent_tokens)
    always_keep_set = set(always_keep)
    
    vote_topk = min(config.vote_topk, seq_len - len(always_keep))
    
    mask = torch.zeros(num_layers, num_kv_heads, seq_len, dtype=torch.bool)
    
    for layer_idx in range(num_layers):
        for kv_head in range(num_kv_heads):
            scores_group = group_scores[layer_idx, kv_head]  # [group_size, seq_len]
            
            available = budget - len(always_keep)
            if available <= 0:
                mask[layer_idx, kv_head, always_keep] = True
                continue
            
            # Budget allocation
            max_rescue_total = group_size * config.rescue_budget
            rescue_budget_total = min(max_rescue_total, max(0, available // 2))
            vote_budget = available - rescue_budget_total
            vote_budget = max(1, vote_budget)
            
            # Get candidates
            candidates_mask = torch.ones(seq_len, dtype=torch.bool, device=scores.device)
            candidates_mask[always_keep] = False
            candidate_indices = torch.where(candidates_mask)[0]
            
            # Voting phase
            vote_counts = torch.zeros(seq_len, dtype=torch.float32, device=scores.device)
            sum_scores = scores_group.sum(dim=0)
            
            for head_in_group in range(group_size):
                head_scores = scores_group[head_in_group]
                candidate_scores = head_scores[candidate_indices]
                
                if len(candidate_indices) > 0 and vote_topk > 0:
                    k = min(vote_topk, len(candidate_indices))
                    _, top_indices = torch.topk(candidate_scores, k, largest=True)
                    voted_tokens = candidate_indices[top_indices]
                    vote_counts[voted_tokens] += 1
            
            # Select vote winners
            composite = vote_counts * 1e6 + sum_scores
            candidate_composite = composite[candidate_indices]
            
            if len(candidate_indices) > 0 and vote_budget > 0:
                k_vote = min(vote_budget, len(candidate_indices))
                _, top_vote_indices = torch.topk(candidate_composite, k_vote, largest=True)
                selected_vote = candidate_indices[top_vote_indices].tolist()
            else:
                selected_vote = []
            
            selected_set = always_keep_set | set(selected_vote)
            
            # Rescue phase: each head adds rescue_budget tokens
            rescue_tokens = []
            for head_in_group in range(group_size):
                head_scores = scores_group[head_in_group]
                
                # Exclude already selected
                rescue_candidates_mask = torch.ones(seq_len, dtype=torch.bool, device=scores.device)
                rescue_candidates_mask[list(selected_set)] = False
                rescue_candidates_mask[rescue_tokens] = False
                rescue_candidate_indices = torch.where(rescue_candidates_mask)[0]
                
                if len(rescue_candidate_indices) > 0 and config.rescue_budget > 0:
                    rescue_candidate_scores = head_scores[rescue_candidate_indices]
                    k_rescue = min(config.rescue_budget, len(rescue_candidate_indices))
                    _, rescue_top_indices = torch.topk(rescue_candidate_scores, k_rescue, largest=True)
                    rescue_from_head = rescue_candidate_indices[rescue_top_indices].tolist()
                    rescue_tokens.extend(rescue_from_head)
            
            # Combine all selected
            final_selected = list(selected_set) + rescue_tokens
            
            # Enforce budget
            if len(final_selected) > budget:
                # Priority: always_keep > vote > rescue
                # Keep always_keep and vote, trim rescue by score
                guaranteed = list(always_keep_set) + selected_vote
                remaining_budget = budget - len(guaranteed)
                
                if remaining_budget > 0:
                    rescue_scores_dict = {
                        idx: scores_group[:, idx].max().item() for idx in rescue_tokens
                    }
                    rescue_sorted = sorted(
                        rescue_tokens, key=lambda x: rescue_scores_dict[x], reverse=True
                    )
                    final_selected = guaranteed + rescue_sorted[:remaining_budget]
                else:
                    final_selected = guaranteed[:budget]
            
            mask[layer_idx, kv_head, final_selected] = True
    
    return mask


def select_gqa_rank_vote(
    scores: torch.Tensor,
    config: VoteKVConfig,
    model_config: Dict[str, int],
) -> torch.Tensor:
    """GQA-Rank-Vote: Rank-weighted voting (Borda count style)
    
    Each query head contributes weighted votes based on token rank in its top-k.
    Unlike simple voting (where top-k tokens get equal weight), this method
    gives more weight to tokens that rank higher.
    
    Weighting: rank 0 → vote_topk points, rank 1 → (vote_topk-1) points, etc.
    
    Example: If token X is rank 0 for head A (128 pts), rank 2 for head B (126 pts),
    it gets 254 total points. Token Y with rank 50 in three heads gets ~234 points.
    Token X wins despite fewer heads voting for it, because it ranks higher.
    
    Args:
        scores: [num_layers, num_attention_heads, seq_len]
        config: VoteKV configuration
        model_config: GQA info
        
    Returns:
        mask: [num_layers, num_key_value_heads, seq_len]
    """
    num_layers, num_attention_heads, seq_len = scores.shape
    num_kv_heads = model_config["num_key_value_heads"]
    group_size = model_config["group_size"]
    
    group_scores = group_scores_by_gqa(scores, num_attention_heads, num_kv_heads)
    budget = config.resolve_budget(seq_len)
    always_keep = get_always_keep_indices(seq_len, config.sink_tokens, config.recent_tokens)
    vote_topk = min(config.vote_topk, seq_len - len(always_keep))
    
    mask = torch.zeros(num_layers, num_kv_heads, seq_len, dtype=torch.bool)
    
    for layer_idx in range(num_layers):
        for kv_head in range(num_kv_heads):
            scores_group = group_scores[layer_idx, kv_head]  # [group_size, seq_len]
            
            rank_scores = torch.zeros(seq_len, dtype=torch.float32, device=scores.device)
            sum_scores = scores_group.sum(dim=0)
            
            candidates_mask = torch.ones(seq_len, dtype=torch.bool, device=scores.device)
            candidates_mask[always_keep] = False
            candidate_indices = torch.where(candidates_mask)[0]
            
            # Rank-based voting
            for head_in_group in range(group_size):
                head_scores = scores_group[head_in_group]
                candidate_scores = head_scores[candidate_indices]
                
                if len(candidate_indices) > 0 and vote_topk > 0:
                    k = min(vote_topk, len(candidate_indices))
                    _, top_indices = torch.topk(candidate_scores, k, largest=True)
                    
                    # Assign rank scores: rank 0 gets vote_topk, rank 1 gets vote_topk-1, etc.
                    for rank, idx_in_candidates in enumerate(top_indices):
                        token_idx = candidate_indices[idx_in_candidates]
                        rank_scores[token_idx] += (vote_topk - rank)
            
            # Composite: rank_score dominates, sum_scores for tie-break
            composite = rank_scores * 1e6 + sum_scores
            
            selected = _select_top_tokens_with_budget(
                composite, budget, always_keep, seq_len
            )
            
            mask[layer_idx, kv_head, selected] = True
    
    return mask


def _select_top_tokens_with_budget(
    score: torch.Tensor,
    budget: int,
    always_keep: list,
    seq_len: int,
) -> list:
    """Helper to select top tokens respecting budget and always-keep constraints
    
    Args:
        score: [seq_len] importance scores
        budget: Total budget
        always_keep: List of indices to always keep
        seq_len: Sequence length
        
    Returns:
        List of selected indices
    """
    always_keep_set = set(always_keep)
    extra_budget = budget - len(always_keep)
    
    if extra_budget <= 0:
        return always_keep
    
    # Get candidates
    candidates_mask = torch.ones(seq_len, dtype=torch.bool, device=score.device)
    candidates_mask[always_keep] = False
    candidate_indices = torch.where(candidates_mask)[0]
    
    if len(candidate_indices) == 0:
        return always_keep
    
    candidate_scores = score[candidate_indices]
    k = min(extra_budget, len(candidate_indices))
    
    _, top_indices = torch.topk(candidate_scores, k, largest=True)
    selected_extra = candidate_indices[top_indices].tolist()
    
    return sorted(list(always_keep_set) + selected_extra)
