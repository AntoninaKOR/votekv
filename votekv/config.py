"""Configuration dataclass for VoteKV"""

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class VoteKVConfig:
    """Configuration for VoteKV KV-cache compression"""
    
    # Model configuration
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    device: str = "cuda"
    dtype: str = "bfloat16"
    use_flash_attention: bool = False
    
    # KV compression method
    kv_method: Literal[
        "full_cache",
        "gqa_mean",
        "gqa_max",
        "gqa_vote",
        "gqa_rank_vote",
        "gqa_vote_rescue"
    ] = "gqa_vote_rescue"
    # Note: All methods (except full_cache) use SnapKV's observation window
    # scoring methodology via compute_snapkv_scores_from_attentions()
    
    # Generation parameters
    max_context_len: int = 8192
    max_new_tokens: int = 64
    observation_window: int = 32
    
    # Budget configuration
    kv_budget: Optional[int] = None
    kv_budget_ratio: float = 0.08
    
    # Voting parameters
    vote_topk: int = 128
    group_budget: Optional[int] = None
    rescue_budget: int = 4
    
    # Always-keep tokens
    sink_tokens: int = 4
    recent_tokens: int = 32
    
    # System parameters
    output_attentions: bool = True
    batch_size: int = 1
    seed: int = 42
    cache_layout: str = "layer_shared"
    
    def resolve_budget(self, seq_len: int) -> int:
        """Calculate actual budget based on config"""
        if self.kv_budget is not None:
            budget = self.kv_budget
        else:
            budget = int(seq_len * self.kv_budget_ratio)
        
        min_budget = self.sink_tokens + self.recent_tokens + 1
        budget = max(budget, min_budget)
        budget = min(budget, seq_len)
        return budget
    
    def get_dtype(self):
        """Convert dtype string to torch dtype"""
        import torch
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.dtype, torch.bfloat16)
