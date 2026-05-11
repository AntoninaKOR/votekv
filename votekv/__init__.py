"""
VoteKV: GQA-aware SnapKV with voting and rescue tokens
Research MVP for inference-time KV-cache compression
"""

__version__ = "0.1.0"

from .config import VoteKVConfig
from .gqa_utils import (
    get_gqa_info,
    build_gqa_groups,
    query_head_to_kv_head,
    group_scores_by_gqa,
)
from .scoring import (
    compute_snapkv_scores_from_attentions,
    compute_snapkv_scores_via_hooks,
    get_always_keep_indices,
)
from .selectors import (
    select_tokens,
    select_full_cache,
    select_gqa_mean,
    select_gqa_max,
    select_gqa_vote,
    select_gqa_vote_rescue,
    select_gqa_rank_vote,
)
from .cache_compression import (
    convert_kv_head_mask_to_layer_indices,
    compress_past_key_values_layer_shared,
    get_cache_info,
)
from .generation import (
    generate_with_compressed_cache,
    simple_greedy_generate,
)
from .model_utils import load_model_and_tokenizer
from .metrics import (
    compute_compression_ratio,
    compute_gqa_disagreement,
    compute_vote_histogram,
    exact_match,
    contains_answer,
)

__all__ = [
    "VoteKVConfig",
    "get_gqa_info",
    "build_gqa_groups",
    "query_head_to_kv_head",
    "group_scores_by_gqa",
    "compute_snapkv_scores_from_attentions",
    "compute_snapkv_scores_via_hooks",
    "get_always_keep_indices",
    "select_tokens",
    "select_full_cache",
    "select_gqa_mean",
    "select_gqa_max",
    "select_gqa_vote",
    "select_gqa_vote_rescue",
    "select_gqa_rank_vote",
    "convert_kv_head_mask_to_layer_indices",
    "compress_past_key_values_layer_shared",
    "get_cache_info",
    "generate_with_compressed_cache",
    "simple_greedy_generate",
    "load_model_and_tokenizer",
    "compute_compression_ratio",
    "compute_gqa_disagreement",
    "compute_vote_histogram",
    "exact_match",
    "contains_answer",
]
