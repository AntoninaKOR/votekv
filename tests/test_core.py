"""Unit tests for VoteKV core utilities"""

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available, skipping tests that require torch")

from votekv.gqa_utils import (
    query_head_to_kv_head,
    build_gqa_groups,
)
from votekv.scoring import get_always_keep_indices
from votekv.config import VoteKVConfig

if TORCH_AVAILABLE:
    from votekv.gqa_utils import get_gqa_info, group_scores_by_gqa


class MockModel:
    """Mock model for testing"""
    class MockConfig:
        def __init__(self):
            self.num_attention_heads = 32
            self.num_key_value_heads = 8
    
    def __init__(self):
        self.config = self.MockConfig()


def test_gqa_grouping():
    """Test GQA group mapping for Mistral-like config"""
    num_attention_heads = 32
    num_key_value_heads = 8
    
    groups = build_gqa_groups(num_attention_heads, num_key_value_heads)
    
    # Check group 0
    assert groups[0] == [0, 1, 2, 3], "KV head 0 should have Q heads [0,1,2,3]"
    
    # Check group 7
    assert groups[7] == [28, 29, 30, 31], "KV head 7 should have Q heads [28,29,30,31]"
    
    # Check all groups have correct size
    for kv_head in range(num_key_value_heads):
        assert len(groups[kv_head]) == 4, f"Each group should have 4 query heads"


def test_query_to_kv_mapping():
    """Test query head to KV head mapping"""
    group_size = 4
    
    assert query_head_to_kv_head(0, group_size) == 0
    assert query_head_to_kv_head(3, group_size) == 0
    assert query_head_to_kv_head(4, group_size) == 1
    assert query_head_to_kv_head(28, group_size) == 7
    assert query_head_to_kv_head(31, group_size) == 7


def test_get_gqa_info():
    """Test extracting GQA info from model"""
    if not TORCH_AVAILABLE:
        print("Skipping test_get_gqa_info (torch not available)")
        return
    
    model = MockModel()
    gqa_info = get_gqa_info(model)
    
    assert gqa_info["num_attention_heads"] == 32
    assert gqa_info["num_key_value_heads"] == 8
    assert gqa_info["group_size"] == 4


def test_always_keep_indices():
    """Test always-keep token selection"""
    seq_len = 100
    sink_tokens = 4
    recent_tokens = 10
    
    always_keep = get_always_keep_indices(seq_len, sink_tokens, recent_tokens)
    
    # Should include first 4 tokens
    assert 0 in always_keep
    assert 1 in always_keep
    assert 2 in always_keep
    assert 3 in always_keep
    
    # Should include last 10 tokens
    assert 90 in always_keep
    assert 91 in always_keep
    assert 99 in always_keep
    
    # Should be sorted
    assert always_keep == sorted(always_keep)


def test_group_scores_by_gqa():
    """Test score grouping by GQA"""
    num_layers = 2
    num_attention_heads = 32
    num_key_value_heads = 8
    seq_len = 100
    
    scores = torch.randn(num_layers, num_attention_heads, seq_len)
    
    grouped = group_scores_by_gqa(scores, num_attention_heads, num_key_value_heads)
    
    # Check shape
    assert grouped.shape == (num_layers, num_key_value_heads, 4, seq_len)
    
    # Check that grouping preserves values
    # KV head 0 should have Q heads 0-3
    assert torch.allclose(grouped[0, 0, 0], scores[0, 0])
    assert torch.allclose(grouped[0, 0, 3], scores[0, 3])
    
    # KV head 1 should have Q heads 4-7
    assert torch.allclose(grouped[0, 1, 0], scores[0, 4])
    assert torch.allclose(grouped[0, 1, 3], scores[0, 7])


def test_budget_resolution():
    """Test budget calculation"""
    config = VoteKVConfig(
        kv_budget_ratio=0.1,
        sink_tokens=4,
        recent_tokens=10,
    )
    
    seq_len = 1000
    budget = config.resolve_budget(seq_len)
    
    # Should be 10% of 1000 = 100
    assert budget == 100
    
    # Test minimum budget
    config2 = VoteKVConfig(
        kv_budget_ratio=0.001,  # Very small ratio
        sink_tokens=4,
        recent_tokens=10,
    )
    budget2 = config2.resolve_budget(1000)
    
    # Should be at least sink + recent + 1
    assert budget2 >= 15


def test_voting_example():
    """Test voting logic with concrete example"""
    # Simulate voting from 4 query heads
    seq_len = 100
    
    # Each head's top-3 tokens
    head_0_top = {7, 25, 41}
    head_1_top = {3, 25, 60}
    head_2_top = {7, 18, 41}
    head_3_top = {25, 41, 60}
    
    # Count votes
    vote_counts = {}
    for token_set in [head_0_top, head_1_top, head_2_top, head_3_top]:
        for token in token_set:
            vote_counts[token] = vote_counts.get(token, 0) + 1
    
    # Expected votes
    assert vote_counts[25] == 3, "Token 25 should have 3 votes"
    assert vote_counts[41] == 3, "Token 41 should have 3 votes"
    assert vote_counts[7] == 2, "Token 7 should have 2 votes"
    assert vote_counts[60] == 2, "Token 60 should have 2 votes"
    assert vote_counts[3] == 1, "Token 3 should have 1 vote"
    assert vote_counts[18] == 1, "Token 18 should have 1 vote"


if __name__ == "__main__":
    # Run tests
    test_gqa_grouping()
    test_query_to_kv_mapping()
    test_get_gqa_info()
    test_always_keep_indices()
    test_group_scores_by_gqa()
    test_budget_resolution()
    test_voting_example()
    
    print("All tests passed!")
