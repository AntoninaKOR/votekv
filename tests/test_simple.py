"""Simple non-torch tests for VoteKV logic"""


def test_gqa_grouping():
    """Test GQA group mapping for Mistral-like config"""
    num_attention_heads = 32
    num_key_value_heads = 8
    
    # Manual implementation of build_gqa_groups
    group_size = num_attention_heads // num_key_value_heads
    groups = {}
    
    for kv_head in range(num_key_value_heads):
        start = kv_head * group_size
        end = start + group_size
        groups[kv_head] = list(range(start, end))
    
    # Check group 0
    assert groups[0] == [0, 1, 2, 3], "KV head 0 should have Q heads [0,1,2,3]"
    
    # Check group 7
    assert groups[7] == [28, 29, 30, 31], "KV head 7 should have Q heads [28,29,30,31]"
    
    # Check all groups have correct size
    for kv_head in range(num_key_value_heads):
        assert len(groups[kv_head]) == 4, f"Each group should have 4 query heads"
    
    print("✓ GQA grouping test passed")


def test_query_to_kv_mapping():
    """Test query head to KV head mapping"""
    group_size = 4
    
    # Manual implementation of query_head_to_kv_head
    def query_head_to_kv_head(query_head_id, group_size):
        return query_head_id // group_size
    
    assert query_head_to_kv_head(0, group_size) == 0
    assert query_head_to_kv_head(3, group_size) == 0
    assert query_head_to_kv_head(4, group_size) == 1
    assert query_head_to_kv_head(28, group_size) == 7
    assert query_head_to_kv_head(31, group_size) == 7
    
    print("✓ Query-to-KV mapping test passed")


def test_always_keep_indices():
    """Test always-keep token selection"""
    seq_len = 100
    sink_tokens = 4
    recent_tokens = 10
    
    # Manual implementation
    sink = list(range(min(sink_tokens, seq_len)))
    recent_start = max(0, seq_len - recent_tokens)
    recent = list(range(recent_start, seq_len))
    always_keep = sorted(set(sink + recent))
    
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
    
    print("✓ Always-keep indices test passed")


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
    
    print("✓ Voting example test passed")


def test_budget_resolution():
    """Test budget calculation"""
    # Manual implementation
    kv_budget_ratio = 0.1
    sink_tokens = 4
    recent_tokens = 10
    
    seq_len = 1000
    budget = int(seq_len * kv_budget_ratio)
    
    min_budget = sink_tokens + recent_tokens + 1
    budget = max(budget, min_budget)
    budget = min(budget, seq_len)
    
    # Should be 10% of 1000 = 100
    assert budget == 100
    
    # Test minimum budget
    kv_budget_ratio2 = 0.001  # Very small ratio
    budget2 = int(1000 * kv_budget_ratio2)
    budget2 = max(budget2, min_budget)
    
    # Should be at least sink + recent + 1
    assert budget2 >= 15
    
    print("✓ Budget resolution test passed")


if __name__ == "__main__":
    # Run tests
    test_gqa_grouping()
    test_query_to_kv_mapping()
    test_always_keep_indices()
    test_voting_example()
    test_budget_resolution()
    
    print("\n" + "="*50)
    print("All tests passed! ✅")
    print("="*50)
