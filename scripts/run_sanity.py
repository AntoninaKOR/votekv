"""Sanity check script for VoteKV

Tests basic functionality with a simple prompt containing a passkey.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import argparse
import logging
from dataclasses import asdict
import time

from votekv.config import VoteKVConfig
from votekv.model_utils import load_model_and_tokenizer
from votekv.gqa_utils import get_gqa_info
from votekv.scoring import compute_snapkv_scores_from_attentions
from votekv.selectors import select_tokens
from votekv.cache_compression import (
    convert_kv_head_mask_to_layer_indices,
    compress_past_key_values_layer_shared,
    get_cache_info,
)
from votekv.generation import generate_with_compressed_cache, simple_greedy_generate
from votekv.logging_utils import setup_logging, log_memory_stats, reset_memory_stats, log_model_info

logger = logging.getLogger(__name__)


def create_sanity_prompt(target_len: int = 1024, passkey: str = "493827") -> str:
    """Create a synthetic prompt with a hidden passkey
    
    Args:
        target_len: Target token length (approximate)
        passkey: Secret passkey to hide in text
        
    Returns:
        Prompt string
    """
    filler = (
        "The quick brown fox jumps over the lazy dog. "
        "This is a sample text used for testing purposes. "
        "We repeat this text multiple times to create a long context. "
    )
    
    # Calculate repetitions needed (rough estimate: ~20 tokens per repetition)
    tokens_per_rep = 20
    needed_reps = target_len // tokens_per_rep
    
    # Insert passkey early in the text
    prompt = f"There is a secret passkey hidden in the text: {passkey}. "
    prompt += "Remember this passkey. " * 3
    prompt += filler * (needed_reps // 2)
    prompt += f"\n\nThe passkey mentioned earlier was: {passkey}.\n\n"
    prompt += filler * (needed_reps // 2)
    prompt += f"\n\nQuestion: What is the secret passkey mentioned in this text?\nAnswer:"
    
    return prompt


@torch.no_grad()
def run_sanity_test(
    config: VoteKVConfig,
    method: str,
    target_len: int = 1024,
):
    """Run sanity test for a specific method
    
    Args:
        config: VoteKV configuration
        method: KV selection method
        target_len: Target prompt length in tokens
        
    Returns:
        Dictionary with results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Running sanity test: {method}")
    logger.info(f"{'='*80}")
    
    # Load model
    device = config.device
    dtype = config.get_dtype()
    
    model, tokenizer = load_model_and_tokenizer(
        config.model_name, device=device, dtype=dtype
    )
    
    # Get GQA info
    gqa_info = get_gqa_info(model)
    logger.info(f"GQA Info: {gqa_info}")
    
    # Create prompt
    passkey = "493827"
    prompt = create_sanity_prompt(target_len=target_len, passkey=passkey)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    prompt_len = input_ids.shape[1]
    logger.info(f"Prompt length: {prompt_len} tokens")
    
    reset_memory_stats(device)
    start_time = time.perf_counter()
    
    # Prefill phase
    prefill_start = time.perf_counter()
    
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        output_attentions=(method != "full_cache"),
        return_dict=True,
    )
    
    past_key_values = outputs.past_key_values
    logits = outputs.logits
    
    prefill_time = time.perf_counter() - prefill_start
    
    # Cache info before compression
    cache_info_before = get_cache_info(past_key_values)
    logger.info(f"Cache before compression: {cache_info_before}")
    
    # Compression phase
    if method == "full_cache":
        compressed_cache = past_key_values
        retained_count = prompt_len
        compression_time = 0.0
    else:
        compression_start = time.perf_counter()
        
        # Compute scores
        attentions = outputs.attentions
        scores = compute_snapkv_scores_from_attentions(
            attentions, config.observation_window
        )
        
        # Select tokens
        mask = select_tokens(scores, method, config, gqa_info)
        
        # Convert to layer indices
        budget = config.resolve_budget(prompt_len)
        selected_indices = convert_kv_head_mask_to_layer_indices(
            mask, scores, budget
        )
        
        # Compress cache
        compressed_cache = compress_past_key_values_layer_shared(
            past_key_values, selected_indices
        )
        
        retained_count = selected_indices[0].shape[0]
        compression_time = time.perf_counter() - compression_start
        
        logger.info(f"Selected tokens (layer 0): {selected_indices[0].tolist()[:20]}...")
        
        # Clean up
        del attentions, scores, mask
        torch.cuda.empty_cache()
    
    # Cache info after compression
    cache_info_after = get_cache_info(compressed_cache)
    logger.info(f"Cache after compression: {cache_info_after}")
    
    # Generation phase
    decode_start = time.perf_counter()
    
    # Get first token from prefill logits
    next_token = logits[:, -1:, :].argmax(dim=-1)
    
    generated_ids, gen_stats = generate_with_compressed_cache(
        model=model,
        tokenizer=tokenizer,
        input_ids=next_token,
        attention_mask=attention_mask,
        past_key_values=compressed_cache,
        max_new_tokens=config.max_new_tokens,
        original_seq_len=prompt_len,
    )
    
    decode_time = time.perf_counter() - decode_start
    total_time = time.perf_counter() - start_time
    
    # Decode output
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    logger.info(f"\nGenerated text: {generated_text}")
    logger.info(f"Correct passkey: {passkey}")
    logger.info(f"Passkey in output: {passkey in generated_text}")
    
    # Memory stats
    memory_stats = {
        "peak_memory_gb": torch.cuda.max_memory_allocated() / (1024 ** 3)
    }
    log_memory_stats(device)
    
    # Timing
    logger.info(f"\nTiming:")
    logger.info(f"  Prefill: {prefill_time:.4f}s")
    logger.info(f"  Compression: {compression_time:.4f}s")
    logger.info(f"  Decode: {decode_time:.4f}s")
    logger.info(f"  Total: {total_time:.4f}s")
    
    # Compression ratio
    compression_ratio = prompt_len / retained_count if retained_count > 0 else 1.0
    logger.info(f"\nCompression ratio: {compression_ratio:.2f}x ({prompt_len} -> {retained_count})")
    
    return {
        "method": method,
        "prompt_len": prompt_len,
        "retained_tokens": retained_count,
        "compression_ratio": compression_ratio,
        "generated_text": generated_text,
        "contains_passkey": passkey in generated_text,
        "prefill_time": prefill_time,
        "compression_time": compression_time,
        "decode_time": decode_time,
        "total_time": total_time,
        "peak_memory_gb": memory_stats["peak_memory_gb"],
    }


def main():
    parser = argparse.ArgumentParser(description="VoteKV Sanity Test")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--methods", nargs="+", default=["full_cache", "gqa_mean", "gqa_max", "gqa_vote", "gqa_vote_rescue"])
    parser.add_argument("--target_len", type=int, default=1024, help="Target prompt length")
    parser.add_argument("--kv_budget_ratio", type=float, default=0.08)
    parser.add_argument("--observation_window", type=int, default=32)
    parser.add_argument("--vote_topk", type=int, default=128)
    parser.add_argument("--rescue_budget", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Create config
    config = VoteKVConfig(
        model_name=args.model_name,
        device=args.device,
        kv_budget_ratio=args.kv_budget_ratio,
        observation_window=args.observation_window,
        vote_topk=args.vote_topk,
        rescue_budget=args.rescue_budget,
    )
    
    log_model_info(None, None, asdict(config))
    
    results = []
    
    for method in args.methods:
        try:
            result = run_sanity_test(config, method, args.target_len)
            results.append(result)
        except Exception as e:
            logger.error(f"Error running {method}: {e}", exc_info=True)
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"{'Method':<25} {'Retained':<10} {'Ratio':<10} {'Passkey':<10} {'Time (s)':<10}")
    logger.info("-"*80)
    
    for r in results:
        logger.info(
            f"{r['method']:<25} {r['retained_tokens']:<10} "
            f"{r['compression_ratio']:<10.2f} {str(r['contains_passkey']):<10} "
            f"{r['total_time']:<10.2f}"
        )


if __name__ == "__main__":
    main()
