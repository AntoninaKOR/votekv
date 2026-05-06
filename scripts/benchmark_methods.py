"""Simple benchmark script to compare all methods"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import argparse
import logging
import json
from pathlib import Path
from dataclasses import asdict
import time

from votekv.config import VoteKVConfig
from votekv.model_utils import load_model_and_tokenizer
from votekv.gqa_utils import get_gqa_info
from votekv.scoring import compute_snapkv_scores_from_attentions
from votekv.selectors import select_tokens
from votekv.cache_compression import convert_kv_head_mask_to_layer_indices, compress_past_key_values_layer_shared
from votekv.generation import generate_with_compressed_cache
from votekv.metrics import compute_gqa_disagreement, compute_vote_histogram
from votekv.logging_utils import setup_logging, reset_memory_stats

logger = logging.getLogger(__name__)


@torch.no_grad()
def benchmark_method(
    model,
    tokenizer,
    config: VoteKVConfig,
    method: str,
    gqa_info: dict,
    prompt: str,
) -> dict:
    """Benchmark a single method on a prompt
    
    Args:
        model: Model
        tokenizer: Tokenizer
        config: Configuration
        method: Method name
        gqa_info: GQA info
        prompt: Input prompt
        
    Returns:
        Results dictionary
    """
    device = config.device
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=config.max_context_len).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    prompt_len = input_ids.shape[1]
    
    reset_memory_stats(device)
    start_time = time.perf_counter()
    
    # Prefill
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        output_attentions=(method != "full_cache"),
        return_dict=True,
    )
    
    past_key_values = outputs.past_key_values
    logits = outputs.logits
    prefill_time = time.perf_counter() - start_time
    
    # Compression
    if method == "full_cache":
        compressed_cache = past_key_values
        retained_count = prompt_len
        compression_time = 0.0
        scores = None
        mask = None
    else:
        compression_start = time.perf_counter()
        
        attentions = outputs.attentions
        scores = compute_snapkv_scores_from_attentions(attentions, config.observation_window)
        
        mask = select_tokens(scores, method, config, gqa_info)
        
        budget = config.resolve_budget(prompt_len)
        selected_indices = convert_kv_head_mask_to_layer_indices(mask, scores, budget)
        
        compressed_cache = compress_past_key_values_layer_shared(past_key_values, selected_indices)
        
        retained_count = selected_indices[0].shape[0]
        compression_time = time.perf_counter() - compression_start
        
        del attentions
        torch.cuda.empty_cache()
    
    # Generation
    decode_start = time.perf_counter()
    next_token = logits[:, -1:, :].argmax(dim=-1)
    
    generated_ids, _ = generate_with_compressed_cache(
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
    
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    result = {
        "method": method,
        "prompt_len": prompt_len,
        "retained_tokens": retained_count,
        "compression_ratio": prompt_len / retained_count if retained_count > 0 else 1.0,
        "prefill_sec": prefill_time,
        "compression_sec": compression_time,
        "decode_sec": decode_time,
        "total_sec": total_time,
        "tokens_per_sec": len(generated_ids[0]) / decode_time if decode_time > 0 else 0,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "output_text": output_text[:200],  # Truncate for logging
    }
    
    # Disagreement analysis for voting methods
    if scores is not None and method in ["gqa_vote", "gqa_vote_rescue", "gqa_rank_vote"]:
        disagreement = compute_gqa_disagreement(scores, gqa_info, config.vote_topk)
        result["avg_disagreement"] = disagreement["average_disagreement"]
        result["avg_jaccard"] = disagreement["average_jaccard"]
        
        if mask is not None:
            vote_hist = compute_vote_histogram(scores, mask, gqa_info, config.vote_topk)
            # Aggregate vote histogram across all layers/heads
            total_hist = {}
            for item in vote_hist["vote_histograms"]:
                for vote_count, num_tokens in item["histogram"].items():
                    total_hist[vote_count] = total_hist.get(vote_count, 0) + num_tokens
            result["vote_histogram"] = total_hist
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark VoteKV methods")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--methods", nargs="+", default=["full_cache", "gqa_mean", "gqa_max", "gqa_vote", "gqa_vote_rescue"])
    parser.add_argument("--prompt_file", type=str, help="Path to file with prompt")
    parser.add_argument("--prompt_len", type=int, default=4096, help="Generate synthetic prompt of this length")
    parser.add_argument("--kv_budget_ratio", type=float, default=0.08)
    parser.add_argument("--observation_window", type=int, default=32)
    parser.add_argument("--vote_topk", type=int, default=128)
    parser.add_argument("--rescue_budget", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="outputs/benchmark")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Load model
    config_base = VoteKVConfig(model_name=args.model_name, device=args.device)
    model, tokenizer = load_model_and_tokenizer(
        args.model_name, device=args.device, dtype=config_base.get_dtype()
    )
    gqa_info = get_gqa_info(model)
    
    # Get prompt
    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompt = f.read()
    else:
        # Generate synthetic prompt
        filler = "This is a test sentence. " * 100
        prompt = filler * (args.prompt_len // 500)
        prompt += "\n\nQuestion: Summarize the above text.\nAnswer:"
    
    # Run benchmarks
    config = VoteKVConfig(
        model_name=args.model_name,
        device=args.device,
        kv_budget_ratio=args.kv_budget_ratio,
        observation_window=args.observation_window,
        vote_topk=args.vote_topk,
        rescue_budget=args.rescue_budget,
        max_new_tokens=64,
    )
    
    results = []
    for method in args.methods:
        logger.info(f"\nBenchmarking: {method}")
        result = benchmark_method(model, tokenizer, config, method, gqa_info, prompt)
        results.append(result)
        
        logger.info(f"  Retained: {result['retained_tokens']}/{result['prompt_len']} ({result['compression_ratio']:.2f}x)")
        logger.info(f"  Time: {result['total_sec']:.2f}s")
        if "avg_disagreement" in result:
            logger.info(f"  Disagreement: {result['avg_disagreement']:.3f}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "benchmark_results.json"
    
    with open(output_file, "w") as f:
        json.dump({
            "config": asdict(config),
            "gqa_info": gqa_info,
            "results": results,
        }, f, indent=2)
    
    logger.info(f"\nResults saved to {output_file}")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"{'Method':<25} {'Retained':<12} {'Ratio':<8} {'Time (s)':<10} {'Mem (GB)':<10}")
    logger.info("-"*80)
    
    for r in results:
        logger.info(
            f"{r['method']:<25} {r['retained_tokens']:<12} "
            f"{r['compression_ratio']:<8.2f} {r['total_sec']:<10.2f} "
            f"{r['peak_memory_gb']:<10.2f}"
        )


if __name__ == "__main__":
    main()
