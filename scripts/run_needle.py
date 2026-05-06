"""Needle-in-haystack evaluation for VoteKV

Evaluates ability to retrieve information at different depths in long contexts.
"""

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
from typing import List, Dict

from votekv.config import VoteKVConfig
from votekv.model_utils import load_model_and_tokenizer
from votekv.gqa_utils import get_gqa_info
from votekv.scoring import compute_snapkv_scores_from_attentions
from votekv.selectors import select_tokens
from votekv.cache_compression import (
    convert_kv_head_mask_to_layer_indices,
    compress_past_key_values_layer_shared,
)
from votekv.generation import generate_with_compressed_cache
from votekv.metrics import exact_match, contains_answer
from votekv.logging_utils import setup_logging, reset_memory_stats

logger = logging.getLogger(__name__)


def create_needle_document(
    needle: str,
    depth: float,
    target_token_len: int,
    tokenizer,
) -> str:
    """Create a document with needle inserted at specified depth
    
    Args:
        needle: The needle sentence to insert
        depth: Depth ratio (0.0 = start, 1.0 = end)
        target_token_len: Target document length in tokens
        tokenizer: Tokenizer for length estimation
        
    Returns:
        Document string with needle inserted
    """
    filler_sentence = (
        "The quick brown fox jumps over the lazy dog. "
        "This is filler text used to create a long context for testing. "
        "We need to make the document sufficiently long to test retrieval capabilities. "
    )
    
    # Estimate tokens per filler repetition
    filler_tokens = len(tokenizer(filler_sentence)["input_ids"])
    needed_reps = target_token_len // filler_tokens
    
    # Create filler text
    filler_text = filler_sentence * needed_reps
    
    # Insert needle at specified depth
    insertion_point = int(len(filler_text) * depth)
    
    document = (
        filler_text[:insertion_point] +
        f"\n{needle}\n" +
        filler_text[insertion_point:]
    )
    
    return document


def create_needle_prompt(
    passkey: str,
    depth: float,
    context_len: int,
    tokenizer,
) -> str:
    """Create needle-in-haystack prompt
    
    Args:
        passkey: The passkey to hide
        depth: Depth ratio (0.0-1.0)
        context_len: Target context length in tokens
        tokenizer: Tokenizer
        
    Returns:
        Full prompt with question
    """
    needle = f"The special passkey is: {passkey}."
    
    document = create_needle_document(
        needle=needle,
        depth=depth,
        target_token_len=context_len - 200,  # Reserve tokens for instructions
        tokenizer=tokenizer,
    )
    
    prompt = (
        "You are given a long document. Read it carefully and answer the question.\n\n"
        f"Document:\n{document}\n\n"
        "Question: What is the special passkey mentioned in the document? "
        "Answer only the passkey, nothing else.\n"
        "Answer:"
    )
    
    return prompt


@torch.no_grad()
def evaluate_needle(
    model,
    tokenizer,
    config: VoteKVConfig,
    method: str,
    gqa_info: Dict,
    passkey: str,
    depth: float,
    context_len: int,
) -> Dict:
    """Evaluate single needle configuration
    
    Args:
        model: Model
        tokenizer: Tokenizer
        config: VoteKV config
        method: Selection method
        gqa_info: GQA information
        passkey: Passkey to find
        depth: Needle depth (0.0-1.0)
        context_len: Context length
        
    Returns:
        Results dictionary
    """
    device = config.device
    
    # Create prompt
    prompt = create_needle_prompt(passkey, depth, context_len, tokenizer)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=context_len).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    actual_prompt_len = input_ids.shape[1]
    
    reset_memory_stats(device)
    start_time = time.perf_counter()
    
    # Prefill
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
    
    # Compression
    if method == "full_cache":
        compressed_cache = past_key_values
        retained_count = actual_prompt_len
        compression_time = 0.0
    else:
        compression_start = time.perf_counter()
        
        attentions = outputs.attentions
        scores = compute_snapkv_scores_from_attentions(
            attentions, config.observation_window
        )
        
        mask = select_tokens(scores, method, config, gqa_info)
        
        budget = config.resolve_budget(actual_prompt_len)
        selected_indices = convert_kv_head_mask_to_layer_indices(
            mask, scores, budget
        )
        
        compressed_cache = compress_past_key_values_layer_shared(
            past_key_values, selected_indices
        )
        
        retained_count = selected_indices[0].shape[0]
        compression_time = time.perf_counter() - compression_start
        
        del attentions, scores, mask
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
        max_new_tokens=32,  # Short answer
        original_seq_len=actual_prompt_len,
    )
    
    decode_time = time.perf_counter() - decode_start
    total_time = time.perf_counter() - start_time
    
    # Decode
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    
    # Evaluate
    correct_exact = exact_match(output_text, passkey)
    correct_contains = contains_answer(output_text, passkey)
    
    logger.info(
        f"[{method}] depth={depth:.2f}, len={actual_prompt_len}: "
        f"output='{output_text}', correct={correct_contains}"
    )
    
    return {
        "method": method,
        "context_len": context_len,
        "actual_prompt_len": actual_prompt_len,
        "depth": depth,
        "passkey": passkey,
        "output": output_text,
        "correct_exact": correct_exact,
        "correct_contains": correct_contains,
        "retained_tokens": retained_count,
        "compression_ratio": actual_prompt_len / retained_count if retained_count > 0 else 1.0,
        "prefill_sec": prefill_time,
        "compression_sec": compression_time,
        "decode_sec": decode_time,
        "total_sec": total_time,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
    }


def main():
    parser = argparse.ArgumentParser(description="VoteKV Needle-in-Haystack Evaluation")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--methods", nargs="+", default=["full_cache", "gqa_mean", "gqa_max", "gqa_vote", "gqa_vote_rescue"])
    parser.add_argument("--context_lengths", nargs="+", type=int, default=[4096, 8192])
    parser.add_argument("--depths", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--budget_ratios", nargs="+", type=float, default=[0.04, 0.08, 0.16])
    parser.add_argument("--observation_window", type=int, default=32)
    parser.add_argument("--vote_topk", type=int, default=128)
    parser.add_argument("--rescue_budget", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="outputs/needle")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--passkey", type=str, default="493827")
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model once
    logger.info(f"Loading model: {args.model_name}")
    config_base = VoteKVConfig(model_name=args.model_name, device=args.device)
    model, tokenizer = load_model_and_tokenizer(
        args.model_name, device=args.device, dtype=config_base.get_dtype()
    )
    gqa_info = get_gqa_info(model)
    logger.info(f"GQA Info: {gqa_info}")
    
    all_results = []
    
    # Run evaluations
    for context_len in args.context_lengths:
        for depth in args.depths:
            for budget_ratio in args.budget_ratios:
                for method in args.methods:
                    logger.info(
                        f"\nEvaluating: len={context_len}, depth={depth}, "
                        f"budget={budget_ratio}, method={method}"
                    )
                    
                    config = VoteKVConfig(
                        model_name=args.model_name,
                        device=args.device,
                        kv_budget_ratio=budget_ratio,
                        observation_window=args.observation_window,
                        vote_topk=args.vote_topk,
                        rescue_budget=args.rescue_budget,
                    )
                    
                    try:
                        result = evaluate_needle(
                            model=model,
                            tokenizer=tokenizer,
                            config=config,
                            method=method,
                            gqa_info=gqa_info,
                            passkey=args.passkey,
                            depth=depth,
                            context_len=context_len,
                        )
                        result["budget_ratio"] = budget_ratio
                        all_results.append(result)
                        
                    except Exception as e:
                        logger.error(f"Error: {e}", exc_info=True)
    
    # Save results
    output_file = output_dir / "needle_results.jsonl"
    with open(output_file, "w") as f:
        for result in all_results:
            f.write(json.dumps(result) + "\n")
    
    logger.info(f"\nResults saved to {output_file}")
    
    # Summary statistics
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    
    for method in args.methods:
        method_results = [r for r in all_results if r["method"] == method]
        if method_results:
            accuracy = sum(r["correct_contains"] for r in method_results) / len(method_results)
            avg_time = sum(r["total_sec"] for r in method_results) / len(method_results)
            logger.info(f"{method}: accuracy={accuracy:.2%}, avg_time={avg_time:.2f}s")


if __name__ == "__main__":
    main()
