"""Sanity check script for VoteKV.

Tests basic functionality with a simple prompt containing a passkey.
Loads the model once and benchmarks every selection method against it.
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
from votekv.generation import generate_with_compressed_cache
from votekv.logging_utils import setup_logging, reset_memory_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Banner helpers — keep stdout readable across many methods.
# ---------------------------------------------------------------------------

def _banner(title: str, char: str = "=", width: int = 80) -> None:
    bar = char * width
    logger.info(bar)
    logger.info(f"  {title}")
    logger.info(bar)


def create_sanity_prompt(target_len: int = 1024, passkey: str = "493827") -> str:
    """Synthetic prompt with a hidden passkey inserted twice (early + midway)."""
    filler = (
        "The quick brown fox jumps over the lazy dog. "
        "This is a sample text used for testing purposes. "
        "We repeat this text multiple times to create a long context. "
    )
    tokens_per_rep = 20
    needed_reps = target_len // tokens_per_rep

    prompt = f"There is a secret passkey hidden in the text: {passkey}. "
    prompt += "Remember this passkey. " * 3
    prompt += filler * (needed_reps // 2)
    prompt += f"\n\nThe passkey mentioned earlier was: {passkey}.\n\n"
    prompt += filler * (needed_reps // 2)
    prompt += f"\n\nQuestion: What is the secret passkey mentioned in this text?\nAnswer:"
    return prompt


@torch.no_grad()
def run_sanity_test(
    model,
    tokenizer,
    gqa_info: dict,
    config: VoteKVConfig,
    method: str,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    passkey: str,
) -> dict:
    """Run a single method on the already-tokenised prompt."""
    _banner(f"METHOD: {method}")

    device = config.device
    prompt_len = input_ids.shape[1]

    reset_memory_stats(device)
    start_time = time.perf_counter()

    # ----- Prefill -----
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

    # ----- Compression -----
    if method == "full_cache":
        compressed_cache = past_key_values
        retained_count = prompt_len
        compression_time = 0.0
        budget = prompt_len
    else:
        compression_start = time.perf_counter()

        attentions = outputs.attentions
        scores = compute_snapkv_scores_from_attentions(
            attentions, config.observation_window
        )
        mask = select_tokens(scores, method, config, gqa_info)
        budget = config.resolve_budget(prompt_len)
        selected_indices = convert_kv_head_mask_to_layer_indices(
            mask, scores, budget
        )
        compressed_cache = compress_past_key_values_layer_shared(
            past_key_values, selected_indices
        )
        retained_count = selected_indices[0].shape[0]
        compression_time = time.perf_counter() - compression_start

        logger.info(
            f"Selected (layer 0, first 10): "
            f"{selected_indices[0][:10].tolist()}"
        )

        del attentions, scores, mask
        torch.cuda.empty_cache()

    compression_ratio = prompt_len / retained_count if retained_count > 0 else 1.0
    logger.info(
        f"Cache: {prompt_len} -> {retained_count} tokens "
        f"(budget={budget}, ratio={compression_ratio:.2f}x)"
    )

    # ----- Generation -----
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

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    num_generated_tokens = generated_ids.shape[1]
    tokens_per_sec = num_generated_tokens / decode_time if decode_time > 0 else 0.0
    contains_passkey = passkey in generated_text

    peak_gb = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
        if torch.cuda.is_available() and device != "cpu"
        else 0.0
    )

    logger.info(f"Generated: {generated_text.strip()[:200]}")
    logger.info(
        f"Passkey {passkey}: {'FOUND' if contains_passkey else 'NOT FOUND'}"
    )
    logger.info(
        f"Timing: prefill={prefill_time:.2f}s | "
        f"compress={compression_time:.2f}s | "
        f"decode={decode_time:.2f}s ({num_generated_tokens} tok, {tokens_per_sec:.1f} tok/s) | "
        f"total={total_time:.2f}s"
    )
    logger.info(f"GPU peak: {peak_gb:.2f} GB")

    # Free compressed cache so the next method starts clean.
    del past_key_values, compressed_cache, outputs, logits, generated_ids
    torch.cuda.empty_cache()

    return {
        "method": method,
        "prompt_len": prompt_len,
        "retained_tokens": retained_count,
        "compression_ratio": compression_ratio,
        "contains_passkey": contains_passkey,
        "num_generated_tokens": num_generated_tokens,
        "tokens_per_sec": tokens_per_sec,
        "prefill_time": prefill_time,
        "compression_time": compression_time,
        "decode_time": decode_time,
        "total_time": total_time,
        "peak_memory_gb": peak_gb,
    }


def main():
    parser = argparse.ArgumentParser(description="VoteKV Sanity Test")
    parser.add_argument(
        "--config", type=str, default=None,
        help="YAML config file with model + VoteKV parameters "
             "(e.g. configs/mistral_7b_votekv.yaml). CLI flags override YAML values.",
    )
    # All VoteKV-related flags default to None so that we can detect whether
    # the user explicitly passed them and let YAML / class defaults fill the gap.
    parser.add_argument("--model_name", type=str, default=None,
                        help="Override model from YAML / default Mistral-7B-Instruct-v0.2")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--kv_budget_ratio", type=float, default=None)
    parser.add_argument("--observation_window", type=int, default=None)
    parser.add_argument("--vote_topk", type=int, default=None)
    parser.add_argument("--rescue_budget", type=int, default=None)

    # Script-only flags (not part of VoteKVConfig).
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["full_cache", "gqa_mean", "gqa_max", "gqa_vote", "gqa_vote_rescue"],
    )
    parser.add_argument("--target_len", type=int, default=1024, help="Target prompt length")
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG logs everywhere (per-layer compression, model config, internals).",
    )
    parser.add_argument(
        "--show-layers", action="store_true",
        help="Enable per-layer compression DEBUG only (one line per layer per method), "
             "without other DEBUG noise.",
    )

    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    if args.show_layers and not args.verbose:
        # Promote only the cache-compression logger to DEBUG so the user sees
        # the per-layer union/budget messages without the rest of the chatter.
        logging.getLogger("votekv.cache_compression").setLevel(logging.DEBUG)

    # Defaults <- YAML <- CLI.
    config = VoteKVConfig.from_args(args, yaml_path=args.config)

    # ----- Load model and tokenizer once -----
    _banner("SETUP")
    model, tokenizer = load_model_and_tokenizer(
        config.model_name, device=config.device, dtype=config.get_dtype()
    )
    gqa_info = get_gqa_info(model)

    passkey = "493827"
    prompt = create_sanity_prompt(target_len=args.target_len, passkey=passkey)
    inputs = tokenizer(prompt, return_tensors="pt").to(config.device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    prompt_len = input_ids.shape[1]
    budget = config.resolve_budget(prompt_len)

    logger.info(
        f"Model: {config.model_name} | "
        f"layers={model.config.num_hidden_layers} "
        f"Q-heads={gqa_info['num_attention_heads']} "
        f"KV-heads={gqa_info['num_key_value_heads']} "
        f"group_size={gqa_info['group_size']}"
    )
    logger.info(
        f"Prompt: {prompt_len} tokens | "
        f"budget={budget} ({config.kv_budget_ratio*100:.1f}%) | "
        f"sink={config.sink_tokens} recent={config.recent_tokens} "
        f"obs_window={config.observation_window} vote_topk={config.vote_topk} "
        f"rescue_budget={config.rescue_budget}"
    )
    logger.info(f"Passkey to retrieve: {passkey}")
    logger.info(f"Methods to run: {', '.join(args.methods)}")

    # ----- Run methods -----
    results = []
    for method in args.methods:
        try:
            result = run_sanity_test(
                model=model,
                tokenizer=tokenizer,
                gqa_info=gqa_info,
                config=config,
                method=method,
                input_ids=input_ids,
                attention_mask=attention_mask,
                passkey=passkey,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Error running {method}: {e}", exc_info=True)

    # ----- Final summary table -----
    _banner("SUMMARY")
    header = (
        f"{'Method':<18} {'Retained':>9} {'Ratio':>7} {'Passkey':>8} "
        f"{'Decode (s)':>11} {'Tok/s':>7} {'Peak GB':>8}"
    )
    logger.info(header)
    logger.info("-" * len(header))
    for r in results:
        logger.info(
            f"{r['method']:<18} {r['retained_tokens']:>9} "
            f"{r['compression_ratio']:>6.2f}x "
            f"{('YES' if r['contains_passkey'] else 'NO'):>8} "
            f"{r['decode_time']:>11.2f} "
            f"{r['tokens_per_sec']:>7.1f} "
            f"{r['peak_memory_gb']:>8.2f}"
        )


if __name__ == "__main__":
    main()
