"""Analysis script for GQA disagreement"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import argparse
import logging
import json
from pathlib import Path

from votekv.config import VoteKVConfig
from votekv.model_utils import load_model_and_tokenizer
from votekv.gqa_utils import get_gqa_info
from votekv.scoring import compute_snapkv_scores_via_hooks
from votekv.selectors import select_tokens
from votekv.metrics import compute_gqa_disagreement, compute_vote_histogram
from votekv.logging_utils import setup_logging

logger = logging.getLogger(__name__)


@torch.no_grad()
def analyze_disagreement(
    model,
    tokenizer,
    config: VoteKVConfig,
    prompt: str,
    output_dir: Path,
):
    """Analyze GQA disagreement for a prompt
    
    Args:
        model: Model
        tokenizer: Tokenizer
        config: Configuration
        prompt: Input prompt
        output_dir: Output directory
    """
    device = config.device
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=config.max_context_len).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    prompt_len = input_ids.shape[1]
    logger.info(f"Prompt length: {prompt_len} tokens")
    
    # Compute scores via hook-based scoring (one layer of attentions live at a time).
    _, scores = compute_snapkv_scores_via_hooks(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        observation_window=config.observation_window,
        use_cache=True,
    )

    # Get GQA info
    gqa_info = get_gqa_info(model)
    
    # Compute disagreement
    disagreement = compute_gqa_disagreement(scores, gqa_info, config.vote_topk)
    
    logger.info(f"Average disagreement: {disagreement['average_disagreement']:.3f}")
    logger.info(f"Average Jaccard: {disagreement['average_jaccard']:.3f}")
    
    # Save full results
    output_file = output_dir / "disagreement_analysis.json"
    with open(output_file, "w") as f:
        json.dump(disagreement, f, indent=2)
    
    logger.info(f"Saved to {output_file}")
    
    # Compute vote histogram for voting methods
    for method in ["gqa_vote", "gqa_vote_rescue"]:
        logger.info(f"\nAnalyzing {method}...")
        mask = select_tokens(scores, method, config, gqa_info)
        
        vote_hist = compute_vote_histogram(scores, mask, gqa_info, config.vote_topk)
        
        # Save
        output_file = output_dir / f"vote_histogram_{method}.json"
        with open(output_file, "w") as f:
            json.dump(vote_hist, f, indent=2)
        
        logger.info(f"Saved to {output_file}")
        
        # Print summary
        total_hist = {}
        for item in vote_hist["vote_histograms"]:
            for vote_count, num_tokens in item["histogram"].items():
                total_hist[vote_count] = total_hist.get(vote_count, 0) + num_tokens
        
        logger.info(f"Vote distribution for {method}:")
        for vote_count in sorted(total_hist.keys()):
            logger.info(f"  {vote_count} votes: {total_hist[vote_count]} tokens")


def main():
    parser = argparse.ArgumentParser(description="Analyze GQA disagreement")
    parser.add_argument(
        "--config", type=str, default=None,
        help="YAML config with model + VoteKV parameters. CLI flags override YAML values.",
    )
    parser.add_argument("--model_name", type=str, default=None,
                        help="Override model from YAML / default Mistral-7B-Instruct-v0.2")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--vote_topk", type=int, default=None)
    parser.add_argument("--observation_window", type=int, default=None)

    # Script-only flags.
    parser.add_argument("--prompt_file", type=str, help="Path to prompt file")
    parser.add_argument("--prompt_len", type=int, default=4096)
    parser.add_argument("--output_dir", type=str, default="outputs/analysis")

    args = parser.parse_args()

    setup_logging()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Defaults <- YAML <- CLI overrides.
    config = VoteKVConfig.from_args(args, yaml_path=args.config)

    model, tokenizer = load_model_and_tokenizer(
        config.model_name, device=config.device, dtype=config.get_dtype()
    )
    
    # Get prompt
    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompt = f.read()
    else:
        filler = "This is sample text for analysis. " * 50
        prompt = filler * (args.prompt_len // 200)
    
    # Analyze
    analyze_disagreement(model, tokenizer, config, prompt, output_dir)


if __name__ == "__main__":
    main()
