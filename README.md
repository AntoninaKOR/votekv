# VoteKV: GQA-aware SnapKV with Voting and Rescue Tokens

Research MVP implementation of VoteKV, an inference-time KV-cache compression method for GQA (Grouped-Query Attention) models.

## Overview

VoteKV addresses a key limitation of existing GQA-aware KV eviction methods: query heads sharing the same KV head may disagree on which tokens are important. Instead of using simple mean or max aggregation, VoteKV uses:

1. **Voting mechanism**: Each query head votes for its top-k important tokens
2. **Rescue tokens**: Head-specific tokens to preserve specialized retrieval signals

**Implementation Note**: This MVP uses layer-shared compression (union of selected tokens within each layer) for PyTorch compatibility. Selection is still per-KV-head, enabling GQA-aware voting.

**SnapKV Integration**: All selection methods use SnapKV's observation window methodology for computing token importance scores (`compute_snapkv_scores_from_attentions`). The difference is in how scores are aggregated within GQA groups: mean (Ada-KV style), max (R-KV style), or voting (VoteKV).

## Project Structure

```
votekv/
├── votekv/
│   ├── __init__.py
│   ├── config.py                  # Configuration dataclass
│   ├── model_utils.py             # Model loading
│   ├── gqa_utils.py               # GQA grouping utilities
│   ├── scoring.py                 # Attention score computation
│   ├── selectors.py               # Token selection methods
│   ├── cache_compression.py       # KV-cache compression
│   ├── generation.py              # Custom generation loop
│   ├── metrics.py                 # Evaluation metrics
│   └── logging_utils.py           # Logging utilities
├── scripts/
│   ├── run_sanity.py              # Sanity check
│   ├── run_needle.py              # Needle-in-haystack evaluation
│   ├── benchmark_methods.py       # Benchmark all methods
│   └── analyze_disagreement.py    # GQA disagreement analysis
├── configs/
│   ├── mistral_7b_votekv.yaml     # Mistral-7B-Instruct-v0.2 (group_size=4)
│   ├── llama_3.1_8b_votekv.yaml   # Llama-3.1-8B-Instruct   (group_size=4)
│   └── qwen2.5_7b_votekv.yaml     # Qwen2.5-7B-Instruct     (group_size=7)
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Supported Methods

1. **full_cache**: No compression (upper bound)
2. **gqa_mean**: Mean aggregation across query heads (Ada-KV style, uniform budget)
3. **gqa_max**: Max aggregation across query heads (R-KV style)
4. **gqa_vote**: Voting-based selection (VoteKV)
5. **gqa_vote_rescue**: Voting + rescue tokens (VoteKV+Rescue, main method)
6. **gqa_rank_vote**: Rank-weighted voting (Borda-style)

## Quick Start

### Choosing a model

Every script accepts a `--config configs/<model>.yaml` flag

```bash
# Mistral-7B (default)
python scripts/run_sanity.py --config configs/mistral_7b_votekv.yaml

# Llama-3.1-8B
python scripts/run_sanity.py --config configs/llama_3.1_8b_votekv.yaml

# Qwen2.5-7B (different group_size=7, uses rescue_budget=3)
python scripts/run_sanity.py --config configs/qwen2.5_7b_votekv.yaml

# Override a single field from a base config
python scripts/run_sanity.py \
  --config configs/llama_3.1_8b_votekv.yaml \
  --kv_budget_ratio 0.04
```

### Sanity Check

```bash
python scripts/run_sanity.py \
  --config configs/mistral_7b_votekv.yaml \
  --methods full_cache gqa_mean gqa_max gqa_vote gqa_vote_rescue \
  --target_len 1024
```

### Needle-in-Haystack Evaluation

```bash
python scripts/run_needle.py \
  --config configs/mistral_7b_votekv.yaml \
  --methods full_cache gqa_mean gqa_max gqa_vote gqa_vote_rescue \
  --context_lengths 4096 8192 \
  --depths 0.0 0.25 0.5 0.75 1.0 \
  --budget_ratios 0.04 0.08 0.16 \
  --output_dir outputs/needle_mistral
```

### Benchmark Methods

```bash
python scripts/benchmark_methods.py \
  --config configs/mistral_7b_votekv.yaml \
  --methods full_cache gqa_mean gqa_max gqa_vote gqa_vote_rescue \
  --prompt_len 4096 \
  --output_dir outputs/benchmark
```

### Disagreement Analysis

```bash
python scripts/analyze_disagreement.py \
  --config configs/mistral_7b_votekv.yaml \
  --prompt_len 4096 \
  --output_dir outputs/analysis
```

### Visualize Results

After running experiments, create comprehensive visualizations:

```bash
# Generate all plots
python scripts/visualize_results.py \
  --needle_results outputs/needle_mistral/needle_results.jsonl \
  --disagreement_results outputs/analysis/disagreement_analysis.json \
  --vote_histogram outputs/analysis/vote_histogram_gqa_vote_rescue.json \
  --plots all \
  --output_dir outputs/plots

# Create publication-ready figures
python scripts/create_paper_figures.py \
  --needle_results outputs/needle_mistral/needle_results.jsonl \
  --disagreement_results outputs/analysis/disagreement_analysis.json \
  --output_dir outputs/paper_figures
```

## Configuration

Key parameters (see `configs/mistral_7b_votekv.yaml`):

- `kv_budget_ratio`: Fraction of tokens to keep (default: 0.08)
- `observation_window`: Recent tokens to use for importance scoring (default: 32)
- `vote_topk`: Top-k tokens per query head for voting (default: 128)
- `rescue_budget`: Rescue tokens per query head (default: 4)
- `sink_tokens`: Initial tokens to always keep (default: 4)
- `recent_tokens`: Recent tokens to always keep (default: 32)

## Key Baselines

- **Full KV**: No compression (quality upper bound)
- **GQA-Mean**: Ada-KV style aggregation
- **GQA-Max**: R-KV style aggregation
- **VoteKV**: Our voting mechanism
- **VoteKV+Rescue**: Voting + rescue tokens

## Supported Models

Selected via `--config configs/<model>.yaml`:

| Model | Config | Q-heads | KV-heads | group_size | Notes |
|-------|--------|---------|----------|------------|-------|
| `mistralai/Mistral-7B-Instruct-v0.2` | `configs/mistral_7b_votekv.yaml` | 32 | 8 | 4 | Default |
| `meta-llama/Llama-3.1-8B-Instruct` | `configs/llama_3.1_8b_votekv.yaml` | 32 | 8 | 4 | Native 128k context, capped at 16k for MVP |
| `Qwen/Qwen2.5-7B-Instruct` | `configs/qwen2.5_7b_votekv.yaml` | 28 | 4 | 7 | Larger group_size → `rescue_budget=3` |

To add a new GQA model: copy one of the YAML files, change `model_name`, and (if `group_size` differs from 4) adjust `rescue_budget` so that `group_size * rescue_budget` stays in a similar range (~16-21 max-rescue per KV-head).

## Limitations (MVP)

1. **Layer-shared compression**: Selection is per-KV-head but compression uses layer-wise union of selected tokens (dense tensor requirement)
2. **Batch size = 1**: Only single-item batches supported
3. **Generation**: Custom generation loop may have compatibility issues with some HF model versions
4. **Memory**: Full attention tensors with `output_attentions=True` may OOM on very long contexts

## TODO

- [ ] True per-KV-head ragged/padded cache support
- [ ] Batch processing
- [ ] LongBench full evaluation

