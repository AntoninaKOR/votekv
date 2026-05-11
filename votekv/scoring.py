"""Attention score computation from observation window"""

import logging
import torch
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


def _self_attn_modules(model) -> list:
    """Collect every `self_attn` submodule of a decoder-only HF model in layer order.

    Works for Mistral / Llama / Qwen2 architectures (`model.model.layers[i].self_attn`).
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        # Fallback for some HF architectures (e.g. GPT-style)
        layers = model.transformer.h
    else:
        raise AttributeError(
            "Cannot locate decoder layers on this model. "
            "Expected `model.model.layers` (Llama/Mistral/Qwen2) "
            "or `model.transformer.h`."
        )

    attn_modules = []
    for layer in layers:
        if hasattr(layer, "self_attn"):
            attn_modules.append(layer.self_attn)
        else:
            raise AttributeError(
                "Decoder layer is missing `self_attn`. "
                "Hook-based scoring needs an eager self_attn block."
            )
    return attn_modules


@torch.no_grad()
def compute_snapkv_scores_via_hooks(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    observation_window: int,
    use_cache: bool = True,
) -> Tuple[Any, torch.Tensor]:
    """Memory-efficient SnapKV scoring using per-layer forward hooks.

    Equivalent in math to `compute_snapkv_scores_from_attentions(outputs.attentions, ...)`
    but only holds one layer's attention tensor on the device at a time:

      * Run the forward with `output_attentions=False`. This bypasses HF's
        `output_capturing.py` wrapper (newer transformers ≥ 5.x), which would
        otherwise accumulate references to the dense [B, H, N, N] tensors of
        every layer in an internal state and prevent them from being freed
        between successive calls — quickly leading to OOM on 40 GB GPUs.
      * Eager-attention `eager_attention_forward` STILL computes and returns
        `(attn_output, attn_weights)` to its caller `self_attn.forward` unconditionally;
        the decoder layer simply discards the second element. Our forward hook
        sits on the `self_attn` module and sees the unmodified return tuple,
        reads `attn_weights`, sums over the last `observation_window` query
        positions in float32, stores the per-layer `[num_heads, seq_len]` score
        on CPU/GPU, and RETURNS A MODIFIED OUTPUT TUPLE with `attn_weights`
        replaced by `None`. The dense tensor becomes unreachable as soon as the
        next decoder layer starts.

    Args:
        model: HF causal LM with `attn_implementation="eager"`.
        input_ids: [batch=1, seq_len]
        attention_mask: optional [batch=1, seq_len]
        observation_window: number of trailing queries used as observers (SnapKV).
        use_cache: pass-through to `model.forward(...)`.

    Returns:
        outputs: the raw HF model output (with `past_key_values`, `logits`, ...).
                 `outputs.attentions` will be a tuple of `None`s — do not use it.
        scores: [num_layers, num_query_heads, seq_len] in float32.
    """
    attn_modules = _self_attn_modules(model)
    num_layers = len(attn_modules)

    captured: list = [None] * num_layers
    hooks = []

    def make_hook(layer_idx: int):
        def _hook(module, inputs, output):
            # Eager self_attn returns either:
            #   (attn_output, attn_weights)             — newer HF
            #   (attn_output, attn_weights, past_kv)    — older HF
            # We need element [1] (the attn_weights) when `output_attentions=True`.
            if not isinstance(output, tuple) or len(output) < 2:
                return output
            attn_weights = output[1]
            if attn_weights is None:
                return output

            # attn_weights: [batch, num_heads, q_len, kv_len]
            seq_len = attn_weights.shape[-2]
            obs_start = max(0, seq_len - observation_window)
            # Cast to float32 BEFORE the reduction so summing many bf16 values
            # does not lose precision (matches `compute_snapkv_scores_from_attentions`).
            score = attn_weights[..., obs_start:, :].float().sum(dim=-2)
            captured[layer_idx] = score.detach()

            # CRITICAL: return a modified tuple with attn_weights replaced by None.
            # PyTorch forward hooks may replace the return value; HF then puts
            # `None` into its `all_self_attns` accumulator and the dense weights
            # tensor becomes unreachable, freeing GPU memory immediately.
            return (output[0], None) + tuple(output[2:])

        return _hook

    for idx, mod in enumerate(attn_modules):
        hooks.append(mod.register_forward_hook(make_hook(idx)))

    try:
        # IMPORTANT: pass output_attentions=False so HF's `output_capturing`
        # wrapper does NOT collect dense attention tensors (which leaks across
        # successive forward calls in transformers ≥ 5.x). Eager attention
        # still returns attn_weights via the self_attn tuple — our forward
        # hook captures it from there and immediately replaces it with None.
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=False,
            return_dict=True,
        )
    finally:
        for h in hooks:
            h.remove()

    missing = [i for i, s in enumerate(captured) if s is None]
    if missing:
        raise RuntimeError(
            f"Hook scoring failed: layers {missing} did not produce attention weights. "
            f"Make sure the model was loaded with attn_implementation='eager'."
        )

    # Stack and drop batch dim. Each entry is [batch, num_heads, seq_len].
    scores = torch.stack(captured, dim=0)  # [num_layers, batch, num_heads, seq_len]
    if scores.dim() == 4:
        if scores.shape[1] != 1:
            raise ValueError(
                f"Only batch_size=1 is supported, got batch={scores.shape[1]}"
            )
        scores = scores.squeeze(1)  # [num_layers, num_heads, seq_len]

    logger.debug(
        f"Hook scoring: captured {num_layers} layers, "
        f"score shape={tuple(scores.shape)}, dtype={scores.dtype}, device={scores.device}"
    )
    return outputs, scores


def compute_snapkv_scores_from_attentions(
    attentions: Tuple[torch.Tensor, ...],
    observation_window: int,
) -> torch.Tensor:
    """Compute importance scores from attention in observation window
    
    Following SnapKV methodology: use attention from the last `observation_window`
    tokens during prefill to determine which earlier tokens are important.
    
    Args:
        attentions: Tuple of length num_layers
            Each element shape: [batch, num_query_heads, seq_len, seq_len]
        observation_window: Number of recent tokens to use as observers
        
    Returns:
        scores: [num_layers, num_query_heads, seq_len]
            Importance score for each token as sum of attention from observation window
    """
    num_layers = len(attentions)
    batch_size, num_heads, seq_len, _ = attentions[0].shape
    
    assert batch_size == 1, f"Only batch_size=1 supported for MVP, got {batch_size}"
    
    all_scores = []
    
    for layer_idx in range(num_layers):
        attn = attentions[layer_idx]  # [batch, num_heads, seq_len, seq_len]

        # Observation window: last `observation_window` tokens as queries.
        obs_start = max(0, seq_len - observation_window)
        obs_end = seq_len

        # Sum attention from observation window to all tokens.
        # attn[:, :, obs_start:obs_end, :] -> [batch, num_heads, obs_window, seq_len]
        # Cast to float32 before reduction: attention weights are typically
        # bfloat16 (eager attention with bf16 weights), and accumulating
        # `observation_window` values in bf16 loses enough precision to make
        # topk unstable for closely-ranked tokens.
        scores = attn[:, :, obs_start:obs_end, :].float().sum(dim=2)

        all_scores.append(scores)
    
    # Stack all layers: [num_layers, batch, num_heads, seq_len]
    all_scores = torch.stack(all_scores, dim=0)
    
    # Remove batch dimension (batch=1)
    all_scores = all_scores.squeeze(1)  # [num_layers, num_heads, seq_len]
    
    return all_scores


def get_always_keep_indices(seq_len: int, sink_tokens: int, recent_tokens: int) -> list:
    """Get indices of tokens that must always be kept
    
    Args:
        seq_len: Total sequence length
        sink_tokens: Number of initial tokens to always keep
        recent_tokens: Number of final tokens to always keep
        
    Returns:
        List of sorted unique indices to always keep
    """
    # Sink tokens: first few tokens
    sink = list(range(min(sink_tokens, seq_len)))
    
    # Recent tokens: last few tokens
    recent_start = max(0, seq_len - recent_tokens)
    recent = list(range(recent_start, seq_len))
    
    # Combine and deduplicate
    return sorted(set(sink + recent))
