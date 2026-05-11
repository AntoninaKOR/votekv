"""Custom generation with compressed KV cache"""

import torch
from typing import Tuple, Optional, Dict
import inspect
import logging

logger = logging.getLogger(__name__)


@torch.no_grad()
def generate_with_compressed_cache(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values: Optional[Tuple] = None,
    max_new_tokens: int = 64,
    original_seq_len: Optional[int] = None,
    use_cache_position: bool = True,
) -> Tuple[torch.Tensor, Dict]:
    """Generate tokens with compressed KV cache
    
    Args:
        model: HuggingFace CausalLM
        tokenizer: Tokenizer
        input_ids: [batch, seq_len] or [batch, 1] if continuing from cache
        attention_mask: Attention mask
        past_key_values: Optional compressed cache from prefill
        max_new_tokens: Number of tokens to generate
        original_seq_len: Original prompt length (for position IDs)
        use_cache_position: Whether to use cache_position parameter
        
    Returns:
        Tuple of (generated_ids, stats)
    """
    device = input_ids.device
    batch_size = input_ids.shape[0]
    
    # Check model forward signature
    forward_signature = inspect.signature(model.forward)
    supports_position_ids = "position_ids" in forward_signature.parameters
    supports_cache_position = "cache_position" in forward_signature.parameters
    
    logger.info(
        f"Model forward supports: position_ids={supports_position_ids}, "
        f"cache_position={supports_cache_position}"
    )
    
    # If starting fresh (no cache), run prefill
    if past_key_values is None:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        logits = outputs.logits
        past_key_values = outputs.past_key_values
        original_seq_len = input_ids.shape[1]
        # First predicted token comes from prefill argmax over the last position.
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    else:
        # Caller already ran prefill (typically with compression) and passes the
        # first predicted token via input_ids.
        next_token = input_ids[:, -1:].clone()

    # Include the first predicted token so the caller does not have to.
    generated = [next_token]

    if next_token.item() == tokenizer.eos_token_id:
        return next_token, {"num_generated": 1}

    for step in range(max_new_tokens - 1):
        # Logical position of the *current* token being fed into the model.
        # generated already holds (step + 1) tokens: the first one was the prefill
        # argmax (logical position = original_seq_len), so this step's token sits
        # at original_seq_len + step.
        current_position = original_seq_len + step

        forward_kwargs = {
            "input_ids": next_token,
            "past_key_values": past_key_values,
            "use_cache": True,
            "return_dict": True,
        }

        if supports_position_ids:
            forward_kwargs["position_ids"] = torch.tensor(
                [[current_position]], device=device, dtype=torch.long
            )

        if supports_cache_position and use_cache_position:
            forward_kwargs["cache_position"] = torch.tensor(
                [current_position], device=device, dtype=torch.long
            )

        # cache_len already reflects any growth from previous decode steps; we
        # only add 1 for the token we are feeding now.
        cache_len = past_key_values.layers[0].keys.shape[2]
        forward_kwargs["attention_mask"] = torch.ones(
            batch_size, cache_len + 1, device=device, dtype=torch.long
        )

        outputs = model(**forward_kwargs)

        logits = outputs.logits
        past_key_values = outputs.past_key_values

        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token)

        if next_token.item() == tokenizer.eos_token_id:
            break

    generated_ids = torch.cat(generated, dim=1)

    stats = {
        "num_generated": generated_ids.shape[1],
    }

    return generated_ids, stats


@torch.no_grad()
def simple_greedy_generate(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 64,
) -> torch.Tensor:
    """Simple greedy generation without cache compression
    
    Args:
        model: HuggingFace CausalLM
        tokenizer: Tokenizer
        input_ids: [batch, seq_len]
        max_new_tokens: Number of tokens to generate
        
    Returns:
        generated_ids: [batch, num_generated]
    """
    device = input_ids.device
    batch_size = input_ids.shape[0]
    
    # Initial forward pass
    outputs = model(input_ids=input_ids, use_cache=True, return_dict=True)
    logits = outputs.logits
    past_key_values = outputs.past_key_values
    
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token]
    
    for _ in range(max_new_tokens - 1):
        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        
        logits = outputs.logits
        past_key_values = outputs.past_key_values
        
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    return torch.cat(generated, dim=1)
