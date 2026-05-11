"""Model loading and utility functions"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_name: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load HuggingFace model and tokenizer
    
    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on
        dtype: Torch dtype for model weights
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {model_name}")
    logger.info(f"Device: {device}, dtype: {dtype}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device if device != "cpu" else None,
        low_cpu_mem_usage=True,
        attn_implementation="eager",  # Required for output_attentions=True
    )
    
    model.eval()
    
    logger.info(f"Model loaded successfully")
    logger.info(f"Model config: {model.config}")
    
    return model, tokenizer
