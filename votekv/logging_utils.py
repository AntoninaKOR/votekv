"""Logging utilities for VoteKV"""

import logging
import sys
import time
import torch
from typing import Dict, Any
from contextlib import contextmanager


def setup_logging(level=logging.INFO):
    """Setup basic logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


@contextmanager
def timer(name: str):
    """Context manager for timing code blocks"""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    elapsed = end - start
    logging.info(f"{name}: {elapsed:.4f} seconds")


def log_memory_stats(device: str = "cuda"):
    """Log GPU memory statistics"""
    if device == "cuda" and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
        
        logging.info(f"GPU Memory - Allocated: {allocated:.2f} GB")
        logging.info(f"GPU Memory - Reserved: {reserved:.2f} GB")
        logging.info(f"GPU Memory - Peak: {max_allocated:.2f} GB")
    else:
        logging.info("CUDA not available")


def get_memory_stats(device: str = "cuda") -> Dict[str, float]:
    """Get GPU memory statistics as dictionary"""
    if device == "cuda" and torch.cuda.is_available():
        return {
            "allocated_gb": torch.cuda.memory_allocated() / (1024 ** 3),
            "reserved_gb": torch.cuda.memory_reserved() / (1024 ** 3),
            "peak_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
        }
    return {}


def reset_memory_stats(device: str = "cuda"):
    """Reset peak memory statistics"""
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def log_model_info(model, tokenizer, config_dict: Dict[str, Any]):
    """Log model and configuration information"""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("MODEL INFORMATION")
    logger.info("=" * 80)
    logger.info(f"Model: {model.config.name_or_path}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    logger.info(f"Layers: {model.config.num_hidden_layers}")
    logger.info(f"Attention Heads: {model.config.num_attention_heads}")
    logger.info(f"KV Heads: {model.config.num_key_value_heads}")
    logger.info(f"Hidden Size: {model.config.hidden_size}")
    logger.info(f"Vocab Size: {model.config.vocab_size}")
    
    logger.info("\n" + "=" * 80)
    logger.info("VOTEKV CONFIGURATION")
    logger.info("=" * 80)
    for key, value in config_dict.items():
        logger.info(f"{key}: {value}")
    logger.info("=" * 80)
