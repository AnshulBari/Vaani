"""
Small LLM Project - A complete implementation of a small Large Language Model

This package provides a ready-to-use implementation of a transformer-based
language model with approximately 4-5 GB size and ~4 billion parameters.

Modules:
    model: Contains the SmallLLM architecture and related components
    tokenizer: Simple tokenizer for text processing
    train: Training pipeline with checkpointing and monitoring
    inference: Interactive chat interface for model testing

Example usage:
    # Training
    python train.py
    
    # Inference
    python inference.py

For more details, see README.md
"""

__version__ = "0.1.0"
__author__ = "Small LLM Project"

from .model import SmallLLM
from .tokenizer import SimpleTokenizer

__all__ = ['SmallLLM', 'SimpleTokenizer']
