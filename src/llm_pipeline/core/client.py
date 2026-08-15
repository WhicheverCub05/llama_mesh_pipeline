#!/usr/bin/env python3
"""
Ollama API client for making LLM requests.

This module is deprecated. Please use llm_client.py instead for backend-agnostic LLM access.
"""

# Re-export for backward compatibility
from .llm_client import (
    OllamaClient,
    GenerationResult,
    save_checkpoint,
    load_checkpoint,
    clear_checkpoint,
    retry_on_failure,
)

__all__ = [
    "OllamaClient",
    "GenerationResult",
    "save_checkpoint",
    "load_checkpoint",
    "clear_checkpoint",
    "retry_on_failure",
]