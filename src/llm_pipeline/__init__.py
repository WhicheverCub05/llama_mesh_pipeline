"""LLM Pipeline Package - A Python-based batch LLM processing pipeline."""

__version__ = "0.3.0"

from .core.llm_client import (
    LLMClient,
    OllamaClient,
    VLLMClient,
    create_llm_client,
    GenerationResult
)
from .core.output import OutputWriter, CSVWriter
from .core.pipeline import run_pipeline, load_config, validate_config, generate_pipeline_diagram
from .utils.wrap_prompts import input_list_to_promptlist, get_prompt_input_list_from_txt_file
from .utils.cache import get_cache, clear_cache

__all__ = [
    "LLMClient",
    "OllamaClient",
    "VLLMClient",
    "create_llm_client",
    "GenerationResult",
    "OutputWriter",
    "CSVWriter",
    "run_pipeline",
    "load_config",
    "validate_config",
    "generate_pipeline_diagram",
    "input_list_to_promptlist",
    "get_prompt_input_list_from_txt_file",
    "get_cache",
    "clear_cache",
]