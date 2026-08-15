#!/usr/bin/env python3
"""
N-stage LLM pipeline with JSON configuration support.
"""

import os
import argparse
import json
import csv
import logging
from pathlib import Path
from typing import Any
from dataclasses import dataclass

# Import existing components
from ..utils.wrap_prompts import (
    get_prompt_input_list_from_txt_file,
    input_list_to_promptlist,
    save_prompt_list_to_txt_file
)
from .llm_client import (
    LLMClient,
    OllamaClient,
    VLLMClient,
    create_llm_client,
    load_checkpoint,
    clear_checkpoint,
    GenerationResult
)
from .output import OutputWriter
from ..config.schema import validate_config
from ..utils.cache import get_cache

logger = logging.getLogger("llm_pipeline")


def load_config(config_path: str) -> dict:
    """Load pipeline configuration from JSON file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_texts_from_csv(csv_path: str) -> list[str]:
    """Extract filtered_response values from a CSV file."""
    extracted_texts = []
    if not os.path.exists(csv_path):
        logger.error(f"{csv_path} not found.")
        return extracted_texts

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("filtered_response", "")
            if text.strip():
                extracted_texts.append(text.strip())

    return extracted_texts


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text (rough approximation).
    Ollama uses ~4 chars per token on average.
    """
    return len(text) // 4


@dataclass
class StageStats:
    """Statistics for a pipeline stage."""
    prompts_processed: int = 0
    prompts_failed: int = 0
    tokens_used: int = 0
    errors: list[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


def run_stage(
    stage_config: dict,
    stage_num: int,
    total_stages: int,
    global_config: dict,
    previous_csv_path: str | None,
    resume_from: int = 0,
    skip_failed: bool = False,
    output_format: str = "csv",
    use_cache: bool = True,
    cache_dir: str = ".prompt_cache"
) -> str:
    """
    Execute a single stage of the pipeline.

    Args:
        stage_config: Configuration for this stage
        stage_num: Current stage number (1-indexed)
        total_stages: Total number of stages
        global_config: Global pipeline configuration
        previous_csv_path: Path to previous stage's CSV (None for first stage)
        resume_from: Index to resume from
        skip_failed: Whether to skip failed prompts instead of failing pipeline
        output_format: Output format (csv, json, jsonl)
        use_cache: Whether to use prompt caching
        cache_dir: Cache directory

    Returns:
        Path to this stage's output CSV
    """
    stage_name = stage_config.get("name", f"stage{stage_num}")
    logger.info(f"\n{'='*40}")
    logger.info(f"--- Stage {stage_num}/{total_stages}: {stage_name} ---")
    logger.info(f"{'='*40}")

    input_config = stage_config["input"]
    prompt_config = stage_config["prompt"]
    output_config = stage_config["output"]
    model_params = stage_config.get("model_params", {})

    # --- Determine input source ---
    if input_config["type"] == "file":
        input_path = input_config["path"]
        delimiter = input_config.get("delimiter", "\n")
        prompt_input_list = get_prompt_input_list_from_txt_file(input_path, delimiter)
    elif input_config["type"] == "previous_csv":
        if previous_csv_path is None:
            raise ValueError(
                f"Stage '{stage_name}' requires previous stage output but none exists"
            )
        delimiter = input_config.get("delimiter", "\n")
        prompt_input_list = extract_texts_from_csv(previous_csv_path)
        if not prompt_input_list:
            logger.warning(f"No valid inputs found in {previous_csv_path}. Ending pipeline.")
            return ""
    else:
        raise ValueError(f"Unknown input type: {input_config['type']}")

    # --- Wrap prompts ---
    prompt_template = prompt_config["template"]
    insert_key = prompt_config.get("insert_key", ":")
    prompt_list = input_list_to_promptlist(
        prompt_template, prompt_input_list, insert_key, use_cache, cache_dir
    )

    # Save wrapped prompts
    wrapped_path = output_config["wrapped_file"]
    save_prompt_list_to_txt_file(wrapped_path, prompt_list, "\n")
    logger.info(f"Wrapped prompts saved to: {wrapped_path}")

    # --- Generate with LLM ---
    model = stage_config["model"]
    logger.info(f"\n--- Generating with model: {model} ---")

    # Determine backend (stage-level overrides global)
    backend = model_params.get("backend") or global_config.get("backend", "ollama")
    host = model_params.get("host") or global_config.get("host", "http://localhost:11434")

    client = create_llm_client(
        backend=backend,
        model=model,
        context_window=global_config.get("context_window", 4096),
        host=host,
        temperature=model_params.get("temperature", 0.7),
        top_p=model_params.get("top_p", 0.9),
        num_predict=model_params.get("num_predict")
    )
    logger.info(f"Using backend: {backend}")

    csv_path = output_config["csv_file"]
    writer = OutputWriter(csv_path, format=output_format)
    stats = StageStats()

    def result_generator():
        for result in client.batch_generate(
            prompt_list,
            verbose=True,
            stage=stage_num,
            resume_from=resume_from,
            parallel=model_params.get("parallel", False),
            max_workers=model_params.get("max_workers", 4),
            stream=model_params.get("stream", False)
        ):
            stats.prompts_processed += 1 if not result.error else 0
            stats.prompts_failed += 1 if result.error else 0
            stats.tokens_used += result.tokens_used
            if result.error:
                stats.errors.append(f"Prompt {result.index}: {result.error}")
                if skip_failed:
                    continue
            yield result

    count = writer.write_results(result_generator())
    logger.info(f"Stage {stage_num} complete. Wrote {count} result(s) to {csv_path}")
    logger.info(f"Stats: {stats.prompts_processed} processed, {stats.prompts_failed} failed, {stats.tokens_used} tokens used")

    # --- Save intermediate file if specified (for next stage input) ---
    if "intermediate_file" in output_config:
        intermediate_path = output_config["intermediate_file"]
        extracted_texts = extract_texts_from_csv(csv_path)
        save_prompt_list_to_txt_file(intermediate_path, extracted_texts, "\n")
        logger.info(f"Intermediate output saved to: {intermediate_path}")

    return csv_path


def run_pipeline(
    config: dict,
    resume: bool = False,
    skip_failed: bool = False,
    output_format: str = "csv",
    use_cache: bool = True,
    cache_dir: str = ".prompt_cache"
) -> bool:
    """
    Run the complete multi-stage pipeline.

    Args:
        config: Pipeline configuration dictionary
        resume: Whether to resume from checkpoint
        skip_failed: Whether to skip failed prompts instead of failing pipeline
        output_format: Output format (csv, json, jsonl)
        use_cache: Whether to use prompt caching
        cache_dir: Cache directory

    Returns:
        True if pipeline completed successfully, False otherwise
    """
    global_config = config.get("global", {})
    stages = config.get("stages", [])

    if not stages:
        logger.error("No stages defined in configuration.")
        return False

    total_stages = len(stages)
    previous_csv_path: str | None = None

    # Load checkpoint if resuming
    resume_stage = 1
    resume_from_index = 0
    if resume:
        checkpoint = load_checkpoint()
        if checkpoint:
            resume_stage = checkpoint.get("stage", 1)
            resume_from_index = checkpoint.get("prompt_index", 0) + 1
            logger.info(f"Resuming from stage {resume_stage}, prompt {resume_from_index}")

    for i, stage_config in enumerate(stages, start=1):
        if i < resume_stage:
            logger.info(f"Skipping stage {i} (resuming from stage {resume_stage})")
            continue

        previous_csv = previous_csv_path if i > 1 else None
        resume_from = resume_from_index if i == resume_stage else 0

        csv_path = run_stage(
            stage_config,
            i,
            total_stages,
            global_config,
            previous_csv,
            resume_from,
            skip_failed,
            output_format,
            use_cache,
            cache_dir
        )

        if csv_path == "":
            return False

        previous_csv_path = csv_path

    logger.info(f"\n{'='*40}\nPIPELINE COMPLETE\n{'='*40}")
    if resume:
        clear_checkpoint()
    return True


def generate_pipeline_diagram(config: dict) -> str:
    """
    Generate a Mermaid.js diagram of the pipeline.

    Args:
        config: Pipeline configuration dictionary

    Returns:
        Mermaid.js diagram string
    """
    stages = config.get("stages", [])
    lines = ["graph TD", "    title LLM Pipeline"]

    for i, stage in enumerate(stages, 1):
        stage_name = stage.get("name", f"stage{i}")
        stage_id = f"S{i}"
        model = stage.get("model", "unknown")
        lines.append(f"    {stage_id}[\"{stage_name}\\n({model})\"]")

        if i > 1:
            prev_id = f"S{i-1}"
            lines.append(f"    {prev_id} --> {stage_id}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="N-stage LLM pipeline with JSON configuration."
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to pipeline configuration JSON file"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        help="Skip failed prompts instead of failing pipeline"
    )
    parser.add_argument(
        "--output-format",
        choices=["csv", "json", "jsonl"],
        default="csv",
        help="Output format for results"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate and print pipeline diagram"
    )

    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        return 1

    try:
        config = load_config(args.config)
        logger.info(f"Validating configuration...")
        validate_config(config)
        logger.info(f"Configuration valid. Starting pipeline.")

        if args.visualize:
            diagram = generate_pipeline_diagram(config)
            print(diagram)
            return 0

        success = run_pipeline(
            config,
            resume=args.resume,
            skip_failed=args.skip_failed,
            output_format=args.output_format
        )
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        return 1


if __name__ == "__main__":
    exit(main())