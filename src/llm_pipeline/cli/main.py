#!/usr/bin/env python3
"""
Main pipeline logic for processing prompts through LLM
"""

import os
import logging
import argparse

from pathlib import Path
from typing import List, Optional

from ..core.llm_client import create_llm_client, LLMClient
from ..core.output import OutputWriter

from ..utils.obj_utils import csv_to_obj

logger = logging.getLogger("llm_pipeline")


def load_prompts(input_files: List[str]) -> List[str]:
    """
    Load prompts from input files.

    Args:
        input_files: List of file paths to read

    Returns:
        List of prompt strings (one per line across all files)
    """
    prompts = []

    for file_path in input_files:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Strip whitespace and filter empty lines
        prompts.extend([line.strip() for line in lines if line.strip()])

    return prompts


def estimate_tokens(text: str) -> int:
    """Estimate token count for text (~4 chars per token)."""
    return len(text) // 4


def run_single_stage_pipeline(
    input_files: List[str],
    output_file: str,
    output_format: str,
    create_obj: Optional[bool],
    obj_folder_path: Optional[str],
    model: Optional[str] = None,
    context_window: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    num_predict: Optional[int] = None,
    dry_run: bool = False,
    estimate_tokens: bool = False,
    stream: bool = False,
    parallel: bool = False,
    max_workers: int = 4,
    skip_failed: bool = False,
    verbose: bool = False,
    host: Optional[str] = None,
    backend: Optional[str] = "ollama",
    use_cache: bool = True,
    cache_dir: str = ".prompt_cache"
) -> None:
    """
    Run the LLM pipeline.

    Args:
        input_files: List of input text files
        output_file: Output file path
        output_format: Output format (csv, json, jsonl)
        model: LLM model to use
        context_window: Context window size in tokens
        temperature: Sampling temperature
        top_p: Nucleus sampling top_p
        num_predict: Max tokens to predict
        dry_run: Print prompts without processing
        estimate_tokens: Estimate token usage in dry-run
        stream: Stream output in real-time
        parallel: Process prompts in parallel
        max_workers: Max parallel workers
        skip_failed: Skip failed prompts
        verbose: Show progress
        host: API host endpoint
        backend: Backend type ('ollama' or 'vllm')
    """
    if verbose:
        logger.info(f"Loading prompts from {len(input_files)} file(s)...")

    # Load prompts
    prompts = load_prompts(input_files)

    if not prompts:
        raise ValueError("No prompts found in input files")

    if verbose:
        logger.info(f"Loaded {len(prompts)} prompt(s)")

    if dry_run:
        logger.info("\n--- Dry run - Prompts ---")
        total_tokens = 0
        for i, prompt in enumerate(prompts, 1):
            prompt_tokens = estimate_tokens(prompt) if estimate_tokens else 0
            total_tokens += prompt_tokens
            logger.info(f"\n[{i}] {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
            if estimate_tokens:
                logger.info(f"    Estimated tokens: {prompt_tokens}")
        logger.info(f"\nTotal: {len(prompts)} prompts")
        if estimate_tokens:
            logger.info(f"Total estimated tokens: {total_tokens}")
        return

    # Initialize LLM client (supports both Ollama and vLLM)
    client = create_llm_client(
        backend=backend,
        model=model,
        context_window=context_window,
        host=host,
        temperature=temperature or 0.7,
        top_p=top_p or 0.9,
        num_predict=num_predict
    )
    logger.info(f"Using backend: {backend}")

    if verbose:
        logger.info(f"Using model: {client.model}")
        logger.info(f"Context window: {client.context_window} tokens")
        logger.info(f"Temperature: {client.temperature}, Top P: {client.top_p}")
        logger.info(f"Ollama host: {client.host}")

        if client.health_check():
            logger.info("Ollama server is running")
        else:
            logger.warning("Could not connect to Ollama server")

    # Initialize output writer
    from ..core.output import OutputWriter
    writer = OutputWriter(output_file, format=output_format)

    # Process prompts and write results
    logger.info(f"Processing {len(prompts)} prompt(s)...")
    count = writer.write_results(
        client.batch_generate(
            prompts,
            verbose=verbose,
            parallel=parallel,
            max_workers=max_workers,
            stream=stream
        )
    )

    logger.info(f"Done! Wrote {count} result(s) to {output_file}")

    if create_obj:
        save_folder = os.path.join(os.path.dirname(output_file), "obj_results")
        if obj_folder_path:
            save_folder = obj_folder_path
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)
        csv_to_obj(output_file, save_folder, 3)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="LLM Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py run -i prompts.txt -o results.csv
  python main.py csv_to_txt --input_csv results.csv --output_txt extracted.txt --column filtered_response
  python main.py csv_to_obj --csv_file results.csv --obj_folder objs --column_index 3
  python main.py visual --input_dir objs --output_dir images
  python main.py pipeline --config pipeline_config.json
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command (single-stage pipeline)
    run_parser = subparsers.add_parser(
        "run",
        help="Process prompts through an LLM model",
        description="Read prompts from text file(s) and generate responses using Ollama"
    )
    run_parser.add_argument(
        "-i", "--input",
        action="append",
        required=True,
        dest="inputs",
        metavar="FILE",
        help="Input text file(s) with prompts (one per line). Can specify multiple files."
    )
    run_parser.add_argument(
        "-o", "--output",
        required=True,
        dest="output",
        metavar="FILE",
        help="Output file to store results"
    )
    run_parser.add_argument(
        "-f", "--format",
        choices=["csv", "json", "jsonl"],
        default="csv",
        dest="output_format",
        help="Output format (default: csv)"
    )
    run_parser.add_argument(
        "-obj", "--create_obj",
        action="store_true",
        dest="create_obj",
        help="Create OBJ files from output in a subfolder"
    )
    run_parser.add_argument(
        "-obj_o", "--obj_folder_output",
        dest="obj_folder_output",
        metavar="DIR",
        help="Directory to save OBJ files (defaults to obj_results next to output CSV)"
    )
    run_parser.add_argument(
        "-m", "--model",
        dest="model",
        metavar="MODEL",
        help="Ollama model to use (e.g., llama3.2:1b). Defaults to MODEL in .env"
    )
    run_parser.add_argument(
        "-c", "--context",
        type=int,
        dest="context_window",
        metavar="SIZE",
        help="Context window size in tokens. Defaults to CONTEXT_WINDOW in .env"
    )
    run_parser.add_argument(
        "--temperature",
        type=float,
        dest="temperature",
        metavar="TEMP",
        help="Sampling temperature (0.0-1.0). Defaults to 0.7"
    )
    run_parser.add_argument(
        "--top-p",
        type=float,
        dest="top_p",
        metavar="TOP_P",
        help="Nucleus sampling top_p value. Defaults to 0.9"
    )
    run_parser.add_argument(
        "--num-predict",
        type=int,
        dest="num_predict",
        metavar="NUM",
        help="Maximum tokens to predict"
    )
    run_parser.add_argument(
        "-d", "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Print prompts without sending to LLM (for validation)"
    )
    run_parser.add_argument(
        "--estimate-tokens",
        action="store_true",
        dest="estimate_tokens",
        help="Estimate token usage in dry-run mode"
    )
    run_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        dest="verbose",
        help="Show progress during processing"
    )
    run_parser.add_argument(
        "--stream",
        action="store_true",
        dest="stream",
        help="Stream output in real-time"
    )
    run_parser.add_argument(
        "--parallel",
        action="store_true",
        dest="parallel",
        help="Process prompts in parallel"
    )
    run_parser.add_argument(
        "--max-workers",
        type=int,
        dest="max_workers",
        default=4,
        help="Maximum parallel workers (default: 4)"
    )
    run_parser.add_argument(
        "--skip-failed",
        action="store_true",
        dest="skip_failed",
        help="Skip failed prompts instead of failing pipeline"
    )
    run_parser.add_argument(
        "--host",
        metavar="HOST",
        help="API host endpoint (e.g., http://localhost:11434 for Ollama, http://localhost:8000 for vLLM). Defaults to LLM_HOST in .env"
    )
    run_parser.add_argument(
        "--backend",
        choices=["ollama", "vllm"],
        default="ollama",
        help="LLM backend to use (default: ollama)"
    )
    run_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable prompt caching"
    )
    run_parser.add_argument(
        "--cache-dir",
        default=".prompt_cache",
        help="Directory for prompt cache (default: .prompt_cache)"
    )

    # CSV to text command
    csv_txt_parser = subparsers.add_parser(
        "csv_to_txt",
        help="Extract CSV column to text file",
        description="Extract a specific column from CSV and save as newline-delimited text"
    )
    csv_txt_parser.add_argument(
        "--input_csv",
        required=True,
        help="Path to input CSV file"
    )
    csv_txt_parser.add_argument(
        "--output_txt",
        required=True,
        help="Path to output text file"
    )
    csv_txt_parser.add_argument(
        "--column",
        required=True,
        help="Name of the column to extract"
    )

    # CSV to OBJ command
    csv_obj_parser = subparsers.add_parser(
        "csv_to_obj",
        help="Convert CSV column to OBJ files",
        description="Create OBJ files from text in a CSV column"
    )
    csv_obj_parser.add_argument(
        "--csv_file",
        required=True,
        help="Path to input CSV file"
    )
    csv_obj_parser.add_argument(
        "--obj_folder",
        required=True,
        help="Directory to save OBJ files"
    )
    csv_obj_parser.add_argument(
        "--column_index",
        type=int,
        required=True,
        help="0-based index of the column containing OBJ text"
    )

    # Visual command
    visual_parser = subparsers.add_parser(
        "visual",
        help="Visualize 3D OBJ files as 2D images",
        description="Render vertices from .obj files into .jpg images"
    )
    visual_parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing .obj files to visualize"
    )
    visual_parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save the resulting .jpg images"
    )

    # Pipeline command
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run multi-stage pipeline from config file",
        description="Execute N-stage LLM pipeline defined in JSON config"
    )
    pipeline_parser.add_argument(
        "--config",
        required=True,
        help="Path to pipeline configuration JSON file"
    )
    pipeline_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint"
    )
    pipeline_parser.add_argument(
        "--skip-failed",
        action="store_true",
        help="Skip failed prompts instead of failing pipeline"
    )
    pipeline_parser.add_argument(
        "--output-format",
        choices=["csv", "json", "jsonl"],
        default="csv",
        help="Output format for results"
    )
    pipeline_parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate and print pipeline diagram"
    )
    pipeline_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable prompt caching"
    )
    pipeline_parser.add_argument(
        "--cache-dir",
        default=".prompt_cache",
        help="Directory for prompt cache (default: .prompt_cache)"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if args.command == "run":
        run_single_stage_pipeline(
            input_files=args.inputs,
            output_file=args.output,
            output_format=args.output_format,
            create_obj=args.create_obj,
            obj_folder_path=args.obj_folder_output,
            model=args.model,
            context_window=args.context_window,
            temperature=args.temperature,
            top_p=args.top_p,
            num_predict=args.num_predict,
            dry_run=args.dry_run,
            estimate_tokens=args.estimate_tokens,
            stream=args.stream,
            parallel=args.parallel,
            max_workers=args.max_workers,
            skip_failed=args.skip_failed,
            verbose=args.verbose,
            host=args.host,
            backend=args.backend,
            use_cache=not args.no_cache,
            cache_dir=args.cache_dir
        )
    elif args.command == "csv_to_txt":
        from ..utils.csv_utils import csv_to_txt
        csv_to_txt(args.input_csv, args.output_txt, args.column)
        print(f"Extracted column '{args.column}' from {args.input_csv} to {args.output_txt}")
    elif args.command == "csv_to_obj":
        csv_to_obj(args.csv_file, args.obj_folder, args.column_index)
        print(f"Created OBJ files from column {args.column_index} of {args.csv_file} in {args.obj_folder}")
    elif args.command == "visual":
        from ..utils.visual import render_obj_to_image
        render_obj_to_image(args.input_dir, args.output_dir)
        print(f"Visualized OBJ files from {args.input_dir} to {args.output_dir}")
    elif args.command == "pipeline":
        from ..core.pipeline import load_config, validate_config, run_pipeline, generate_pipeline_diagram
        config = load_config(args.config)
        validate_config(config)

        if args.visualize:
            diagram = generate_pipeline_diagram(config)
            print(diagram)
            return

        success = run_pipeline(
            config,
            resume=args.resume,
            skip_failed=args.skip_failed,
            output_format=args.output_format,
            use_cache=not args.no_cache,
            cache_dir=args.cache_dir
        )
        if success:
            print("Pipeline completed successfully")
        else:
            print("Pipeline failed")
            exit(1)
    else:
        parser.print_help()
        return


if __name__ == "__main__":
    main()