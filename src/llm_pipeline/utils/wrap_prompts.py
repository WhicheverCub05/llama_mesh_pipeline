"""
Prompt wrapping utilities for LLM pipeline.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

from .cache import get_cache, PromptCache

logger = logging.getLogger("llm_pipeline")


def get_prompt_input_list_from_txt_file(file_path: str, delimiter: str) -> list:
    """
    Load prompts from a text file.

    Args:
        file_path: Path to the input file
        delimiter: Delimiter to split prompts (comma or newline)

    Returns:
        List of prompt strings
    """
    logger.debug(f"Reading file: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    logger.debug(f"File contents ({len(text)} chars): {text[:200]}...")

    if delimiter == ",":
        prompt_input = text.split(",")
    else:  # newline
        prompt_input = text.split("\n")

    return [p.strip() for p in prompt_input if p.strip()]


def save_prompt_list_to_txt_file(file_path: str, prompts: list, delim: str = "\n") -> None:
    """
    Save a list of prompts to a text file.

    Args:
        file_path: Path to output file
        prompts: List of prompt strings
        delim: Delimiter between prompts
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for prompt in prompts:
            f.write(prompt + delim)


def input_list_to_promptlist(
    prompt_wrapper: str,
    prompt_input_list: list,
    insert_key: str,
    use_cache: bool = True,
    cache_dir: str = ".prompt_cache"
) -> list:
    """
    Convert plain input texts into wrapped prompts for LLM ingestion.

    Args:
        prompt_wrapper: Template with insert_key placeholder
        prompt_input_list: List of input texts to wrap
        insert_key: Placeholder to replace with input text
        use_cache: Whether to use prompt caching
        cache_dir: Directory for cache files

    Returns:
        List of wrapped prompts
    """
    if not prompt_wrapper:
        raise ValueError("Prompt wrapper template cannot be empty")

    # Try to get from cache first
    if use_cache:
        cache = get_cache(cache_dir)
        cached = cache.get(prompt_wrapper, insert_key, tuple(prompt_input_list))
        if cached:
            logger.info(f"Loaded {len(cached)} prompts from cache")
            return cached

    prompts = []
    insert_index = prompt_wrapper.find(insert_key)
    insert_at_end = insert_index == -1

    logger.debug(f"Prompt wrapper: {prompt_wrapper[:100] if len(prompt_wrapper) < 100 else prompt_wrapper[:100]}...")
    logger.debug(f"Input list has {len(prompt_input_list)} items")

    for input_prompt in prompt_input_list:
        if not input_prompt:
            continue
        if insert_at_end:
            prompt = prompt_wrapper + input_prompt
        else:
            prompt = prompt_wrapper.replace(insert_key, input_prompt)
        prompts.append(prompt)

    # Cache the result
    if use_cache:
        cache = get_cache(cache_dir)
        cache.set(prompt_wrapper, insert_key, tuple(prompt_input_list), prompts)

    return prompts


def wrap_prompts_from_file(
    input_file: str,
    output_file: str,
    template: str,
    insert_key: str,
    delimiter: str = "\n",
    use_cache: bool = True,
    cache_dir: str = ".prompt_cache"
) -> int:
    """
    Wrap prompts from a file and save to output.

    Args:
        input_file: Path to input file with raw prompts
        output_file: Path to output file for wrapped prompts
        template: Prompt template with insert_key
        insert_key: Placeholder to replace
        delimiter: Delimiter in input file
        use_cache: Whether to use caching
        cache_dir: Cache directory

    Returns:
        Number of prompts wrapped
    """
    prompt_input_list = get_prompt_input_list_from_txt_file(input_file, delimiter)
    prompt_list = input_list_to_promptlist(
        template, prompt_input_list, insert_key, use_cache, cache_dir
    )
    save_prompt_list_to_txt_file(output_file, prompt_list, "\n")
    return len(prompt_list)



def main():
    parser = argparse.ArgumentParser(description="Part of LLM Pipeline, use this tool to wrap promppts and create a textfile, with each line being a prompt",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="""
        examples:
    %(prog)s python .\prep_prompts.py run -i sample_prompts.txt -o test_output.txt --prompt "Create a 3D obj file using the following description:<>. directly output the obj file:" -key "<>"
        """)
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    run_parser = subparsers.add_parser(
        "run",
        help="wrap prompts to give to an LLM model",
        description="Read prompts from text file(s) and wrap them with prompt at input key"
    )

    run_parser.add_argument(
        "-i", "--input",
        action="append",
        required=True,
        dest="inputs",
        metavar="FILE",
        help="Input text file with prompts (one per line)"
    )
    run_parser.add_argument(
        "-d", "--dilemma",
        type=str,
        dest="dilemma",
        default="",
        help="Delimiter for reading prompts (comma or newline)"
    )

    run_parser.add_argument(
        "-o", "--output",
        required=True,
        dest="output",
        metavar="FILE",
        help="Output CSV file to store results"
    )

    run_parser.add_argument(
        "-k", "--insert_key",
        type=str,
        dest="key",
        default=":",
        help="key to instert prompt in wrapper"
    )

    run_parser.add_argument(
        "-p", "--prompt",
        type=str,
        dest="prompt",
        default="",
        help="prompt to wrap"
    )

    args = parser.parse_args()

    if args.command != "run":
        parser.print_help()
        return

    if not hasattr(args, "dilemma"):
        args.dilemma = "\n"

    prompt_input_list = get_prompt_input_list_from_txt_file(args.inputs[0], args.dilemma)
    prompt_list = input_list_to_promptlist(args.prompt if args.prompt else "", prompt_input_list, args.key)
    save_prompt_list_to_txt_file(args.output, prompt_list)


if __name__=="__main__":
    main()