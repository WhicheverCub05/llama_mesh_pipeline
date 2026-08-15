#!/usr/bin/env python3
"""
Output handler for writing results in multiple formats.
"""

import csv
import json
from pathlib import Path
from typing import Generator, Union
from dataclasses import asdict

from .client import GenerationResult


class OutputWriter:
    """Handles writing results to files in various formats."""

    SUPPORTED_FORMATS = ["csv", "json", "jsonl"]

    def __init__(self, output_path: str, format: str = "csv"):
        """
        Initialize the output writer.

        Args:
            output_path: Path to output file
            format: Output format (csv, json, jsonl)
        """
        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format}. Supported: {self.SUPPORTED_FORMATS}")

        self.output_path = Path(output_path)
        self.format = format
        self._ensure_parent_dir()

    def _ensure_parent_dir(self) -> None:
        """Create parent directories if they don't exist."""
        if not self.output_path.parent.exists():
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_output_path(self) -> Path:
        """Get the appropriate output path based on format."""
        if self.format == "csv":
            return self.output_path.with_suffix(".csv")
        elif self.format == "json":
            return self.output_path.with_suffix(".json")
        elif self.format == "jsonl":
            return self.output_path.with_suffix(".jsonl")
        return self.output_path

    def write_results(
        self,
        results: Generator[GenerationResult, None, None],
        append: bool = False
    ) -> int:
        """
        Write prompts and responses to output file.

        Args:
            results: Generator of GenerationResult objects
            append: Whether to append to existing file

        Returns:
            Number of records written
        """
        output_path = self._get_output_path()
        first_write = not output_path.exists()
        mode = "a" if append else "w"
        write_header = not append or first_write or (append and output_path.stat().st_size == 0)
        count = 0

        with open(output_path, mode=mode, encoding="utf-8") as f:
            if self.format == "csv":
                count = self._write_csv(f, results, write_header)
            elif self.format == "json":
                count = self._write_json(f, results, write_header)
            elif self.format == "jsonl":
                count = self._write_jsonl(f, results, write_header)

        return count

    def _write_csv(
        self,
        f,
        results: Generator[GenerationResult, None, None],
        write_header: bool
    ) -> int:
        """Write results in CSV format."""
        writer = csv.writer(f)

        if write_header:
            writer.writerow(["index", "prompt", "response", "filtered_response", "tokens_used", "error"])

        count = 0
        for result in results:
            writer.writerow([
                result.index,
                result.prompt,
                result.response,
                result.filtered_response,
                result.tokens_used,
                result.error
            ])
            count += 1

        return count

    def _write_json(
        self,
        f,
        results: Generator[GenerationResult, None, None],
        write_header: bool
    ) -> int:
        """Write results in JSON format (array of objects)."""
        results_list = []
        count = 0

        for result in results:
            results_list.append({
                "index": result.index,
                "prompt": result.prompt,
                "response": result.response,
                "filtered_response": result.filtered_response,
                "tokens_used": result.tokens_used,
                "error": result.error
            })
            count += 1

        if write_header or not f.tell():
            f.write(json.dumps(results_list, indent=2, ensure_ascii=False))

        return count

    def _write_jsonl(
        self,
        f,
        results: Generator[GenerationResult, None, None],
        write_header: bool
    ) -> int:
        """Write results in JSONL format (one JSON object per line)."""
        count = 0

        for result in results:
            line = json.dumps({
                "index": result.index,
                "prompt": result.prompt,
                "response": result.response,
                "filtered_response": result.filtered_response,
                "tokens_used": result.tokens_used,
                "error": result.error
            }, ensure_ascii=False)
            f.write(line + "\n")
            count += 1

        return count


# Backward compatibility alias
CSVWriter = OutputWriter