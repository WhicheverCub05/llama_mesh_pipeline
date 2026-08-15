#!/usr/bin/env python3
"""
CSV utilities for extraction and transformation
"""

import csv
from pathlib import Path

def csv_to_txt(input_csv: str, output_txt: str, column_name: str) -> None:
    """
    Extracts a specific column from a CSV and saves it as a newline-delimited text file.

    Args:
        input_csv: Path to the source CSV file.
        output_txt: Path to the destination text file.
        column_name: The name of the column to extract.
    """
    input_path = Path(input_csv)
    output_path = Path(output_txt)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV file not found: {input_csv}")

    with open(input_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError(f"CSV file {input_csv} is empty or has no headers.")

        if column_name not in reader.fieldnames:
            raise ValueError(f"Column '{column_name}' not found in CSV. Available columns: {', '.join(reader.fieldnames)}")

        with open(output_path, mode="w", encoding="utf-8") as out_f:
            for row in reader:
                val = row[column_name]
                if val:  # Only write non-empty values
                    out_f.write(f"{val}\n")