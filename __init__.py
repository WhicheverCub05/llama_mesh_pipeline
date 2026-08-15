# LLM Pipeline - Text Processing with Ollama
"""
LLM Pipeline: Process text prompts through Ollama models.

Example:
    python -m llm_pipeline run -i prompts.txt -o results.csv
"""

import logging
from typing import Optional

__version__ = "0.1.0"


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Configure logging for the LLM pipeline.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file for file output

    Returns:
        Configured logger instance
    """
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create logger
    logger = logging.getLogger("llm_pipeline")
    logger.setLevel(getattr(logging, level.upper()))

    # Clear any existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Get default logger
logger = setup_logging()
