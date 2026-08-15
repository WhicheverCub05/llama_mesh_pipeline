"""
JSON schemas for pipeline configuration validation.
"""

import jsonschema
from jsonschema import validate, ValidationError

# Schema for global configuration
GLOBAL_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "host": {"type": "string", "pattern": "^https?://"},
        "context_window": {"type": "integer", "minimum": 1},
        "backend": {"type": "string", "enum": ["ollama", "vllm"]}
    },
    "additionalProperties": False
}

# Schema for input configuration
INPUT_CONFIG_SCHEMA = {
    "type": "object",
    "required": ["type"],
    "properties": {
        "type": {"enum": ["file", "previous_csv"]},
        "path": {"type": "string"},
        "delimiter": {"type": "string"}
    },
    "additionalProperties": False,
    "if": {
        "properties": {"type": {"const": "file"}},
        "required": ["path"]
    },
    "then": {
        "required": ["path"]
    }
}

# Schema for prompt configuration
PROMPT_CONFIG_SCHEMA = {
    "type": "object",
    "required": ["template"],
    "properties": {
        "template": {"type": "string"},
        "insert_key": {"type": "string"}
    },
    "additionalProperties": False
}

# Schema for output configuration
OUTPUT_CONFIG_SCHEMA = {
    "type": "object",
    "required": ["wrapped_file", "csv_file"],
    "properties": {
        "wrapped_file": {"type": "string"},
        "csv_file": {"type": "string"},
        "intermediate_file": {"type": "string"}
    },
    "additionalProperties": False
}

# Schema for model parameters
MODEL_PARAMS_SCHEMA = {
    "type": "object",
    "properties": {
        "temperature": {"type": "number", "minimum": 0, "maximum": 1},
        "top_p": {"type": "number", "minimum": 0, "maximum": 1},
        "num_predict": {"type": "integer", "minimum": 1},
        "parallel": {"type": "boolean"},
        "max_workers": {"type": "integer", "minimum": 1, "maximum": 32},
        "stream": {"type": "boolean"},
        "backend": {"type": "string", "enum": ["ollama", "vllm"]}
    },
    "additionalProperties": False
}

# Schema for a single stage
STAGE_CONFIG_SCHEMA = {
    "type": "object",
    "required": ["input", "prompt", "model", "output"],
    "properties": {
        "name": {"type": "string"},
        "input": INPUT_CONFIG_SCHEMA,
        "prompt": PROMPT_CONFIG_SCHEMA,
        "model": {"type": "string"},
        "model_params": MODEL_PARAMS_SCHEMA,
        "output": OUTPUT_CONFIG_SCHEMA
    },
    "additionalProperties": False
}

# Complete pipeline configuration schema
PIPELINE_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "global": GLOBAL_CONFIG_SCHEMA,
        "stages": {
            "type": "array",
            "minItems": 1,
            "items": STAGE_CONFIG_SCHEMA
        }
    },
    "required": ["stages"],
    "additionalProperties": False
}


def validate_config(config: dict) -> None:
    """
    Validate pipeline configuration against JSON schema.

    Args:
        config: Pipeline configuration dictionary

    Raises:
        ValidationError: If configuration is invalid
    """
    try:
        validate(instance=config, schema=PIPELINE_CONFIG_SCHEMA)
    except ValidationError as e:
        error_msg = f"Configuration validation error:\n"
        error_msg += f"  Location: {' -> '.join(str(p) for p in e.path)}\n"
        error_msg += f"  Message: {e.message}"
        raise ValidationError(error_msg) from e


def validate_config_file(config_path: str) -> dict:
    """
    Load and validate a pipeline configuration file.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Validated configuration dictionary

    Raises:
        ValidationError: If configuration is invalid
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If JSON is malformed
    """
    import json

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    validate_config(config)
    return config