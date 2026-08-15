"""
Prompt caching utilities to avoid regeneration of identical prompts.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger("llm_pipeline")


class PromptCache:
    """Cache for wrapped prompts to avoid regeneration."""

    def __init__(self, cache_dir: str = ".prompt_cache"):
        """
        Initialize the prompt cache.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / "cache.json"
        self._ensure_cache_dir()
        self._load_cache()

    def _ensure_cache_dir(self) -> None:
        """Create cache directory if it doesn't exist."""
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_cache(self) -> None:
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                logger.debug(f"Loaded cache with {len(self.cache)} entries")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load cache: {e}")
                self.cache = {}
        else:
            self.cache = {}

    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
        except IOError as e:
            logger.warning(f"Failed to save cache: {e}")

    def _compute_hash(self, template: str, insert_key: str, inputs: tuple) -> str:
        """Compute a hash for the prompt configuration."""
        content = f"{template}::{insert_key}::{json.dumps(inputs)}"
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def get(self, template: str, insert_key: str, inputs: tuple) -> Optional[str]:
        """
        Get cached wrapped prompts if available.

        Args:
            template: Prompt template
            insert_key: Key to replace in template
            inputs: List of input texts

        Returns:
            Cached wrapped prompts or None if not found
        """
        cache_key = self._compute_hash(template, insert_key, inputs)
        return self.cache.get(cache_key)

    def set(self, template: str, insert_key: str, inputs: tuple, wrapped_prompts: list) -> None:
        """
        Cache wrapped prompts.

        Args:
            template: Prompt template
            insert_key: Key to replace in template
            inputs: List of input texts
            wrapped_prompts: List of wrapped prompts
        """
        cache_key = self._compute_hash(template, insert_key, inputs)
        self.cache[cache_key] = wrapped_prompts
        self._save_cache()
        logger.debug(f"Cached {len(wrapped_prompts)} prompts")

    def clear(self) -> None:
        """Clear the cache."""
        self.cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("Cache cleared")

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self.cache),
            "cache_file": str(self.cache_file)
        }


# Global cache instance
_cache_instance: Optional[PromptCache] = None


def get_cache(cache_dir: str = ".prompt_cache") -> PromptCache:
    """Get or create the global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = PromptCache(cache_dir)
    return _cache_instance


def clear_cache() -> None:
    """Clear the global cache."""
    global _cache_instance
    if _cache_instance:
        _cache_instance.clear()
        _cache_instance = None
