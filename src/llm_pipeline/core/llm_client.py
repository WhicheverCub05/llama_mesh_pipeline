"""
Base LLM client interface and implementations for different backends.
"""

import json
import requests
import os
import time
import hashlib
from functools import wraps
from pathlib import Path
from typing import Optional, Generator, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

import logging
from tqdm import tqdm
from ..utils.text_filters import filter_model_output

logger = logging.getLogger("llm_pipeline")

CHECKPOINT_FILE = ".pipeline_checkpoint.json"


@dataclass
class GenerationResult:
    """Result of a generation request."""
    index: int
    prompt: str
    response: str
    filtered_response: str
    tokens_used: int = 0
    error: str = ""


def save_checkpoint(stage: int, prompt_index: int):
    """Save checkpoint after successful prompt processing."""
    checkpoint = {"stage": stage, "prompt_index": prompt_index}
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f)


def load_checkpoint() -> dict | None:
    """Load checkpoint if it exists."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return None


def clear_checkpoint():
    """Clear checkpoint file."""
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)


def retry_on_failure(max_attempts: int = 3, backoff: float = 2.0):
    """
    Decorator that retries a function on requests.RequestException.

    Args:
        max_attempts: Maximum number of retry attempts
        backoff: Base wait time multiplier (exponential backoff)

    Returns:
        Decorated function with retry logic
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except requests.RequestException as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        wait_time = backoff ** attempt
                        logger.warning(f"Request failed, retrying in {wait_time:.1f}s... (attempt {attempt + 1}/{max_attempts})")
                        time.sleep(wait_time)
            raise last_exception
        return wrapper
    return decorator


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(
        self,
        model: Optional[str] = None,
        context_window: Optional[int] = None,
        host: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_predict: Optional[int] = None,
        max_retries: int = 3,
        retry_backoff: float = 2.0
    ):
        """
        Initialize the LLM client.

        Args:
            model: Model name
            context_window: Maximum tokens for context window
            host: API endpoint URL
            temperature: Sampling temperature (0.0-1.0)
            top_p: Nucleus sampling top_p value
            num_predict: Maximum tokens to predict
            max_retries: Maximum retry attempts for failed requests
            retry_backoff: Base backoff time for retries
        """
        self._load_env(Path(__file__).parent.parent.parent / ".env")

        self.host = host or os.getenv("LLM_HOST", "http://localhost:11434")
        self.model = model or os.getenv("MODEL", "llama3.2:1b")
        self.context_window = context_window or int(
            os.getenv("CONTEXT_WINDOW", "4096")
        )
        self.temperature = temperature
        self.top_p = top_p
        self.num_predict = num_predict
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

        # Strip trailing slash if present
        if self.host.endswith("/"):
            self.host = self.host.rstrip("/")

    def _load_env(self, env_file: Path) -> None:
        """Load environment variables from .env file."""
        if not env_file.exists():
            return

        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())

    @abstractmethod
    def generate(self, prompt: str, stream: bool = False) -> tuple[str, int]:
        """
        Generate a response from the LLM.

        Args:
            prompt: The input prompt text
            stream: Whether to stream the response

        Returns:
            Tuple of (response text, tokens used)
        """
        pass

    @abstractmethod
    def generate_streaming(self, prompt: str) -> Generator[str, None, None]:
        """
        Generate a response with streaming output.

        Args:
            prompt: The input prompt text

        Yields:
            Individual tokens as they are generated
        """
        pass

    @abstractmethod
    def batch_generate(
        self,
        prompts: list[str],
        verbose: bool = True,
        stage: int = 0,
        resume_from: int = 0,
        parallel: bool = False,
        max_workers: int = 4,
        stream: bool = False
    ) -> Generator[GenerationResult, None, None]:
        """
        Generate responses for multiple prompts.

        Args:
            prompts: List of prompt strings
            verbose: Whether to show progress bar and log progress
            stage: Current stage number (for checkpointing)
            resume_from: Index to resume from (for checkpoint recovery)
            parallel: Whether to process prompts in parallel
            max_workers: Maximum parallel workers (if parallel=True)
            stream: Whether to stream output

        Yields:
            GenerationResult objects with index, prompt, response, filtered_response, tokens_used, error
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the LLM server is running."""
        pass

    @abstractmethod
    def check_model_available(self) -> bool:
        """Check if the specified model is available."""
        pass


class OllamaClient(LLMClient):
    """Client for interacting with Ollama API."""

    def _check_model_available(self) -> bool:
        """Check if the specified model is available."""
        try:
            response = requests.get(
                f"{self.host}/api/tags",
                timeout=10
            )
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            return self.model.split(":")[0] in [m.split(":")[0] for m in models]
        except (requests.RequestException, KeyError):
            return False

    def _build_params(self, prompt: str, stream: bool = False) -> Dict[str, Any]:
        """Build request parameters for Ollama API."""
        params = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "num_ctx": self.context_window,
                "temperature": self.temperature,
                "top_p": self.top_p
            }
        }
        if self.num_predict is not None:
            params["options"]["num_predict"] = self.num_predict
        return params

    @retry_on_failure(max_attempts=3, backoff=2.0)
    def generate(self, prompt: str, stream: bool = False) -> tuple[str, int]:
        """
        Generate a response from the LLM.

        Args:
            prompt: The input prompt text
            stream: Whether to stream the response

        Returns:
            Tuple of (response text, tokens used)
        """
        params = self._build_params(prompt, stream)

        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json=params,
                timeout=300,
                stream=stream
            )
            response.raise_for_status()

            if stream:
                full_response = []
                total_tokens = 0
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line.decode('utf-8'))
                        if chunk.get("response"):
                            full_response.append(chunk["response"])
                        if chunk.get("total_duration"):
                            total_tokens = chunk["total_duration"] / 1e9
                return "".join(full_response), int(total_tokens)
            else:
                result = response.json()
                tokens = result.get("eval_count", 0)
                return result.get("response", ""), tokens
        except requests.RequestException as e:
            if "404" in str(e) or "not found" in str(e).lower():
                raise RuntimeError(
                    f"Model '{self.model}' not found. Run 'ollama pull {self.model}' first."
                ) from e
            raise RuntimeError(f"Ollama API error: {e}") from e

    def generate_streaming(self, prompt: str) -> Generator[str, None, None]:
        """
        Generate a response with streaming output.

        Args:
            prompt: The input prompt text

        Yields:
            Individual tokens as they are generated
        """
        params = self._build_params(prompt, stream=True)

        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json=params,
                timeout=300,
                stream=True
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode('utf-8'))
                    if chunk.get("response"):
                        yield chunk["response"]
        except requests.RequestException as e:
            raise RuntimeError(f"Ollama API error: {e}") from e

    def batch_generate(
        self,
        prompts: list[str],
        verbose: bool = True,
        stage: int = 0,
        resume_from: int = 0,
        parallel: bool = False,
        max_workers: int = 4,
        stream: bool = False
    ) -> Generator[GenerationResult, None, None]:
        """
        Generate responses for multiple prompts.
        """
        import concurrent.futures

        total = len(prompts)
        start_index = resume_from if resume_from > 0 else 0

        progress_bar = tqdm(
            range(start_index, total),
            desc=f"Stage {stage}: Generating",
            disable=not verbose,
            unit="prompt",
            initial=start_index,
            total=total
        )

        if parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for i in range(start_index, total):
                    prompt = prompts[i]
                    future = executor.submit(self.generate, prompt, stream=stream)
                    futures[future] = i

                for future in concurrent.futures.as_completed(futures):
                    idx = futures[future]
                    try:
                        response, tokens = future.result()
                        filtered_response = filter_model_output(input=response, model=self.model)
                        result = GenerationResult(
                            index=idx,
                            prompt=prompts[idx],
                            response=response,
                            filtered_response=filtered_response,
                            tokens_used=tokens
                        )
                        yield result
                        save_checkpoint(stage, idx)
                    except Exception as e:
                        logger.error(f"Error processing prompt {idx + 1}: {e}")
                        yield GenerationResult(
                            index=idx,
                            prompt=prompts[idx],
                            response=f"[ERROR: {str(e)}]",
                            filtered_response=f"[ERROR: {str(e)}]",
                            error=str(e)
                        )
        else:
            for i in progress_bar:
                prompt = prompts[i]
                progress_bar.set_description(f"Stage {stage}: Generating ({len(prompt)} chars)")

                try:
                    response, tokens = self.generate(prompt, stream=stream)
                    filtered_response = filter_model_output(input=response, model=self.model)
                    yield GenerationResult(
                        index=i,
                        prompt=prompt,
                        response=response,
                        filtered_response=filtered_response,
                        tokens_used=tokens
                    )
                    save_checkpoint(stage, i)
                except Exception as e:
                    if verbose:
                        logger.error(f"Error processing prompt {i + 1}: {e}")
                    yield GenerationResult(
                        index=i,
                        prompt=prompt,
                        response=f"[ERROR: {str(e)}]",
                        filtered_response=f"[ERROR: {str(e)}]",
                        error=str(e)
                    )

    def health_check(self) -> bool:
        """Check if the Ollama server is running."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False


class VLLMClient(LLMClient):
    """Client for interacting with vLLM API (OpenAI-compatible)."""

    def _check_model_available(self) -> bool:
        """Check if the specified model is available."""
        try:
            response = requests.get(
                f"{self.host}/v1/models",
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            models = [m["id"] for m in data.get("data", [])]
            return self.model in models
        except (requests.RequestException, KeyError):
            return False

    def _build_params(self, prompt: str, stream: bool = False) -> Dict[str, Any]:
        """Build request parameters for vLLM/OpenAI API."""
        params = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "max_tokens": self.num_predict or (self.context_window - 100),
            "temperature": self.temperature,
            "top_p": self.top_p
        }
        return params

    @retry_on_failure(max_attempts=3, backoff=2.0)
    def generate(self, prompt: str, stream: bool = False) -> tuple[str, int]:
        """
        Generate a response from the LLM.

        Args:
            prompt: The input prompt text
            stream: Whether to stream the response

        Returns:
            Tuple of (response text, tokens used)
        """
        params = self._build_params(prompt, stream)

        try:
            response = requests.post(
                f"{self.host}/v1/completions",
                json=params,
                timeout=300,
                stream=stream
            )
            response.raise_for_status()

            if stream:
                full_response = []
                total_tokens = 0
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line.decode('utf-8'))
                        if chunk.get("choices"):
                            delta = chunk["choices"][0].get("text", "")
                            full_response.append(delta)
                        if chunk.get("usage"):
                            total_tokens = chunk["usage"].get("total_tokens", 0)
                return "".join(full_response), total_tokens
            else:
                result = response.json()
                tokens = result.get("usage", {}).get("total_tokens", 0)
                text = result.get("choices", [{}])[0].get("text", "")
                return text, tokens
        except requests.RequestException as e:
            if "404" in str(e) or "not found" in str(e).lower():
                raise RuntimeError(
                    f"Model '{self.model}' not found on vLLM server."
                ) from e
            raise RuntimeError(f"vLLM API error: {e}") from e

    def generate_streaming(self, prompt: str) -> Generator[str, None, None]:
        """
        Generate a response with streaming output.

        Args:
            prompt: The input prompt text

        Yields:
            Individual tokens as they are generated
        """
        params = self._build_params(prompt, stream=True)

        try:
            response = requests.post(
                f"{self.host}/v1/completions",
                json=params,
                timeout=300,
                stream=True
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode('utf-8'))
                    if chunk.get("choices"):
                        delta = chunk["choices"][0].get("text", "")
                        yield delta
        except requests.RequestException as e:
            raise RuntimeError(f"vLLM API error: {e}") from e

    def batch_generate(
        self,
        prompts: list[str],
        verbose: bool = True,
        stage: int = 0,
        resume_from: int = 0,
        parallel: bool = False,
        max_workers: int = 4,
        stream: bool = False
    ) -> Generator[GenerationResult, None, None]:
        """
        Generate responses for multiple prompts.
        """
        import concurrent.futures

        total = len(prompts)
        start_index = resume_from if resume_from > 0 else 0

        progress_bar = tqdm(
            range(start_index, total),
            desc=f"Stage {stage}: Generating",
            disable=not verbose,
            unit="prompt",
            initial=start_index,
            total=total
        )

        if parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for i in range(start_index, total):
                    prompt = prompts[i]
                    future = executor.submit(self.generate, prompt, stream=stream)
                    futures[future] = i

                for future in concurrent.futures.as_completed(futures):
                    idx = futures[future]
                    try:
                        response, tokens = future.result()
                        filtered_response = filter_model_output(input=response, model=self.model)
                        result = GenerationResult(
                            index=idx,
                            prompt=prompts[idx],
                            response=response,
                            filtered_response=filtered_response,
                            tokens_used=tokens
                        )
                        yield result
                        save_checkpoint(stage, idx)
                    except Exception as e:
                        logger.error(f"Error processing prompt {idx + 1}: {e}")
                        yield GenerationResult(
                            index=idx,
                            prompt=prompts[idx],
                            response=f"[ERROR: {str(e)}]",
                            filtered_response=f"[ERROR: {str(e)}]",
                            error=str(e)
                        )
        else:
            for i in progress_bar:
                prompt = prompts[i]
                progress_bar.set_description(f"Stage {stage}: Generating ({len(prompt)} chars)")

                try:
                    response, tokens = self.generate(prompt, stream=stream)
                    filtered_response = filter_model_output(input=response, model=self.model)
                    yield GenerationResult(
                        index=i,
                        prompt=prompt,
                        response=response,
                        filtered_response=filtered_response,
                        tokens_used=tokens
                    )
                    save_checkpoint(stage, i)
                except Exception as e:
                    if verbose:
                        logger.error(f"Error processing prompt {i + 1}: {e}")
                    yield GenerationResult(
                        index=i,
                        prompt=prompt,
                        response=f"[ERROR: {str(e)}]",
                        filtered_response=f"[ERROR: {str(e)}]",
                        error=str(e)
                    )

    def health_check(self) -> bool:
        """Check if the vLLM server is running."""
        try:
            response = requests.get(f"{self.host}/v1/models", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False


def create_llm_client(
    backend: str,
    model: Optional[str] = None,
    context_window: Optional[int] = None,
    host: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_predict: Optional[int] = None,
    max_retries: int = 3,
    retry_backoff: float = 2.0
) -> LLMClient:
    """
    Factory function to create the appropriate LLM client based on backend.

    Args:
        backend: Backend type - "ollama" or "vllm"
        model: Model name
        context_window: Maximum tokens for context window
        host: API endpoint URL
        temperature: Sampling temperature
        top_p: Nucleus sampling top_p
        num_predict: Maximum tokens to predict
        max_retries: Maximum retry attempts
        retry_backoff: Base backoff time

    Returns:
        Instance of the appropriate LLM client

    Raises:
        ValueError: If an unknown backend is specified
    """
    backend = backend.lower()
    if backend == "ollama":
        return OllamaClient(
            model=model,
            context_window=context_window,
            host=host,
            temperature=temperature,
            top_p=top_p,
            num_predict=num_predict,
            max_retries=max_retries,
            retry_backoff=retry_backoff
        )
    elif backend == "vllm":
        return VLLMClient(
            model=model,
            context_window=context_window,
            host=host,
            temperature=temperature,
            top_p=top_p,
            num_predict=num_predict,
            max_retries=max_retries,
            retry_backoff=retry_backoff
        )
    else:
        raise ValueError(f"Unknown backend: {backend}. Supported backends: 'ollama', 'vllm'")
