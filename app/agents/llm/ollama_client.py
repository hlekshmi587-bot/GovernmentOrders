"""
Ollama HTTP Client for Local LLM Interaction.

This module provides a direct, lightweight interface to the Ollama server
running locally at http://localhost:11434. It handles health checks, model verification,
and text generation without relying on external cloud APIs.
"""

from typing import Any, Dict, List, Optional
import requests


class OllamaClient:
    """
    Client for interacting with local Ollama API endpoints.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "llama3.2:3b",
        timeout: int = 120,
        default_format: Optional[str] = None,
    ) -> None:
        """
        Initialize the Ollama client.

        Args:
            base_url: URL where Ollama server is listening (default: http://localhost:11434).
            default_model: Name of the local model to use (default: llama3.2:3b).
            timeout: Request timeout in seconds.
            default_format: Default response format (e.g. 'json' or None).
        """
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout
        self.default_format = default_format

    def is_server_running(self) -> bool:
        """
        Check if the local Ollama server is reachable.

        Returns:
            True if Ollama server responds, False otherwise.
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False

    def list_models(self) -> List[str]:
        """
        List all model names currently available in Ollama.

        Returns:
            List of model names (e.g. ['llama3.2:3b']).
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            data = response.json()
            models = data.get("models", [])
            return [m.get("name") for m in models if "name" in m]
        except Exception as e:
            raise RuntimeError(f"Failed to list Ollama models: {e}") from e

    def is_model_available(self, model_name: Optional[str] = None) -> bool:
        """
        Check if a specific model is downloaded and ready in Ollama.

        Args:
            model_name: Target model name, or self.default_model if None.

        Returns:
            True if the model is available, False otherwise.
        """
        target = model_name or self.default_model
        available = self.list_models()
        # Check both exact match and match without tag (e.g. 'llama3.2:3b' vs 'llama3.2:latest')
        return any(target == m or target in m for m in available)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
        format: Optional[str] = None,
        num_predict: Optional[int] = None,
        num_ctx: Optional[int] = None,
    ) -> str:
        """
        Generate a text completion for a given prompt using the local LLM.

        Args:
            prompt: User/task prompt text.
            system_prompt: Optional instructions describing the agent's role.
            model: Model name to use (defaults to self.default_model).
            temperature: Sampling temperature (0.0 - 1.0). Lower means more deterministic.
            format: Optional response format (e.g. 'json').
            num_predict: Optional maximum number of tokens to predict.
            num_ctx: Optional context window size in tokens.

        Returns:
            The generated response string.
        """
        selected_model = model or self.default_model
        target_format = format or self.default_format

        options: Dict[str, Any] = {
            "temperature": temperature,
        }
        if num_predict is not None:
            options["num_predict"] = num_predict
        if num_ctx is not None:
            options["num_ctx"] = num_ctx

        payload: Dict[str, Any] = {
            "model": selected_model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }

        if target_format:
            payload["format"] = target_format

        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except requests.ConnectionError as ce:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. Please ensure Ollama is running."
            ) from ce
        except requests.Timeout as te:
            raise TimeoutError(
                f"Ollama request timed out after {self.timeout} seconds."
            ) from te
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}") from e

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.1,
        num_predict: Optional[int] = None,
        num_ctx: Optional[int] = None,
    ) -> str:
        """
        Send a conversational list of messages to the local LLM.

        Args:
            messages: List of message dictionaries, e.g. [{'role': 'user', 'content': 'Hello'}]
            model: Model name to use (defaults to self.default_model).
            temperature: Sampling temperature (0.0 - 1.0).
            num_predict: Optional maximum number of tokens to predict.
            num_ctx: Optional context window size in tokens.

        Returns:
            The assistant's response string.
        """
        selected_model = model or self.default_model

        options: Dict[str, Any] = {
            "temperature": temperature,
        }
        if num_predict is not None:
            options["num_predict"] = num_predict
        if num_ctx is not None:
            options["num_ctx"] = num_ctx

        payload = {
            "model": selected_model,
            "messages": messages,
            "stream": False,
            "options": options,
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            message = data.get("message", {})
            return message.get("content", "").strip()
        except requests.ConnectionError as ce:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. Please ensure Ollama is running."
            ) from ce
        except requests.Timeout as te:
            raise TimeoutError(
                f"Ollama chat timed out after {self.timeout} seconds."
            ) from te
        except Exception as e:
            raise RuntimeError(f"Ollama chat failed: {e}") from e
