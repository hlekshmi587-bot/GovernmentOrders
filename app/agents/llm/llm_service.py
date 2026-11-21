"""
LLM Service for Structured Prompting and JSON Output Parsing.

This module provides high-level reasoning capabilities for AI agents. It ensures that
the local LLM returns reliably formatted structured JSON data required for autonomous
planning, decision making, verification, and tool selection.
"""

import json
import re
from typing import Any, Dict, Optional, Type, TypeVar
from pydantic import BaseModel, ValidationError

from app.agents.llm.ollama_client import OllamaClient

T = TypeVar("T", bound=BaseModel)


class LLMService:
    """
    High-level LLM service providing structured reasoning and robust JSON parsing.
    """

    def __init__(
        self,
        client: Optional[OllamaClient] = None,
        model: str = "llama3.2:3b",
    ) -> None:
        """
        Initialize the LLM service.

        Args:
            client: An existing OllamaClient instance, or creates a default one.
            model: Target local LLM name.
        """
        self.client = client or OllamaClient(default_model=model)
        self.model = model

    @staticmethod
    def extract_json_string(raw_text: str) -> str:
        """
        Extract a valid JSON substring from LLM response text.
        Handles cases where the LLM wraps JSON in markdown blocks (```json ... ```)
        or includes introductory/closing remarks.

        Args:
            raw_text: The complete output from the LLM.

        Returns:
            The extracted JSON string candidate.
        """
        text = raw_text.strip()

        # 1. Look for markdown code fence with json tag
        markdown_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if markdown_match:
            return markdown_match.group(1).strip()

        # 2. Look for first '{' and last '}'
        start_brace = text.find("{")
        end_brace = text.rfind("}")
        if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
            return text[start_brace : end_brace + 1].strip()

        # 3. Look for first '[' and last ']' (in case of JSON array)
        start_bracket = text.find("[")
        end_bracket = text.rfind("]")
        if start_bracket != -1 and end_bracket != -1 and end_bracket > start_bracket:
            return text[start_bracket : end_bracket + 1].strip()

        return text

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        num_predict: Optional[int] = None,
        num_ctx: Optional[int] = None,
    ) -> str:
        """
        Generate plain text using the LLM.

        Args:
            prompt: User/agent instruction.
            system_prompt: Role/system instructions.
            temperature: Sampling temperature.
            num_predict: Optional maximum number of tokens to predict.
            num_ctx: Optional context window size in tokens.

        Returns:
            Response text string.
        """
        client_kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "model": self.model,
            "temperature": temperature,
        }
        if num_predict is not None:
            client_kwargs["num_predict"] = num_predict
        if num_ctx is not None:
            client_kwargs["num_ctx"] = num_ctx

        return self.client.generate(**client_kwargs)

    def generate_structured(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        schema_model: Optional[Type[T]] = None,
        temperature: float = 0.1,
        max_retries: int = 2,
        num_predict: Optional[int] = None,
        num_ctx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Prompt the LLM and return guaranteed parsed JSON dictionary.
        Optionally validates against a Pydantic schema model.

        Args:
            prompt: The specific task prompt.
            system_prompt: Guidelines instructing the LLM to output valid JSON.
            schema_model: Optional Pydantic model class for schema validation.
            temperature: Sampling temperature (default 0.1 for high determinism).
            max_retries: Number of retry attempts if JSON is invalid.
            num_predict: Optional maximum number of tokens to predict.
            num_ctx: Optional context window size in tokens.

        Returns:
            Parsed dictionary of the JSON response.

        Raises:
            ValueError: If the LLM output cannot be parsed into valid JSON after retries.
        """
        json_guidance = (
            "\n\nIMPORTANT: Respond ONLY with a valid JSON object. "
            "Do NOT include any explanatory text, markdown notes, or conversation outside the JSON block."
        )

        full_prompt = prompt + json_guidance

        system_instruction = (
            (system_prompt + "\n" if system_prompt else "")
            + "You are an analytical AI system. You always respond strictly with valid JSON conforming to the requested structure."
        )

        last_error = ""
        current_prompt = full_prompt

        client_kwargs: Dict[str, Any] = {
            "prompt": current_prompt,
            "system_prompt": system_instruction,
            "model": self.model,
            "temperature": temperature,
            "format": "json",
        }
        if num_predict is not None:
            client_kwargs["num_predict"] = num_predict
        if num_ctx is not None:
            client_kwargs["num_ctx"] = num_ctx

        for attempt in range(max_retries + 1):
            client_kwargs["prompt"] = current_prompt
            raw_response = self.client.generate(**client_kwargs)

            json_str = self.extract_json_string(raw_response)

            try:
                parsed_data = json.loads(json_str)

                # Validate with Pydantic schema if provided
                if schema_model is not None:
                    validated = schema_model.model_validate(parsed_data)
                    return validated.model_dump()

                if isinstance(parsed_data, dict):
                    return parsed_data
                elif isinstance(parsed_data, list):
                    return {"items": parsed_data}
                else:
                    raise ValueError(f"Expected JSON object or array, got {type(parsed_data).__name__}")

            except (json.JSONDecodeError, ValidationError, ValueError) as err:
                last_error = str(err)
                # If we have retry attempts left, instruct the model to correct its previous output
                if attempt < max_retries:
                    current_prompt = (
                        f"{full_prompt}\n\n"
                        f"Your previous response caused this parsing error: {last_error}\n"
                        f"Previous invalid output was:\n{raw_response}\n\n"
                        f"Please output ONLY valid, corrected JSON now."
                    )

        raise ValueError(
            f"Failed to produce valid structured JSON from LLM after {max_retries + 1} attempts. Error: {last_error}"
        )
