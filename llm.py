"""Gemini adapter for the bounded agent loop.

Only tool *declarations* are given to Gemini and automatic function calling is
disabled, so the model can only request a tool; agent.py decides whether to run
it and executes it itself.
"""
from __future__ import annotations

from typing import Any

from agent import ModelError, ModelReply, ToolCall

DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_FALLBACK_MODEL = "gemini-4-hydrogen"


class GeminiChat:
    def __init__(self, api_key: str, model_name: str, system_prompt: str,
                 declarations: list[dict[str, Any]], timeout_seconds: int = 120):
        if not api_key:
            raise ModelError(
                "not_configured",
                "No Gemini API key is configured on the server (GEMINI_API_KEY secret).",
            )
        from google import genai
        from google.genai import types

        self._types = types
        self.model_name = model_name
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=timeout_seconds * 1000,
                # Retry only temporary server overloads; never retry 429 quota errors.
                retry_options=types.HttpRetryOptions(
                    attempts=3, initial_delay=2.0, max_delay=8.0, http_status_codes=[500, 502, 503, 504]
                ),
            ),
        )
        self.config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2,
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=d["name"],
                            description=d["description"],
                            parameters_json_schema=d["parameters"],
                        )
                        for d in declarations
                    ]
                )
            ],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        self.contents: list[Any] = []

    # -- ChatModel interface ------------------------------------------------
    def send_text(self, text: str) -> ModelReply:
        types = self._types
        self.contents.append(types.Content(role="user", parts=[types.Part.from_text(text=text)]))
        return self._generate()

    def send_tool_results(self, results: list[tuple[str, dict[str, Any]]]) -> ModelReply:
        types = self._types
        parts = [types.Part.from_function_response(name=name, response=payload) for name, payload in results]
        self.contents.append(types.Content(role="user", parts=parts))
        return self._generate()

    # -- internals ------------------------------------------------------------
    def _generate(self) -> ModelReply:
        from google.genai import errors

        try:
            response = self.client.models.generate_content(
                model=self.model_name, contents=self.contents, config=self.config
            )
        except errors.APIError as error:
            raise _translate_api_error(error, self.model_name) from error
        except Exception as error:  # network problems, timeouts from the HTTP layer
            if "timeout" in type(error).__name__.lower() or "timed out" in str(error).lower():
                raise ModelError("timeout", "The model service did not answer in time.") from error
            raise ModelError("unavailable", f"Could not reach the model service ({type(error).__name__}).") from error

        candidate = response.candidates[0] if response.candidates else None
        if candidate is None or candidate.content is None or not candidate.content.parts:
            reason = getattr(candidate, "finish_reason", None) if candidate else None
            raise ModelError("empty", f"The model returned an empty reply (finish reason: {reason}).")

        # Keep the model's own turn (including any thought signatures) in the history.
        self.contents.append(candidate.content)
        calls = [ToolCall(name=fc.name or "", args=dict(fc.args or {})) for fc in (response.function_calls or [])]
        text = "".join(
            part.text for part in candidate.content.parts if getattr(part, "text", None) and not getattr(part, "thought", False)
        )
        return ModelReply(text=text.strip(), calls=calls)


def _translate_api_error(error: Any, model_name: str) -> ModelError:
    code = getattr(error, "code", None)
    if code == 429:
        return ModelError("quota", "The free-tier quota or rate limit for the model was reached (HTTP 429).")
    if code in (401, 403):
        return ModelError("auth", f"The model service rejected the server's credential (HTTP {code}).")
    if code == 404:
        return ModelError("model_not_found", f"Model '{model_name}' is not available for this key (HTTP 404).")
    if code == 400 and "api key" in str(error).lower():
        return ModelError("auth", "The configured API key is not valid (HTTP 400).")
    if isinstance(code, int) and code >= 500:
        return ModelError("unavailable", f"The model service is temporarily unavailable (HTTP {code}).")
    return ModelError("bad_request", f"The model service returned an error (HTTP {code}).")


class UnavailableModel:
    """Stand-in used by the in-app 'simulate outage' switch to test error handling."""

    def send_text(self, text: str) -> ModelReply:
        raise ModelError("unavailable", "Simulated outage: the model service is switched off for this demo run.")

    def send_tool_results(self, results):
        raise ModelError("unavailable", "Simulated outage: the model service is switched off for this demo run.")
