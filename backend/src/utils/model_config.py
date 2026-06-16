"""Runtime model configuration helpers."""

from __future__ import annotations

import os
from typing import Any


DEFAULT_RUNTIME_OPENAI_MODEL = "qwen3.5-flash"
_ENABLE_THINKING_MODEL_PREFIXES = ("qwen",)


def _normalize_model_name(value: Any) -> str:
    return str(value or "").strip()


def _supports_enable_thinking_toggle(model_name: Any) -> bool:
    """Return whether the model family accepts Qwen-style enable_thinking."""
    normalized = _normalize_model_name(model_name).lower()
    if not normalized:
        return False
    return normalized.startswith(_ENABLE_THINKING_MODEL_PREFIXES)


def resolve_openai_chat_model(
    explicit_model: Any = None,
    *,
    env_var: str = "OPENAI_MODEL_NAME",
    default_model: str = DEFAULT_RUNTIME_OPENAI_MODEL,
) -> str:
    """Resolve the runtime chat model with explicit override precedence.

    Order:
    1. explicit fallback passed by caller
    2. environment variable
    3. repository runtime default
    """
    explicit = _normalize_model_name(explicit_model)
    if explicit:
        return explicit

    env_model = _normalize_model_name(os.environ.get(env_var))
    if env_model:
        return env_model

    return _normalize_model_name(default_model) or DEFAULT_RUNTIME_OPENAI_MODEL


def build_runtime_openai_chat_config(
    *,
    model_name: Any = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Build runtime chat config for OpenAI-compatible backends.

    Only inject provider-specific reasoning toggles for model families that
    are known to accept them. This avoids passing non-standard parameters
    like ``enable_thinking`` to models such as ``gpt-5-mini``.
    """
    config: dict[str, Any] = {}
    if temperature is not None:
        config["temperature"] = temperature
    if max_tokens is not None:
        config["max_tokens"] = max_tokens
    if _supports_enable_thinking_toggle(model_name):
        config["extra_body"] = {"enable_thinking": False}
    return config
