"""Runtime compatibility patches for third-party chat agents."""

from __future__ import annotations

from typing import Any


def _normalize_token_usage_value(value: Any) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def normalize_usage_dict(usage_dict: Any) -> dict[str, Any]:
    usage = dict(usage_dict or {})
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        usage[key] = _normalize_token_usage_value(usage.get(key))
    return usage


def patch_chat_agent_usage_cache(chat_agent: Any) -> None:
    """Patch token cache updates to tolerate providers returning None usage fields."""
    if chat_agent is None or getattr(chat_agent, "_simlaw_usage_cache_patch", False):
        return

    original = getattr(chat_agent, "_update_token_cache", None)
    if not callable(original):
        return

    def _safe_update_token_cache(usage_dict: Any, message_count: int) -> Any:
        return original(normalize_usage_dict(usage_dict), message_count)

    setattr(chat_agent, "_update_token_cache", _safe_update_token_cache)
    setattr(chat_agent, "_simlaw_usage_cache_patch", True)


def patch_chat_agent_usage_serialization() -> None:
    """Patch CAMEL usage serialization so usage dicts never carry None token fields."""
    try:
        import camel.agents.chat_agent as chat_agent_module
    except Exception:
        return

    if getattr(chat_agent_module, "_simlaw_safe_model_dump_patch", False):
        return

    original = getattr(chat_agent_module, "safe_model_dump", None)
    if not callable(original):
        return

    def _safe_model_dump_with_normalized_usage(value: Any, *args: Any, **kwargs: Any) -> Any:
        dumped = original(value, *args, **kwargs)
        if isinstance(dumped, dict):
            return normalize_usage_dict(dumped)
        return dumped

    setattr(chat_agent_module, "safe_model_dump", _safe_model_dump_with_normalized_usage)
    setattr(chat_agent_module, "_simlaw_safe_model_dump_patch", True)
