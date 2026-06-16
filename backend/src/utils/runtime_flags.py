"""Runtime flags for controlling verbosity and debug output."""

from __future__ import annotations

import os


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
_VALID_FRONTEND_MODES = {"auto", "legacy", "player_v2"}


def env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable with a conservative fallback."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return default


def scenario_verbose_enabled() -> bool:
    """Whether scenarios should print every dialogue turn to stdout."""
    return env_flag("SIMLAW_VERBOSE_SCENARIOS", default=False)


def system_prompt_print_enabled() -> bool:
    """Whether agent activation should dump full system prompts to stdout."""
    return env_flag("SIMLAW_PRINT_SYSTEM_PROMPTS", default=False)


def stage_summary_enabled() -> bool:
    """Whether per-stage chat summaries should be generated and injected."""
    return env_flag("SIMLAW_ENABLE_STAGE_SUMMARY", default=False)


def player_lawyer_ai_surrogate_enabled() -> bool:
    """Whether AI should explicitly act as the current-side player lawyer."""
    return env_flag("SIMLAW_PLAYER_LAWYER_AI_SURROGATE", default=False)


def player_lawyer_mode() -> str:
    """Return the active player-lawyer mode string (e.g. 'plaintiff').

    Empty string means the feature is disabled (AI-only mode).
    """
    return os.environ.get("SIMLAW_PLAYER_LAWYER_MODE", "").strip().lower()


def normalize_frontend_mode(value: str | None) -> str:
    """Normalize frontend runtime mode names used by backend feature gates."""
    normalized = str(value or "").strip().lower().replace("-", "_")
    return normalized if normalized in _VALID_FRONTEND_MODES else "auto"


def player_lawyer_mode_for_frontend(
    *,
    frontend_mode: str | None = None,
    has_player_v2_client: bool = False,
) -> str:
    """Return player-lawyer mode only when the runtime is allowed to be player-v2."""
    mode = player_lawyer_mode()
    if mode != "plaintiff":
        return ""

    resolved_frontend_mode = normalize_frontend_mode(
        frontend_mode if frontend_mode is not None else os.environ.get("SIMLAW_FRONTEND_MODE", "auto")
    )
    if resolved_frontend_mode == "player_v2":
        return mode
    if resolved_frontend_mode == "auto" and has_player_v2_client:
        return mode
    return ""
