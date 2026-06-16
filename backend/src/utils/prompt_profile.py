"""Prompt profile runtime helpers."""

from __future__ import annotations

import os


PROMPT_PROFILE_ENV = "SIMLAW_PROMPT_PROFILE"
PROMPT_PROFILE_PROD = "prod"
PROMPT_PROFILE_TEST = "test"
_SUPPORTED_PROMPT_PROFILES = {
    PROMPT_PROFILE_PROD,
    PROMPT_PROFILE_TEST,
}

_TEST_PROFILE_MAX_TURNS = {
    "LC": 12,
    "CD": 15,
    "DD": 15,
    "AD": 15,
    "AR": 15,
    "CI": 6,
    "CIA": 6,
    "RECEPTION": 2,
}

TEST_STAGE_SUMMARY_MAX_CHARS = 48


def get_prompt_profile() -> str:
    raw_value = str(os.getenv(PROMPT_PROFILE_ENV, PROMPT_PROFILE_PROD) or "").strip().lower()
    if raw_value in _SUPPORTED_PROMPT_PROFILES:
        return raw_value
    return PROMPT_PROFILE_PROD


def is_test_prompt_profile() -> bool:
    return get_prompt_profile() == PROMPT_PROFILE_TEST


def resolve_prompt_profile_max_turns(stage_code: str, prod_default: int) -> int:
    if not is_test_prompt_profile():
        return prod_default

    normalized_stage_code = str(stage_code or "").strip().upper()
    return _TEST_PROFILE_MAX_TURNS.get(normalized_stage_code, prod_default)


def use_lightweight_stage_summary() -> bool:
    return is_test_prompt_profile()


def use_lightweight_eval_judge_prompt() -> bool:
    return is_test_prompt_profile()
