import asyncio
import json
import os
from pathlib import Path
from typing import Any, Optional

from openai import AsyncOpenAI, OpenAI


MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
API_BASE_URL = os.getenv("OPENAI_BASE_URL")
SYSTEM_PROMPT = os.getenv(
    "LEGAL_DATASET_SYSTEM_PROMPT",
    "你是一个专业的法律文书分析助手，擅长从裁判文书中提取结构化信息。",
)
MAX_RETRIES = int(os.getenv("LEGAL_DATASET_MAX_RETRIES", "3"))
MAX_TOKENS = int(os.getenv("LEGAL_DATASET_MAX_TOKENS", "16384"))
TEMPERATURE = float(os.getenv("LEGAL_DATASET_TEMPERATURE", "0"))


def configure(
    model_name: Optional[str] = None,
    api_base_url: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> None:
    """Override runtime LLM settings without editing source files."""
    global MODEL_NAME, API_BASE_URL, SYSTEM_PROMPT
    if model_name:
        MODEL_NAME = model_name
    if api_base_url:
        API_BASE_URL = api_base_url
    if system_prompt:
        SYSTEM_PROMPT = system_prompt


def load_json(file_path: str | Path) -> Any:
    with Path(file_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, save_path: str | Path) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def _build_messages(prompt: str, history: Optional[list[tuple[str, str]]] = None) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": prompt})
    return messages


def _build_request_kwargs(
    messages: list[dict[str, str]],
    flag: int,
    model_name: Optional[str] = None,
) -> dict[str, Any]:
    request_kwargs: dict[str, Any] = {
        "messages": messages,
        "model": model_name or MODEL_NAME,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }
    if flag == 1:
        request_kwargs["response_format"] = {"type": "json_object"}
    return request_kwargs


def _client_kwargs() -> dict[str, str]:
    kwargs: dict[str, str] = {}
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        kwargs["api_key"] = api_key
    if API_BASE_URL:
        kwargs["base_url"] = API_BASE_URL
    return kwargs


async def aget_completion(
    prompt: str,
    history: Optional[list[tuple[str, str]]] = None,
    flag: int = 1,
    model_name: Optional[str] = None,
) -> tuple[str, list[tuple[str, str]]]:
    if history is None:
        history = []

    client = AsyncOpenAI(**_client_kwargs())
    messages = _build_messages(prompt, history)
    last_error: Optional[Exception] = None

    try:
        for attempt in range(MAX_RETRIES):
            try:
                chat_completion = await client.chat.completions.create(
                    **_build_request_kwargs(messages, flag, model_name=model_name)
                )
                response = chat_completion.choices[0].message.content or ""
                history.append((prompt, response))
                return response, history
            except Exception as exc:
                last_error = exc
                if attempt == MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(1.5 * (attempt + 1))
    finally:
        await client.close()

    raise RuntimeError("LLM request failed") from last_error


def get_completion(
    prompt: str,
    history: Optional[list[tuple[str, str]]] = None,
    flag: int = 1,
    model_name: Optional[str] = None,
) -> tuple[str, list[tuple[str, str]]]:
    if history is None:
        history = []

    client = OpenAI(**_client_kwargs())
    messages = _build_messages(prompt, history)
    last_error: Optional[Exception] = None

    try:
        for attempt in range(MAX_RETRIES):
            try:
                chat_completion = client.chat.completions.create(
                    **_build_request_kwargs(messages, flag, model_name=model_name)
                )
                response = chat_completion.choices[0].message.content or ""
                history.append((prompt, response))
                return response, history
            except Exception as exc:
                last_error = exc
                if attempt == MAX_RETRIES - 1:
                    raise
                import time

                time.sleep(1.5 * (attempt + 1))
    finally:
        client.close()

    raise RuntimeError("LLM request failed") from last_error
