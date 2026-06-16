"""YAML memory initialization helpers for sandbox startup and reset."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .live_card_memory import (
    CLIENT_MEMORY_OWNER,
    LAWYER_MEMORY_OWNER,
    bootstrap_memory_from_legacy,
    build_default_memory,
    has_meaningful_memory,
    normalize_memory_payload,
    render_memory_yaml,
)

if TYPE_CHECKING:
    from ..core.file_storage_manager import FileStorageManager

logger = logging.getLogger(__name__)


def _normalize_case_key(case_id: Any) -> str:
    raw = str(case_id or "").strip()
    if not raw:
        return "case_unknown"
    return raw if raw.startswith("case_") else f"case_{raw}"


def _resolve_client_slot(client_dir: Path) -> str:
    if client_dir.parent.name.startswith("case_") and client_dir.name in {"plaintiff", "defendant"}:
        return f"{client_dir.parent.name}_{client_dir.name}"
    return client_dir.name


def _resolve_lawyer_slot(config: dict[str, Any], lawyer_dir: Path) -> str:
    profile = config.get("profile", {})
    if isinstance(profile, dict):
        lawyer_id = str(profile.get("lawyer_id", "") or "").strip()
        if lawyer_id:
            return lawyer_id
    return lawyer_dir.name


def _memory_path(storage: "FileStorageManager", case_key: str, slot: str) -> Path:
    return Path(storage.base_dir) / "output" / case_key / slot / "memory.yaml"


def _load_existing_yaml(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        import yaml

        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def initialize_client_memory(storage: "FileStorageManager", client_path: str) -> None:
    """Create client memory.yaml when missing, optionally migrating legacy config memory."""
    client_dir = Path(client_path)
    config = storage.load_agent_config(client_dir)
    case_key = _normalize_case_key(config.get("case_id"))
    slot = _resolve_client_slot(client_dir)
    memory_path = _memory_path(storage, case_key, slot)

    existing_payload = _load_existing_yaml(memory_path)
    if has_meaningful_memory(existing_payload or {}):
        return

    legacy_payload = config.get("long_term_memory", {}) if isinstance(config.get("long_term_memory"), dict) else {}
    if legacy_payload:
        payload = bootstrap_memory_from_legacy(CLIENT_MEMORY_OWNER, legacy_payload)
        default_payload = build_default_memory(CLIENT_MEMORY_OWNER, None)
        seeded_self_narrative = str(default_payload["case_knowledge"].get("self_narrative", "") or "").strip()
        if not payload["case_knowledge"]["self_narrative"] and seeded_self_narrative:
            payload["case_knowledge"]["self_narrative"] = seeded_self_narrative
    else:
        payload = build_default_memory(CLIENT_MEMORY_OWNER, None)

    dataset_path = str(config.get("dataset_path", "") or "").strip()
    if dataset_path and not payload["case_knowledge"]["self_narrative"]:
        try:
            from ..data.data_loader import DataLoader

            data_loader = DataLoader(dataset_path)
            case = data_loader.resolve_case_for_config(config)
            payload["case_knowledge"]["self_narrative"] = str(
                data_loader.extract_case_background(case) or ""
            ).strip()
        except Exception as exc:
            logger.warning("[MemoryInit] Failed to seed client self_narrative for %s: %s", client_path, exc)

    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text(
        render_memory_yaml(CLIENT_MEMORY_OWNER, payload),
        encoding="utf-8",
    )
    logger.info("[MemoryInit] Client %s: memory.yaml initialized at %s", client_path, memory_path)


def initialize_lawyer_memory(storage: "FileStorageManager", lawyer_path: str) -> None:
    """Create lawyer memory.yaml when missing, optionally migrating legacy config memory."""
    lawyer_dir = Path(lawyer_path)
    config = storage.load_agent_config(lawyer_dir)
    current_case_id = str(config.get("current_handling_case", "") or config.get("case_id", "")).strip()
    if not current_case_id:
        return
    case_key = _normalize_case_key(current_case_id)
    slot = _resolve_lawyer_slot(config, lawyer_dir)
    memory_path = _memory_path(storage, case_key, slot)

    existing_payload = _load_existing_yaml(memory_path)
    if has_meaningful_memory(existing_payload or {}):
        return

    legacy_payload = config.get("long_term_memory", {}) if isinstance(config.get("long_term_memory"), dict) else {}
    payload = (
        bootstrap_memory_from_legacy(LAWYER_MEMORY_OWNER, legacy_payload)
        if legacy_payload
        else build_default_memory(LAWYER_MEMORY_OWNER, None)
    )
    payload = normalize_memory_payload(LAWYER_MEMORY_OWNER, payload)

    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text(
        render_memory_yaml(LAWYER_MEMORY_OWNER, payload),
        encoding="utf-8",
    )
    logger.info("[MemoryInit] Lawyer %s: memory.yaml initialized at %s", lawyer_path, memory_path)


