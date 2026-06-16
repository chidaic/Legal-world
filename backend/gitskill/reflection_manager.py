from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from src.utils.file_io import write_json


_INVALID_PATH_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1F]')


def sanitize_path_component(value: str, fallback: str = "unknown") -> str:
    """Return a filesystem-safe path component while preserving Chinese text."""
    cleaned = _INVALID_PATH_CHARS_RE.sub("_", str(value or "").strip())
    cleaned = cleaned.rstrip(". ").strip()
    return cleaned or fallback


@dataclass
class ReflectionManager:
    """Manage one reflection.md and its sidecar JSON files for a single case."""

    reflection_root: Path
    case_cause: str
    case_id: str

    def __post_init__(self) -> None:
        self.reflection_root = Path(self.reflection_root).resolve()
        self.case_cause_dir = sanitize_path_component(self.case_cause, fallback="unknown_case_cause")
        self.case_id_dir = sanitize_path_component(self.case_id, fallback="unknown_case")

    @property
    def case_dir(self) -> Path:
        case_prefix = self.case_id_dir
        if not case_prefix.lower().startswith("case_"):
            case_prefix = f"case_{case_prefix}"
        return self.reflection_root / f"{case_prefix}__{self.case_cause_dir}"

    @property
    def reflection_path(self) -> Path:
        return self.case_dir / "reflection.md"

    @property
    def developer_trace_path(self) -> Path:
        return self.case_dir / "trace.json"

    @property
    def result_path(self) -> Path:
        return self.case_dir / "result.json"

    def exists(self) -> bool:
        return self.reflection_path.exists()

    def read_reflection(self, default: str = "") -> str:
        if not self.reflection_path.exists():
            return default
        return self.reflection_path.read_text(encoding="utf-8")

    def write_reflection(self, markdown: str) -> Path:
        self.case_dir.mkdir(parents=True, exist_ok=True)
        self.reflection_path.write_text(str(markdown or "").strip() + "\n", encoding="utf-8")
        return self.reflection_path

    def delete_reflection(self) -> bool:
        if not self.reflection_path.exists():
            return False
        self.reflection_path.unlink()
        return True

    def write_developer_trace(self, payload: dict[str, Any]) -> Path:
        self.case_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.developer_trace_path, payload)
        return self.developer_trace_path

    def write_result(self, payload: dict[str, Any]) -> Path:
        self.case_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.result_path, payload)
        return self.result_path
