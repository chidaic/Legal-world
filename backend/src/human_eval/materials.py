from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


class HumanEvalMaterialNotFoundError(FileNotFoundError):
    """Raised when a requested human evaluation material is missing."""


class HumanEvalMaterials:
    """Read-only access to prepared human-evaluation case materials."""

    def __init__(self, root: Path | None = None) -> None:
        backend_dir = Path(__file__).resolve().parents[2]
        self.root = root or backend_dir / "data" / "human_eval"

    @staticmethod
    def read_json(path: Path) -> Any:
        text = path.read_text(encoding="utf-8-sig")
        return json.loads(text)

    def _resolve(self, relative_path: str) -> Path:
        path = (self.root / relative_path).resolve()
        root = self.root.resolve()
        if root not in path.parents and path != root:
            raise HumanEvalMaterialNotFoundError(f"非法评测材料路径: {relative_path}")
        if not path.exists():
            raise HumanEvalMaterialNotFoundError(f"评测材料不存在: {relative_path}")
        return path

    @lru_cache(maxsize=1)
    def load_manifest(self) -> dict[str, Any]:
        return self.read_json(self._resolve("manifest.json"))

    @lru_cache(maxsize=1)
    def list_cases(self) -> list[dict[str, Any]]:
        payload = self.read_json(self._resolve("case_flow_index.json"))
        if not isinstance(payload, list):
            raise ValueError("case_flow_index.json must contain a list")
        return payload

    @lru_cache(maxsize=1)
    def load_schema(self) -> dict[str, Any]:
        payload = self.read_json(self._resolve("questionnaire_schema.json"))
        if not isinstance(payload, dict):
            raise ValueError("questionnaire_schema.json must contain an object")
        return payload

    def load_case(self, case_id: int | str) -> dict[str, Any]:
        normalized_case_id = int(case_id)
        path = self._resolve(f"cases/case_{normalized_case_id}_flow.json")
        payload = self.read_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"case_{normalized_case_id}_flow.json must contain an object")
        return payload

    def read_questionnaire_template(self) -> str:
        return self._resolve("questionnaire_sheet.csv").read_text(encoding="utf-8-sig")
