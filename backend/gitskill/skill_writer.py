from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import shutil

import yaml

from camel.toolkits import FunctionTool

from .reflection_manager import sanitize_path_component


def _split_frontmatter(contents: str) -> tuple[Optional[str], str]:
    lines = contents.splitlines()
    if not lines or lines[0].strip() != "---":
        return None, contents

    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return "\n".join(lines[1:index]), "\n".join(lines[index + 1 :])

    return None, contents


def _validate_skill_markdown(skill_markdown: str) -> tuple[dict[str, Any], str]:
    contents = str(skill_markdown or "").strip()
    frontmatter_text, body = _split_frontmatter(contents)
    if frontmatter_text is None:
        raise ValueError("SKILL.md must start with YAML frontmatter.")

    metadata = yaml.safe_load(frontmatter_text) or {}
    if not isinstance(metadata, dict):
        raise ValueError("Skill frontmatter must be a YAML mapping.")

    name = metadata.get("name")
    description = metadata.get("description")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Skill frontmatter requires a non-empty name.")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("Skill frontmatter requires a non-empty description.")
    if not body.strip():
        raise ValueError("Skill body cannot be empty.")

    return metadata, body.strip()


def _normalize_relative_skill_dir(relative_skill_dir: str) -> str:
    raw = str(relative_skill_dir or "").strip().replace("\\", "/")
    if not raw:
        raise ValueError("relative_skill_dir is required.")

    normalized = Path(raw)
    if normalized.is_absolute():
        raise ValueError("relative_skill_dir must be relative.")

    parts = [part for part in normalized.parts if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise ValueError("relative_skill_dir cannot escape the private skill root.")
    if parts[-1].lower() == "skill.md":
        raise ValueError("relative_skill_dir must point to a directory, not SKILL.md.")

    return "/".join(parts)


@dataclass
class SkillWriter:
    """Create or update a single SKILL.md under the private skill root."""

    private_skill_root: Path
    case_cause: str
    main_skill_root: Optional[Path] = None
    last_result: Optional[dict[str, Any]] = field(default=None, init=False)
    results: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.private_skill_root = Path(self.private_skill_root).resolve()
        self.main_skill_root = Path(self.main_skill_root).resolve() if self.main_skill_root else None
        self.case_cause_dir = sanitize_path_component(self.case_cause, fallback="unknown_case_cause")

    def _build_tool(self) -> FunctionTool:
        tool = FunctionTool(self.upsert_skill_file)
        schema = tool.get_openai_tool_schema()
        schema["function"]["description"] = (
            "Create or update one private skill folder under the current case-cause directory. "
            f"The relative_skill_dir must start with '{self.case_cause_dir}/'. "
            "You may call this tool multiple times in one reflection if you decide to write multiple skills. "
            "If action=update and the private copy does not exist yet, the system will first look for the same "
            "skill under main and copy that baseline into private before writing the new version."
        )
        tool.set_openai_tool_schema(schema)
        return tool

    def get_tool(self) -> FunctionTool:
        return self._build_tool()

    def reset(self) -> None:
        self.last_result = None
        self.results = []

    def has_successful_results(self) -> bool:
        return any(item.get("status") == "success" for item in self.results)

    def has_error_results(self) -> bool:
        return any(item.get("status") == "error" for item in self.results)

    def _resolve_private_target(self, parts: list[str]) -> Path:
        return self.private_skill_root / Path(*parts) / "SKILL.md"

    def _resolve_main_source(self, parts: list[str]) -> Optional[Path]:
        if self.main_skill_root is None:
            return None
        candidate = self.main_skill_root / Path(*parts) / "SKILL.md"
        if candidate.exists():
            return candidate
        return None

    def upsert_skill_file(self, action: str, relative_skill_dir: str, skill_markdown: str) -> str:
        """Create or update one SKILL.md file.

        Args:
            action: Either "new" or "update".
            relative_skill_dir: Relative directory under the private skill root.
            skill_markdown: Complete SKILL.md contents including frontmatter.
        """
        normalized_action = str(action or "").strip().lower()
        if normalized_action not in {"new", "update"}:
            self.last_result = {
                "status": "error",
                "action": normalized_action,
                "error": "action must be either 'new' or 'update'.",
            }
            self.results.append(dict(self.last_result))
            return "Error: action must be either 'new' or 'update'."

        try:
            normalized_dir = _normalize_relative_skill_dir(relative_skill_dir)
            parts = normalized_dir.split("/")
            if not parts or parts[0] != self.case_cause_dir:
                return (
                    "Error: relative_skill_dir must stay under the current case-cause directory "
                    f"'{self.case_cause_dir}/'."
                )

            _validate_skill_markdown(skill_markdown)

            target_dir = self.private_skill_root / Path(*parts)
            target_path = self._resolve_private_target(parts)
            copied_from_main = False
            main_source_path = self._resolve_main_source(parts)

            if normalized_action == "new" and target_path.exists():
                return f"Error: target skill already exists at {target_path}."
            if normalized_action == "update" and not target_path.exists():
                if main_source_path is None:
                    return (
                        "Error: target skill does not exist in private, and no same-path baseline was found in main. "
                        f"private_target={target_path}"
                    )
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(main_source_path, target_path)
                copied_from_main = True

            target_dir.mkdir(parents=True, exist_ok=True)
            target_path.write_text(str(skill_markdown or "").strip() + "\n", encoding="utf-8")

            self.last_result = {
                "status": "success",
                "action": normalized_action,
                "relative_skill_dir": normalized_dir,
                "skill_path": str(target_path),
                "copied_from_main": copied_from_main,
                "main_source_path": str(main_source_path) if main_source_path else None,
            }
            self.results.append(dict(self.last_result))
            return f"Success: {normalized_action} skill at {target_path}"
        except Exception as exc:
            self.last_result = {
                "status": "error",
                "action": normalized_action,
                "error": str(exc),
            }
            self.results.append(dict(self.last_result))
            return f"Error: {exc}"
