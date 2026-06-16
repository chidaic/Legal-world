"""Utilities for loading flat CAMEL-style SKILL files as a common tool."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import yaml

logger = logging.getLogger(__name__)


def normalize_skill_dirs(skill_dirs: Iterable[str | Path | None]) -> list[str]:
    """Normalize and deduplicate skill directories while preserving order."""
    normalized: list[str] = []
    seen: set[str] = set()

    for skill_dir in skill_dirs:
        if not skill_dir:
            continue
        resolved = str(Path(skill_dir).resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        normalized.append(resolved)

    return normalized


def _resolve_skills_root(base_dir: str | Path) -> Optional[Path]:
    """Resolve a user-provided path to an actual skills root."""
    resolved = Path(base_dir).resolve()
    if not resolved.is_dir():
        return None

    return resolved


class _FlatSkillToolkit:
    """Merge multiple flat skill trees into a single toolkit."""

    def __init__(
        self,
        skill_dirs: list[str],
        usage_recorder: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> None:
        self.skill_dirs = normalize_skill_dirs(skill_dirs)
        self._usage_recorder = usage_recorder
        self._skills_by_name: dict[str, dict[str, Any]] = {}
        self._scan_skills()

    def has_skills(self) -> bool:
        return bool(self._skills_by_name)

    def _scan_skills(self) -> None:
        for base_dir in self.skill_dirs:
            skills_root = _resolve_skills_root(base_dir)
            if skills_root is None:
                logger.debug("[SkillLoader] No visible skills root found in %s, skipping", base_dir)
                continue

            for skill_path in sorted(skills_root.rglob("SKILL.md")):
                if self._is_hidden_path(skill_path, skills_root):
                    continue

                parsed = self._parse_skill(skill_path)
                if not parsed:
                    continue

                skill_rel_path = str(skill_path.parent.relative_to(skills_root)).replace("\\", "/").strip(".")
                skill_rel_path = skill_rel_path.strip("/")

                skill_record = {
                    "name": parsed["name"],
                    "description": parsed["description"],
                    "body": parsed["body"],
                    "path": str(skill_path),
                    "base_dir": str(skill_path.parent),
                    "source_root": str(Path(base_dir).resolve()),
                    "skill_path": skill_rel_path,
                }

                existing = self._skills_by_name.get(parsed["name"])
                if existing:
                    logger.info(
                        "[SkillLoader] Skill '%s' overridden by %s",
                        parsed["name"],
                        skill_path,
                    )

                self._skills_by_name[parsed["name"]] = skill_record

    @staticmethod
    def _is_hidden_path(path: Path, root: Path) -> bool:
        try:
            relative = path.relative_to(root)
        except ValueError:
            relative = path
        return any(part.startswith(".") for part in relative.parts[:-1])

    @staticmethod
    def _split_frontmatter(contents: str) -> tuple[Optional[str], str]:
        normalized = contents.lstrip("\ufeff")
        lines = normalized.splitlines()
        if not lines or lines[0].strip() != "---":
            return None, contents

        for index in range(1, len(lines)):
            if lines[index].strip() == "---":
                return "\n".join(lines[1:index]), "\n".join(lines[index + 1 :])

        return None, contents

    def _parse_skill(self, path: Path) -> Optional[dict[str, str]]:
        try:
            contents = path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            logger.warning("[SkillLoader] Failed to read %s: %s", path, exc)
            return None

        frontmatter_text, body = self._split_frontmatter(contents)
        if frontmatter_text is None:
            logger.warning("[SkillLoader] Missing YAML frontmatter in %s", path)
            return None

        try:
            data = yaml.safe_load(frontmatter_text) or {}
        except yaml.YAMLError as exc:
            logger.warning("[SkillLoader] Invalid YAML frontmatter in %s: %s", path, exc)
            return None

        if not isinstance(data, dict):
            logger.warning("[SkillLoader] Frontmatter must be a mapping in %s", path)
            return None

        name = data.get("name")
        description = data.get("description")
        if not isinstance(name, str) or not isinstance(description, str):
            logger.warning("[SkillLoader] Skill missing name/description in %s", path)
            return None

        return {
            "name": name.strip(),
            "description": description.strip(),
            "body": body.strip(),
        }

    def list_skills(self) -> list[dict[str, Any]]:
        """Return all visible skills with name, description, and path metadata."""
        return [
            {
                "name": skill["name"],
                "description": skill["description"],
                "skill_path": skill["skill_path"],
                "path": skill["path"],
                "source_root": skill["source_root"],
            }
            for _, skill in sorted(self._skills_by_name.items())
        ]

    def _resolve_skill(self, name_or_path: str) -> Optional[dict[str, Any]]:
        raw = (name_or_path or "").strip()
        if not raw:
            return None

        direct = self._skills_by_name.get(raw)
        if direct:
            return direct

        normalized = raw.replace("\\", "/").strip("/")
        for skill in self._skills_by_name.values():
            if skill["skill_path"] == normalized:
                return skill
        return None

    def load_skill(self, names: list[str]) -> str:
        """Load one or more skills by exact name or skill path."""
        requested = [str(item).strip() for item in (names or []) if str(item).strip()]
        if not requested:
            return "Error: No skill names provided."

        resolved_skills: list[dict[str, Any]] = []
        missing: list[str] = []
        for item in requested:
            skill = self._resolve_skill(item)
            if skill is None:
                missing.append(item)
                continue
            resolved_skills.append(skill)

        if not resolved_skills:
            available = ", ".join(sorted(self._skills_by_name.keys())) or "none"
            return f"Error: None of the requested skills were found. Available: {available}"

        blocks: list[str] = []
        if missing:
            blocks.append(f"Warning: These skills were not found: {', '.join(missing)}")

        self._record_usage(requested, resolved_skills, missing)
        for skill in resolved_skills:
            blocks.append(self._render_skill(skill))

        if len(blocks) == 1:
            return blocks[0]
        return ("\n\n" + ("=" * 80) + "\n\n").join(blocks)

    def _record_usage(
        self,
        requested: list[str],
        resolved_skills: list[dict[str, Any]],
        missing: list[str],
    ) -> None:
        if self._usage_recorder is None or not resolved_skills:
            return

        payload = {
            "requested": list(requested),
            "missing": list(missing),
            "resolved": [
                {
                    "name": skill["name"],
                    "skill_path": skill["skill_path"],
                    "path": skill["path"],
                    "source_root": skill["source_root"],
                }
                for skill in resolved_skills
            ],
        }
        try:
            self._usage_recorder(payload)
        except Exception:
            logger.exception("[SkillLoader] Failed to record skill usage")

    @staticmethod
    def _render_skill(skill: dict[str, Any]) -> str:
        base_dir = Path(skill["base_dir"])
        entries = []
        for item in sorted(base_dir.iterdir()):
            if item.name == "SKILL.md":
                continue
            entries.append(f"  - {item.name}/" if item.is_dir() else f"  - {item.name}")

        lines = [
            f"## Skill: {skill['name']}",
            "",
            f"**Catalog path**: {skill['skill_path'] or '<root>'}",
            f"**Base directory**: {base_dir}",
        ]

        if entries:
            lines.extend(["", "**Available files:**", *entries])

        lines.extend(["", skill["body"]])
        return "\n".join(lines).strip()

    def _build_load_skill_description(self) -> str:
        visible_skills = self.list_skills()
        if not visible_skills:
            return (
                "Load one or more skills by exact name or skill path. "
                "No skills are currently available."
            )

        lines = [
            "Load one or more skills directly.",
            "Pass a list of exact skill names or skill paths in the `names` field.",
            "All currently visible skills are listed below so you can choose without a separate listing step.",
            "",
            "Visible skills:",
        ]
        for skill in visible_skills:
            path_hint = skill["skill_path"] or skill["name"]
            lines.append(f"- {skill['name']} ({path_hint}): {skill['description']}")
        return "\n".join(lines)

    @staticmethod
    def _normalize_nullable_parameter_types(schema: dict[str, Any]) -> dict[str, Any]:
        function = schema.get("function", {})
        parameters = function.get("parameters", {})
        properties = parameters.get("properties", {})
        for prop_schema in properties.values():
            prop_type = prop_schema.get("type")
            if isinstance(prop_type, list):
                non_null_types = [item for item in prop_type if item != "null"]
                if len(non_null_types) == 1:
                    prop_schema["type"] = non_null_types[0]
        return schema

    def get_tools(self) -> list:
        from camel.toolkits import FunctionTool

        load_schema = {
            "type": "function",
            "function": {
                "name": "load_skill",
                "description": self._build_load_skill_description(),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "names": {
                            "type": "array",
                            "description": (
                                "A list of exact skill names or skill paths to load into "
                                "the current context."
                            ),
                            "items": {
                                "type": "string",
                                "description": "One exact skill name or skill path.",
                            },
                        }
                    },
                    "required": ["names"],
                    "additionalProperties": False,
                },
            },
        }
        load_schema = self._normalize_nullable_parameter_types(load_schema)
        load_tool = FunctionTool(self.load_skill, openai_tool_schema=load_schema)
        load_tool.set_openai_tool_schema(load_schema)
        return [load_tool]


def load_agent_skills(
    skill_dirs: list[str],
    usage_recorder: Optional[Callable[[dict[str, Any]], None]] = None,
) -> list:
    """Load merged flat skill tools from multiple directories."""
    normalized_dirs = normalize_skill_dirs(skill_dirs)
    if not normalized_dirs:
        return []

    try:
        toolkit = _FlatSkillToolkit(normalized_dirs, usage_recorder=usage_recorder)
        if not toolkit.has_skills():
            logger.info("[SkillLoader] No skills found in %s", normalized_dirs)
            return []

        visible_skills = toolkit.list_skills()
        logger.info(
            "[SkillLoader] Loaded %d flat skills from %s: %s",
            len(visible_skills),
            normalized_dirs,
            [skill["name"] for skill in visible_skills],
        )
        return toolkit.get_tools()
    except ImportError:
        logger.warning("camel.toolkits.FunctionTool not available, skipping skill loading")
        return []
    except Exception as exc:
        logger.warning("[SkillLoader] Failed to load skills from %s: %s", normalized_dirs, exc)
        return []


__all__ = [
    "load_agent_skills",
    "normalize_skill_dirs",
]
