from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Optional

import yaml


DEFAULT_SHARED_SKILL_ROOT = Path(__file__).resolve().parent / "skillhub" / "shared"
DEFAULT_TRACE_PATH = Path(__file__).resolve().parent / "runs" / "skill_management_trace.jsonl"

_FRONTMATTER_RE = re.compile(r"\A\s*---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)
_SECRET_RE = re.compile(
    r"(api[_-]?key|secret|password|token|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})",
    re.IGNORECASE,
)
_LOCAL_PATH_RE = re.compile(r"([A-Za-z]:\\|/(?:Users|home)/|\\\\)")
_CONFLICT_RE = re.compile(r"(ignore previous|bypass|disable safety|忽略.*规则|不要遵守|覆盖.*系统)", re.IGNORECASE)
_CASE_SPECIFIC_RE = re.compile(r"(case_\d+|案件\s*\d+|本案当事人|本案金额|[A-Za-z]:\\|backend/batch_runs)")
_DESCRIPTION_TRIGGER_RE = re.compile(r"(use when|when|trigger|scenario|stage|适用|使用|触发|场景|阶段|用于|当.+时)")


@dataclass
class SkillRecord:
    name: str
    description: str
    body: str
    path: str
    relative_dir: str
    sha256: str
    metadata: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_trace(event: dict[str, Any], trace_path: str | Path | None = None) -> None:
    target = Path(trace_path or DEFAULT_TRACE_PATH).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": _utc_now(), **event}
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _split_skill(contents: str) -> tuple[dict[str, Any], str]:
    match = _FRONTMATTER_RE.match(contents.lstrip("\ufeff"))
    if not match:
        raise ValueError("SKILL.md must start with YAML frontmatter.")
    metadata = yaml.safe_load(match.group(1)) or {}
    if not isinstance(metadata, dict):
        raise ValueError("Skill frontmatter must be a YAML mapping.")
    return metadata, match.group(2).strip()


def load_skill_record(skill_path: str | Path, root: str | Path | None = None) -> SkillRecord:
    path = Path(skill_path).resolve()
    if path.is_dir():
        path = path / "SKILL.md"
    contents = path.read_text(encoding="utf-8-sig")
    metadata, body = _split_skill(contents)
    name = str(metadata.get("name") or "").strip()
    description = str(metadata.get("description") or "").strip()
    if not name or not description:
        raise ValueError(f"Skill missing name/description: {path}")
    root_path = Path(root).resolve() if root else path.parents[1]
    try:
        relative_dir = str(path.parent.relative_to(root_path)).replace("\\", "/")
    except ValueError:
        relative_dir = path.parent.name
    return SkillRecord(
        name=name,
        description=description,
        body=body,
        path=str(path),
        relative_dir=relative_dir,
        sha256=hashlib.sha256(contents.encode("utf-8")).hexdigest(),
        metadata=dict(metadata),
    )


def discover_skills(skill_root: str | Path = DEFAULT_SHARED_SKILL_ROOT) -> list[SkillRecord]:
    root = Path(skill_root).resolve()
    if not root.exists():
        return []
    records: list[SkillRecord] = []
    for path in sorted(root.rglob("SKILL.md")):
        if any(part.startswith(".") for part in path.relative_to(root).parts[:-1]):
            continue
        if (path.parent / ".gitskill_retired.json").exists():
            continue
        try:
            records.append(load_skill_record(path, root=root))
        except Exception:
            continue
    return records


def _issue(
    severity: str,
    rule_id: str,
    message: str,
    skill: SkillRecord | None = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    payload = {
        "severity": severity,
        "rule_id": rule_id,
        "message": message,
    }
    if skill is not None:
        payload.update({"skill": skill.name, "skill_path": skill.path, "relative_dir": skill.relative_dir})
    if extra:
        payload.update(extra)
    return payload


def check_skill_library(
    skill_root: str | Path = DEFAULT_SHARED_SKILL_ROOT,
    *,
    trace_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(skill_root).resolve()
    skills = discover_skills(root)
    issues: list[dict[str, Any]] = []

    if not root.exists():
        issues.append(_issue("error", "missing_root", f"Skill root does not exist: {root}"))

    by_name: dict[str, list[SkillRecord]] = {}
    by_hash: dict[str, list[SkillRecord]] = {}
    for skill in skills:
        by_name.setdefault(skill.name, []).append(skill)
        by_hash.setdefault(skill.sha256, []).append(skill)

        if len(skill.description) < 20:
            issues.append(_issue("warn", "short_description", "Description is too short to guide selection.", skill))
        if not _DESCRIPTION_TRIGGER_RE.search(skill.description):
            issues.append(
                _issue(
                    "warn",
                    "weak_description_trigger",
                    "Description should state the loading scenario, stage, or trigger condition.",
                    skill,
                )
            )
        if len(skill.body) < 120:
            issues.append(_issue("warn", "thin_body", "Skill body is too short to be actionable.", skill))
        if len(skill.body) > 6000:
            issues.append(_issue("warn", "large_body", "Skill body is large and may add context cost.", skill))
        if _SECRET_RE.search(skill.body) or _SECRET_RE.search(skill.description):
            issues.append(_issue("error", "possible_secret", "Skill appears to contain a secret/token pattern.", skill))
        if _LOCAL_PATH_RE.search(skill.body):
            issues.append(_issue("warn", "local_path_leak", "Skill body contains a local filesystem path.", skill))
        if _CONFLICT_RE.search(skill.body):
            issues.append(_issue("error", "instruction_conflict", "Skill contains potentially unsafe override language.", skill))
        if _CASE_SPECIFIC_RE.search(skill.body):
            issues.append(_issue("warn", "case_specific_leak", "Skill may contain case/run-specific references.", skill))

    for name, duplicates in by_name.items():
        if len(duplicates) > 1:
            issues.append(
                _issue(
                    "warn",
                    "duplicate_name",
                    f"Duplicate skill name '{name}' appears {len(duplicates)} times.",
                    extra={"paths": [item.path for item in duplicates]},
                )
            )

    for digest, duplicates in by_hash.items():
        if len(duplicates) > 1:
            issues.append(
                _issue(
                    "warn",
                    "duplicate_content",
                    f"Identical skill content appears {len(duplicates)} times.",
                    extra={"sha256": digest, "paths": [item.path for item in duplicates]},
                )
            )

    summary = {
        "skills": len(skills),
        "errors": sum(1 for item in issues if item["severity"] == "error"),
        "warnings": sum(1 for item in issues if item["severity"] == "warn"),
        "issues": len(issues),
    }
    report = {
        "generated_at": _utc_now(),
        "skill_root": str(root),
        "summary": summary,
        "issues": issues,
        "skills": [asdict(skill) for skill in skills],
    }
    _write_trace({"event": "check", "skill_root": str(root), "summary": summary}, trace_path)
    return report


def _terms(text: str) -> set[str]:
    normalized = re.sub(r"\s+", " ", text.lower())
    words = set(re.findall(r"[a-zA-Z0-9_]{3,}", normalized))
    cjk = re.findall(r"[\u4e00-\u9fff]", normalized)
    bigrams = {"".join(cjk[index : index + 2]) for index in range(max(len(cjk) - 1, 0))}
    return words | bigrams


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _skill_terms(skill: SkillRecord) -> dict[str, set[str]]:
    headings = "\n".join(line for line in skill.body.splitlines() if line.lstrip().startswith("#"))
    return {
        "name": _terms(skill.name),
        "description": _terms(skill.description),
        "headings": _terms(headings),
        "body": _terms(skill.body[:4000]),
    }


def scan_skill_library(
    skill_root: str | Path = DEFAULT_SHARED_SKILL_ROOT,
    *,
    min_score: float = 0.08,
    trace_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(skill_root).resolve()
    skills = discover_skills(root)
    term_cache = {skill.path: _skill_terms(skill) for skill in skills}
    candidates: list[dict[str, Any]] = []

    for left_index, left in enumerate(skills):
        for right in skills[left_index + 1 :]:
            lt = term_cache[left.path]
            rt = term_cache[right.path]
            score = (
                0.15 * _jaccard(lt["name"], rt["name"])
                + 0.35 * _jaccard(lt["description"], rt["description"])
                + 0.20 * _jaccard(lt["headings"], rt["headings"])
                + 0.30 * _jaccard(lt["body"], rt["body"])
            )
            if score < min_score:
                continue
            overlap_terms = sorted((lt["description"] | lt["headings"]) & (rt["description"] | rt["headings"]))[:20]
            candidates.append(
                {
                    "score": round(score, 4),
                    "rating": "***" if score >= 0.20 else "**" if score >= 0.12 else "*",
                    "left": asdict(left),
                    "right": asdict(right),
                    "overlap_terms": overlap_terms,
                    "recommendation": "merge_candidate" if score >= 0.25 else "review_candidate",
                }
            )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    report = {
        "generated_at": _utc_now(),
        "skill_root": str(root),
        "summary": {"skills": len(skills), "candidates": len(candidates), "min_score": min_score},
        "candidates": candidates,
    }
    _write_trace(
        {"event": "scan", "skill_root": str(root), "summary": report["summary"], "top_score": candidates[0]["score"] if candidates else None},
        trace_path,
    )
    return report


def _dedupe_lines(texts: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for text in texts:
        for line in text.splitlines():
            normalized = re.sub(r"\s+", " ", line.strip())
            if not normalized:
                if output and output[-1] != "":
                    output.append("")
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            output.append(line.rstrip())
    while output and output[-1] == "":
        output.pop()
    return output


def _safe_dir_name(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", value.strip())
    cleaned = cleaned.rstrip(". ").strip()
    return cleaned or "merged-skill"


def build_merged_skill_markdown(left: SkillRecord, right: SkillRecord, *, merged_name: str | None = None) -> str:
    name = merged_name or f"{left.name}与{right.name}整合规范"
    description = (
        f"整合“{left.name}”与“{right.name}”的重叠经验，形成一个更少冲突、"
        "更适合跨案件复用的操作规范。"
    )
    merged_lines = _dedupe_lines(
        [
            "# 整合目标\n\n"
            f"- 来源技能一：{left.name}\n"
            f"- 来源技能二：{right.name}\n"
            "- 使用本技能时，优先抽取可迁移的流程、检查点和风险提示，不要复述单案事实。\n",
            "## 来源技能一要点\n\n" + left.body,
            "## 来源技能二要点\n\n" + right.body,
            "## 合并使用准则\n\n"
            "- 如果两个来源技能给出不同优先级，先保留更具体的检查步骤，再把宽泛表述降级为背景提示。\n"
            "- 遇到文书起草或庭审发言任务，先判断当前阶段是否真的触发本技能；不相关时不要加载。\n"
            "- 输出给当事人或法院的内容必须回到当前案件事实、证据状态和诉讼阶段，不要机械套用技能标题。\n",
        ]
    )
    return (
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        "---\n\n"
        + "\n".join(merged_lines).strip()
        + "\n"
    )


def build_merge_prompt(left: SkillRecord, right: SkillRecord, *, merged_name: str | None = None) -> str:
    target_name = merged_name or f"{left.name}与{right.name}整合规范"
    return (
        "你是 GitSkill 共享技能库的治理 Agent。请审查下面两个 SKILL.md，判断是否应合并。\n\n"
        "要求：\n"
        "1. 如果二者目标、触发条件或操作步骤高度重叠，输出一个更强、更少冲突的合并版 SKILL.md。\n"
        "2. 如果二者只是主题相关但适用场景不同，明确输出不合并，并说明保留边界。\n"
        "3. 合并版必须保留 YAML frontmatter，name 和 description 要简洁可检索；description 还要写清适用阶段、触发条件和使用目标。\n"
        "4. 删除单案事实、路径、当事人姓名、金额日期等不可复用内容。\n"
        "5. 不要引入来源 Skill 没有支撑的新法律结论。\n\n"
        f"建议合并技能名：{target_name}\n\n"
        f"## 来源 Skill A\n\n路径：{left.relative_dir}\n\n"
        f"```markdown\n---\nname: {left.name}\ndescription: {left.description}\n---\n\n{left.body}\n```\n\n"
        f"## 来源 Skill B\n\n路径：{right.relative_dir}\n\n"
        f"```markdown\n---\nname: {right.name}\ndescription: {right.description}\n---\n\n{right.body}\n```\n"
    )


def merge_skills(
    left_skill: str | Path,
    right_skill: str | Path,
    *,
    skill_root: str | Path = DEFAULT_SHARED_SKILL_ROOT,
    output_dir: str | Path | None = None,
    merged_name: str | None = None,
    write: bool = False,
    mode: str = "deterministic",
    prompt_output: str | Path | None = None,
    retire_sources: bool = False,
    trace_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(skill_root).resolve()
    left = load_skill_record(left_skill, root=root)
    right = load_skill_record(right_skill, root=root)
    normalized_mode = str(mode or "deterministic").strip().lower()
    if normalized_mode not in {"deterministic", "prompt"}:
        raise ValueError("mode must be 'deterministic' or 'prompt'.")
    markdown = build_merged_skill_markdown(left, right, merged_name=merged_name)
    merge_prompt = build_merge_prompt(left, right, merged_name=merged_name)

    if output_dir is None:
        case_cause_dir = left.relative_dir.split("/")[0] if "/" in left.relative_dir else Path(left.relative_dir).name
        output_dir = root / case_cause_dir / _safe_dir_name(merged_name or f"{left.name}与{right.name}整合规范")
    target_dir = Path(output_dir).resolve()
    target_path = target_dir / "SKILL.md"
    prompt_path = Path(prompt_output).resolve() if prompt_output else target_dir / "MERGE_PROMPT.md"

    result = {
        "generated_at": _utc_now(),
        "status": "written" if write else "dry_run",
        "mode": normalized_mode,
        "left": asdict(left),
        "right": asdict(right),
        "target_path": str(target_path),
        "merged_markdown": markdown,
        "merge_prompt": merge_prompt,
        "prompt_path": str(prompt_path),
        "retire_sources": bool(retire_sources),
    }
    if normalized_mode == "prompt" or prompt_output:
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(merge_prompt, encoding="utf-8")
    if write:
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path.write_text(markdown, encoding="utf-8")
        (target_dir / ".gitskill_merge.json").write_text(
            json.dumps(
                {
                    "generated_at": result["generated_at"],
                    "source_skills": [left.path, right.path],
                    "source_sha256": [left.sha256, right.sha256],
                    "merge_strategy": f"{normalized_mode}_dedupe_v1",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if retire_sources:
            for source in (left, right):
                source_dir = Path(source.path).parent
                (source_dir / ".gitskill_retired.json").write_text(
                    json.dumps(
                        {
                            "generated_at": result["generated_at"],
                            "reason": "merged_into_new_skill",
                            "merged_target_path": str(target_path),
                            "merged_target_relative_dir": str(
                                target_dir.relative_to(root)
                            ).replace("\\", "/")
                            if target_dir.is_relative_to(root)
                            else str(target_dir),
                            "source_skill": source.path,
                            "source_sha256": source.sha256,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )

    _write_trace(
        {
            "event": "merge",
            "status": result["status"],
            "left": left.path,
            "right": right.path,
            "target_path": str(target_path),
            "retire_sources": bool(retire_sources),
        },
        trace_path,
    )
    return result


def write_json_report(payload: dict[str, Any], path: str | Path) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target
