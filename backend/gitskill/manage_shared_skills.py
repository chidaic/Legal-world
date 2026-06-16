from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from gitskill.skill_manager import (  # noqa: E402
    DEFAULT_SHARED_SKILL_ROOT,
    check_skill_library,
    discover_skills,
    merge_skills,
    scan_skill_library,
    write_json_report,
)


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _resolve_skill_path(root: Path, value: str) -> Path:
    raw = Path(value)
    if raw.exists():
        return raw.resolve()
    candidate = root / value
    if candidate.is_dir():
        candidate = candidate / "SKILL.md"
    if candidate.exists():
        return candidate.resolve()
    for skill in discover_skills(root):
        if value in {skill.name, skill.relative_dir}:
            return Path(skill.path).resolve()
    raise FileNotFoundError(f"Cannot resolve skill: {value}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage the local shared GitSkill library.")
    parser.add_argument("--skill-root", default=str(DEFAULT_SHARED_SKILL_ROOT))
    parser.add_argument("--trace-path", default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List installed shared skills.")

    check_parser = subparsers.add_parser("check", help="Audit skills for rule conflicts and safety issues.")
    check_parser.add_argument("--output", default="")

    scan_parser = subparsers.add_parser("scan", help="Find semantically overlapping skills.")
    scan_parser.add_argument("--min-score", type=float, default=0.08)
    scan_parser.add_argument("--output", default="")

    merge_parser = subparsers.add_parser("merge", help="Merge two similar skills.")
    merge_parser.add_argument("left")
    merge_parser.add_argument("right")
    merge_parser.add_argument("--merged-name", default="")
    merge_parser.add_argument("--output-dir", default="")
    merge_parser.add_argument("--mode", choices=["deterministic", "prompt"], default="deterministic")
    merge_parser.add_argument("--prompt-output", default="", help="Optional path for the LLM merge prompt.")
    merge_parser.add_argument("--write", action="store_true", help="Write the merged SKILL.md. Omit for dry-run.")
    merge_parser.add_argument(
        "--retire-sources",
        action="store_true",
        help="When used with --write, hide source skills from future discovery without deleting them.",
    )
    merge_parser.add_argument("--output", default="", help="Optional JSON report path.")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    root = Path(args.skill_root).resolve()

    if args.command == "list":
        payload = {
            "skill_root": str(root),
            "skills": [
                {
                    "name": skill.name,
                    "description": skill.description,
                    "relative_dir": skill.relative_dir,
                    "path": skill.path,
                    "sha256": skill.sha256,
                }
                for skill in discover_skills(root)
            ],
        }
        _print_json(payload)
        return

    if args.command == "check":
        payload = check_skill_library(root, trace_path=args.trace_path)
        if args.output:
            write_json_report(payload, args.output)
        _print_json(payload)
        return

    if args.command == "scan":
        payload = scan_skill_library(root, min_score=args.min_score, trace_path=args.trace_path)
        if args.output:
            write_json_report(payload, args.output)
        _print_json(payload)
        return

    if args.command == "merge":
        left = _resolve_skill_path(root, args.left)
        right = _resolve_skill_path(root, args.right)
        payload = merge_skills(
            left,
            right,
            skill_root=root,
            output_dir=args.output_dir or None,
            merged_name=args.merged_name or None,
            write=bool(args.write),
            mode=args.mode,
            prompt_output=args.prompt_output or None,
            retire_sources=bool(args.retire_sources),
            trace_path=args.trace_path,
        )
        if args.output:
            write_json_report(payload, args.output)
        _print_json(payload)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
