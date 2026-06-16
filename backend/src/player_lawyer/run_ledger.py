"""Unified ledger for player-lawyer responsibility turns and review reports."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "player-run-ledger-v1"
LEDGER_FILENAME = "player_run_ledger.json"


def _now() -> str:
    return datetime.utcnow().isoformat()


def _string(value: Any) -> str:
    return str(value or "").strip()


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_string(item) for item in value if _string(item)]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _escape_markdown_cell(value: Any) -> str:
    text = str(value or "").replace("\n", "<br>").strip()
    return text.replace("|", "\\|")


class PlayerRunLedger:
    def __init__(self, *, storage_root: str | Path) -> None:
        self.storage_root = Path(storage_root)

    def case_output_dir(self, case_id: str) -> Path:
        return self.storage_root / "output" / _string(case_id)

    def player_dir(self, case_id: str) -> Path:
        return self.case_output_dir(case_id) / "_player_lawyer"

    def ledger_path(self, case_id: str) -> Path:
        return self.player_dir(case_id) / LEDGER_FILENAME

    def load(self, case_id: str) -> dict[str, Any]:
        path = self.ledger_path(case_id)
        payload = _read_json(path) if path.exists() else {}
        return self._normalize_payload(case_id, payload)

    def save(self, case_id: str, payload: dict[str, Any]) -> Path:
        normalized = self._normalize_payload(case_id, payload)
        normalized["updated_at"] = _now()
        path = self.ledger_path(case_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def record_responsibility_turn(
        self,
        *,
        case_id: str,
        request_id: str,
        stage: str,
        role: str,
        speaker_label: str,
        prompt: str,
        context_summary: str,
        status: str = "pending",
        created_at: str = "",
    ) -> dict[str, Any]:
        payload = self.load(case_id)
        request_id = _string(request_id)
        if not request_id:
            raise ValueError("request_id is required")
        turn = {
            "turn_id": request_id,
            "request_id": request_id,
            "stage": _string(stage),
            "role": _string(role),
            "speaker_label": _string(speaker_label),
            "prompt": str(prompt or ""),
            "context_summary": str(context_summary or ""),
            "created_at": _string(created_at) or _now(),
            "status": _string(status) or "pending",
        }
        items = [
            item
            for item in payload["responsibility_turns"]
            if item.get("request_id") != request_id
        ]
        items.append(turn)
        payload["responsibility_turns"] = sorted(
            items,
            key=lambda item: item.get("created_at") or "",
        )
        self.save(case_id, payload)
        return turn

    def record_submission(
        self,
        *,
        case_id: str,
        request_id: str,
        stage: str,
        submission_type: str,
        original_message: str,
        polished_message: str,
        final_message: str,
        hint_ids: list[str] | None = None,
        used_ai_polish: bool = False,
        submitted_at: str = "",
    ) -> dict[str, Any]:
        payload = self.load(case_id)
        request_id = _string(request_id)
        if not request_id:
            raise ValueError("request_id is required")
        item = {
            "request_id": request_id,
            "stage": _string(stage),
            "submission_type": _string(submission_type) or "dialogue",
            "original_message": str(original_message or ""),
            "polished_message": str(polished_message or ""),
            "final_message": str(final_message or ""),
            "used_ai_polish": bool(used_ai_polish),
            "hint_ids": _string_list(hint_ids or []),
            "submitted_at": _string(submitted_at) or _now(),
        }
        payload["submissions"] = [
            entry
            for entry in payload["submissions"]
            if entry.get("request_id") != request_id
        ]
        payload["submissions"].append(item)
        for turn in payload["responsibility_turns"]:
            if turn.get("request_id") == request_id:
                turn["status"] = "submitted"
        self.save(case_id, payload)
        return item

    def record_followup(
        self,
        *,
        case_id: str,
        request_id: str,
        stage: str,
        question: str,
        answer: str,
        created_at: str = "",
    ) -> dict[str, Any]:
        payload = self.load(case_id)
        item = {
            "request_id": _string(request_id),
            "stage": _string(stage),
            "question": str(question or ""),
            "answer": str(answer or ""),
            "created_at": _string(created_at) or _now(),
        }
        payload["followups"].append(item)
        payload["flow_events"].append({
            "event_type": "document_followup",
            "stage": item["stage"],
            "message": item["question"],
            "created_at": item["created_at"],
            "metadata": {"request_id": item["request_id"]},
        })
        self.save(case_id, payload)
        return item

    def record_document_confirmation(
        self,
        *,
        case_id: str,
        request_id: str,
        stage: str,
        document_type: str,
        document_text: str,
        result_json_path: str = "",
        pdf_path: str = "",
        confirmed_at: str = "",
    ) -> dict[str, Any]:
        payload = self.load(case_id)
        request_id = _string(request_id)
        item = {
            "request_id": request_id,
            "stage": _string(stage),
            "document_type": _string(document_type),
            "document_text": str(document_text or ""),
            "result_json_path": _string(result_json_path),
            "pdf_path": _string(pdf_path),
            "confirmed_at": _string(confirmed_at) or _now(),
        }
        payload["documents"] = [
            entry
            for entry in payload["documents"]
            if entry.get("request_id") != request_id
        ]
        payload["documents"].append(item)
        if request_id:
            self._upsert_document_submission(payload, item)
        self.save(case_id, payload)
        return item

    def record_flow_event(
        self,
        *,
        case_id: str,
        event_type: str,
        stage: str = "",
        message: str = "",
        metadata: dict[str, Any] | None = None,
        created_at: str = "",
    ) -> dict[str, Any]:
        payload = self.load(case_id)
        item = {
            "event_type": _string(event_type),
            "stage": _string(stage),
            "message": str(message or ""),
            "created_at": _string(created_at) or _now(),
            "metadata": dict(metadata or {}),
        }
        payload["flow_events"].append(item)
        self.save(case_id, payload)
        return item

    def record_evaluation(self, *, case_id: str, evaluation: dict[str, Any]) -> dict[str, Any]:
        payload = self.load(case_id)
        payload["evaluation"] = dict(evaluation or {})
        self.save(case_id, payload)
        return payload["evaluation"]

    def load_player_turns(self, case_id: str) -> list[dict[str, Any]]:
        payload = self.load(case_id)
        turns = self._player_turns_from_ledger(payload)
        if turns:
            return turns
        return self._player_turns_from_legacy_files(case_id)

    def build_markdown_report(
        self,
        *,
        case_id: str,
        case_entry: dict[str, Any],
        documents: list[dict[str, Any]],
        transcript: list[dict[str, Any]],
        evaluation: dict[str, Any] | None = None,
    ) -> str:
        payload = self.load(case_id)
        turns = self.load_player_turns(case_id)
        eval_payload = evaluation or payload.get("evaluation") or {}
        lines = [
            "# 玩家法律实训复盘报告",
            "",
            "## 案件信息",
            "",
            f"- 案件：{_string(case_entry.get('title')) or _string(case_id)}",
            f"- 原告：{_string(case_entry.get('plaintiff_name') or case_entry.get('plaintiffName')) or '未记录'}",
            f"- 被告：{_string(case_entry.get('defendant_name') or case_entry.get('defendantName')) or '未记录'}",
            f"- 类型：{_string(case_entry.get('training_category') or case_entry.get('trainingCategory')) or '法律实训'}",
            f"- 难度：{_string(case_entry.get('difficulty')) or '未标注'}",
            "",
            "## 流程概览",
            "",
        ]
        if payload["flow_events"]:
            for item in payload["flow_events"]:
                stage = _string(item.get("stage")) or "全局"
                message = _string(item.get("message")) or _string(item.get("event_type"))
                lines.append(f"- {stage}：{message}")
        else:
            lines.append("- 未记录到额外流程事件。")
        lines.extend(["", "## 玩家职责提交记录", ""])
        if turns:
            lines.extend(["| 阶段 | 角色 | 任务 | 最终提交 |", "| --- | --- | --- | --- |"])
            for turn in turns:
                lines.append(
                    f"| {_escape_markdown_cell(turn.get('stage'))} "
                    f"| {_escape_markdown_cell(turn.get('speaker_label') or turn.get('role'))} "
                    f"| {_escape_markdown_cell(turn.get('prompt'))} "
                    f"| {_escape_markdown_cell(turn.get('final_message'))} |"
                )
        else:
            lines.append("未记录到玩家提交。")
        lines.extend(["", "## 完整对话流程", ""])
        if transcript:
            for item in transcript:
                stage = _string(item.get("stage")) or "未分阶段"
                speaker = _string(item.get("speaker") or item.get("role")) or "未知角色"
                content = str(item.get("content") or "").strip()
                if content:
                    lines.append(f"### {stage} · {speaker}")
                    lines.extend(["", content, ""])
        else:
            lines.append("该案件未记录到完整对白。")
        lines.extend(["", "## 文书成果", ""])
        available_documents = [item for item in documents if item.get("available")]
        if available_documents:
            for item in available_documents:
                title = _string(item.get("title") or item.get("document_key"))
                lines.append(f"- {_string(item.get('stage')) or '阶段'}：{title}")
        else:
            lines.append("暂无可用文书。")
        for item in payload["documents"]:
            if _string(item.get("document_text")):
                lines.extend([
                    "",
                    f"### {_string(item.get('stage'))} · {_string(item.get('document_type'))}",
                    "",
                    str(item.get("document_text") or "").strip(),
                ])
        lines.extend(["", "## 评分结果", ""])
        if eval_payload:
            lines.append(f"- 总分：{eval_payload.get('overall_score', '未记录')}")
            summary = _string(eval_payload.get("summary"))
            if summary:
                lines.append(f"- 总评：{summary}")
        else:
            lines.append("评分尚未生成。")
        lines.extend(["", "## 改进建议", ""])
        improvements = eval_payload.get("improvements") if isinstance(eval_payload, dict) else []
        if isinstance(improvements, list) and improvements:
            for item in improvements:
                lines.append(f"- {_string(item)}")
        else:
            lines.append("- 暂无改进建议。")
        return "\n".join(lines).rstrip() + "\n"

    def _normalize_payload(self, case_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": _string(payload.get("schema_version")) or SCHEMA_VERSION,
            "case_id": _string(payload.get("case_id")) or _string(case_id),
            "created_at": _string(payload.get("created_at")) or _now(),
            "updated_at": _string(payload.get("updated_at")) or _now(),
            "responsibility_turns": list(payload.get("responsibility_turns") or []),
            "submissions": list(payload.get("submissions") or []),
            "followups": list(payload.get("followups") or []),
            "documents": list(payload.get("documents") or []),
            "flow_events": list(payload.get("flow_events") or []),
            "evaluation": payload.get("evaluation"),
            "report": payload.get("report"),
        }

    def _upsert_document_submission(self, payload: dict[str, Any], document: dict[str, Any]) -> None:
        request_id = _string(document.get("request_id"))
        payload["submissions"] = [
            entry
            for entry in payload["submissions"]
            if entry.get("request_id") != request_id
        ]
        payload["submissions"].append({
            "request_id": request_id,
            "stage": _string(document.get("stage")),
            "submission_type": "document",
            "original_message": "",
            "polished_message": "",
            "final_message": str(document.get("document_text") or ""),
            "used_ai_polish": False,
            "hint_ids": [],
            "submitted_at": _string(document.get("confirmed_at")) or _now(),
        })
        for turn in payload["responsibility_turns"]:
            if turn.get("request_id") == request_id:
                turn["status"] = "submitted"

    def _player_turns_from_ledger(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        turn_by_request = {
            item.get("request_id"): dict(item)
            for item in payload.get("responsibility_turns") or []
        }
        results = []
        for submission in payload.get("submissions") or []:
            request_id = _string(submission.get("request_id"))
            turn = turn_by_request.get(request_id, {})
            final_message = str(submission.get("final_message") or "").strip()
            if not final_message:
                continue
            results.append({
                "request_id": request_id,
                "stage": _string(submission.get("stage") or turn.get("stage")),
                "role": _string(turn.get("role")),
                "speaker_label": _string(turn.get("speaker_label")),
                "prompt": str(turn.get("prompt") or ""),
                "context_summary": str(turn.get("context_summary") or ""),
                "final_message": final_message,
                "user_original_message": str(submission.get("original_message") or ""),
                "created_at": _string(turn.get("created_at")),
                "resolved_at": _string(submission.get("submitted_at")),
            })
        return sorted(
            results,
            key=lambda item: item.get("resolved_at") or item.get("created_at") or "",
        )

    def _player_turns_from_legacy_files(self, case_id: str) -> list[dict[str, Any]]:
        player_dir = self.player_dir(case_id)
        if not player_dir.exists():
            return []
        turns = []
        for request_path in sorted(player_dir.glob("*.json")):
            if request_path.name in {LEDGER_FILENAME, "closing_evaluation.json"}:
                continue
            request_payload = _read_json(request_path)
            if _string(request_payload.get("status")).lower() != "submitted":
                continue
            request_id = _string(request_payload.get("request_id") or request_path.stem)
            assist_payload = _read_json(player_dir / "response_assists" / f"{request_id}.json")
            final_message = str(
                assist_payload.get("final_submitted_message")
                or request_payload.get("message")
                or ""
            ).strip()
            if not final_message:
                continue
            turns.append({
                "request_id": request_id,
                "stage": _string(request_payload.get("stage") or assist_payload.get("stage")),
                "role": _string(request_payload.get("role") or assist_payload.get("role")),
                "speaker_label": _string(
                    request_payload.get("speaker_label")
                    or assist_payload.get("speaker_label")
                ),
                "prompt": str(
                    request_payload.get("prompt")
                    or assist_payload.get("prompt")
                    or ""
                ).strip(),
                "context_summary": str(
                    request_payload.get("context_summary")
                    or assist_payload.get("context_summary")
                    or ""
                ).strip(),
                "final_message": final_message,
                "user_original_message": str(assist_payload.get("user_original_message") or "").strip(),
                "created_at": _string(request_payload.get("created_at")),
                "resolved_at": _string(request_payload.get("resolved_at")),
            })
        return turns


__all__ = ["LEDGER_FILENAME", "SCHEMA_VERSION", "PlayerRunLedger"]
