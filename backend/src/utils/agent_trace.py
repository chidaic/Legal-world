from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRACE_SCHEMA_VERSION = "simlaw-agent-trace-v1"
CASE_TRACE_SCHEMA_VERSION = "simlaw-case-trace-v1"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _normalize_payload(payload: Any) -> Any:
    if payload is None:
        return None
    if isinstance(payload, dict):
        return {
            str(key): _normalize_payload(value)
            for key, value in payload.items()
            if not str(key).startswith("_")
        }
    if isinstance(payload, list):
        return [_normalize_payload(item) for item in payload]
    if isinstance(payload, tuple):
        return [_normalize_payload(item) for item in payload]
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, (bool, int, float)):
        return payload

    text_value = str(payload)
    stripped = text_value.strip()
    if not stripped:
        return ""
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            return _normalize_payload(json.loads(stripped))
        except Exception:
            return text_value
    return text_value


def _coerce_tool_call_mapping(record: Any) -> dict[str, Any]:
    if isinstance(record, dict):
        return dict(record)
    for attr_name in ("model_dump", "dict"):
        serializer = getattr(record, attr_name, None)
        if callable(serializer):
            try:
                payload = serializer()
            except Exception:
                payload = None
            if isinstance(payload, dict):
                return dict(payload)
    if hasattr(record, "__dict__"):
        try:
            payload = vars(record)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            return dict(payload)
    return {}


def _tool_call_value(record: Any, mapping: dict[str, Any], *field_names: str) -> Any:
    for field_name in field_names:
        if field_name in mapping:
            value = mapping.get(field_name)
            if value not in (None, ""):
                return value
        value = getattr(record, field_name, None)
        if value not in (None, ""):
            return value
    return None


def serialize_tool_call_record(record: Any) -> dict[str, Any]:
    mapping = _coerce_tool_call_mapping(record)
    tool_name = str(
        _tool_call_value(record, mapping, "tool_name", "name", "tool", "function_name")
        or type(record).__name__
    ).strip()
    kwargs = _tool_call_value(
        record,
        mapping,
        "kwargs",
        "parameters",
        "arguments",
        "input_kwargs",
    )
    args = _tool_call_value(record, mapping, "args", "input_args")
    result = _tool_call_value(record, mapping, "result", "return_value", "output")
    error = _tool_call_value(record, mapping, "error", "exception")
    status = str(_tool_call_value(record, mapping, "status") or "").strip().lower()
    normalized_result = _normalize_payload(result)
    normalized_error = str(error).strip() if error not in (None, "") else ""
    if not status:
        if normalized_error:
            status = "failed"
        elif isinstance(normalized_result, str) and normalized_result.startswith("Tool execution failed:"):
            status = "failed"
        else:
            status = "completed"
    return {
        "tool_name": tool_name,
        "tool_call_id": str(
            _tool_call_value(record, mapping, "tool_call_id", "call_id", "id") or ""
        ).strip(),
        "args": _normalize_payload(args),
        "kwargs": _normalize_payload(kwargs),
        "result": normalized_result,
        "error": normalized_error,
        "status": status,
        "raw_record": _normalize_payload(mapping),
    }


def bind_agent_trace_context(
    agent: Any,
    *,
    recorder: "CaseAgentTraceRecorder",
    output_dir: str | Path,
    stage_code: str,
    stage_key: str | None = None,
) -> None:
    setattr(agent, "_simlaw_trace_recorder", recorder)
    setattr(agent, "_simlaw_trace_output_dir", str(Path(output_dir).resolve()))
    setattr(agent, "_simlaw_trace_stage_code", str(stage_code or "").strip().upper())
    setattr(
        agent,
        "_simlaw_trace_stage_key",
        str(stage_key or stage_code or "").strip().upper(),
    )


def serialize_agent_chat_history(agent: Any) -> list[dict[str, Any]]:
    chat_agent = getattr(agent, "chat_agent", None)
    if chat_agent is None:
        return []
    try:
        context_records = chat_agent.memory.retrieve()
    except Exception:
        return []

    serialized: list[dict[str, Any]] = []
    for item in list(context_records or []):
        record = getattr(item, "memory_record", None)
        message = getattr(record, "message", None)
        if record is None or message is None:
            continue
        serialized.append(
            {
                "role_at_backend": str(getattr(record, "role_at_backend", "") or ""),
                "role_name": str(getattr(message, "role_name", "") or ""),
                "role_type": str(getattr(message, "role_type", "") or ""),
                "content": _normalize_text(getattr(message, "content", "")),
            }
        )
    return serialized


def _role_tags(value: Any) -> set[str]:
    lowered = str(value or "").strip().lower()
    tags: set[str] = set()
    if not lowered:
        return tags
    if any(token in lowered for token in ("lawyer", "律师")):
        tags.add("lawyer")
    if any(token in lowered for token in ("judge", "法官", "审判")):
        tags.add("judge")
    if any(token in lowered for token in ("plaintiff", "原告")):
        tags.add("plaintiff")
    if any(token in lowered for token in ("defendant", "被告")):
        tags.add("defendant")
    if any(token in lowered for token in ("appellant", "上诉人")):
        tags.add("appellant")
    if any(token in lowered for token in ("appellee", "被上诉")):
        tags.add("appellee")
    if any(token in lowered for token in ("client", "当事人")):
        tags.add("client")
    if "plaintiff_lawyer" in lowered:
        tags.add("plaintiff_lawyer")
        tags.add("lawyer")
        tags.add("plaintiff")
    if "defendant_lawyer" in lowered:
        tags.add("defendant_lawyer")
        tags.add("lawyer")
        tags.add("defendant")
    if "appellant_lawyer" in lowered:
        tags.add("appellant_lawyer")
        tags.add("lawyer")
        tags.add("appellant")
    if "appellee_lawyer" in lowered:
        tags.add("appellee_lawyer")
        tags.add("lawyer")
        tags.add("appellee")
    return tags or {lowered}


def _agent_role_tags(agent: Any) -> set[str]:
    values = [
        getattr(agent, "agent_id", ""),
        getattr(agent, "name", ""),
        getattr(agent, "_simlaw_stage_role", ""),
        getattr(agent, "role", ""),
        getattr(agent, "court_role", ""),
        type(agent).__name__,
    ]
    tags: set[str] = set()
    for value in values:
        tags.update(_role_tags(value))
    return tags


def _dialog_matches_agent(agent: Any, dialog_entry: dict[str, Any]) -> bool:
    dialog_tags = _role_tags(dialog_entry.get("role"))
    agent_tags = _agent_role_tags(agent)
    if dialog_tags & agent_tags:
        return True
    dialog_role = str(dialog_entry.get("role", "") or "").strip().lower()
    if dialog_role == "lawyer" and "lawyer" in agent_tags:
        return True
    return False


def _write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False))
            handle.write("\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _aggregate_llm_calls(records: list[dict[str, Any]]) -> dict[str, Any]:
    llm_call_count = len(list(records or []))
    prompt_tokens = sum(int(item.get("prompt_tokens") or 0) for item in list(records or []))
    completion_tokens = sum(int(item.get("completion_tokens") or 0) for item in list(records or []))
    total_tokens = sum(int(item.get("total_tokens") or 0) for item in list(records or []))
    context_tokens = sum(int(item.get("context_tokens") or 0) for item in list(records or []))
    duration_seconds_sum = round(
        sum(float(item.get("duration_seconds") or 0.0) for item in list(records or [])),
        4,
    )
    missing_usage_count = sum(
        1
        for item in list(records or [])
        if not int(item.get("prompt_tokens") or 0)
        and not int(item.get("completion_tokens") or 0)
        and not int(item.get("total_tokens") or 0)
    )
    return {
        "llm_call_count": llm_call_count,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "context_tokens": context_tokens,
        "duration_seconds_sum": duration_seconds_sum,
        "missing_usage_count": missing_usage_count,
    }


def _tool_names_from_calls(records: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for record in list(records or []):
        for tool_call in list(record.get("tool_calls") or []):
            tool_name = str(tool_call.get("tool_name") or "").strip()
            if tool_name and tool_name not in names:
                names.append(tool_name)
    return names


def _dedupe_strings(values: list[str]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for raw in list(values or []):
        value = str(raw or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        names.append(value)
    return names


def _stage_label(stage_code: str) -> str:
    labels = {
        "RECEPTION": "前台导引",
        "LC": "法律咨询",
        "CD": "起诉状起草",
        "DD": "答辩状起草",
        "TIA": "一审庭前信息分析",
        "CI": "一审庭审",
        "AD": "上诉状起草",
        "AR": "上诉答辩状起草",
        "TIAA": "二审庭前信息分析",
        "CIA": "二审庭审",
    }
    normalized = str(stage_code or "").strip().upper()
    return labels.get(normalized) or normalized


class CaseAgentTraceRecorder:
    def __init__(self, case_output_dir: str | Path) -> None:
        self.case_output_dir = Path(case_output_dir).resolve()
        self.case_output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._llm_calls: list[dict[str, Any]] = []
        self._dialog_entries: list[dict[str, Any]] = []
        self._runtime_tool_events: list[dict[str, Any]] = []
        self._stage_exports: list[dict[str, Any]] = []

    @property
    def debug_dir(self) -> Path:
        path = self.case_output_dir / "_debug"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _stage_records_for_export(
        self,
        *,
        stage_code: str,
        stage_key: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        normalized_stage_key = str(stage_key or stage_code or "").strip().upper()
        stage_llm_calls = [
            dict(item)
            for item in self._llm_calls
            if str(item.get("stage_key", "") or "").strip().upper() == normalized_stage_key
        ]
        stage_dialogs = [
            dict(item)
            for item in self._dialog_entries
            if str(item.get("stage_key", "") or "").strip().upper() == normalized_stage_key
        ]
        return stage_llm_calls, stage_dialogs

    def _runtime_tool_records_for_export(
        self,
        *,
        stage_code: str,
        stage_key: str,
    ) -> list[dict[str, Any]]:
        normalized_stage_key = str(stage_key or stage_code or "").strip().upper()
        return [
            dict(item)
            for item in self._runtime_tool_events
            if str(item.get("stage_key", "") or "").strip().upper() == normalized_stage_key
        ]

    @staticmethod
    def _agent_participant_summary(
        agent: Any,
        agent_llm_calls: list[dict[str, Any]],
        runtime_tool_events: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        tool_names = _dedupe_strings(
            [
                *_tool_names_from_calls(agent_llm_calls),
                *[
                    tool_name
                    for item in list(runtime_tool_events or [])
                    for tool_name in list(item.get("tool_names") or [])
                ],
            ]
        )
        return {
            "agent_id": str(getattr(agent, "agent_id", "") or ""),
            "agent_name": str(getattr(agent, "name", "") or ""),
            "agent_class": type(agent).__name__,
            "agent_role": str(getattr(agent, "_simlaw_stage_role", "") or ""),
            "role": str(getattr(agent, "role", "") or ""),
            "court_role": str(getattr(agent, "court_role", "") or ""),
            "tool_names": [
                tool.get_function_name()
                for tool in list(getattr(agent, "tools", []) or [])
                if hasattr(tool, "get_function_name")
            ],
            "actual_tool_calls": tool_names,
            "actual_tool_call_count": sum(
                len(list(item.get("tool_calls") or []))
                for item in list(agent_llm_calls or [])
            ) + sum(len(list(item.get("tool_names") or [])) for item in list(runtime_tool_events or [])),
        }

    def _upsert_stage_export(self, record: dict[str, Any]) -> None:
        stage_key = str(record.get("stage_key") or "").strip().upper()
        for index, existing in enumerate(self._stage_exports):
            if str(existing.get("stage_key") or "").strip().upper() == stage_key:
                self._stage_exports[index] = record
                return
        self._stage_exports.append(record)

    def _build_case_index(self) -> dict[str, Any]:
        llm_calls = [dict(item) for item in self._llm_calls]
        dialog_entries = [dict(item) for item in self._dialog_entries]
        runtime_tool_events = [dict(item) for item in self._runtime_tool_events]
        stage_exports = [dict(item) for item in self._stage_exports]

        run_status = "not_started"
        if stage_exports:
            run_status = "failed" if any(item.get("status") == "failed" for item in stage_exports) else "completed"

        owner_buckets: dict[str, list[dict[str, Any]]] = {}
        for item in llm_calls:
            owner = dict(item.get("agent") or {})
            agent_id = str(owner.get("agent_id") or "").strip() or "anonymous"
            owner_buckets.setdefault(agent_id, []).append(item)

        agent_summaries: list[dict[str, Any]] = []
        for agent_id, records in owner_buckets.items():
            agent = dict(records[0].get("agent") or {})
            actual_tool_calls = _tool_names_from_calls(records)
            agent_summaries.append(
                {
                    "agent_id": agent_id,
                    "agent_name": str(agent.get("agent_name") or ""),
                    "agent_class": str(agent.get("agent_class") or ""),
                    "agent_role": str(agent.get("agent_role") or ""),
                    "role": str(agent.get("role") or ""),
                    "court_role": str(agent.get("court_role") or ""),
                    "stage_codes": sorted(
                        {
                            str(item.get("stage_code") or "").strip().upper()
                            for item in records
                            if str(item.get("stage_code") or "").strip()
                        }
                    ),
                    "actual_tool_calls": actual_tool_calls,
                    "actual_tool_call_count": sum(
                        len(list(item.get("tool_calls") or []))
                        for item in list(records or [])
                    ),
                    **_aggregate_llm_calls(records),
                }
            )
        agent_summaries.sort(
            key=lambda item: (
                -int(item.get("total_tokens") or 0),
                str(item.get("agent_id") or ""),
            )
        )

        ordered_stages = sorted(
            stage_exports,
            key=lambda item: (
                str(item.get("started_at") or item.get("finished_at") or ""),
                str(item.get("stage_key") or ""),
            ),
        )
        return _normalize_payload(
            {
                "schema_version": CASE_TRACE_SCHEMA_VERSION,
                "generated_at": _utc_timestamp(),
                "case_output_dir": str(self.case_output_dir),
                "debug_dir": str(self.debug_dir),
                "run_status": run_status,
                "llm_totals": _aggregate_llm_calls(llm_calls),
                "dialog_totals": {
                    "dialog_count": len(dialog_entries),
                },
                "runtime_tool_event_count": len(runtime_tool_events),
                "stage_count": len(ordered_stages),
                "agent_count": len(agent_summaries),
                "stages": ordered_stages,
                "agents": agent_summaries,
                "runtime_tool_events": runtime_tool_events,
                "artifacts": {
                    "trace_path": str(self.debug_dir / "trace.jsonl"),
                    "index_path": str(self.debug_dir / "index.json"),
                    "agent_traces_dir": str(self.debug_dir / "agent_traces"),
                },
            }
        )

    def _build_case_trace_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for stage in list(self._stage_exports or []):
            participants = list(stage.get("participating_agents") or [])
            for agent in participants:
                timestamp = str(stage.get("started_at") or stage.get("finished_at") or _utc_timestamp())
                events.append(
                    {
                        "event_type": "agent_activation",
                        "timestamp": timestamp,
                        "stage_code": stage.get("stage_code"),
                        "stage_key": stage.get("stage_key"),
                        **dict(agent),
                    }
                )
            if stage.get("started_at"):
                events.append(
                    {
                        "event_type": "stage_start",
                        "timestamp": stage.get("started_at"),
                        **dict(stage),
                    }
                )
            events.append(
                {
                    "event_type": "stage_end",
                    "timestamp": stage.get("finished_at") or stage.get("started_at") or _utc_timestamp(),
                    **dict(stage),
                }
            )
        events.extend({"event_type": "dialogue", **dict(item)} for item in list(self._dialog_entries or []))
        events.extend({"event_type": "llm_call", **dict(item)} for item in list(self._llm_calls or []))
        events.extend({"event_type": "runtime_tool_event", **dict(item)} for item in list(self._runtime_tool_events or []))
        events.sort(
            key=lambda item: (
                str(item.get("timestamp") or ""),
                str(item.get("event_type") or ""),
                str(item.get("stage_key") or item.get("stage_code") or ""),
            )
        )
        return [_normalize_payload(item) for item in events]

    def flush_case_debug_bundle(self) -> None:
        with self._lock:
            index_payload = self._build_case_index()
            trace_events = self._build_case_trace_events()
        _write_json(self.debug_dir / "index.json", index_payload)
        _write_jsonl(self.debug_dir / "trace.jsonl", trace_events)

    def log_agent_step(
        self,
        *,
        agent: Any,
        instruction: str,
        response_text: str,
        info: dict[str, Any] | None,
        duration_seconds: float,
        error: str = "",
    ) -> None:
        info = dict(info or {})
        usage = dict(info.get("usage") or {})
        stage_code = str(
            getattr(agent, "_simlaw_trace_stage_code", "")
            or getattr(agent, "_simlaw_stage_code", "")
            or getattr(agent, "scenario_type", "")
            or ""
        ).strip().upper()
        stage_key = str(
            getattr(agent, "_simlaw_trace_stage_key", "")
            or stage_code
            or ""
        ).strip().upper()
        output_dir = str(getattr(agent, "_simlaw_trace_output_dir", "") or "").strip()

        record = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "timestamp": _utc_timestamp(),
            "case_output_dir": str(self.case_output_dir),
            "stage_code": stage_code,
            "stage_key": stage_key,
            "agent": {
                "agent_id": str(getattr(agent, "agent_id", "") or ""),
                "agent_name": str(getattr(agent, "name", "") or ""),
                "agent_class": type(agent).__name__,
                "agent_role": str(getattr(agent, "_simlaw_stage_role", "") or ""),
                "role": str(getattr(agent, "role", "") or ""),
                "court_role": str(getattr(agent, "court_role", "") or ""),
                "scenario_type": str(getattr(agent, "scenario_type", "") or ""),
                "trace_output_dir": output_dir,
            },
            "duration_seconds": round(float(duration_seconds or 0.0), 4),
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
            "context_tokens": int(info.get("num_tokens") or 0),
            "finish_reasons": list(info.get("termination_reasons") or []),
            "tool_call_count": len(list(info.get("tool_calls") or [])),
            "tool_calls": [
                serialize_tool_call_record(item)
                for item in list(info.get("tool_calls") or [])
            ],
            "input_text": _normalize_text(instruction),
            "response_text": _normalize_text(response_text),
            "error": str(error or "").strip(),
        }
        with self._lock:
            self._llm_calls.append(record)

    def log_dialog(
        self,
        *,
        stage_code: str,
        stage_key: str,
        scenario: str,
        turn: int,
        role: str,
        content: str,
        timestamp: str = "",
    ) -> None:
        record = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "timestamp": str(timestamp or _utc_timestamp()),
            "case_output_dir": str(self.case_output_dir),
            "stage_code": str(stage_code or "").strip().upper(),
            "stage_key": str(stage_key or stage_code or "").strip().upper(),
            "scenario": str(scenario or "").strip(),
            "turn": int(turn or 0),
            "role": str(role or "").strip(),
            "content": _normalize_text(content),
        }
        with self._lock:
            self._dialog_entries.append(record)

    def log_runtime_tool_event(
        self,
        *,
        case_id: str,
        stage_code: str,
        tool_names: list[str] | None = None,
        skill_names: list[str] | None = None,
        message: str = "",
        detail: str = "",
        metadata: dict[str, Any] | None = None,
        stage_key: str | None = None,
    ) -> None:
        normalized_stage_code = str(stage_code or "").strip().upper()
        normalized_stage_key = str(stage_key or normalized_stage_code).strip().upper()
        record = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "timestamp": _utc_timestamp(),
            "case_output_dir": str(self.case_output_dir),
            "case_id": str(case_id or "").strip(),
            "stage_code": normalized_stage_code,
            "stage_key": normalized_stage_key,
            "tool_names": _dedupe_strings(list(tool_names or [])),
            "skill_names": _dedupe_strings(list(skill_names or [])),
            "message": _normalize_text(message),
            "detail": _normalize_text(detail),
            "metadata": _normalize_payload(dict(metadata or {})),
        }
        with self._lock:
            self._runtime_tool_events.append(record)

    def export_stage(
        self,
        *,
        stage_code: str,
        stage_key: str | None = None,
        agents: list[Any] | None = None,
        stage_result: Any = None,
        stage_result_path: str | Path | None = None,
        status: str = "completed",
        error: str = "",
    ) -> list[dict[str, Any]]:
        normalized_stage_code = str(stage_code or "").strip().upper()
        normalized_stage_key = str(stage_key or stage_code or "").strip().upper()
        agent_list = [agent for agent in list(agents or []) if agent is not None]

        with self._lock:
            stage_llm_calls, stage_dialogs = self._stage_records_for_export(
                stage_code=normalized_stage_code,
                stage_key=normalized_stage_key,
            )
            stage_runtime_tool_events = self._runtime_tool_records_for_export(
                stage_code=normalized_stage_code,
                stage_key=normalized_stage_key,
            )

        exports: list[dict[str, Any]] = []
        participating_agents: list[dict[str, Any]] = []
        for agent in agent_list:
            agent_id = str(getattr(agent, "agent_id", "") or "").strip() or "agent"
            trace_output_dir_raw = str(getattr(agent, "_simlaw_trace_output_dir", "") or "").strip()
            trace_output_dir = (
                Path(trace_output_dir_raw).resolve()
                if trace_output_dir_raw
                else (self.case_output_dir / agent_id / "agent_trace").resolve()
            )
            trace_output_dir.mkdir(parents=True, exist_ok=True)

            agent_llm_calls = [
                dict(item)
                for item in stage_llm_calls
                if str(item.get("agent", {}).get("agent_id", "") or "").strip() == agent_id
            ]
            participating_agents.append(
                self._agent_participant_summary(agent, agent_llm_calls, stage_runtime_tool_events)
            )
            tool_events: list[dict[str, Any]] = []
            for llm_index, call in enumerate(agent_llm_calls, start=1):
                for tool_index, tool_call in enumerate(list(call.get("tool_calls") or []), start=1):
                    tool_events.append(
                        {
                            "event_type": "tool_call",
                            "stage_code": normalized_stage_code,
                            "stage_key": normalized_stage_key,
                            "llm_call_index": llm_index,
                            "tool_call_index": tool_index,
                            "timestamp": str(call.get("timestamp", "") or ""),
                            **dict(tool_call),
                        }
                    )

            agent_dialogs = [
                dict(item)
                for item in stage_dialogs
                if _dialog_matches_agent(agent, item)
            ]
            chat_history = serialize_agent_chat_history(agent)
            stage_jsonl_path = trace_output_dir / f"{normalized_stage_key}.jsonl"

            payload = {
                "schema_version": TRACE_SCHEMA_VERSION,
                "case_output_dir": str(self.case_output_dir),
                "stage_code": normalized_stage_code,
                "stage_key": normalized_stage_key,
                "status": str(status or "").strip().lower() or "completed",
                "error": str(error or "").strip(),
                "stage_result_path": str(Path(stage_result_path).resolve()) if stage_result_path else "",
                "stage_result": _normalize_payload(stage_result),
                "agent": {
                    "agent_id": agent_id,
                    "agent_name": str(getattr(agent, "name", "") or ""),
                    "agent_class": type(agent).__name__,
                    "agent_role": str(getattr(agent, "_simlaw_stage_role", "") or ""),
                    "role": str(getattr(agent, "role", "") or ""),
                    "court_role": str(getattr(agent, "court_role", "") or ""),
                    "scenario_type": str(getattr(agent, "scenario_type", "") or ""),
                    "trace_output_dir": str(trace_output_dir),
                    "memory_yaml_path": str(getattr(agent, "memory_yaml_path", "") or ""),
                    "long_term_memory_path": str(getattr(agent, "long_term_memory_path", "") or ""),
                },
                "system_prompt": str(getattr(agent, "system_prompt", "") or ""),
                "llm_call_count": len(agent_llm_calls),
                "tool_call_count": len(tool_events),
                "runtime_tool_event_count": len(stage_runtime_tool_events),
                "dialog_count": len(agent_dialogs),
                "stage_dialog_count": len(stage_dialogs),
                "llm_calls": agent_llm_calls,
                "tool_calls": tool_events,
                "runtime_tool_events": stage_runtime_tool_events,
                "agent_dialogues": agent_dialogs,
                "stage_dialogues": stage_dialogs,
                "chat_history": chat_history,
            }

            event_stream: list[dict[str, Any]] = []
            event_stream.append(
                {
                    "event_type": "stage_context",
                    "schema_version": TRACE_SCHEMA_VERSION,
                    "case_output_dir": str(self.case_output_dir),
                    "stage_code": normalized_stage_code,
                    "stage_key": normalized_stage_key,
                    "status": payload["status"],
                    "error": payload["error"],
                    "stage_result_path": payload["stage_result_path"],
                    "stage_result": payload["stage_result"],
                    "agent": payload["agent"],
                    "system_prompt": payload["system_prompt"],
                    "chat_history": payload["chat_history"],
                    "stage_dialogues": payload["stage_dialogues"],
                }
            )
            event_stream.append(
                {
                    "event_type": "stage_summary",
                    "schema_version": TRACE_SCHEMA_VERSION,
                    "stage_code": normalized_stage_code,
                    "stage_key": normalized_stage_key,
                    "status": payload["status"],
                    "agent_id": agent_id,
                    "agent_role": payload["agent"]["agent_role"],
                    "llm_call_count": len(agent_llm_calls),
                    "tool_call_count": len(tool_events),
                    "dialog_count": len(agent_dialogs),
                    "stage_result_path": payload["stage_result_path"],
                }
            )
            event_stream.extend(
                {
                    "event_type": "llm_call",
                    **dict(item),
                }
                for item in agent_llm_calls
            )
            event_stream.extend(
                {
                    "event_type": "dialogue",
                    **dict(item),
                }
                for item in agent_dialogs
            )
            event_stream.extend(tool_events)
            event_stream.extend(
                {
                    "event_type": "runtime_tool_event",
                    **dict(item),
                }
                for item in stage_runtime_tool_events
            )
            _write_jsonl(stage_jsonl_path, event_stream)

            exports.append(
                {
                    "agent_id": agent_id,
                    "stage_code": normalized_stage_code,
                    "stage_key": normalized_stage_key,
                    "jsonl_path": str(stage_jsonl_path),
                }
            )

        timestamps = [
            str(item.get("timestamp") or "").strip()
            for item in list(stage_llm_calls or []) + list(stage_dialogs or [])
            if str(item.get("timestamp") or "").strip()
        ]
        started_at = min(timestamps) if timestamps else _utc_timestamp()
        finished_at = max(timestamps) if timestamps else _utc_timestamp()
        stage_summary = {
            "schema_version": CASE_TRACE_SCHEMA_VERSION,
            "stage_code": normalized_stage_code,
            "stage_key": normalized_stage_key,
            "stage_name": _stage_label(normalized_stage_code),
            "status": str(status or "").strip().lower() or "completed",
            "error": str(error or "").strip(),
            "started_at": started_at,
            "finished_at": finished_at,
            "stage_result_path": str(Path(stage_result_path).resolve()) if stage_result_path else "",
            "result_file_exists": bool(stage_result_path and Path(stage_result_path).exists()),
            "participating_agents": participating_agents,
            "participating_agent_count": len(participating_agents),
            "actual_tool_calls": _dedupe_strings(
                [
                    *_tool_names_from_calls(stage_llm_calls),
                    *[
                        tool_name
                        for item in list(stage_runtime_tool_events or [])
                        for tool_name in list(item.get("tool_names") or [])
                    ],
                ]
            ),
            "actual_tool_call_count": sum(
                len(list(item.get("tool_calls") or []))
                for item in list(stage_llm_calls or [])
            ) + sum(len(list(item.get("tool_names") or [])) for item in list(stage_runtime_tool_events or [])),
            "runtime_tool_events": stage_runtime_tool_events,
            "runtime_tool_event_count": len(stage_runtime_tool_events),
            "dialog_count": len(stage_dialogs),
            **_aggregate_llm_calls(stage_llm_calls),
        }
        with self._lock:
            self._upsert_stage_export(stage_summary)
        self.flush_case_debug_bundle()

        return exports
