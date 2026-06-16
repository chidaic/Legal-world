"""Appeal Drafting (AD) scenario."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .base_scenario import BaseScenario
from ..tools.legal import (
    get_document_drafting_tool_name,
    render_document_drafting_payload,
)
from ..tools.legal.document_drafting_support import extract_document_body
from ..utils.prompt_profile import resolve_prompt_profile_max_turns
from .drafting_runtime import (
    DRAFTING_MAX_TURNS,
    build_forced_document_prompt,
    capture_drafting_tool_payload,
    has_document_payload,
    is_stalled_drafting_dialogue,
    missing_document_error,
)


logger = logging.getLogger(__name__)


class AppealDraftingScenario(BaseScenario):
    END_MARKER = "【起草结束】"
    OPENING_PROMPT = "请自然开始当前交流。"
    DOCUMENT_TITLE = "民事上诉状"
    scenario_type = "AD"

    def __init__(
        self,
        appellant_agent,
        lawyer_agent,
        max_turns: Optional[int] = None,
        output_path: Optional[str] = None,
        verbose: bool = False,
        **kwargs,
    ):
        agents = {
            "appellant": appellant_agent,
            "lawyer": lawyer_agent,
        }
        configured_max_turns = resolve_prompt_profile_max_turns(
            self.scenario_type,
            DRAFTING_MAX_TURNS,
        )
        resolved_max_turns = (
            configured_max_turns if max_turns is None else min(max_turns, configured_max_turns)
        )
        super().__init__(agents=agents, max_turns=resolved_max_turns, verbose=verbose, **kwargs)
        self.output_path = output_path
        self.appeal_statement = ""
        self._drafted_document_payload: Dict[str, str] = {}
        self.finish_reason = "max_turns"

    def _require_runtime_tools(self, lawyer: Any) -> None:
        existing_names = {
            tool.get_function_name()
            for tool in list(getattr(lawyer, "tools", []) or [])
            if tool is not None
        }
        tool_name = get_document_drafting_tool_name(self.scenario_type)
        if tool_name not in existing_names:
            raise RuntimeError(
                f"Appeal drafting requires runtime tool '{tool_name}' to be preloaded."
            )

    def _capture_drafting_tool_result(self, lawyer: Any) -> None:
        try:
            payload = capture_drafting_tool_payload(lawyer, scenario_type=self.scenario_type)
        except Exception:
            return

        if has_document_payload(payload):
            self._drafted_document_payload = payload

    @staticmethod
    def _is_retryable_step_error(exc: Exception) -> bool:
        return isinstance(exc, json.JSONDecodeError) or exc.__class__.__name__ == "JSONDecodeError"

    def _step_with_retry(self, agent: Any, prompt: str) -> str:
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                return agent.step(prompt)
            except Exception as exc:
                last_error = exc
                if not self._is_retryable_step_error(exc) or attempt > 0:
                    raise
                logger.warning(
                    "Retrying AD agent step after JSON decode error: agent=%s attempt=%s error=%s",
                    getattr(agent, "name", getattr(agent, "agent_id", "unknown")),
                    attempt + 1,
                    exc,
                )
        if last_error is not None:
            raise last_error
        return ""

    def execute(self) -> Dict[str, Any]:
        appellant = self.agents["appellant"]
        lawyer = self.agents["lawyer"]

        self._log("开始上诉状起草场景")
        self._require_runtime_tools(lawyer)

        self._check_pause_sync()
        appellant_message = self._step_with_retry(appellant, self.OPENING_PROMPT)
        self._add_dialog("appellant", appellant_message)

        while self.turn_count < self.max_turns:
            self._check_pause_sync()
            lawyer_response = self._step_with_retry(lawyer, appellant_message)
            self._add_dialog("lawyer", lawyer_response)
            self._capture_drafting_tool_result(lawyer)
            if self.END_MARKER in lawyer_response:
                if self._has_final_appeal(lawyer_response):
                    self.completed = True
                    self.finish_reason = "end_marker"
                    break
                logger.warning("Ignoring AD end marker without full appeal body.")
            elif has_document_payload(self._drafted_document_payload):
                self.completed = True
                self.finish_reason = "draft_tool"
                break
            elif is_stalled_drafting_dialogue(
                party_message=appellant_message,
                lawyer_response=lawyer_response,
                turn_count=self.turn_count,
            ):
                logger.warning("Forcing AD document after stalled drafting dialogue.")
                forced_response = self._step_with_retry(
                    lawyer,
                    build_forced_document_prompt(
                        scenario_type=self.scenario_type,
                        document_title=self.DOCUMENT_TITLE,
                        end_marker=self.END_MARKER,
                    ),
                )
                self._add_dialog("lawyer", forced_response)
                self._capture_drafting_tool_result(lawyer)
                if self._has_final_appeal(forced_response):
                    self.completed = True
                    self.finish_reason = "forced_document_after_stalled_dialogue"
                    break

            self.turn_count += 1
            self._check_pause_sync()
            appellant_message = self._step_with_retry(appellant, lawyer_response)
            self._add_dialog("appellant", appellant_message)

        if not self.completed:
            self.finish_reason = "turn_limit_reached"
            self._force_final_appeal(lawyer, appellant_message)

        if not self.completed:
            self.completed = True

        self.appeal_statement = self._extract_appeal() or str(
            self._drafted_document_payload.get("document_text", "") or ""
        ).strip()
        if not self.appeal_statement.strip():
            self.completed = False
            self.finish_reason = f"{self.finish_reason}_without_document"
            result = self._build_result()
            if self.output_path:
                self._save_result(result)
            raise missing_document_error(
                scenario_type=self.scenario_type,
                document_label=self.DOCUMENT_TITLE,
                finish_reason=self.finish_reason,
                turn_count=self.turn_count,
            )
        self._ensure_pdf_output(lawyer)
        result = self._build_result()
        if self.output_path:
            self._save_result(result)
        return result

    def _extract_appeal(self) -> str:
        for entry in reversed(self.dialog_history):
            if entry["role"] != "lawyer":
                continue
            text = extract_document_body(
                entry.get("content", ""),
                document_title=self.DOCUMENT_TITLE,
                end_marker=self.END_MARKER,
            )
            if text:
                return text
        return ""

    def _has_final_appeal(self, response_text: str) -> bool:
        if extract_document_body(
            response_text,
            document_title=self.DOCUMENT_TITLE,
            end_marker=self.END_MARKER,
        ):
            return True
        return bool(str(self._drafted_document_payload.get("document_text", "") or "").strip())

    def _ensure_pdf_output(self, lawyer: Any) -> None:
        if self._drafted_document_payload.get("pdf_path") or not self.appeal_statement.strip():
            return
        try:
            self._drafted_document_payload = render_document_drafting_payload(
                lawyer,
                document_type=self.scenario_type,
                document_text=self.appeal_statement,
            )
        except Exception as exc:
            logger.warning("Failed to backfill appeal PDF output: %s", exc)

    def _build_result(self) -> Dict[str, Any]:
        return {
            "scenario_type": self.scenario_type,
            "dialog_history": self.dialog_history,
            "turn_count": self.turn_count,
            "completed": self.completed,
            "finish_reason": self.finish_reason,
            "appeal_statement": self.appeal_statement,
            "drafted_document_payload": self._drafted_document_payload,
            "pdf_path": str(self._drafted_document_payload.get("pdf_path", "") or ""),
        }

    def _save_result(self, result: Dict[str, Any]) -> None:
        if not self.output_path:
            return
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
        self._log(f"结果已保存到 {self.output_path}")

    def _force_final_appeal(self, lawyer: Any, appellant_message: str) -> None:
        prompt = build_forced_document_prompt(
            party_message=appellant_message,
            document_title=self.DOCUMENT_TITLE,
            end_marker=self.END_MARKER,
            tool_name=get_document_drafting_tool_name(self.scenario_type),
        )
        lawyer_response = self._step_with_retry(lawyer, prompt)
        self._add_dialog("lawyer", lawyer_response)
        self._capture_drafting_tool_result(lawyer)
        if self.END_MARKER in lawyer_response and self._has_final_appeal(lawyer_response):
            self.completed = True
            self.finish_reason = "forced_end_marker"
        elif has_document_payload(self._drafted_document_payload):
            self.completed = True
            self.finish_reason = "forced_draft_tool"
        else:
            self.finish_reason = "forced_draft_failed"

    def _build_checkpoint_data(self) -> Dict[str, Any]:
        return {
            "scenario_type": self.scenario_type,
            "dialog_history": self.dialog_history,
            "turn_count": self.turn_count,
            "completed": self.completed,
            "appeal_statement": self.appeal_statement,
            "drafted_document_payload": self._drafted_document_payload,
            "finish_reason": self.finish_reason,
        }

    async def resume_from_checkpoint(self, checkpoint_data: Dict[str, Any]) -> Dict[str, Any]:
        self.dialog_history = checkpoint_data.get("dialog_history", [])
        self.turn_count = checkpoint_data.get("turn_count", 0)
        self.completed = checkpoint_data.get("completed", False)
        self.appeal_statement = checkpoint_data.get("appeal_statement", "")
        self._drafted_document_payload = checkpoint_data.get("drafted_document_payload", {}) or {}
        self.finish_reason = checkpoint_data.get("finish_reason", self.finish_reason)

        if self.completed:
            return self._build_result()

        return self.execute()
