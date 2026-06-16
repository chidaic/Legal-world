"""Legal Consultation (LC) scenario."""

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

from .base_scenario import BaseScenario
from ..player_lawyer.responsibility_marker import build_player_responsibility_marker
from ..utils.runtime_flags import player_lawyer_ai_surrogate_enabled, player_lawyer_mode_for_frontend
from ..utils.prompt_profile import resolve_prompt_profile_max_turns

logger = logging.getLogger(__name__)

DEFAULT_AGENT_STEP_TIMEOUT_SECONDS = 90.0


class LegalConsultationScenario(BaseScenario):
    """Free-form legal consultation without runtime control tokens."""

    DEFAULT_MAX_TURNS = 15
    END_MARKER = "【咨询结束】"
    END_MARKER_PATTERN = re.compile(r"(?:【\s*)?(?:咨询结束|结束对话)(?:\s*】)?")
    OPENING_PROMPT = "请自然开始当前咨询。"
    LAWYER_OPENING_PROMPT = "请以当前接待律师身份自然开启咨询，先简短问候，再请当事人说明主要情况。"
    scenario_type = "LC"
    CLIENT_REPLY_MARKERS = (
        "请律师回复",
        "请律师回答",
        "律师回复",
        "律师答复",
    )
    CLIENT_ROLE_PREFIXES = re.compile(r"^\s*(?:律师回复|律师答复|律师)[：:]\s*")

    def __init__(
        self,
        client_agent,
        lawyer_agent,
        max_turns: Optional[int] = None,
        output_path: Optional[str] = None,
        verbose: bool = False,
        map_engine: Optional[Any] = None,
        checkpoint_manager: Optional[Any] = None,
        scenario_id: Optional[str] = None,
        **kwargs,
    ):
        agents = {
            "client": client_agent,
            "lawyer": lawyer_agent,
        }
        resolved_max_turns = (
            resolve_prompt_profile_max_turns(self.scenario_type, self.DEFAULT_MAX_TURNS)
            if max_turns is None
            else max_turns
        )
        super().__init__(
            agents=agents,
            max_turns=resolved_max_turns,
            verbose=verbose,
            map_engine=map_engine,
            checkpoint_manager=checkpoint_manager,
            scenario_id=scenario_id,
            **kwargs,
        )
        self.output_path = output_path
        self.finish_reason = "max_turns"

    def _agent_step_timeout_seconds(self) -> float:
        try:
            return max(
                1.0,
                float(os.getenv("LC_AGENT_STEP_TIMEOUT_SECONDS", DEFAULT_AGENT_STEP_TIMEOUT_SECONDS)),
            )
        except (TypeError, ValueError):
            return DEFAULT_AGENT_STEP_TIMEOUT_SECONDS

    def _fallback_agent_step_response(self, agent: Any, instruction: str) -> str:
        role = "client" if agent is self.agents.get("client") else "lawyer"
        party_role = self._resolve_party_role()
        if party_role == "defendant":
            if role == "client":
                if self.dialog_history:
                    return "我明白了，会按您说的准备答辩材料。谢谢律师。【咨询结束】"
                return "我收到了起诉材料，想咨询如何应诉、准备证据和答辩。"
            return (
                "我先按起诉状和现有材料为您梳理应诉思路：核对原告诉请和证据，"
                "准备反驳事实、己方证据和答辩要点，下一步起草答辩状。"
            )
        if role == "client":
            if self.dialog_history:
                return "我明白了，会按您说的准备起诉材料和证据。谢谢律师。【咨询结束】"
            return "我想咨询一下这起纠纷该怎么维权、准备哪些证据以及如何起诉。"
        return (
            "我先按现有材料为您梳理维权思路：核对事故经过、责任认定、损失凭证和赔偿项目，"
            "下一步准备起诉状及配套证据。"
        )

    def _resolve_party_role(self) -> str:
        if self.trace_stage_code == "DLC" or self.trace_stage_key.startswith("DLC"):
            return "defendant"
        if self.trace_stage_code == "PLC" or self.trace_stage_key.startswith("PLC"):
            return "plaintiff"

        client = self.agents.get("client")
        raw_party_role = getattr(client, "party_role", "")
        if callable(raw_party_role):
            try:
                raw_party_role = raw_party_role()
            except Exception:
                raw_party_role = ""
        if str(raw_party_role or "").strip().lower() == "defendant":
            return "defendant"

        scenario_data = getattr(client, "scenario_data", None)
        if isinstance(scenario_data, dict):
            raw_party_role = scenario_data.get("party_role")
            if str(raw_party_role or "").strip().lower() == "defendant":
                return "defendant"

        for value in (
            getattr(client, "agent_id", ""),
            getattr(client, "config_path", ""),
        ):
            if "defendant" in str(value or "").lower():
                return "defendant"

        return "plaintiff"

    async def _run_agent_step(self, agent: Any, instruction: str) -> str:
        if self._agent_step_expects_player_input(agent):
            return await asyncio.to_thread(agent.step, instruction)
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(agent.step, instruction),
                timeout=self._agent_step_timeout_seconds(),
            )
        except asyncio.TimeoutError:
            logger.warning(
                "[LegalConsultationScenario] Agent step timed out: agent=%s; using fallback response",
                getattr(agent, "agent_id", getattr(agent, "name", "")),
            )
            return self._fallback_agent_step_response(agent, instruction)

    def _agent_step_expects_player_input(self, agent: Any) -> bool:
        checker = getattr(agent, "expects_player_input_for_current_step", None)
        if not callable(checker):
            return False
        try:
            return bool(checker())
        except Exception:
            logger.exception(
                "[LegalConsultationScenario] Failed to inspect player input state: agent=%s",
                getattr(agent, "agent_id", getattr(agent, "name", "")),
            )
            return False

    async def _run_lawyer_opening_step(self, lawyer: Any) -> str:
        auto_opening = getattr(lawyer, "build_auto_opening_response", None)
        if callable(auto_opening):
            return str(auto_opening(self.LAWYER_OPENING_PROMPT)).strip()
        return await self._run_agent_step(lawyer, self.LAWYER_OPENING_PROMPT)

    @classmethod
    def _sanitize_client_message(cls, content: str) -> str:
        normalized = str(content or "").strip()
        if not normalized:
            return ""

        for marker in cls.CLIENT_REPLY_MARKERS:
            marker_index = normalized.find(marker)
            if marker_index > 0:
                normalized = normalized[:marker_index].rstrip("：: \n")
                break

        return cls.CLIENT_ROLE_PREFIXES.sub("", normalized).strip()

    @classmethod
    def _contains_end_marker(cls, content: str) -> bool:
        return bool(cls.END_MARKER_PATTERN.search(str(content or "")))

    async def _stream_dialogue(
        self,
        role: str,
        content: str,
        *,
        duration: Optional[float] = None,
    ) -> None:
        if not self.map_engine or not content:
            return

        agent = self.agents.get(role)
        if not agent or not getattr(agent, "agent_id", ""):
            return

        client = self.agents.get("client")
        case_id = getattr(client, "case_id", "") or getattr(agent, "case_id", "")
        speaker_name = getattr(agent, "name", role)
        turn = self.dialog_history[-1].get("turn", self.turn_count) if self.dialog_history else self.turn_count
        generation_duration_seconds = (
            self.dialog_history[-1].get("generation_duration_seconds")
            if self.dialog_history
            and str(self.dialog_history[-1].get("role", "") or "") == role
            and str(self.dialog_history[-1].get("content", "") or "").strip() == str(content or "").strip()
            else None
        )
        generation_total_tokens = (
            self.dialog_history[-1].get("generation_total_tokens")
            if self.dialog_history
            and str(self.dialog_history[-1].get("role", "") or "") == role
            and str(self.dialog_history[-1].get("content", "") or "").strip() == str(content or "").strip()
            else None
        )

        if case_id and hasattr(self.map_engine, "broadcast_dialogue"):
            marker = build_player_responsibility_marker(
                role=role,
                stage=self.trace_stage_code or self.scenario_type,
                player_lawyer_enabled=self._player_lawyer_enabled_for_frontend(),
                ai_surrogate_enabled=player_lawyer_ai_surrogate_enabled(),
                content=content,
            )
            await self.map_engine.broadcast_dialogue(
                case_id,
                agent.agent_id,
                speaker_name,
                content,
                turn,
                scenario_type=self.trace_stage_code or self.scenario_type,
                generation_duration_seconds=generation_duration_seconds,
                generation_total_tokens=generation_total_tokens,
                **(marker or {}),
            )

        if hasattr(self.map_engine, "send_update_dialogue"):
            await self.map_engine.send_update_dialogue(agent.agent_id, content, duration)
        elif hasattr(self.map_engine, "show_bubble"):
            await self.map_engine.show_bubble(agent.agent_id, content, duration or 3.0)

    def _player_lawyer_enabled_for_frontend(self) -> bool:
        if not self.map_engine:
            return False
        supports_fn = getattr(self.map_engine, "supports_player_v2_runtime", None)
        supports_player_v2 = bool(supports_fn()) if callable(supports_fn) else False
        mode = player_lawyer_mode_for_frontend(
            frontend_mode=getattr(self.map_engine, "_frontend_mode", None),
            has_player_v2_client=supports_player_v2,
        )
        return mode == "plaintiff"

    async def execute(self) -> Dict[str, Any]:
        client = self.agents["client"]
        lawyer = self.agents["lawyer"]

        self._log("开始法律咨询场景")

        await self._check_pause()
        lawyer_opening = await self._run_lawyer_opening_step(lawyer)
        self._add_dialog("lawyer", lawyer_opening)
        await self._stream_dialogue("lawyer", lawyer_opening, duration=3.0)

        await self._check_pause()
        client_message = self._sanitize_client_message(
            await self._run_agent_step(client, lawyer_opening)
        )
        self._add_dialog("client", client_message)
        await self._stream_dialogue("client", client_message, duration=2.0)

        if self._contains_end_marker(client_message):
            self.completed = True
            self.finish_reason = "end_marker"
            result = self._build_result()
            if self.output_path:
                self._save_result(result)
            return result

        while self.turn_count < self.max_turns:
            await self._check_pause()
            self._save_checkpoint_if_needed()
            if hasattr(lawyer, "set_stage_turn_index"):
                lawyer.set_stage_turn_index(self.turn_count)
            lawyer_response = await self._run_agent_step(lawyer, client_message)
            self._add_dialog("lawyer", lawyer_response)
            await self._stream_dialogue("lawyer", lawyer_response, duration=3.0)

            self.turn_count += 1
            if self.turn_count >= self.max_turns:
                self.completed = True
                self.finish_reason = "turn_limit_reached"
                break

            await self._check_pause()
            client_message = self._sanitize_client_message(
                await self._run_agent_step(client, lawyer_response)
            )
            self._add_dialog("client", client_message)
            await self._stream_dialogue("client", client_message, duration=2.0)

            if self._contains_end_marker(client_message):
                self.completed = True
                self.finish_reason = "end_marker"
                break

        if not self.completed:
            self.completed = True
            self.finish_reason = "turn_limit_reached"

        result = self._build_result()
        if self.output_path:
            self._save_result(result)
        return result

    def _build_result(self) -> Dict[str, Any]:
        return {
            "scenario_type": self.scenario_type,
            "dialog_history": self.dialog_history,
            "turn_count": self.turn_count,
            "completed": self.completed,
            "finish_reason": self.finish_reason if self.completed else "max_turns",
        }

    def _save_result(self, result: Dict[str, Any]) -> None:
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
        self._log(f"结果已保存到 {self.output_path}")

    def _build_checkpoint_data(self) -> Dict[str, Any]:
        client = self.agents.get("client")
        lawyer = self.agents.get("lawyer")
        return {
            "scenario_type": self.scenario_type,
            "case_id": getattr(client, "case_id", ""),
            "client_id": getattr(client, "agent_id", ""),
            "lawyer_id": getattr(lawyer, "agent_id", ""),
            "dialog_history": self.dialog_history,
            "turn_count": self.turn_count,
            "completed": self.completed,
            "finish_reason": self.finish_reason,
        }

    async def resume_from_checkpoint(self, checkpoint_data: Dict[str, Any]) -> Dict[str, Any]:
        self.dialog_history = checkpoint_data.get("dialog_history", [])
        self.turn_count = checkpoint_data.get("turn_count", 0)
        self.completed = checkpoint_data.get("completed", False)
        self.finish_reason = checkpoint_data.get("finish_reason", self.finish_reason)

        if self.completed:
            return self._build_result()

        if not self.dialog_history:
            return await self.execute()

        last_dialog = self.dialog_history[-1]
        client = self.agents["client"]
        lawyer = self.agents["lawyer"]

        if last_dialog.get("role") == "lawyer":
            await self._check_pause()
            client_message = self._sanitize_client_message(
                await self._run_agent_step(client, str(last_dialog.get("content") or ""))
            )
            self._add_dialog("client", client_message)
            await self._stream_dialogue("client", client_message, duration=2.0)
            if self._contains_end_marker(client_message):
                self.completed = True
                self.finish_reason = "end_marker"
        elif last_dialog.get("role") == "client":
            client_message = self._sanitize_client_message(str(last_dialog.get("content") or ""))
        else:
            return await self.execute()

        if self.turn_count >= self.max_turns and not self.completed:
            self.completed = True
            self.finish_reason = "turn_limit_reached"

        while self.turn_count < self.max_turns and client_message:
            await self._check_pause()
            self._save_checkpoint_if_needed()
            if hasattr(lawyer, "set_stage_turn_index"):
                lawyer.set_stage_turn_index(self.turn_count)
            lawyer_response = await self._run_agent_step(lawyer, client_message)
            self._add_dialog("lawyer", lawyer_response)
            await self._stream_dialogue("lawyer", lawyer_response, duration=3.0)

            self.turn_count += 1
            if self.turn_count >= self.max_turns:
                self.completed = True
                self.finish_reason = "turn_limit_reached"
                break

            await self._check_pause()
            client_message = self._sanitize_client_message(
                await self._run_agent_step(client, lawyer_response)
            )
            self._add_dialog("client", client_message)
            await self._stream_dialogue("client", client_message, duration=2.0)

            if self._contains_end_marker(client_message):
                self.completed = True
                self.finish_reason = "end_marker"
                break

        if not self.completed:
            self.completed = True
            self.finish_reason = "turn_limit_reached"

        result = self._build_result()
        if self.output_path:
            self._save_result(result)
        return result
