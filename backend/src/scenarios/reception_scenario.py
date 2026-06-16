"""前台导引场景 (ReceptionScenario)。

前导场景：前台接待员与当事人进行 1 轮对话，
了解法律需求并从律所名册中推荐最合适的律师。
"""

import asyncio
import json
import re
import logging
from typing import Any, Dict, Optional
from pathlib import Path

from .base_scenario import BaseScenario


logger = logging.getLogger(__name__)


class ReceptionScenario(BaseScenario):
    """前台导引场景 — 1 轮对话完成律师推荐。

    Flow:
    1. 前台问候并询问需求，当事人陈述，前台直接推荐律师（输出【推荐律师：lawyer_xxx】）

    Attributes:
        scenario_type: 场景类型标识
        MAX_ROUNDS: 最大对话轮数
        matched_lawyer_id: 匹配到的律师 ID
    """

    scenario_type = "RECEPTION"
    MAX_ROUNDS = 1
    CLIENT_REPLY_MARKERS = ("请律师回复", "请前台回复", "律师回复", "前台回复")
    CLIENT_ROLE_PREFIXES = re.compile(r"^\s*(?:律师回复|前台回复|律师|前台)[：:]\s*")

    def __init__(
        self,
        receptionist_agent,
        client_agent,
        lawyer_roster: dict,
        preferred_lawyer_id: str = "",
        preferred_lawyer_name: str = "",
        output_path: Optional[str] = None,
        verbose: bool = False,
        map_engine: Optional[Any] = None,
        trace_recorder: Optional[Any] = None,
        trace_stage_code: Optional[str] = None,
        trace_stage_key: Optional[str] = None,
    ):
        agents = {
            "receptionist": receptionist_agent,
            "client": client_agent,
        }
        super().__init__(
            agents=agents,
            max_turns=self.MAX_ROUNDS,
            verbose=verbose,
            map_engine=map_engine,
            trace_recorder=trace_recorder,
            trace_stage_code=trace_stage_code,
            trace_stage_key=trace_stage_key,
        )
        self.lawyer_roster = lawyer_roster
        self.preferred_lawyer_id = str(preferred_lawyer_id or "").strip()
        self.preferred_lawyer_name = str(preferred_lawyer_name or "").strip()
        self.output_path = output_path
        self.matched_lawyer_id = ""
        self.match_status = "unresolved"
        self._dialogue_sequence = 0

    async def _stream_dialogue(self, role: str, content: str, duration: float = 2.0) -> None:
        if not self.map_engine:
            return

        agent = self.agents[role]
        case_id = getattr(self.agents.get("client"), "case_id", "") or getattr(agent, "case_id", "")
        self._dialogue_sequence += 1
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
            await self.map_engine.broadcast_dialogue(
                case_id,
                agent.agent_id,
                getattr(agent, "name", agent.agent_id),
                content,
                self._dialogue_sequence,
                scenario_type="RECEPTION",
                generation_duration_seconds=generation_duration_seconds,
                generation_total_tokens=generation_total_tokens,
            )

        if hasattr(self.map_engine, "send_update_dialogue"):
            await self.map_engine.send_update_dialogue(agent.agent_id, content, duration)

        await asyncio.sleep(duration)

    @classmethod
    def _sanitize_client_need(cls, content: str) -> str:
        normalized = str(content or "").strip()
        if not normalized:
            return ""

        for marker in cls.CLIENT_REPLY_MARKERS:
            marker_index = normalized.find(marker)
            if marker_index > 0:
                normalized = normalized[:marker_index].rstrip("：: \n")
                break

        normalized = cls.CLIENT_ROLE_PREFIXES.sub("", normalized).strip()
        return normalized

    def _resolve_preferred_lawyer_name(self) -> str:
        if self.preferred_lawyer_name:
            return self.preferred_lawyer_name
        if not self.preferred_lawyer_id:
            return ""

        for lawyer in self.lawyer_roster.get("lawyers", []):
            if str(lawyer.get("id", "") or "").strip() == self.preferred_lawyer_id:
                return str(lawyer.get("name", "") or "").strip()
        return ""

    def _normalize_recommendation(self, recommendation: str) -> tuple[str, str, str]:
        lawyer_id, status = self._extract_match_result(recommendation)
        if self.preferred_lawyer_id and lawyer_id != self.preferred_lawyer_id:
            preferred_name = self._resolve_preferred_lawyer_name() or self.preferred_lawyer_id
            recommendation = (
                f"【推荐律师：{self.preferred_lawyer_id}】\n"
                f"建议优先由{preferred_name}律师承办当前案件。"
            )
            return recommendation, self.preferred_lawyer_id, "matched"
        return recommendation, lawyer_id, status

    async def execute(self) -> Dict[str, Any]:
        """执行前台导引场景（异步）。"""
        receptionist = self.agents["receptionist"]
        client = self.agents["client"]

        self._log("开始前台导引场景")

        roster_str = self._format_roster()

        # ── Round 1: 前台问候并询问需求 → 当事人陈述 → 前台直接推荐 ──
        self.turn_count = 1

        # 检查暂停状态
        await self._check_pause()

        # 前台问候
        greeting = await asyncio.to_thread(
            receptionist.step,
            f"一位当事人来到律所前台寻求法律帮助。\n"
            f"律所律师名册：\n{roster_str}\n\n"
            f"请问候当事人并询问其法律需求。"
        )
        self._add_dialog("receptionist", greeting)
        self._log(f"[Round 1] 前台: {greeting}")
        await self._stream_dialogue("receptionist", greeting)

        # 检查暂停状态
        await self._check_pause()

        # 当事人陈述需求
        client_need = self._sanitize_client_need(await asyncio.to_thread(
            client.step,
            "前台刚刚问候并询问你的法律需求。\n"
            f"前台的话：{greeting}\n\n"
            "请你只用当事人口吻、一两句话说明本次来访的法律需求。"
            "不要代替前台或律师发言，不要写“律师：”“前台：”或模拟下一轮回复。"
        ))
        self._add_dialog("client", client_need)
        self._log(f"[Round 1] 当事人: {client_need}")
        await self._stream_dialogue("client", client_need)

        # 检查暂停状态
        await self._check_pause()

        # 前台直接推荐律师
        preferred_hint = ""
        if self.preferred_lawyer_id:
            preferred_name = self._resolve_preferred_lawyer_name() or self.preferred_lawyer_id
            preferred_hint = (
                "\n系统提示：当前案件已在本所预配置律师中匹配到"
                f"{preferred_name}（{self.preferred_lawyer_id}）；"
                "如该律师状态为 available，必须优先推荐该律师。"
            )
        recommendation = await asyncio.to_thread(
            receptionist.step,
            f"当事人说：「{client_need}」\n\n"
            f"请根据当事人需求和律所名册，推荐最合适的律师。\n"
            f"必须输出推荐律师ID，格式：【推荐律师：lawyer_xxx】\n"
            f"{preferred_hint}\n"
            f"并简要说明推荐理由。"
        )
        recommendation, self.matched_lawyer_id, self.match_status = self._normalize_recommendation(recommendation)
        self._add_dialog("receptionist", recommendation)
        self._log(f"[Round 1] 前台推荐: {recommendation}")
        await self._stream_dialogue("receptionist", recommendation)
        if self.map_engine:
            # 清除气泡
            await self.map_engine.send_end_interaction(receptionist.agent_id)
            await self.map_engine.send_end_interaction(client.agent_id)

        self.completed = True
        self._log(f"前台导引完成，推荐律师: {self.matched_lawyer_id}")

        result = self._build_result()
        if self.output_path:
            self._save_result(result)
        return result

    def _format_roster(self) -> str:
        """将律师名册格式化为可读文本。"""
        lawyers = self.lawyer_roster.get("lawyers", [])
        lines = []
        for lawyer in lawyers:
            name = lawyer.get("name", "")
            lid = lawyer.get("id", "")
            specialty = "、".join(lawyer.get("specialty", []))
            seniority = lawyer.get("seniority", "")
            status = lawyer.get("status", "")
            case_hint = " | 当前案件：优先承办" if lid and lid == self.preferred_lawyer_id else ""
            lines.append(
                f"- {name}（{lid}）| 专长：{specialty or '综合'} | "
                f"资历：{seniority} | 状态：{status}{case_hint}"
            )
        return "\n".join(lines) if lines else "暂无律师信息"

    def _extract_match_result(self, text: str) -> tuple[str, str]:
        """从前台推荐文本中提取 lawyer_id 或无匹配结论。"""
        normalized_text = str(text or "")
        if re.search(r"【推荐律师[：:]\s*暂无匹配】", normalized_text):
            return "", "no_match"

        # 匹配【推荐律师：lawyer_xxx】
        match = re.search(r'【推荐律师[：:]\s*(lawyer_\w+)】', normalized_text)
        if match:
            return match.group(1), "matched"
        # 仅在推荐字段附近做宽松匹配，避免误吃律师名册中的 ID
        match = re.search(r'(?:推荐律师|律师编号)[：:\s]*(lawyer_\w+)', normalized_text)
        if match:
            return match.group(1), "matched"
        parenthesized_ids = {
            lawyer_id.strip()
            for lawyer_id in re.findall(r'[（(]\s*(lawyer_\w+)\s*[)）]', normalized_text)
            if lawyer_id.strip()
        }
        if len(parenthesized_ids) == 1:
            return next(iter(parenthesized_ids)), "matched"

        lawyer_name_map = {
            str(lawyer.get("name", "")).strip(): str(lawyer.get("id", "")).strip()
            for lawyer in self.lawyer_roster.get("lawyers", [])
            if str(lawyer.get("name", "")).strip() and str(lawyer.get("id", "")).strip()
        }
        name_match = re.search(r"(?:推荐律师|推荐|律师)[：:\s]*([^\s，。,；;】）)]+)", normalized_text)
        if name_match:
            lawyer_id = lawyer_name_map.get(name_match.group(1).strip())
            if lawyer_id:
                return lawyer_id, "matched"

        no_match_hints = (
            "均不直接覆盖",
            "均不涵盖",
            "暂无合适律师",
            "暂无匹配",
            "专业方向均不",
            "协调其他",
            "合作律所",
        )
        if any(hint in normalized_text for hint in no_match_hints):
            return "", "no_match"
        logger.warning("[ReceptionScenario] 未能从推荐文本中提取 lawyer_id")
        return "", "unresolved"

    def _build_result(self) -> Dict[str, Any]:
        return {
            "scenario_type": self.scenario_type,
            "matched_lawyer_id": self.matched_lawyer_id,
            "match_status": self.match_status,
            "dialog_history": self.dialog_history,
            "turn_count": self.turn_count,
            "completed": self.completed,
        }

    def _save_result(self, result: Dict[str, Any]) -> None:
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        self._log(f"结果已保存到 {self.output_path}")

    def _build_checkpoint_data(self) -> Dict[str, Any]:
        """构建检查点数据。"""
        return {
            "scenario_type": self.scenario_type,
            "matched_lawyer_id": self.matched_lawyer_id,
            "match_status": self.match_status,
            "dialog_history": self.dialog_history,
            "turn_count": self.turn_count,
            "completed": self.completed,
            "lawyer_roster": self.lawyer_roster,
            "output_path": self.output_path,
            "dialogue_sequence": self._dialogue_sequence,
        }

    async def resume_from_checkpoint(self, checkpoint_data: Dict[str, Any]) -> Dict[str, Any]:
        """从检查点恢复场景执行。

        前台导引场景通常很短（1轮），不需要复杂的恢复逻辑。
        如果已完成，直接返回结果；否则重新执行。
        """
        self.matched_lawyer_id = checkpoint_data.get("matched_lawyer_id", "")
        self.match_status = checkpoint_data.get("match_status", "unresolved")
        self.dialog_history = checkpoint_data.get("dialog_history", [])
        self.turn_count = checkpoint_data.get("turn_count", 0)
        self.completed = checkpoint_data.get("completed", False)
        self._dialogue_sequence = checkpoint_data.get("dialogue_sequence", 0)

        if self.completed:
            self._log("前台导引场景已完成，直接返回结果")
            return self._build_result()

        # 未完成则重新执行
        self._log("前台导引场景未完成，重新执行")
        return await self.execute()
