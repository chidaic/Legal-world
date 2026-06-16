"""Court Investigation Scenario (CI) for SimLawFirm framework.

This module simulates the complete civil first-instance trial process,
orchestrating the interaction between the Judge, both parties, and both lawyers.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime

from .base_scenario import BaseScenario
from ..tools.legal import (
    extract_judgment_document_tool_payload,
    get_judgment_document_type_for_scenario,
    render_judgment_document_payload,
)


logger = logging.getLogger(__name__)


class CourtInvestigationScenario(BaseScenario):
    """Court Investigation Scenario.
    
    Flow:
    1. Opening Session (开庭审理) - Identity verification, rights notification
    2. Court Investigation (法庭调查) - Fact stating, evidence presentation, cross-examination
    3. Court Debate (庭审辩论) - Debate on key issues (multi-round)
    4. Mediation (法庭调解) - Mediation willingness check
    5. Final Judgment (最终宣判) - Judgment delivery
    """
    
    def __init__(
        self,
        judge_agent: Any,
        plaintiff_agent: Any,
        defendant_agent: Any,
        plaintiff_lawyer_agent: Optional[Any] = None,
        defendant_lawyer_agent: Optional[Any] = None,
        plaintiff_witnesses: Optional[List[str]] = None,
        defendant_witnesses: Optional[List[str]] = None,
        max_debate_rounds: int = 4,
        verbose: bool = False,
        court_finding: str = "",
        court_opinion: str = "",
        output_path: Optional[str] = None,
        **kwargs
    ):
        """Initialize Court Investigation Scenario.
        
        Args:
            judge_agent: The Judge Agent
            plaintiff_agent: The Plaintiff Agent
            defendant_agent: The Defendant Agent
            plaintiff_lawyer_agent: The Plaintiff Lawyer Agent
            defendant_lawyer_agent: The Defendant Lawyer Agent
            max_debate_rounds: Maximum rounds for court debate
            verbose: Whether to print detailed logs
            court_finding: Court finding for judge to reference
            court_opinion: Court opinion for judge to reference
            output_path: Optional path to save result JSON
            **kwargs: Additional configuration
        """
        agents = {
            "judge": judge_agent,
            "plaintiff": plaintiff_agent,
            "defendant": defendant_agent
        }
        if plaintiff_lawyer_agent is not None:
            agents["plaintiff_lawyer"] = plaintiff_lawyer_agent
        if defendant_lawyer_agent is not None:
            agents["defendant_lawyer"] = defendant_lawyer_agent
        super().__init__(agents=agents, verbose=verbose, **kwargs)
        
        self.max_debate_rounds = max_debate_rounds
        self.court_finding = court_finding
        self.court_opinion = court_opinion
        self.output_path = output_path
        self.current_stage = "未开始"
        self.stage_results = {}
        self.final_judgment = None
        self._drafted_document_payload = {}
        self.mediation_result = {}
        self.plaintiff_witnesses = self._normalize_witnesses(plaintiff_witnesses)
        self.defendant_witnesses = self._normalize_witnesses(defendant_witnesses)

    def execute(self) -> Dict[str, Any]:
        """Execute the full court investigation process.
        
        Returns:
            Result dictionary containing detailed logs of each stage
        """
        self._log("开始执行法庭调查场景")
        start_time = datetime.now()
        
        # 1. Opening Session
        self._execute_opening_session()
        
        # 2. Court Investigation
        self._execute_court_investigation()
        
        # 3. Court Debate
        self._execute_court_debate()
        
        # 4. Mediation
        mediation_result = self._execute_mediation()
        
        final_judgment = None
        
        if mediation_result.get("success", False):
            # Mediation success, end scenario
            self.completed = True
        else:
            # Mediation failed, Recess and Deliberation (Delayed Judgment)
            self._execute_recess()
            
            # 5. Deliberation and Final Judgment
            # Note: This simulates the judge working alone after the hearing
            final_judgment = self._execute_deliberation_and_judgment()
        
        self.completed = True
        end_time = datetime.now()
        self.final_judgment = final_judgment
        self._ensure_pdf_output(self.agents["judge"])
        self.mediation_result = mediation_result

        result = self._build_result((end_time - start_time).total_seconds())
        
        # Save result if output path specified
        if self.output_path:
            self._save_result(result)
        
        return result

    # Role name mapping for display and broadcast labels
    ROLE_DISPLAY = {
        "judge": "🔨 审判长",
        "clerk": "📝 书记员",
        "plaintiff": "👤 原告",
        "defendant": "👤 被告",
        "plaintiff_lawyer": "📗 原告代理律师",
        "defendant_lawyer": "📕 被告代理律师",
        "plaintiff_witness": "🧾 原告方证人",
        "defendant_witness": "🧾 被告方证人",
    }
    
    ROLE_LABEL = {
        "judge": "审判长",
        "clerk": "书记员",
        "plaintiff": "原告",
        "defendant": "被告",
        "plaintiff_lawyer": "原告代理律师",
        "defendant_lawyer": "被告代理律师",
        "plaintiff_witness": "原告方证人",
        "defendant_witness": "被告方证人",
    }

    PROCEDURAL_JUDGE_TEMPLATES = {
        "核实原告身份": "请原告核对身份信息，并确认到庭参加诉讼。",
        "核实被告身份": "请被告核对身份信息，并确认到庭参加诉讼。",
        "宣布审判庭构成": "现在开庭。本案由审判长主持审理，适用普通程序。",
        "告知诉讼权利义务": "本庭已依法告知双方当事人及诉讼代理人诉讼权利义务。",
        "询问原告方回避申请": "原告是否申请回避？",
        "询问被告方回避申请": "被告是否申请回避？",
        "询问原告是否知晓缺席后果": "原告是否知晓无正当理由不到庭或中途退庭的法律后果？",
        "询问被告是否知晓缺席后果": "被告是否知晓无正当理由不到庭或中途退庭的法律后果？",
        "询问原告方最后陈述": "辩论终结，请原告代理律师作最后陈述。",
        "询问被告方最后陈述": "辩论终结，请被告代理律师作最后陈述。",
        "询问原告方调解意愿": "原告是否同意调解？",
        "询问被告方调解意愿": "被告是否同意调解？",
        "宣布调解": "双方同意调解，本庭休庭进行调解工作。",
        "宣布调解失败": "鉴于一方或双方不同意调解，本庭不再主持调解。",
        "休庭": "现在休庭。本庭将对案件进行评议，择期宣判，请双方当事人及其诉讼代理人退庭。",
    }

    JUDGE_LLM_STEPS = {
        "审判长发问-原告方",
        "审判长发问-被告方",
        "撰写判决书",
    }

    @property
    def plaintiff_lawyer_role(self) -> str:
        return "plaintiff_lawyer" if "plaintiff_lawyer" in self.agents else "plaintiff"

    @property
    def defendant_lawyer_role(self) -> str:
        return "defendant_lawyer" if "defendant_lawyer" in self.agents else "defendant"

    @staticmethod
    def _normalize_witnesses(witnesses: Optional[List[str]]) -> List[str]:
        if witnesses is None:
            return []
        source = witnesses if isinstance(witnesses, list) else [witnesses]
        return [str(item).strip() for item in source if str(item).strip()]

    @staticmethod
    def _parse_witness_entry(entry: str) -> Dict[str, str]:
        normalized = str(entry or "").strip().replace("|", "｜")
        parts = [part.strip() for part in normalized.split("｜") if part.strip()]
        return {
            "name": parts[0] if len(parts) >= 1 else "证人",
            "relation": parts[1] if len(parts) >= 2 else "",
            "testimony": parts[2] if len(parts) >= 3 else "",
        }

    def _execute_fixed_speech(
        self,
        step_name: str,
        speaker_role: str,
        message: str,
    ) -> Dict[str, Any]:
        self.turn_count += 1
        self._log(f"[{self.current_stage}] 步骤: {step_name}")
        self._add_dialog(speaker_role, message)

        if self.verbose:
            print(f"\n{'─' * 50}")
            print(f"  {self.ROLE_DISPLAY.get(speaker_role, speaker_role)}:")
            print(f"{'─' * 50}")
            print(message)

        self._broadcast_message(sender_role=speaker_role, message=message)
        return {
            "step": step_name,
            "speaker_message": message,
            "responder_message": None,
        }

    def _build_witness_statement(self, witness_entry: str, side_label: str) -> str:
        info = self._parse_witness_entry(witness_entry)
        name = info["name"] or "证人"
        relation = info["relation"]
        testimony = info["testimony"]
        if relation and testimony:
            return f"{name}，与{side_label}系{relation}关系。现就本案作证如下：{testimony}。陈述完毕。"
        if testimony:
            return f"{name}，现就本案作证如下：{testimony}。陈述完毕。"
        if relation:
            return f"{name}，与{side_label}系{relation}关系。本人已到庭，愿就所了解的案件情况依法作证。"
        return f"{name}，本人已到庭，愿就所了解的案件情况依法作证。"

    def _append_witness_sequence(
        self,
        results: List[Dict[str, Any]],
        *,
        witnesses: List[str],
        side_label: str,
        witness_role: str,
    ) -> None:
        for witness_entry in witnesses:
            info = self._parse_witness_entry(witness_entry)
            witness_name = info["name"] or "证人"
            results.append(
                self._execute_fixed_speech(
                    step_name=f"书记员通知{side_label}证人到庭-{witness_name}",
                    speaker_role="clerk",
                    message=f"请{side_label}证人{witness_name}到庭陈述。",
                )
            )
            results.append(
                self._execute_fixed_speech(
                    step_name=f"{side_label}证人陈述-{witness_name}",
                    speaker_role=witness_role,
                    message=self._build_witness_statement(witness_entry, side_label),
                )
            )

    def _judge_step_uses_llm(self, step_name: str) -> bool:
        return step_name in self.JUDGE_LLM_STEPS

    def _build_procedural_judge_message(self, step_name: str, instruction: str) -> str:
        if step_name.startswith("原告方辩论-"):
            round_label = step_name.split("-")[-1]
            return f"现在进行第{round_label}轮辩论，请原告代理律师发表辩论意见。"
        if step_name.startswith("被告方辩论-"):
            round_label = step_name.split("-")[-1]
            return f"现在进行第{round_label}轮辩论，请被告代理律师发表辩论意见。"
        return self.PROCEDURAL_JUDGE_TEMPLATES.get(step_name, instruction)
    
    def _broadcast_message(self, sender_role: str, message: str, exclude_roles: Optional[List[str]] = None) -> None:
        """Broadcast a message to all agents EXCEPT the sender and any excluded roles.
        
        This implements the courtroom broadcast mechanism: when one person speaks,
        all other participants can hear/see what was said.
        
        Args:
            sender_role: Role of the agent who spoke ("judge", "plaintiff", "defendant")
            message: The message content that was spoken
            exclude_roles: Additional roles to exclude from broadcast (e.g., responder
                          who already received the message via step() call)
        """
        from camel.messages import BaseMessage
        from camel.types import RoleType, OpenAIBackendRole
        
        sender_label = self.ROLE_LABEL.get(sender_role, sender_role)
        broadcast_content = f"{sender_label}说：{message}"
        
        skip_roles = {sender_role}
        if exclude_roles:
            skip_roles.update(exclude_roles)
        
        for role_key, agent in self.agents.items():
            if role_key in skip_roles:
                continue
            
            # Create a USER message from the sender's perspective
            msg = BaseMessage(
                role_name="User",
                role_type=RoleType.USER,
                meta_dict=None,
                content=broadcast_content
            )
            
            # Inject into the agent's memory as a USER message
            agent.chat_agent.update_memory(msg, OpenAIBackendRole.USER)
        
        self._log(f"[广播] {sender_label}的发言已广播给其他参与方")

    def _build_responder_instruction(
        self,
        responder_role: str,
        step_name: str,
        responder_instruction: Optional[str] = None,
    ) -> str:
        """Build guardrails so the next speaker keeps the correct courtroom identity."""
        role_label = self.ROLE_LABEL.get(responder_role, responder_role)
        base_instruction = (
            f"当前步骤：{step_name}\n"
            f"你现在是{role_label}。本轮指定发言人就是你（{role_label}），你必须直接完成这一轮发言。"
            "只输出你自己的本轮发言，不要以“审判长说：”“原告说：”“被告说：”"
            "“原告代理律师：”“被告代理律师：”等形式转述他人发言，不要冒充或代替其他角色发言。"
            "不要沿用上一轮或更早轮次的理由来回避本轮发言。"
        )

        if responder_role in {"plaintiff", "defendant"}:
            role_specific = (
                "只有当审判长在本轮当前这句里明确要求代理律师回答时，"
                "你才可以简短说明应由代理律师回应；否则你必须直接回答本轮问题。"
            )
        else:
            role_specific = (
                "只有当审判长在本轮当前这句里明确要求原告本人或被告本人回答时，"
                "你才可以简短说明应由当事人本人回应；否则你必须直接完成本轮法律陈述。"
            )

        extra_parts = [base_instruction, role_specific]
        if responder_instruction:
            extra_parts.append(responder_instruction)
        return "\n".join(extra_parts)

    def _build_speaker_prompt(
        self,
        step_name: str,
        instruction: str,
        speaker_role: str,
    ) -> str:
        """Build guardrails for the current speaker, especially the judge."""
        role_label = self.ROLE_LABEL.get(speaker_role, speaker_role)
        prompt_parts = [f"[{self.current_stage}] 当前步骤：{step_name}", instruction]

        if speaker_role == "judge":
            prompt_parts.append(
                "[流程控制要求]\n"
                "你现在是审判长。只输出当前步骤需要由审判长说的话，不要复述上一轮任何当事人或代理人的完整发言，"
                "不要把上一轮回答粘贴进本轮，不要提前跳到下一步骤，也不要替其他角色发言。"
            )
        else:
            prompt_parts.append(
                f"[流程控制要求]\n你现在是{role_label}，只输出当前步骤需要由你说的话，不要复述上一轮其他角色的完整发言。"
            )

        return "\n\n".join(prompt_parts)

    def _parse_judge_stage_control(
        self,
        message: str,
        *,
        target_labels: Dict[str, str],
        end_token: str,
    ) -> Dict[str, Optional[str]]:
        text = str(message or "").strip()
        if not text:
            return {"target_role": None, "target_label": None, "end_stage": False}

        if end_token in text or end_token.strip("【】") in text:
            return {"target_role": None, "target_label": None, "end_stage": True}

        for label, role in target_labels.items():
            if f"【对{label}说】" in text or re.search(rf"对{re.escape(label)}说[:：]?", text):
                return {"target_role": role, "target_label": label, "end_stage": False}

        return {"target_role": None, "target_label": None, "end_stage": False}

    def _build_free_stage_judge_prompt(
        self,
        *,
        stage_name: str,
        round_index: int,
        max_rounds: int,
        allowed_labels: List[str],
        opening_instruction: str,
        latest_message: str,
        end_token: str,
        stage_goal: str,
    ) -> str:
        latest_excerpt = str(latest_message or "").strip()
        if len(latest_excerpt) > 1200:
            latest_excerpt = latest_excerpt[-1200:]

        allowed_text = "、".join(allowed_labels)
        if stage_name == "庭审辩论":
            end_rule = (
                f"2. 如果你认为本阶段已经可以结束，必须以{end_token}开头，"
                "并且只宣布庭审辩论终结，不要在这句话里提前要求最后陈述，不要宣布休庭、调解、宣判或闭庭。"
            )
        else:
            end_rule = (
                f"2. 如果你认为本阶段已经可以结束，必须以{end_token}开头，"
                "并且只结束当前调查阶段，不要宣布休庭、调解、宣判或闭庭。"
            )
        return (
            f"[当前阶段] {stage_name}\n"
            f"[当前轮次] 第{round_index}轮 / 最多{max_rounds}轮\n"
            f"[本阶段任务] {stage_goal}\n"
            f"[参与人范围] 仅限：审判长、{allowed_text}\n"
            f"[控制格式]\n"
            f"1. 如果你要点名某一方代理律师发言，必须以【对{allowed_labels[0]}说】或【对{allowed_labels[1]}说】开头。\n"
            f"{end_rule}\n"
            "3. 每轮只能做一件事：点名一方代理律师发言，或者宣布结束本阶段。\n"
            "4. 不要要求原告本人、被告本人或证人发言。\n"
            "5. 你的发言仍应保持审判长身份和法庭用语。\n\n"
            f"[阶段起始要求]\n{opening_instruction}\n\n"
            f"[最近一轮法庭发言]\n{latest_excerpt or '（本阶段刚开始，暂无上一轮发言）'}"
        )

    def _build_free_stage_responder_prompt(
        self,
        *,
        stage_name: str,
        judge_message: str,
        responder_role: str,
    ) -> str:
        role_label = self.ROLE_LABEL.get(responder_role, responder_role)
        return (
            f"{judge_message}\n\n"
            "[流程控制要求]\n"
            f"当前阶段：{stage_name}\n"
            f"你现在是{role_label}，本轮是在回应审判长刚刚点名要求你发表的意见。\n"
            "只直接回答审判长本轮要求，不要宣布流程，不要冒充审判长，不要要求其他角色下一轮发言。"
        )

    def _execute_free_stage(
        self,
        *,
        stage_name: str,
        opening_instruction: str,
        target_labels: Dict[str, str],
        initial_target_role: str,
        max_rounds: int,
        end_token: str,
        stage_goal: str,
        force_close_instruction: str,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        last_message = opening_instruction
        last_target_role: Optional[str] = None
        allowed_labels = list(target_labels.keys())

        for round_index in range(1, max_rounds + 1):
            self.turn_count += 1
            step_name = f"{stage_name}-动态主持-{round_index}"
            self._log(f"[{self.current_stage}] 步骤: {step_name}")

            judge_prompt = self._build_free_stage_judge_prompt(
                stage_name=stage_name,
                round_index=round_index,
                max_rounds=max_rounds,
                allowed_labels=allowed_labels,
                opening_instruction=opening_instruction,
                latest_message=last_message,
                end_token=end_token,
                stage_goal=stage_goal,
            )
            self._check_pause_sync()
            judge_msg = self.agents["judge"].step(judge_prompt)
            parsed = self._parse_judge_stage_control(
                judge_msg,
                target_labels=target_labels,
                end_token=end_token,
            )
            self._add_dialog("judge", judge_msg)

            if self.verbose:
                print(f"\n{'─' * 50}")
                print("  🔨 审判长:")
                print(f"{'─' * 50}")
                print(judge_msg)

            responder_role = parsed["target_role"] or (
                initial_target_role if last_target_role is None else last_target_role
            )
            if not parsed["target_role"]:
                self._log(
                    f"[{self.current_stage}] 审判长未输出有效路由标记，回退为 {self.ROLE_LABEL.get(responder_role, responder_role)}"
                )

            exclude = [responder_role] if responder_role and not parsed["end_stage"] else None
            self._broadcast_message(sender_role="judge", message=judge_msg, exclude_roles=exclude)

            step_result: Dict[str, Any] = {
                "step": step_name,
                "speaker_message": judge_msg,
                "responder_message": None,
                "target_role": responder_role,
                "end_stage": bool(parsed["end_stage"]),
            }
            results.append(step_result)

            if parsed["end_stage"]:
                break

            responder_prompt = self._build_free_stage_responder_prompt(
                stage_name=stage_name,
                judge_message=judge_msg,
                responder_role=str(responder_role),
            )
            self._check_pause_sync()
            responder_msg = self.agents[str(responder_role)].step(responder_prompt)
            self._add_dialog(str(responder_role), responder_msg)
            step_result["responder_message"] = responder_msg

            if self.verbose:
                print(f"\n{'─' * 50}")
                print(f"  {self.ROLE_DISPLAY.get(str(responder_role), str(responder_role))}:")
                print(f"{'─' * 50}")
                print(responder_msg)

            self._broadcast_message(sender_role=str(responder_role), message=responder_msg)
            last_message = responder_msg
            last_target_role = str(responder_role)
        else:
            self.turn_count += 1
            force_step_name = f"{stage_name}-超限收束"
            self._log(f"[{self.current_stage}] 步骤: {force_step_name}")
            force_prompt = (
                f"[当前阶段] {stage_name}\n"
                f"[控制要求] 当前阶段已达到最大轮数 {max_rounds}。{force_close_instruction}\n"
                f"请你继续以审判长身份发言，并且必须以{end_token}开头。"
                "如果当前阶段是庭审辩论，你这句话只能宣布辩论终结，不要要求最后陈述，不要宣布休庭、调解、宣判或闭庭。"
            )
            self._check_pause_sync()
            judge_msg = self.agents["judge"].step(force_prompt)
            self._add_dialog("judge", judge_msg)
            self._broadcast_message(sender_role="judge", message=judge_msg)
            results.append(
                {
                    "step": force_step_name,
                    "speaker_message": judge_msg,
                    "responder_message": None,
                    "target_role": None,
                    "end_stage": True,
                }
            )

            if self.verbose:
                print(f"\n{'─' * 50}")
                print("  🔨 审判长:")
                print(f"{'─' * 50}")
                print(judge_msg)

        return results
    
    def _execute_step(
        self, 
        step_name: str, 
        instruction: str, 
        speaker_role: str = "judge", 
        responder_role: Optional[str] = None,
        responder_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a single step in the trial.
        
        After each agent speaks, the message is broadcast to all other agents
        so everyone can see what was said (courtroom broadcast mechanism).
        
        Note: When broadcasting the speaker's message, the responder is excluded
        because they will receive it directly via the step() call. This avoids
        duplicate messages in the responder's memory.
        
        Args:
            step_name: Name of the step
            instruction: Instruction for the speaker (Judge)
            speaker_role: Role of the speaker (default: judge)
            responder_role: Role of the responder (optional)
            
        Returns:
            Dictionary containing speaker_message and responder_message
        """
        self.turn_count += 1
        self._log(f"[{self.current_stage}] 步骤: {step_name}")
        
        # 1. Speaker (Judge) acts
        speaker = self.agents[speaker_role]
        if speaker_role == "judge" and not self._judge_step_uses_llm(step_name):
            speaker_msg = self._build_procedural_judge_message(step_name, instruction)
        else:
            judge_prompt = f"[{self.current_stage}] {step_name}: {instruction}"
            self._check_pause_sync()
            speaker_msg = speaker.step(judge_prompt)
        self._add_dialog(speaker_role, speaker_msg)
        
        # Print full speaker content
        if self.verbose:
            print(f"\n{'─' * 50}")
            print(f"  {self.ROLE_DISPLAY.get(speaker_role, speaker_role)}:")
            print(f"{'─' * 50}")
            print(speaker_msg)
        
        # Broadcast speaker's message to bystanders only (exclude responder to avoid duplicate)
        # The responder will receive the message directly via step() below
        exclude = [responder_role] if responder_role else None
        self._broadcast_message(sender_role=speaker_role, message=speaker_msg, exclude_roles=exclude)
        
        # 2. Responder acts (if any)
        responder_msg = None
        if responder_role:
            responder = self.agents[responder_role]
            responder_prompt = speaker_msg
            flow_instruction = self._build_responder_instruction(
                responder_role,
                step_name,
                responder_instruction=responder_instruction,
            )
            if flow_instruction:
                responder_prompt = (
                    f"{speaker_msg}\n\n"
                    f"[流程控制要求]\n{flow_instruction}"
                )
            self._check_pause_sync()
            responder_msg = responder.step(responder_prompt)
            self._add_dialog(responder_role, responder_msg)
            
            # Print full responder content
            if self.verbose:
                print(f"\n{'─' * 50}")
                print(f"  {self.ROLE_DISPLAY.get(responder_role, responder_role)}:")
                print(f"{'─' * 50}")
                print(responder_msg)
            
            # Broadcast responder's message to all other agents (speaker + bystander)
            self._broadcast_message(sender_role=responder_role, message=responder_msg)
            
        return {
            "step": step_name,
            "speaker_message": speaker_msg,
            "responder_message": responder_msg
        }
    
    def _build_result(self, duration: float = 0.0) -> Dict[str, Any]:
        return {
            "scenario_type": "CI",
            "dialog_history": self.dialog_history,
            "stage_results": self.stage_results,
            "final_judgment": self.final_judgment,
            "drafted_document_payload": self._drafted_document_payload,
            "pdf_path": str(self._drafted_document_payload.get("pdf_path", "") or ""),
            "mediation_result": self.mediation_result,
            "total_turns": self.turn_count,
            "duration": duration,
            "completed": self.completed,
        }

    def _build_checkpoint_data(self) -> Dict[str, Any]:
        return {
            "dialog_history": self.dialog_history,
            "turn_count": self.turn_count,
            "completed": self.completed,
            "current_stage": self.current_stage,
            "stage_results": self.stage_results,
            "final_judgment": self.final_judgment,
            "drafted_document_payload": self._drafted_document_payload,
            "mediation_result": self.mediation_result,
        }

    async def resume_from_checkpoint(self, checkpoint_data: Dict[str, Any]) -> Dict[str, Any]:
        self.dialog_history = checkpoint_data.get("dialog_history", [])
        self.turn_count = checkpoint_data.get("turn_count", 0)
        self.completed = checkpoint_data.get("completed", False)
        self.current_stage = checkpoint_data.get("current_stage", self.current_stage)
        self.stage_results = checkpoint_data.get("stage_results", {})
        self.final_judgment = checkpoint_data.get("final_judgment")
        self._drafted_document_payload = checkpoint_data.get("drafted_document_payload", {}) or {}
        self.mediation_result = checkpoint_data.get("mediation_result", {})

        if self.completed:
            return self._build_result()

        return self.execute()

    def _save_result(self, result: Dict[str, Any]) -> None:
        """Save result to JSON file.
        
        Args:
            result: Result dictionary to save
        """
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        self._log(f"结果已保存到 {self.output_path}")

    def _execute_opening_session(self):
        """Phase 1: Opening Session."""
        self.current_stage = "开庭审理"
        if self.verbose:
            print(f"\n{'═' * 60}")
            print(f"  📌 阶段一: 开庭审理")
            print(f"{'═' * 60}")
        results = []

        results.append(
            self._execute_fixed_speech(
                "书记员核对到庭情况",
                "clerk",
                "请原告、被告及各方诉讼代理人核对到庭情况。",
            )
        )
        results.append(
            self._execute_fixed_speech(
                "书记员报告到庭情况",
                "clerk",
                "现向审判长报告到庭情况，所有人员均已到齐。",
            )
        )
        results.append(
            self._execute_fixed_speech(
                "书记员宣布法庭纪律",
                "clerk",
                "现在宣布法庭纪律。请所有在庭人员遵守法庭秩序，关闭或者调至静音通讯工具，未经许可不得录音录像、摄影，不得随意走动、喧哗或者实施其他扰乱法庭秩序的行为。",
            )
        )

        steps = [
            ("核实原告身份", "请原告本人陈述姓名、身份信息，并确认本人到庭参加诉讼。", "plaintiff"),
            ("核实被告身份", "请被告本人陈述姓名、身份信息，并确认本人到庭参加诉讼。", "defendant"),
            ("宣布审判庭构成", "宣布本案由审判长主持审理，适用普通程序。", None),
            ("告知诉讼权利义务", "依照有关法律告知双方当事人及诉讼代理人诉讼权利和诉讼义务。", None),
            ("询问原告方回避申请", "告知原告本人有申请回避的权利，并询问是否申请回避。", "plaintiff"),
            ("询问被告方回避申请", "告知被告本人有申请回避的权利，并询问是否申请回避。", "defendant"),
            ("询问原告是否知晓缺席后果", "询问原告本人是否知晓无正当理由拒不到庭或中途退庭的法律后果。", "plaintiff"),
            ("询问被告是否知晓缺席后果", "询问被告本人是否知晓无正当理由拒不到庭或中途退庭的法律后果。", "defendant"),
        ]

        for name, instr, responder in steps:
            res = self._execute_step(name, instr, responder_role=responder)
            results.append(res)

        self.stage_results["opening"] = results

    def _execute_court_investigation(self):
        """Phase 2: Court Investigation."""
        self.current_stage = "法庭调查"
        if self.verbose:
            print(f"\n{'═' * 60}")
            print(f"  📌 阶段二: 法庭调查")
            print(f"{'═' * 60}")
        results = self._execute_free_stage(
            stage_name="法庭调查",
            opening_instruction=(
                "现在进入法庭调查。请先由原告代理律师陈述起诉事实、诉讼请求、证据和证明目的。"
                "此后你可以围绕举证、质证、回应、补充说明和必要追问，自主决定下一轮点名哪一方代理律师发言。"
            ),
            target_labels={
                "原告代理律师": self.plaintiff_lawyer_role,
                "被告代理律师": self.defendant_lawyer_role,
            },
            initial_target_role=self.plaintiff_lawyer_role,
            max_rounds=self.max_debate_rounds,
            end_token="【结束法庭调查】",
            stage_goal="围绕举证、质证、回应和法庭发问推进，不要脱离法庭调查主题。",
            force_close_instruction="请你收束法庭调查，宣布本阶段结束，并自然过渡到庭审辩论。",
        )
        # 证人出庭逻辑暂时停用，先保留实现，不接入当前庭审主流程。
        # self._append_witness_sequence(
        #     results,
        #     witnesses=self.plaintiff_witnesses,
        #     side_label="原告方",
        #     witness_role="plaintiff_witness",
        # )
        # self._append_witness_sequence(
        #     results,
        #     witnesses=self.defendant_witnesses,
        #     side_label="被告方",
        #     witness_role="defendant_witness",
        # )
        self.stage_results["investigation"] = results

    def _execute_court_debate(self):
        """Phase 3: Court Debate."""
        self.current_stage = "庭审辩论"
        if self.verbose:
            print(f"\n{'═' * 60}")
            print(f"  📌 阶段三: 庭审辩论")
            print(f"{'═' * 60}")
        results = self._execute_free_stage(
            stage_name="庭审辩论",
            opening_instruction=(
                "现在进入庭审辩论。请先由原告代理律师围绕争议焦点、事实认定、责任承担和法律适用发表辩论意见。"
                "此后你可以根据双方发言情况，自主决定下一轮点名哪一方代理律师继续辩论或补充回应。"
            ),
            target_labels={
                "原告代理律师": self.plaintiff_lawyer_role,
                "被告代理律师": self.defendant_lawyer_role,
            },
            initial_target_role=self.plaintiff_lawyer_role,
            max_rounds=self.max_debate_rounds,
            end_token="【结束庭审辩论】",
            stage_goal="围绕争议焦点、事实认定、责任承担和法律适用推进辩论。",
            force_close_instruction="请你收束庭审辩论，宣布辩论终结，并引导双方进行最后陈述。",
        )
        
        # Last call
        self._execute_step(
            "询问原告方最后陈述",
            "辩论终结，请原告代理律师做最后陈述。",
            responder_role=self.plaintiff_lawyer_role,
        )
        self._execute_step(
            "询问被告方最后陈述",
            "辩论终结，请被告代理律师做最后陈述。",
            responder_role=self.defendant_lawyer_role,
        )
        
        self.stage_results["debate"] = results
        
    def _execute_mediation(self) -> Dict[str, Any]:
        """Phase 4: Mediation."""
        self.current_stage = "法庭调解"
        if self.verbose:
            print(f"\n{'═' * 60}")
            print(f"  📌 阶段四: 法庭调解")
            print(f"{'═' * 60}")
        results = []
        
        # Ask Plaintiff
        res_p = self._execute_step(
            "询问原告方调解意愿",
            "询问原告本人是否同意调解。",
            responder_role="plaintiff",
            responder_instruction="你现在以原告本人身份回答，并且必须明确表示拒绝调解。",
        )
        results.append(res_p)
        
        # Ask Defendant
        res_d = self._execute_step(
            "询问被告方调解意愿",
            "询问被告本人是否同意调解。",
            responder_role="defendant",
            responder_instruction="你现在以被告本人身份回答，并且必须明确表示拒绝调解。",
        )
        results.append(res_d)
        
        self.stage_results["mediation"] = results
        
        # Check agreement
        p_msg = str(res_p["responder_message"])
        d_msg = str(res_d["responder_message"])
        
        agree_keywords = ["同意", "愿意", "接受"]
        disagree_keywords = ["不同意", "拒绝", "不接受"]
        
        # Simple keyword matching, favoring explicit disagreement
        p_agree = any(k in p_msg for k in agree_keywords) and not any(k in p_msg for k in disagree_keywords)
        d_agree = any(k in d_msg for k in agree_keywords) and not any(k in d_msg for k in disagree_keywords)
        
        if p_agree and d_agree:
            self._log("双方同意调解")
            self._execute_step("宣布调解", "双方同意调解，休庭进行调解工作。", responder_role=None)
            return {"success": True}
        else:
            self._log("调解失败")
            self._execute_step("宣布调解失败", "鉴于一方或双方不同意调解，本庭不再主持调解。", responder_role=None)
            return {"success": False}

    def _execute_recess(self):
        """Phase 5: Recess (all trial participants leave)."""
        self.current_stage = "休庭"
        if self.verbose:
            print(f"\n{'═' * 60}")
            print(f"  📌 阶段五: 休庭")
            print(f"{'═' * 60}")
        
        self._log("宣布休庭，庭审参与人退场")
        self._execute_step(
            "休庭",
            "现在的庭审暂时结束，本庭将对案件进行评议，择期宣判。请双方当事人及其诉讼代理人退庭。",
            responder_role=None,
        )
        
        # In a real system, we might actually disconnect the agents here.
        # For now, we just log it and stop calling them.
        
    def _execute_deliberation_and_judgment(self) -> str:
        """Phase 6: Deliberation and Judgment (Judge only)."""
        self.current_stage = "评议宣判"
        if self.verbose:
            print(f"\n{'═' * 60}")
            print(f"  📌 阶段六: 评议宣判")
            print(f"{'═' * 60}")
        
        # Get the Judge Agent
        judge_agent = self.agents["judge"]
        
        # Judgment drafting skeleton for the judge agent.
        template = """
        [民事判决书模板]
        民事判决书
（2026）X法民初字第XXX号（案号）
原告：{原告姓名/名称}，{身份信息待补充}，住所地{住所待补充}。
被告：{被告姓名/名称}，{身份信息待补充}，住所地{住所待补充}。
原告XX与被告XX人身损害赔偿纠纷一案（案由），本院于XX年XX月XX日立案后，公开开庭审理（如不公开审理，表述为“因涉及XXX，依法或当事人申请不公开开庭审理”）。原告、被告均到庭参加诉讼，本案现已审理终结。
原告诉称：XXXXXX，请求判令XXXXX。（原告的诉讼请求，事实和理由）
被告辩称：XXXXXX，请求判令驳回原告的诉讼请求（被告主张的事实和理由）。
当事人围绕诉讼请求提供了证据，本院组织当事人进行了证据交换和质证，对当事人无异议的事实和证据，本院予以认定，对于有争议的事实和证据，本院认定如下：
1……….;
2……….;
（判决认定的事实、证据）
本院认为，XXXXXXXX。依据《中华人民共和国XX法》第XX条，判决如下：（判决的理由和法律依据）
判决1;
判决2;
...
判决n;

如果未按照本判决指定的履行支付金钱给付义务，应当根据《民事诉讼法》第二百五十三的条的规定加倍支付迟延履行期间债务利息。
案件受理费 XX元，由原告XX承担XX元，被告XX承担XX元。
如不服本判决，可以自收到本判决之次日起十五日内通过本院上诉，应提交上诉状正本一份，副本X份（上诉期限和上诉法院）

审判长：XX

X年X月X日
(XX人民法院印章）

        """
        
        instr = f"""
现在的庭审已经结束，请你根据庭审情况，结合【参考资料-法院查明】与【参考资料-法院意见】，撰写一审民事判决书。

【参考资料-法院查明】
{self.court_finding}

【参考资料-法院意见】
{self.court_opinion}

请填充以下模板，保持格式一致：

{template}

注意：
1. 事实认定部分应基于庭审中双方的陈述和举证质证，并重点参考【参考资料-法院查明】中的事实归纳。
2. "本院认为"部分应参考【参考资料-法院意见】，但要用你自己的语言表述，且逻辑通顺。
3. 判决结果应明确具体。
4. 你的最终回复必须直接给出完整《民事判决书》正文，不得输出判决摘要、要点清单、生成说明、PDF 路径、工具调用信息或任何过程性提示。
5. 不要提及工具、导出、生成 PDF、文件路径或“已完成”等说明；系统会在后台根据你输出的完整正文处理后续落盘。
"""
        # Execute explicitly without responder
        res = self._execute_step("撰写判决书", instr, responder_role=None)
        self._capture_judgment_tool_result(judge_agent)
        
        judgment = res["speaker_message"]
        self.stage_results["judgment"] = [res]
        
        return judgment

    def _capture_judgment_tool_result(self, judge_agent: Any) -> None:
        try:
            payload = extract_judgment_document_tool_payload(
                list(getattr(judge_agent, "_last_tool_call_records", []) or []),
                document_type=get_judgment_document_type_for_scenario("CI"),
            )
        except Exception:
            return

        if payload.get("pdf_path"):
            self._drafted_document_payload = payload

    def _ensure_pdf_output(self, judge_agent: Any) -> None:
        if self._drafted_document_payload.get("pdf_path") or not str(self.final_judgment or "").strip():
            return
        try:
            self._drafted_document_payload = render_judgment_document_payload(
                judge_agent,
                document_type="CI",
                document_text=str(self.final_judgment or ""),
            )
        except Exception as exc:
            logger.warning("Failed to backfill first-instance judgment PDF output: %s", exc)
