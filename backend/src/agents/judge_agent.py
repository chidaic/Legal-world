"""Judge agent for court proceedings.

Slim sandbox version: metadata + event handlers only.
Prompt building and scenario execution are handled by PromptAssembler
and ScenarioOrchestrator respectively.
"""

import os
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base_agent import BaseAgent
from ..pipeline.stage_tool_resolver import build_agent_default_tools

if TYPE_CHECKING:
    from ..core.event_bus import EventBus
    from ..core.file_storage_manager import FileStorageManager


logger = logging.getLogger(__name__)


def _build_default_judge_tools(
    agent: "JudgeAgent",
    provided_tools: Optional[List[Any]] = None,
) -> List[Any]:
    """Build the default toolset for judge agents."""
    return build_agent_default_tools("judge", agent, provided_tools=provided_tools)


class JudgeAgent(BaseAgent):
    """Judge Agent — shell mode by default, activated per-scenario.

    Profile fields (court_name, court_level) are loaded from config.yaml
    in sandbox mode.
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        court_name: str = "人民法院",
        court_level: str = "basic",
        years_of_experience: Optional[int] = None,
        personality: str = "",
        speaking_style: str = "",
        # Pipeline-only params
        system_prompt: str = "",
        scenario_type: Optional[str] = None,
        scenario_data: Optional[Dict[str, Any]] = None,
        work_memory_path: Optional[str] = None,
        **kwargs
    ):
        config_path = kwargs.get("config_path")
        storage = kwargs.get("storage")
        provided_tools = kwargs.pop("tools", None)
        self.name = name

        # Load profile from sandbox config.yaml if available
        if config_path and storage and os.path.exists(config_path):
            config = storage.load_agent_config(config_path)
            profile = config.get("profile", {})
            self.court_name = court_name or profile.get("court_name", "人民法院")
            self.court_level = court_level or profile.get("court_level", "basic")
            self.years_of_experience = years_of_experience or profile.get("years_of_experience")
            self.personality = personality or profile.get("personality", "")
            self.speaking_style = speaking_style or profile.get("speaking_style", "")
        else:
            self.court_name = court_name
            self.court_level = court_level
            self.years_of_experience = years_of_experience
            self.personality = personality
            self.speaking_style = speaking_style

        # Pipeline-mode scenario state
        self.scenario_type = scenario_type
        self.scenario_data = scenario_data or {}

        # Pipeline compat: if scenario_type given but no system_prompt, build it
        if scenario_type and not system_prompt:
            system_prompt = self._build_pipeline_prompt()

        super().__init__(
            agent_id=agent_id,
            name=name,
            system_prompt=system_prompt,
            work_memory_path=work_memory_path,
            tools=_build_default_judge_tools(self, provided_tools),
            **kwargs
        )

    def _build_pipeline_prompt(self) -> str:
        """Build system prompt for pipeline mode using PromptAssembler."""
        from ..prompts.prompt_assembler import PromptAssembler

        scenario_prompt = PromptAssembler.build_scenario_prompt(
            "judge", self.scenario_type, self.scenario_data
        )
        return PromptAssembler.build(
            profile={
                "name": self.name,
                "occupation": "审判员",
                "court_name": self.court_name,
                "court_level": self.court_level,
                "years_of_experience": self.years_of_experience,
                "personality": self.personality,
                "speaking_style": self.speaking_style,
            },
            scenario_prompt=scenario_prompt,
        )

    # ══════════════════════════════════════════════════════════
    #  沙盒模式：事件驱动回调 & 案件队列管理
    # ══════════════════════════════════════════════════════════

    @property
    def current_handling_case(self) -> Optional[str]:
        if not self.storage or not self.config_path:
            return None
        config = self.storage.load_agent_config(self.config_path)
        return config.get("current_handling_case")

    @property
    def case_queue(self) -> List[str]:
        if not self.storage or not self.config_path:
            return []
        config = self.storage.load_agent_config(self.config_path)
        return config.get("case_queue", [])

    def register_sandbox_events(self) -> None:
        if not self.event_bus:
            return
        from ..core.event_bus import EventType
        self.event_bus.subscribe(EventType.LAWSUIT_FILED, self._on_lawsuit_filed)
        self.event_bus.subscribe(EventType.DEFENSE_FILED, self._on_defense_filed)
        self.event_bus.subscribe(EventType.APPEAL_FILED, self._on_appeal_filed)

    async def _on_lawsuit_filed(self, payload: dict) -> None:
        court_level = payload.get("court_level", "basic")
        if court_level != self.court_level:
            return
        case_id = payload.get("case_id", "")
        logger.info(f"[法官 {self.name}] 受理案件: {case_id}")
        if self.storage and self.config_path:
            self.storage.append_to_queue(self.config_path, "case_queue", case_id)

    async def _on_defense_filed(self, payload: dict) -> None:
        case_id = payload.get("case_id", "")
        if case_id not in self.case_queue:
            return
        logger.info(f"[法官 {self.name}] 收到答辩状，案件 {case_id} 诉辩双方就绪")
        await self.schedule_trial(case_id, "first_instance")

    async def _on_appeal_filed(self, payload: dict) -> None:
        court_level = payload.get("court_level", "intermediate")
        if court_level != self.court_level:
            return
        case_id = payload.get("case_id", "")
        logger.info(f"[法官 {self.name}] 受理上诉案件: {case_id}")
        if self.storage and self.config_path:
            self.storage.append_to_queue(self.config_path, "case_queue", case_id)
        await self.schedule_trial(case_id, "second_instance")

    async def schedule_trial(self, case_id: str, instance: str) -> None:
        logger.info(
            "[法官 %s] 记录排期开庭请求: %s (%s)，等待编排器统一触发开庭",
            self.name,
            case_id,
            instance,
        )

    async def on_trial_completed(self, case_id: str) -> None:
        if self.storage and self.config_path:
            self.storage.update_agent_field(self.config_path, "current_handling_case", None)
        self.clear_case_memory()
        if self._is_active:
            self.deactivate()
        logger.info(f"[法官 {self.name}] 案件 {case_id} 审理完毕")
