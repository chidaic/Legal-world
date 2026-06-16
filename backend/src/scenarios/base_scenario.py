"""Base scenario class for SimLawFirm framework.

This module provides the foundational BaseScenario class that all specific
scenarios inherit from.
"""

import inspect
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .pause_control import get_runtime_pause_controller


logger = logging.getLogger(__name__)


class BaseScenario(ABC):
    """Base class for all scenarios in SimLawFirm."""

    def __init__(
        self,
        agents: Dict[str, Any],
        max_turns: int = 99,
        verbose: bool = False,
        map_engine: Optional[Any] = None,
        checkpoint_manager: Optional[Any] = None,
        scenario_id: Optional[str] = None,
        bubble_publisher: Optional[Callable[[str, str], None]] = None,
        trace_recorder: Optional[Any] = None,
        trace_stage_code: Optional[str] = None,
        trace_stage_key: Optional[str] = None,
    ):
        self.agents = agents
        self.max_turns = max_turns
        self.verbose = verbose
        self.map_engine = map_engine
        self.checkpoint_manager = checkpoint_manager
        self.scenario_id = scenario_id
        self.bubble_publisher = bubble_publisher
        self.trace_recorder = trace_recorder
        self.trace_stage_code = str(trace_stage_code or "").strip().upper()
        self.trace_stage_key = str(trace_stage_key or trace_stage_code or "").strip().upper()

        self.dialog_history: List[Dict[str, Any]] = []
        self.turn_count = 0
        self.completed = False

    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """Execute the scenario."""

    def _add_dialog(self, role: str, content: str) -> None:
        timestamp = datetime.now().isoformat()
        entry = {
            "turn": self.turn_count,
            "role": role,
            "content": content,
            "timestamp": timestamp,
        }
        generation_duration_seconds = self._resolve_generation_duration_seconds(role, content)
        if generation_duration_seconds is not None:
            entry["generation_duration_seconds"] = generation_duration_seconds
        generation_total_tokens = self._resolve_generation_total_tokens(role, content)
        if generation_total_tokens is not None:
            entry["generation_total_tokens"] = generation_total_tokens
        self.dialog_history.append(entry)
        if self.trace_recorder and content:
            try:
                self.trace_recorder.log_dialog(
                    stage_code=self.trace_stage_code or getattr(self, "scenario_type", ""),
                    stage_key=self.trace_stage_key or self.trace_stage_code or getattr(self, "scenario_type", ""),
                    scenario=type(self).__name__,
                    turn=int(self.turn_count or 0),
                    role=role,
                    content=content,
                    timestamp=timestamp,
                )
            except Exception as exc:
                logger.warning(
                    "[%s] Failed to record trace dialogue: role=%s, error=%s",
                    self.__class__.__name__,
                    role,
                    exc,
                )
        if self.bubble_publisher and content:
            try:
                if self._bubble_publisher_accepts_entry():
                    self.bubble_publisher(role, content, entry)
                else:
                    self.bubble_publisher(role, content)
            except Exception as exc:
                logger.warning(
                    "[%s] Failed to publish dialogue bubble: role=%s, error=%s",
                    self.__class__.__name__,
                    role,
                    exc,
                )

    def _resolve_generation_duration_seconds(self, role: str, content: str) -> float | None:
        agent = self.agents.get(role)
        if agent is None:
            return None
        if str(getattr(agent, "_simlaw_last_step_response_text", "") or "").strip() != str(content or "").strip():
            return None
        duration = getattr(agent, "_simlaw_last_step_duration_seconds", None)
        try:
            numeric_duration = float(duration)
        except (TypeError, ValueError):
            return None
        if numeric_duration <= 0:
            return None
        return round(numeric_duration, 4)

    def _resolve_generation_total_tokens(self, role: str, content: str) -> int | None:
        agent = self.agents.get(role)
        if agent is None:
            return None
        if str(getattr(agent, "_simlaw_last_step_response_text", "") or "").strip() != str(content or "").strip():
            return None
        total_tokens = getattr(agent, "_simlaw_last_step_total_tokens", None)
        try:
            numeric_total = int(total_tokens)
        except (TypeError, ValueError):
            return None
        if numeric_total <= 0:
            return None
        return numeric_total

    def _bubble_publisher_accepts_entry(self) -> bool:
        if not self.bubble_publisher:
            return False
        try:
            signature = inspect.signature(self.bubble_publisher)
        except (TypeError, ValueError):
            return False
        return len(signature.parameters) >= 3

    def _resolve_pause_controller(self) -> Any | None:
        controller = self.map_engine
        if controller and hasattr(controller, "_paused") and hasattr(controller, "_resumed_event"):
            return controller
        return get_runtime_pause_controller()

    def _save_checkpoint_if_needed(self) -> None:
        if not self.checkpoint_manager or not self.scenario_id:
            return
        try:
            checkpoint_data = self._build_checkpoint_data()
            self.checkpoint_manager.save_scenario_checkpoint(self.scenario_id, checkpoint_data)
        except Exception as exc:
            logger.error("[Scenario] Failed to save checkpoint: %s", exc)

    async def _check_pause(self) -> None:
        controller = self._resolve_pause_controller()
        if controller and getattr(controller, "_paused", False):
            logger.info("[Scenario] Pause requested, waiting for resume")
            self._save_checkpoint_if_needed()
            await controller._resumed_event.wait()
            logger.info("[Scenario] Resumed")

    def _check_pause_sync(self) -> None:
        controller = self._resolve_pause_controller()
        if not controller or not getattr(controller, "_paused", False):
            return

        logger.info("[Scenario] Pause requested, waiting for resume")
        self._save_checkpoint_if_needed()

        resumed_event = getattr(controller, "_resumed_event", None)
        wait_sync = getattr(resumed_event, "wait_sync", None) if resumed_event is not None else None
        if callable(wait_sync):
            wait_sync()
            logger.info("[Scenario] Resumed")
            return

        logger.warning("[Scenario] Pause controller does not support sync wait")

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[{self.__class__.__name__}] {message}")
        logger.debug(f"[{self.__class__.__name__}] {message}")

    @abstractmethod
    def _build_checkpoint_data(self) -> Dict[str, Any]:
        """Build checkpoint data for this scenario."""

    @abstractmethod
    async def resume_from_checkpoint(self, checkpoint_data: Dict[str, Any]) -> Dict[str, Any]:
        """Resume scenario execution from a checkpoint."""
