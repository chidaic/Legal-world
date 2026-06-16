"""Player plaintiff-lawyer agent adapter.

Occupies the same slot as a ``LawyerAgent`` in scenario loops
but delegates ``step()`` to the player input gateway instead of
calling the LLM.

Feature flag
────────────
Controlled by ``SIMLAW_PLAYER_LAWYER_MODE``.  When the env-var is
set to ``plaintiff`` (case-insensitive), the orchestrator should
instantiate this adapter instead of the registry AI lawyer for the
plaintiff side.  Default (unset / empty) keeps current AI behavior.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional

from .input_gateway import PlayerInputGateway
from .models import PlayerLawyerRequest

logger = logging.getLogger(__name__)

# ── Feature flag ──────────────────────────────────────────────────

_PLAYER_MODE_ENV = "SIMLAW_PLAYER_LAWYER_MODE"


def is_player_plaintiff_mode() -> bool:
    """Return True when the backend should use a human player as plaintiff lawyer."""
    return os.environ.get(_PLAYER_MODE_ENV, "").strip().lower() == "plaintiff"


# ── Agent adapter ─────────────────────────────────────────────────


class _NoOpChatAgent:
    """Minimal stub that satisfies courtroom broadcast's ``.chat_agent`` access.

    CI / appeal-CI scenarios call ``agent.chat_agent.update_memory()`` to
    inject courtroom speech into every agent's memory.  The player adapter
    has no LLM memory, so this is intentionally a no-op.
    """

    def update_memory(self, msg: Any, role: Any) -> None:
        pass


class PlayerPlaintiffLawyerAgent:
    """Drop-in replacement for LawyerAgent on the plaintiff side.

    It exposes the minimal surface that existing scenarios depend on
    so the orchestrator can treat it as a regular agent.

    ``step(instruction)`` creates a pending request in the gateway,
    optionally broadcasts a ``player_lawyer_input_required`` event,
    then blocks until the player submits a response.
    """

    def __init__(
        self,
        *,
        agent_id: str,
        name: str,
        law_firm: str = "",
        firm_id: str = "",
        gateway: PlayerInputGateway,
        case_id: str = "",
        sandbox_id: int = 0,
        broadcast_fn: Optional[Callable[..., Any]] = None,
    ) -> None:
        # Identity — matches LawyerAgent surface
        self.agent_id = agent_id
        self.name = name
        self.law_firm = law_firm
        self.firm_id = firm_id
        self.config_path: Optional[str] = None
        self.storage: Any = None

        # Scenario metadata — written by orchestrator before use
        self.scenario_type: Optional[str] = None
        self.scenario_data: Dict[str, Any] = {}
        self.tools: List[Any] = []
        self._last_tool_call_records: List[Any] = []
        self.current_scenario_id: Optional[str] = None
        self.system_prompt: str = ""
        self.skill_usage_log: List[Dict[str, Any]] = []

        # Player gateway
        self._gateway = gateway
        self._case_id = case_id
        self._sandbox_id = sandbox_id
        self._broadcast_fn = broadcast_fn

        # Active flag
        self._is_active = False
        self._current_stage: str = ""

        # No-op chat_agent adapter — satisfies courtroom broadcast
        self.chat_agent = _NoOpChatAgent()

    # ── Agent protocol surface ────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def agent_type(self) -> str:
        return "lawyer"

    def activate(
        self,
        system_prompt: str = "",
        model_platform: Any = None,
        model_type: Any = None,
        *,
        tools: Any = None,
        skill_dirs: Any = None,
        debug_output_dir: Any = None,
        scenario_id: Optional[str] = None,
        step_timeout_seconds: Any = None,
    ) -> None:
        """Mark the adapter as active — no LLM needed."""
        self.system_prompt = system_prompt
        if scenario_id:
            self.current_scenario_id = scenario_id
        self._is_active = True
        logger.info("[PlayerAdapter %s] Activated (player mode)", self.name)

    def deactivate(self) -> None:
        self._is_active = False
        self.current_scenario_id = None
        self.system_prompt = ""
        logger.info("[PlayerAdapter %s] Deactivated", self.name)

    def recover_from_error(self) -> None:
        """No-op: nothing to reset since there is no LLM state."""
        logger.info("[PlayerAdapter %s] recover_from_error (no-op)", self.name)

    def reset_memory(self) -> None:
        """No-op."""

    def get_prompt_info(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.name,
            "agent_class": "PlayerPlaintiffLawyerAgent",
            "system_prompt": self.system_prompt,
        }

    def get_skill_usage_report(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.name,
            "tool_call_count": 0,
            "skill_load_count": 0,
            "skills": [],
            "tool_calls": [],
        }

    def reset_skill_usage_report(self) -> None:
        self.skill_usage_log = []

    # ── Long-term memory (no-ops for player) ──────────────────────

    def extract_and_save_long_term_memory(
        self,
        filepath: Optional[str] = None,
        raise_on_error: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Player has no LLM memory to checkpoint."""
        return None

    # ── Core: step() blocks on player input ───────────────────────

    def step(
        self,
        instruction: str,
        response_format: Any = None,
        image_list: Any = None,
        context: Any = None,
    ) -> str:
        """Block until the human player submits a response.

        1. Creates a pending request in the gateway.
        2. Calls ``broadcast_fn`` (if set) to notify frontend/WS.
        3. Blocks until ``gateway.wait_for_response`` returns.
        4. Returns the player's message as the "lawyer response".
        """
        stage = self._current_stage or self.scenario_type or ""
        req = self._gateway.find_reusable_request(
            case_id=self._case_id,
            stage=stage,
            prompt=instruction,
        )
        if req is not None and req.status.value == "submitted":
            logger.info(
                "[PlayerAdapter %s] Reusing submitted player input (request=%s)",
                self.name,
                req.request_id,
            )
            return req.message

        if req is None:
            req = self._gateway.create_request(
                case_id=self._case_id,
                stage=stage,
                role="plaintiff_lawyer",
                speaker_label=self.name,
                prompt=instruction,
                context_summary=f"案件 {self._case_id} · {stage} 阶段",
            )
        else:
            logger.info(
                "[PlayerAdapter %s] Reusing pending player input (request=%s)",
                self.name,
                req.request_id,
            )

        # Notify frontends that player input is required
        if self._broadcast_fn is not None:
            try:
                self._broadcast_fn(
                    "player_lawyer_input_required",
                    req.to_dict(),
                )
            except Exception as exc:
                logger.warning(
                    "[PlayerAdapter %s] broadcast failed: %s",
                    self.name,
                    exc,
                )

        logger.info(
            "[PlayerAdapter %s] Waiting for player input (request=%s)",
            self.name,
            req.request_id,
        )
        message = self._gateway.wait_for_response(req.request_id)
        logger.info(
            "[PlayerAdapter %s] Got player input (%d chars)",
            self.name,
            len(message),
        )
        return message

    def build_auto_opening_response(self, instruction: str = "") -> str:
        """Return a non-substantive greeting that should not require player input."""
        return f"您好，我是{self.name}，请您先说一下这次想咨询的问题。"

    # ── Convenience setters ───────────────────────────────────────

    def set_stage(self, stage: str) -> None:
        self._current_stage = stage
        self.scenario_data["case_id"] = self._case_id
        self.scenario_data["current_handling_case"] = self._case_id

    @property
    def case_id(self) -> str:
        return self._case_id

    @property
    def current_handling_case(self) -> str:
        return self._case_id

    def expects_player_input_for_current_step(self) -> bool:
        return True

    def set_case_id(self, case_id: str) -> None:
        self._case_id = case_id
        self.scenario_data["case_id"] = self._case_id
        self.scenario_data["current_handling_case"] = self._case_id
