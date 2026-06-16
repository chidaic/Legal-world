"""Receptionist agent that routes cases to lawyers."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any, Optional

from .base_agent import BaseAgent
from ..prompts.prompt_assembler import PromptAssembler
from ..utils.agent_trace import CaseAgentTraceRecorder, bind_agent_trace_context

if TYPE_CHECKING:
    from ..core.event_bus import EventBus
    from ..core.file_storage_manager import FileStorageManager

logger = logging.getLogger(__name__)

DEFAULT_LLM_MATCH_TIMEOUT_SECONDS = 25.0


class ReceptionistAgent(BaseAgent):
    """Front-desk agent for a law firm."""

    def __init__(
        self,
        firm_id: str,
        event_bus: "EventBus",
        storage: "FileStorageManager",
        config_path: Optional[str] = None,
        map_engine: Optional[Any] = None,
    ):
        self.firm_id = firm_id
        self._firm_dir: Optional[str] = None
        self.map_engine = map_engine
        self._assignment_lock = asyncio.Lock()
        self._reserved_lawyers: dict[str, str] = {}
        self._front_desk_busy = False
        self._front_desk_queue: list[dict] = []
        self._queued_client_sofas: dict[str, str] = {}
        self._queued_client_wait_spots: dict[str, str] = {}
        self._last_assigned_lawyer_id = ""
        self.runtime_issue_reporter = None

        super().__init__(
            agent_id=f"receptionist_{firm_id}",
            name=f"receptionist_{firm_id}",
            event_bus=event_bus,
            storage=storage,
            config_path=config_path,
        )

        from ..core.event_bus import EventType

        self.event_bus.subscribe(EventType.PLAINTIFF_ARRIVED, self.on_client_arrived)
        self.event_bus.subscribe(EventType.DEFENDANT_ARRIVED, self.on_client_arrived)
        logger.info("ReceptionistAgent '%s' initialized", self.firm_id)

    def _clear_lawyer_reservation(self, lawyer_id: str, case_id: str) -> None:
        if lawyer_id and self._reserved_lawyers.get(lawyer_id) == case_id:
            self._reserved_lawyers.pop(lawyer_id, None)

    def _finalize_dispatch_task(self, task: asyncio.Task, lawyer_id: str, case_id: str) -> None:
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            exc = None

        if exc is not None:
            logger.error(
                "[FrontDesk %s] dispatch task failed for case=%s lawyer=%s: %s",
                self.firm_id,
                case_id,
                lawyer_id,
                exc,
                exc_info=exc,
            )
        self._clear_lawyer_reservation(lawyer_id, case_id)
        return

        super().__init__(
            agent_id=f"receptionist_{firm_id}",
            name=f"前台_{firm_id}",
            event_bus=event_bus,
            storage=storage,
            config_path=config_path,
        )

        from ..core.event_bus import EventType

        self.event_bus.subscribe(EventType.PLAINTIFF_ARRIVED, self.on_client_arrived)
        self.event_bus.subscribe(EventType.DEFENDANT_ARRIVED, self.on_client_arrived)
        logger.info("ReceptionistAgent '%s' initialized", self.firm_id)

    async def on_client_arrived(self, payload: dict) -> None:
        """Handle a client arriving at this firm's front desk."""
        target_firm = payload.get("target_firm")
        if target_firm != self.firm_id:
            return

        async with self._assignment_lock:
            if self._front_desk_busy:
                await self._enqueue_front_desk_waiter(payload)
                return
            self._front_desk_busy = True

        await self._process_front_desk_arrival(payload)

    async def _report_runtime_issue(
        self,
        *,
        case_id: str,
        scenario_type: str,
        exc: Exception,
        stage_label: str = "",
    ) -> bool:
        reporter = getattr(self, "runtime_issue_reporter", None)
        if not callable(reporter):
            return False
        try:
            return bool(
                await reporter(
                    case_id=case_id,
                    scenario_type=scenario_type,
                    exc=exc,
                    stage_label=stage_label,
                )
            )
        except Exception as report_exc:
            logger.warning(
                "[FrontDesk %s] failed to report runtime issue for %s/%s: %s",
                self.firm_id,
                case_id,
                scenario_type,
                report_exc,
            )
            return False

    def _resolve_map_prefix(self, payload: dict) -> str:
        map_prefix = str(payload.get("map_prefix", "") or "").strip()
        compact = "".join(ch for ch in map_prefix.lower() if ch.isalnum())
        if compact == "lawfirmb":
            return "lawfirmB"
        if compact == "lawfirma":
            return "lawfirmA"

        firm_key = "".join(ch for ch in str(self.firm_id or "").lower() if ch.isalnum())
        return "lawfirmB" if firm_key == "lawfirmb" else "lawfirmA"

    def _get_reserved_sofa_ids(self, map_prefix: str) -> set[str]:
        reserved = set(self._queued_client_sofas.values())
        if not self.map_engine or not getattr(self.map_engine, "registry", None):
            return reserved

        prefix = f"{map_prefix}_sofa"
        sofa_locations = {
            loc_id: loc
            for loc_id, loc in self.map_engine.registry.lawfirm_sofas.items()
            if loc_id.startswith(prefix)
        }
        if not sofa_locations:
            return reserved

        for state in getattr(self.map_engine, "_agent_states", {}).values():
            sitting = state.get("sitting") or {}
            x = sitting.get("x")
            y = sitting.get("y")
            if x is None or y is None:
                continue
            for loc_id, loc in sofa_locations.items():
                if abs(loc.x - x) < 0.5 and abs(loc.y - y) < 0.5:
                    reserved.add(loc_id)
        return reserved

    def _get_reserved_wait_spot_ids(self, map_prefix: str) -> set[str]:
        reserved = set(self._queued_client_wait_spots.values())
        if not self.map_engine or not getattr(self.map_engine, "registry", None):
            return reserved

        prefix = f"{map_prefix}_wait_"
        wait_locations = {
            loc_id: loc
            for loc_id, loc in self.map_engine.registry.lawfirm_waiting_spots.items()
            if loc_id.startswith(prefix)
        }
        if not wait_locations:
            return reserved

        for state in getattr(self.map_engine, "_agent_states", {}).values():
            x = state.get("x")
            y = state.get("y")
            if x is None or y is None:
                continue
            for loc_id, loc in wait_locations.items():
                if abs(loc.x - x) < 0.5 and abs(loc.y - y) < 0.5:
                    reserved.add(loc_id)
        return reserved

    async def _move_client_to_queue_sofa(self, payload: dict) -> None:
        if not self.map_engine or not getattr(self.map_engine, "registry", None):
            return

        client_id = payload.get("client_id", "")
        if not client_id or client_id in self._queued_client_sofas:
            return

        map_prefix = self._resolve_map_prefix(payload)
        sofa_id = self.map_engine.registry.get_available_sofa(
            map_prefix,
            self._get_reserved_sofa_ids(map_prefix),
        )
        if not sofa_id:
            logger.warning("[FrontDesk %s] no sofa available for queued client %s", self.firm_id, client_id)
            return

        self._queued_client_sofas[client_id] = sofa_id
        try:
            await self.map_engine.stand_agent(client_id)
            moved = await self.map_engine.move_to_location(client_id, sofa_id)
            if moved:
                sofa_direction = "left" if map_prefix == "lawfirmB" else None
                await self.map_engine.sit_agent(client_id, sofa_id, direction_override=sofa_direction)
            else:
                self._queued_client_sofas.pop(client_id, None)
        except Exception:
            self._queued_client_sofas.pop(client_id, None)
            raise

    async def _move_client_to_queue_wait_spot(self, payload: dict) -> bool:
        if not self.map_engine or not getattr(self.map_engine, "registry", None):
            return False

        client_id = payload.get("client_id", "")
        if not client_id or client_id in self._queued_client_wait_spots:
            return False

        map_prefix = self._resolve_map_prefix(payload)
        wait_spot_id, wait_spot = self.map_engine.registry.get_available_waiting_spot(
            map_prefix,
            self._get_reserved_wait_spot_ids(map_prefix),
        )
        if not wait_spot_id or not wait_spot:
            logger.warning("[FrontDesk %s] no standing queue spot available for client %s", self.firm_id, client_id)
            return False

        self._queued_client_wait_spots[client_id] = wait_spot_id
        try:
            await self.map_engine.stand_agent(client_id)
            moved = await self.map_engine.move_to_location(client_id, wait_spot_id)
            if moved:
                await self.map_engine.stand_agent(
                    client_id,
                    direction_override=getattr(wait_spot, "direction", "") or "down",
                )
                return True
            self._queued_client_wait_spots.pop(client_id, None)
            return False
        except Exception:
            self._queued_client_wait_spots.pop(client_id, None)
            raise

    async def _enqueue_front_desk_waiter(self, payload: dict) -> None:
        client_id = payload.get("client_id", "")
        case_id = payload.get("case_id", "")
        already_queued = any(
            item.get("client_id") == client_id and item.get("case_id") == case_id
            for item in self._front_desk_queue
        )
        if not already_queued:
            queued_payload = dict(payload)
            queued_payload["_queued_front_desk"] = True
            self._front_desk_queue.append(queued_payload)
            logger.info(
                "[FrontDesk %s] desk busy, queued client=%s case=%s (queue=%d)",
                self.firm_id,
                client_id,
                case_id,
                len(self._front_desk_queue),
            )
        await self._move_client_to_queue_sofa(payload)
        if client_id not in self._queued_client_sofas:
            await self._move_client_to_queue_wait_spot(payload)

    async def _bring_client_to_front_desk(self, payload: dict) -> bool:
        if not self.map_engine or not getattr(self.map_engine, "registry", None):
            logger.warning("[FrontDesk %s] map engine unavailable, cannot move client to front desk", self.firm_id)
            return False

        client_id = payload.get("client_id", "")
        if not client_id:
            return False

        sofa_id = self._queued_client_sofas.pop(client_id, None)
        if sofa_id:
            await self.map_engine.stand_agent(client_id)
        wait_spot_id = self._queued_client_wait_spots.pop(client_id, None)
        if wait_spot_id:
            await self.map_engine.stand_agent(client_id)

        front_desk_id = f"{self._resolve_map_prefix(payload)}_front_desk"
        if not self.map_engine.registry.get(front_desk_id):
            logger.warning("[FrontDesk %s] front desk location missing: %s", self.firm_id, front_desk_id)
            return False

        moved = await self.map_engine.move_to_location(client_id, front_desk_id)
        if not moved:
            logger.warning("[FrontDesk %s] client %s failed to reach %s", self.firm_id, client_id, front_desk_id)
            return False
        return True

    async def _process_front_desk_arrival(self, payload: dict) -> None:
        client_id = payload.get("client_id", "")
        case_id = payload.get("case_id", "")
        party_role = payload.get("party_role", "plaintiff")

        logger.info("[FrontDesk %s] handling arrival: client=%s case=%s", self.firm_id, client_id, case_id)

        lawyer_id = ""
        needs_rule_recommendation = False
        try:
            moved_to_front_desk = await self._bring_client_to_front_desk(payload)
            if not moved_to_front_desk:
                logger.warning(
                    "[FrontDesk %s] skip reception for client=%s case=%s because front-desk movement failed",
                    self.firm_id,
                    client_id,
                    case_id,
                )
                return
            if payload.get("_queued_front_desk"):
                logger.info("[FrontDesk %s] queue backlog detected, using fast assignment for %s", self.firm_id, case_id)
                lawyer_id = self._rule_match(case_id, party_role)
                needs_rule_recommendation = True
            else:
                try:
                    logger.info("[FrontDesk %s] trying LLM lawyer matching", self.firm_id)
                    lawyer_id = await asyncio.wait_for(
                        self._llm_match(payload),
                        timeout=self._llm_match_timeout_seconds(),
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "[FrontDesk %s] LLM match timed out for case=%s; falling back to rule match",
                        self.firm_id,
                        case_id,
                    )
                    needs_rule_recommendation = True
                except Exception as exc:
                    logger.warning("[FrontDesk %s] LLM match failed: %s", self.firm_id, exc, exc_info=True)
                    needs_rule_recommendation = True

            if lawyer_id is None:
                logger.info(
                    "[FrontDesk %s] no suitable lawyer matched for client=%s case=%s; keeping case at front desk",
                    self.firm_id,
                    client_id,
                    case_id,
                )
                return

            if not lawyer_id:
                lawyer_id = self._rule_match(case_id, party_role)
                needs_rule_recommendation = True

            requested_lawyer_id = str(lawyer_id or "").strip()
            lawyer_id = self._resolve_lawyer_assignment(requested_lawyer_id, case_id, party_role)
            if lawyer_id:
                self._reserved_lawyers[lawyer_id] = case_id
                self._last_assigned_lawyer_id = lawyer_id
            if needs_rule_recommendation:
                await self._publish_rule_match_recommendation(lawyer_id, case_id, client_id, party_role)
            elif requested_lawyer_id and requested_lawyer_id != lawyer_id:
                await self._publish_final_assignment_correction(
                    requested_lawyer_id,
                    lawyer_id,
                    case_id,
                )

            await self._dispatch_case(lawyer_id, case_id, client_id, party_role, payload)
        finally:
            self._clear_lawyer_reservation(lawyer_id, case_id)
            next_payload = None
            async with self._assignment_lock:
                self._front_desk_busy = False
                if self._front_desk_queue:
                    next_payload = self._front_desk_queue.pop(0)
                    next_payload["_queued_front_desk"] = True
                    self._front_desk_busy = True
            if next_payload:
                await self._process_front_desk_arrival(next_payload)

    def _llm_match_timeout_seconds(self) -> float:
        try:
            return max(
                1.0,
                float(os.getenv("RECEPTION_LLM_MATCH_TIMEOUT_SECONDS", DEFAULT_LLM_MATCH_TIMEOUT_SECONDS)),
            )
        except (TypeError, ValueError):
            return DEFAULT_LLM_MATCH_TIMEOUT_SECONDS

    async def _llm_match(self, payload: dict) -> Optional[str]:
        """Run the reception scenario and return the matched lawyer id.

        Returns ``None`` when the scenario explicitly concludes that no suitable
        lawyer exists in the current roster.
        """
        from ..scenarios.reception_scenario import ReceptionScenario

        roster = self._load_roster()
        if not roster:
            return ""

        payload_case_id = str(payload.get("case_id", "") or "")
        payload_party_role = str(payload.get("party_role", "plaintiff") or "plaintiff")
        preferred_lawyer_id = self._get_case_designated_lawyer_id(
            payload_case_id,
            payload_party_role,
            require_available=True,
        )
        preferred_lawyer_name = self._lookup_lawyer_name(preferred_lawyer_id, roster)

        client_agent = self._find_client_agent(payload.get("client_id", ""))
        if not client_agent:
            logger.warning("[FrontDesk %s] client agent not found, skipping LLM match", self.firm_id)
            return ""

        roster_str = self._format_roster_for_prompt(
            roster,
            case_id=payload_case_id,
            party_role=payload_party_role,
        )
        receptionist_prompt = PromptAssembler.build_scenario_prompt(
            "receptionist",
            "RECEPTION",
            {"lawyer_roster": roster_str},
        )

        client_config = {}
        if client_agent.config_path and client_agent.storage:
            client_config = client_agent.storage.load_agent_config(client_agent.config_path)

        case_background = ""
        dataset_path = client_config.get("dataset_path", "")
        case_id = client_config.get("case_id", payload_case_id or "0")
        if dataset_path:
            try:
                from ..data.data_loader import DataLoader

                data_loader = DataLoader(dataset_path)
                case = data_loader.resolve_case_for_config(client_config, fallback_name=client_agent.name)
                case_background = data_loader.extract_case_background(case)
            except Exception as exc:
                logger.warning("[FrontDesk %s] failed to load case background: %s", self.firm_id, exc)

        client_scenario = PromptAssembler.build_scenario_prompt(
            "client",
            "RECEPTION",
            {"case_background": case_background},
        )
        client_prompt = PromptAssembler.build(
            profile={
                "name": client_agent.name,
                "gender": getattr(client_agent, "gender", ""),
                "birth_date": getattr(client_agent, "birth_date", ""),
                "address": getattr(client_agent, "address", ""),
            },
            scenario_prompt=client_scenario,
        )

        self.activate(receptionist_prompt)
        client_agent.activate(client_prompt)

        try:
            output_path = None
            trace_recorder = None
            trace_stage_key = f"RECEPTION_{payload.get('party_role', 'plaintiff')}".upper()
            if self.storage:
                output_dir = self.storage.base_dir / "output" / payload.get("case_id", "unknown")
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = str(output_dir / "RECEPTION_result.json")
                trace_recorder = CaseAgentTraceRecorder(output_dir)
                bind_agent_trace_context(
                    self,
                    recorder=trace_recorder,
                    output_dir=output_dir / "_debug" / "agent_traces" / self.agent_id,
                    stage_code="RECEPTION",
                    stage_key=trace_stage_key,
                )
                bind_agent_trace_context(
                    client_agent,
                    recorder=trace_recorder,
                    output_dir=output_dir / "_debug" / "agent_traces" / client_agent.agent_id,
                    stage_code="RECEPTION",
                    stage_key=trace_stage_key,
                )

            scenario = ReceptionScenario(
                receptionist_agent=self,
                client_agent=client_agent,
                lawyer_roster=roster,
                preferred_lawyer_id=preferred_lawyer_id,
                preferred_lawyer_name=preferred_lawyer_name,
                output_path=output_path,
                verbose=True,
                map_engine=self.map_engine,
                trace_recorder=trace_recorder,
                trace_stage_code="RECEPTION",
                trace_stage_key=trace_stage_key,
            )
            result = await scenario.execute()
            if trace_recorder is not None:
                trace_recorder.export_stage(
                    stage_code="RECEPTION",
                    stage_key=trace_stage_key,
                    agents=[self, client_agent],
                    stage_result=result,
                    stage_result_path=output_path,
                    status="completed",
                )
            if result.get("match_status") == "no_match":
                return None
            return str(result.get("matched_lawyer_id", "") or "")
        except Exception as exc:
            logger.error("[Receptionist] Reception scenario failed: %s", exc)
            if trace_recorder is not None:
                trace_recorder.export_stage(
                    stage_code="RECEPTION",
                    stage_key=trace_stage_key,
                    agents=[self, client_agent],
                    stage_result=None,
                    stage_result_path=output_path,
                    status="failed",
                    error=repr(exc),
                )
            self.recover_from_error()
            client_agent.recover_from_error()
            if await self._report_runtime_issue(
                case_id=str(payload.get("case_id", "") or ""),
                scenario_type="RECEPTION",
                exc=exc,
                stage_label="前台导引",
            ):
                return None
            return ""
        finally:
            if self.is_active:
                self.deactivate()
            if client_agent.is_active:
                client_agent.deactivate()

    def _get_conflicted_lawyer_ids(self, case_id: str, party_role: str) -> set[str]:
        if not self.storage or not case_id:
            return set()

        opposing_role = "defendant" if party_role == "plaintiff" else "plaintiff"
        try:
            opposing_path = self.storage.get_case_agent_path(case_id, opposing_role)
            opposing_config = self.storage.load_agent_config(opposing_path)
        except FileNotFoundError:
            return set()
        except Exception:
            logger.warning(
                "[FrontDesk %s] failed to inspect opposing counsel for case=%s role=%s",
                self.firm_id,
                case_id,
                party_role,
                exc_info=True,
            )
            return set()

        conflicted_lawyer_ids = {
            str(opposing_config.get(field, "") or "").strip()
            for field in ("assigned_lawyer_id", "designated_lawyer_id")
        }
        return {lawyer_id for lawyer_id in conflicted_lawyer_ids if lawyer_id}

    def _get_live_available_lawyer_ids(self, case_id: str, party_role: str = "plaintiff") -> list[str]:
        return [
            lawyer_id
            for lawyer_id in self._get_ranked_lawyer_ids()
            if self._is_lawyer_available(lawyer_id, case_id)
            and lawyer_id not in self._get_conflicted_lawyer_ids(case_id, party_role)
        ]

    def _choose_available_lawyer(self, available_lawyer_ids: list[str], preferred_lawyer_id: str = "") -> str:
        if not available_lawyer_ids:
            return ""

        ordered_ids = [lawyer_id for lawyer_id in self._get_ranked_lawyer_ids() if lawyer_id in available_lawyer_ids]
        if not ordered_ids:
            ordered_ids = list(available_lawyer_ids)

        # 前台已明确推荐具体律师时，优先尊重推荐结果；
        # 轮转均衡只适用于没有明确推荐的兜底分配。
        if preferred_lawyer_id in ordered_ids:
            return preferred_lawyer_id

        if len(ordered_ids) > 1 and self._last_assigned_lawyer_id in ordered_ids:
            pivot = ordered_ids.index(self._last_assigned_lawyer_id)
            ordered_ids = ordered_ids[pivot + 1:] + ordered_ids[:pivot + 1]

        return ordered_ids[0]

    def _rule_match(self, case_id: str = "", party_role: str = "plaintiff") -> str:
        """Fallback lawyer selection using roster order."""
        available_lawyer_ids = self._get_live_available_lawyer_ids(case_id, party_role) if case_id else []
        if available_lawyer_ids:
            lawyer_id = self._choose_available_lawyer(available_lawyer_ids)
            logger.info("[FrontDesk %s] rule match selected live-available %s", self.firm_id, lawyer_id)
            return lawyer_id

        roster = self._load_roster()
        if not roster:
            return ""

        lawyers = roster.get("lawyers", [])
        for lawyer in lawyers:
            if lawyer.get("status") == "available":
                logger.info(
                    "[FrontDesk %s] rule match selected %s (%s)",
                    self.firm_id,
                    lawyer.get("id", ""),
                    lawyer.get("name", ""),
                )
                return str(lawyer.get("id", "") or "")

        return str(lawyers[0].get("id", "") or "") if lawyers else ""

    def _resolve_lawyer_assignment(self, preferred_lawyer_id: str, case_id: str, party_role: str) -> str:
        """Select a live available lawyer for concurrent multi-case execution."""
        ranked_lawyer_ids = self._get_ranked_lawyer_ids()
        if not ranked_lawyer_ids:
            return preferred_lawyer_id

        conflicted_lawyer_ids = self._get_conflicted_lawyer_ids(case_id, party_role)
        available_lawyer_ids = self._get_live_available_lawyer_ids(case_id, party_role)
        if available_lawyer_ids:
            lawyer_id = self._choose_available_lawyer(available_lawyer_ids, preferred_lawyer_id)
            if lawyer_id in conflicted_lawyer_ids:
                lawyer_id = next(
                    (
                        candidate_id
                        for candidate_id in available_lawyer_ids
                        if candidate_id not in conflicted_lawyer_ids
                    ),
                    "",
                )
            if preferred_lawyer_id and preferred_lawyer_id != lawyer_id:
                logger.info(
                    "[FrontDesk %s] rerouted %s from preferred lawyer %s to idle lawyer %s",
                    self.firm_id,
                    case_id,
                    preferred_lawyer_id,
                    lawyer_id,
                )
            return lawyer_id

        if preferred_lawyer_id and preferred_lawyer_id in conflicted_lawyer_ids:
            ranked_lawyer_ids = [
                lawyer_id for lawyer_id in ranked_lawyer_ids
                if lawyer_id not in conflicted_lawyer_ids
            ]

        fallback_lawyer_id = preferred_lawyer_id or ranked_lawyer_ids[0]
        if fallback_lawyer_id in conflicted_lawyer_ids:
            fallback_lawyer_id = ranked_lawyer_ids[0] if ranked_lawyer_ids else ""
        logger.info(
            "[FrontDesk %s] all lawyers are busy, keeping %s for queued handling of %s",
            self.firm_id,
            fallback_lawyer_id,
            case_id,
        )
        return fallback_lawyer_id

    def _build_rule_match_client_need(self, party_role: str) -> str:
        if str(party_role or "").strip().lower() == "defendant":
            return (
                "你好，我刚收到法院送达的起诉状，说我被起诉了。"
                "我想咨询一下接下来应该怎么应对、需要准备什么材料。"
            )
        return "你好，我想咨询一下这个案件该怎么处理，麻烦帮我安排合适的律师。"

    async def _publish_rule_match_recommendation(
        self,
        lawyer_id: str,
        case_id: str,
        client_id: str,
        party_role: str,
    ) -> None:
        """Show a front-desk recommendation when matching falls back to rules."""
        if not lawyer_id or not case_id or not self.map_engine:
            return

        roster = self._load_roster()
        lawyer_name = self._lookup_lawyer_name(lawyer_id, roster) or lawyer_id
        client = self._find_client_agent(client_id)
        client_name = str(getattr(client, "name", "") or "当事人").strip()
        client_need = self._build_rule_match_client_need(party_role)
        text = (
            f"我了解了。您这个情况适合由{lawyer_name}律师处理，"
            f"我现在帮您分配给{lawyer_name}律师。"
        )
        if hasattr(self.map_engine, "broadcast_dialogue"):
            await self.map_engine.broadcast_dialogue(
                case_id,
                client_id,
                client_name,
                client_need,
                2,
                scenario_type="RECEPTION",
            )
            await self.map_engine.broadcast_dialogue(
                case_id,
                self.agent_id,
                "律所前台",
                text,
                3,
                scenario_type="RECEPTION",
            )
        if hasattr(self.map_engine, "send_update_dialogue"):
            await self.map_engine.send_update_dialogue(client_id, client_need, 2.0)
            await self.map_engine.send_update_dialogue(self.agent_id, text, 2.0)

    async def _publish_final_assignment_correction(
        self,
        original_lawyer_id: str,
        final_lawyer_id: str,
        case_id: str,
    ) -> None:
        """Announce the actual lawyer if availability rerouted a visible recommendation."""
        if not final_lawyer_id or original_lawyer_id == final_lawyer_id or not case_id or not self.map_engine:
            return

        roster = self._load_roster()
        final_lawyer_name = self._lookup_lawyer_name(final_lawyer_id, roster) or final_lawyer_id
        text = (
            f"刚才推荐的律师当前无法立即接待，"
            f"最终为您安排【推荐律师：{final_lawyer_id}】{final_lawyer_name}律师接续办理。"
        )
        if hasattr(self.map_engine, "broadcast_dialogue"):
            await self.map_engine.broadcast_dialogue(
                case_id,
                self.agent_id,
                "律所前台",
                text,
                4,
                scenario_type="RECEPTION",
            )
        if hasattr(self.map_engine, "send_update_dialogue"):
            await self.map_engine.send_update_dialogue(self.agent_id, text, 2.0)

    def _get_case_designated_lawyer_ids(self, case_id: str, party_role: str = "plaintiff") -> list[str]:
        if not case_id:
            return []

        ranked_ids = self._get_ranked_lawyer_ids()
        designated_ids: list[str] = []

        configured_lawyer_id = self._get_case_configured_lawyer_id(case_id, party_role)
        if configured_lawyer_id and (not ranked_ids or configured_lawyer_id in ranked_ids):
            designated_ids.append(configured_lawyer_id)

        event_bus = getattr(self, "event_bus", None)
        registry = getattr(event_bus, "_registry", None)
        if not registry:
            return designated_ids
        for lawyer in registry.get_agents_by_type("lawyer"):
            lawyer_id = str(getattr(lawyer, "agent_id", "") or "").strip()
            if not lawyer_id or getattr(lawyer, "firm_id", "") != self.firm_id:
                continue

            case_queue = [
                str(item).strip()
                for item in (getattr(lawyer, "case_queue", []) or [])
                if str(item).strip()
            ]
            if case_id in case_queue and lawyer_id not in designated_ids:
                designated_ids.append(lawyer_id)

        if ranked_ids:
            designated_ids.sort(
                key=lambda lawyer_id: ranked_ids.index(lawyer_id)
                if lawyer_id in ranked_ids
                else len(ranked_ids)
            )
        return designated_ids

    def _get_case_configured_lawyer_id(self, case_id: str, party_role: str) -> str:
        storage = getattr(self, "storage", None)
        if not storage or not case_id:
            return ""

        try:
            agent_path = storage.get_case_agent_path(case_id, party_role)
            agent_config = storage.load_agent_config(agent_path)
        except FileNotFoundError:
            return ""
        except Exception:
            logger.warning(
                "[FrontDesk %s] failed to inspect configured lawyer for case=%s role=%s",
                self.firm_id,
                case_id,
                party_role,
                exc_info=True,
            )
            return ""

        return str(
            agent_config.get("designated_lawyer_id", "") or agent_config.get("assigned_lawyer_id", "") or ""
        ).strip()

    def _get_case_designated_lawyer_id(
        self,
        case_id: str,
        party_role: str = "plaintiff",
        require_available: bool = False,
    ) -> str:
        designated_ids = [
            lawyer_id
            for lawyer_id in self._get_case_designated_lawyer_ids(case_id, party_role)
            if lawyer_id not in self._get_conflicted_lawyer_ids(case_id, party_role)
        ]
        if not designated_ids:
            return ""

        if require_available:
            available_ids = [
                lawyer_id
                for lawyer_id in designated_ids
                if self._is_lawyer_available(lawyer_id, case_id)
            ]
            if not available_ids:
                return ""
            return self._choose_available_lawyer(available_ids, preferred_lawyer_id=available_ids[0])

        return designated_ids[0]

    def _lookup_lawyer_name(self, lawyer_id: str, roster: Optional[dict] = None) -> str:
        if lawyer_id and roster:
            for lawyer in roster.get("lawyers", []):
                if str(lawyer.get("id", "") or "").strip() == lawyer_id:
                    return str(lawyer.get("name", "") or "").strip()

        event_bus = getattr(self, "event_bus", None)
        registry = getattr(event_bus, "_registry", None)
        if not registry or not lawyer_id:
            return ""

        lawyer = registry.get_agent(lawyer_id)
        return str(getattr(lawyer, "name", "") or "").strip() if lawyer else ""

    def _get_ranked_lawyer_ids(self) -> list[str]:
        roster = self._load_roster()
        ranked_ids = [
            str(lawyer.get("id", "")).strip()
            for lawyer in roster.get("lawyers", [])
            if str(lawyer.get("id", "")).strip()
        ]
        if ranked_ids:
            return ranked_ids

        event_bus = getattr(self, "event_bus", None)
        registry = getattr(event_bus, "_registry", None)
        if not registry:
            return []

        return [
            agent.agent_id
            for agent in registry.get_agents_by_type("lawyer")
            if getattr(agent, "firm_id", "") == self.firm_id
        ]

    def _is_lawyer_available(self, lawyer_id: str, case_id: str) -> bool:
        reservation_case_id = self._reserved_lawyers.get(lawyer_id)
        if reservation_case_id and reservation_case_id != case_id:
            return False

        registry = getattr(self.event_bus, "_registry", None)
        if not registry:
            return False

        lawyer = registry.get_agent(lawyer_id)
        if not lawyer:
            return False
        if getattr(lawyer, "firm_id", "") not in {"", self.firm_id}:
            return False
        if self.event_bus.is_agent_busy(lawyer_id):
            return False

        current_case_id = getattr(lawyer, "current_handling_case", None)
        if current_case_id and current_case_id != case_id:
            return False

        return True

    def _load_roster(self) -> dict:
        """Load the firm roster."""
        storage = getattr(self, "storage", None)
        if not storage:
            return {}
        roster_path = storage.base_dir / "law_firms" / self.firm_id / "lawyer_roster.yaml"
        try:
            return storage.load_yaml(roster_path)
        except FileNotFoundError:
            logger.warning("[FrontDesk %s] roster not found: %s", self.firm_id, roster_path)
            return {}

    def _find_client_agent(self, client_id: str):
        """Look up a client agent through the registry stored on the event bus."""
        registry = getattr(self.event_bus, "_registry", None)
        if registry:
            return registry.get_agent(client_id)
        return None

    def _format_roster_for_prompt(
        self,
        roster: dict,
        case_id: str = "",
        party_role: str = "plaintiff",
    ) -> str:
        """Render roster data for the reception prompt."""
        lawyers = roster.get("lawyers", [])
        designated_lawyer_id = self._get_case_designated_lawyer_id(case_id, party_role) if case_id else ""
        lines = []
        for lawyer in lawyers:
            name = lawyer.get("name", "")
            lawyer_id = lawyer.get("id", "")
            specialty = "、".join(lawyer.get("specialty", []))
            seniority = lawyer.get("seniority", "")
            status = lawyer.get("status", "")
            if lawyer_id and case_id:
                status = "available" if self._is_lawyer_available(lawyer_id, case_id) else "busy"
            case_hint = " | 当前案件：优先承办" if lawyer_id and lawyer_id == designated_lawyer_id else ""
            lines.append(
                f"- {name}（{lawyer_id}） | 专长：{specialty or '综合'} | 资历：{seniority} | 状态：{status}{case_hint}"
            )
        return "\n".join(lines) if lines else "暂无律师信息"

    async def _dispatch_case(
        self,
        lawyer_id: str,
        case_id: str,
        client_id: str,
        party_role: str,
        original_payload: dict,
    ) -> None:
        """Publish CASE_ASSIGNED."""
        from ..core.event_bus import EventType

        payload = {
            **original_payload,
            "lawyer_id": lawyer_id,
            "case_id": case_id,
            "client_id": client_id,
            "firm_id": self.firm_id,
            "party_role": party_role,
        }
        logger.info("[FrontDesk %s] dispatch case %s to %s", self.firm_id, case_id, lawyer_id)
        await self.event_bus.publish(EventType.CASE_ASSIGNED, payload)
