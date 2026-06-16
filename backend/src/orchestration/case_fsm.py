"""案件状态机 (CaseStateMachine) — 管控案件流转合法性。

维护状态迁移图，校验每次状态变更的合法性，
并通过 FileStorageManager 将状态实时落盘到当事人的 config.yaml。
"""

import logging
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional, Set

from ..core.event_bus import EventBus, EventType
from ..core.file_storage_manager import FileStorageManager
from ..utils.case_progress import infer_case_state_from_artifacts

logger = logging.getLogger(__name__)

FIRST_TRIAL_WAITING_STATE = "等待一审开庭"
SECOND_TRIAL_WAITING_STATE = "等待二审开庭"


class CaseState:
    """案件状态常量 — 业务流程节点，非物理移动。"""

    IDLE = "空闲"

    # ── 原告阶段 ──
    WAITING_FOR_RECEPTION = "等待前台接待"           # 原告在律所等待（可能在沙发）
    PLAINTIFF_CONSULTATION = "原告咨询中"            # 前台已分配，律师咨询进行中
    COMPLAINT_DRAFTING = "起诉状起草中"              # 咨询完成，律师起草起诉状
    COMPLAINT_FILED = "起诉状已递交"                 # 起诉状已递交基层法院
    DEFENDANT_WAITING = "等待被告"                   # 额外支持：等待被告激活

    # ── 被告阶段 ──
    DEFENDANT_SUMMONED = "被告已传唤"                # 法院已向被告送达起诉状副本
    DEFENDANT_CONSULTATION = "被告咨询中"            # 被告律师咨询进行中
    DEFENSE_DRAFTING = "答辩状起草中"                # 被告律师起草答辩状
    DEFENSE_FILED = "答辩状已递交"                   # 答辩状已递交法院

    # ── 一审准备与审判阶段 ──
    WAITING_FOR_FIRST_TRIAL = FIRST_TRIAL_WAITING_STATE
    TRIAL_FIRST_INSTANCE = "一审庭审中"              # 一审开庭审理
    FIRST_INSTANCE_VERDICT = "一审判决"              # 一审判决完成

    # ── 上诉决策阶段 ──
    APPEAL_DECISION = "上诉决策中"                   # 当事人决定是否上诉

    # ── 上诉阶段 ──
    APPEAL_DRAFTING = "上诉状起草中"                 # 上诉人律师起草上诉状
    APPEAL_FILED = "上诉状已递交"                    # 上诉状已递交中级法院
    APPEAL_RESPONSE_DRAFTING = "上诉答辩状起草中"    # 被上诉人律师起草上诉答辩状
    APPEAL_RESPONSE_FILED = "上诉答辩状已递交"       # 上诉答辩状已递交法院

    # ── 二审准备与审判阶段 ──
    WAITING_FOR_SECOND_TRIAL = SECOND_TRIAL_WAITING_STATE
    TRIAL_SECOND_INSTANCE = "二审庭审中"             # 二审开庭审理
    FINAL_VERDICT = "终审判决"                       # 二审判决（终审）
    CLOSED = "已结案"                                # 案件归档


# 合法状态迁移图：(当前状态) -> {可达的下一状态集合}
VALID_TRANSITIONS: Dict[str, Set[str]] = {
    CaseState.IDLE: {CaseState.WAITING_FOR_RECEPTION},
    CaseState.WAITING_FOR_RECEPTION: {
        CaseState.PLAINTIFF_CONSULTATION,
        CaseState.DEFENDANT_CONSULTATION,
    },
    CaseState.PLAINTIFF_CONSULTATION: {CaseState.COMPLAINT_DRAFTING},
    CaseState.COMPLAINT_DRAFTING: {CaseState.COMPLAINT_FILED},
    CaseState.COMPLAINT_FILED: {
        CaseState.DEFENDANT_SUMMONED,
        CaseState.DEFENDANT_WAITING,
        CaseState.WAITING_FOR_FIRST_TRIAL,
    },
    CaseState.DEFENDANT_WAITING: {CaseState.DEFENDANT_SUMMONED},
    CaseState.DEFENDANT_SUMMONED: {CaseState.DEFENDANT_CONSULTATION},
    CaseState.DEFENDANT_CONSULTATION: {CaseState.DEFENSE_DRAFTING},
    CaseState.DEFENSE_DRAFTING: {CaseState.DEFENSE_FILED},
    CaseState.DEFENSE_FILED: {CaseState.WAITING_FOR_FIRST_TRIAL},

    # ── 一审准备与审判 ──
    CaseState.WAITING_FOR_FIRST_TRIAL: {CaseState.TRIAL_FIRST_INSTANCE},
    CaseState.TRIAL_FIRST_INSTANCE: {CaseState.FIRST_INSTANCE_VERDICT},
    CaseState.FIRST_INSTANCE_VERDICT: {CaseState.APPEAL_DECISION},

    # ── 上诉决策分支 ──
    CaseState.APPEAL_DECISION: {
        CaseState.CLOSED,              # 服判 → 直接结案
        CaseState.APPEAL_DRAFTING,     # 不服 → 上诉
        CaseState.APPEAL_RESPONSE_DRAFTING,
    },

    # ── 上诉流程 ──
    CaseState.APPEAL_DRAFTING: {CaseState.APPEAL_FILED},
    CaseState.APPEAL_FILED: {
        CaseState.APPEAL_RESPONSE_DRAFTING,
        CaseState.WAITING_FOR_SECOND_TRIAL,
    },
    CaseState.APPEAL_RESPONSE_DRAFTING: {CaseState.APPEAL_RESPONSE_FILED},
    CaseState.APPEAL_RESPONSE_FILED: {CaseState.WAITING_FOR_SECOND_TRIAL},

    # ── 二审准备与审判 ──
    CaseState.WAITING_FOR_SECOND_TRIAL: {CaseState.TRIAL_SECOND_INSTANCE},
    CaseState.TRIAL_SECOND_INSTANCE: {CaseState.FINAL_VERDICT},
    CaseState.FINAL_VERDICT: {CaseState.CLOSED},
    CaseState.CLOSED: set(),  # 终态，无后续
}

SHARED_CASE_STATES: Set[str] = {
    CaseState.WAITING_FOR_FIRST_TRIAL,
    CaseState.TRIAL_FIRST_INSTANCE,
    CaseState.FIRST_INSTANCE_VERDICT,
    CaseState.APPEAL_DECISION,
    CaseState.APPEAL_DRAFTING,
    CaseState.APPEAL_FILED,
    CaseState.APPEAL_RESPONSE_DRAFTING,
    CaseState.APPEAL_RESPONSE_FILED,
    CaseState.WAITING_FOR_SECOND_TRIAL,
    CaseState.TRIAL_SECOND_INSTANCE,
    CaseState.FINAL_VERDICT,
    CaseState.CLOSED,
}


class CaseStateMachine:
    """案件状态机。

    监听各阶段完成事件，校验并推进案件状态，
    将状态变更实时写入当事人的 config.yaml。
    """

    def __init__(
        self,
        event_bus: EventBus,
        storage: FileStorageManager,
        state_change_notifier: Callable[..., Awaitable[None]] | None = None,
    ):
        self.event_bus = event_bus
        self.storage = storage
        self.state_change_notifier = state_change_notifier
        self._register_listeners()

    def _register_listeners(self) -> None:
        """注册对各阶段完成事件的监听。"""
        # 每个完成事件映射到对应的下一状态
        event_to_next_state = {
            EventType.PLAINTIFF_ARRIVED: CaseState.WAITING_FOR_RECEPTION,
            EventType.DEFENDANT_ARRIVED: CaseState.WAITING_FOR_RECEPTION,
            EventType.CASE_ASSIGNED: None,
            EventType.PLAINTIFF_CONSULTATION_COMPLETED: CaseState.COMPLAINT_DRAFTING,
            EventType.COMPLAINT_DRAFTING_COMPLETED: CaseState.COMPLAINT_FILED,
            EventType.LAWSUIT_FILED: CaseState.DEFENDANT_SUMMONED,
            EventType.DEFENDANT_SERVED: CaseState.DEFENDANT_CONSULTATION,
            EventType.DEFENDANT_CONSULTATION_COMPLETED: CaseState.DEFENSE_DRAFTING,
            EventType.DEFENSE_DRAFTING_COMPLETED: CaseState.DEFENSE_FILED,
            EventType.DEFENSE_FILED: CaseState.WAITING_FOR_FIRST_TRIAL,

            # ── 一审准备与审判 ──
            EventType.ENTER_TRIAL_FIRST_INSTANCE: CaseState.TRIAL_FIRST_INSTANCE,
            EventType.TRIAL_FIRST_INSTANCE_COMPLETED: CaseState.FIRST_INSTANCE_VERDICT,
            EventType.FIRST_INSTANCE_VERDICT_ISSUED: CaseState.APPEAL_DECISION,

            # ── 上诉决策与流程 ──
            EventType.APPEAL_DECISION_MADE: None,  # 需要根据 payload 决定
            EventType.APPEAL_DRAFTING_COMPLETED: CaseState.APPEAL_FILED,
            EventType.APPEAL_FILED: CaseState.APPEAL_RESPONSE_DRAFTING,
            EventType.APPEAL_RESPONSE_DRAFTING_COMPLETED: CaseState.APPEAL_RESPONSE_FILED,
            EventType.APPEAL_RESPONSE_FILED: CaseState.WAITING_FOR_SECOND_TRIAL,

            # ── 二审准备与审判 ──
            EventType.ENTER_TRIAL_SECOND_INSTANCE: CaseState.TRIAL_SECOND_INSTANCE,
            EventType.TRIAL_SECOND_INSTANCE_COMPLETED: CaseState.FINAL_VERDICT,
            EventType.FINAL_VERDICT_ISSUED: CaseState.CLOSED,
            EventType.CASE_CLOSED: CaseState.CLOSED,
        }

        for event_type in event_to_next_state:
            self.event_bus.subscribe(
                event_type,
                self._make_handler(event_type, event_to_next_state[event_type]),
                priority=100,
            )

    def _make_handler(self, event_type: EventType, default_next: Optional[str]):
        """为每个事件创建处理器闭包。"""

        async def handler(payload: dict):
            case_id = payload.get("case_id")
            client_path = payload.get("client_path")
            if not case_id or not client_path:
                logger.warning(f"[FSM] 事件 {event_type} 缺少 case_id 或 client_path")
                return

            # 确定目标状态
            next_state = default_next
            party_role = self._resolve_party_role(payload)

            # 特殊处理：上诉决策事件根据 payload 决定下一状态
            if event_type == EventType.APPEAL_DECISION_MADE:
                will_appeal = payload.get("will_appeal", False)
                if will_appeal and payload.get("player_current_appeal_stage") == "AR":
                    next_state = CaseState.APPEAL_RESPONSE_DRAFTING
                else:
                    next_state = CaseState.APPEAL_DRAFTING if will_appeal else CaseState.CLOSED
                logger.info(f"[FSM] 上诉决策: {'上诉' if will_appeal else '服判'} → {next_state}")
            elif event_type == EventType.CASE_ASSIGNED:
                next_state = (
                    CaseState.DEFENDANT_CONSULTATION
                    if party_role == "defendant"
                    else CaseState.PLAINTIFF_CONSULTATION
                )
            elif event_type == EventType.DEFENDANT_ARRIVED and party_role == "defendant":
                next_state = CaseState.DEFENDANT_SUMMONED

            if not next_state:
                next_state = payload.get("next_state")
            if not next_state:
                logger.warning(f"[FSM] 事件 {event_type} 无法确定下一状态")
                return

            transitioned, from_state, case_runtime = await self.transition(
                case_id,
                client_path,
                next_state,
                party_role=party_role,
            )
            if transitioned and self.state_change_notifier:
                await self.state_change_notifier(
                    case_id=case_id,
                    event=event_type,
                    from_state=from_state,
                    to_state=next_state,
                    party_role=case_runtime.get("active_party_role", party_role),
                    overall_state=case_runtime.get("overall_state", next_state),
                )

        return handler

    async def transition(
        self,
        case_id: str,
        client_path: str,
        next_state: str,
        party_role: str = "",
    ) -> tuple[bool, str, dict[str, Any]]:
        """执行状态迁移。

        Args:
            case_id: 案件 ID
            client_path: 当事人 config.yaml 所在目录路径
            next_state: 目标状态

        Returns:
            迁移是否成功
        """
        config = self.storage.load_agent_config(client_path)
        current_state = config.get("case_state", CaseState.IDLE)
        current_state = self._repair_transition_precondition(
            case_id, client_path, current_state, next_state
        )

        # 如果当前状态和目标状态相同，直接返回成功（幂等性）
        if current_state == next_state:
            logger.info(f"[FSM] 案件 {case_id}: 状态已经是 {next_state}，跳过迁移")
            return True, current_state, self._load_case_runtime(case_id)

        if not self._validate_transition(current_state, next_state):
            logger.error(
                f"[FSM] 非法状态迁移: {current_state} → {next_state} (case={case_id})"
            )
            return False, current_state, self._load_case_runtime(case_id)

        # 落盘状态变更
        self.storage.update_agent_field(client_path, "case_state", next_state)
        if next_state in SHARED_CASE_STATES:
            self._sync_shared_case_state(case_id, client_path, next_state)
        case_runtime = self._update_case_runtime(case_id, next_state, party_role)
        logger.info(f"[FSM] 案件 {case_id}: {current_state} → {next_state}")

        # 结案时通知 EventBus
        if next_state == CaseState.CLOSED:
            self.event_bus.mark_case_closed(case_id)

        return True, current_state, case_runtime

    def _repair_transition_precondition(
        self,
        case_id: str,
        client_path: str,
        current_state: str,
        next_state: str,
    ) -> str:
        """修复共享阶段被错误复位后的前置状态，避免恢复事件被 FSM 拒绝。"""
        current_state = self._repair_state_from_artifacts(
            case_id, client_path, current_state, next_state
        )

        if next_state != CaseState.TRIAL_FIRST_INSTANCE:
            return current_state

        if current_state == CaseState.WAITING_FOR_FIRST_TRIAL:
            return current_state
        if current_state == CaseState.TRIAL_FIRST_INSTANCE:
            return current_state

        repaired_state = self._infer_first_instance_precondition(case_id)
        if not repaired_state:
            return current_state

        if current_state != repaired_state:
            self.storage.update_agent_field(client_path, "case_state", repaired_state)
            if repaired_state in SHARED_CASE_STATES:
                self._sync_shared_case_state(case_id, client_path, repaired_state)
            logger.warning(
                "[FSM] 修补一审共享状态: %s -> %s (case=%s)",
                current_state,
                repaired_state,
                case_id,
            )

        return repaired_state

    def _repair_state_from_artifacts(
        self,
        case_id: str,
        client_path: str,
        current_state: str,
        next_state: str,
    ) -> str:
        """在状态被错误重置后，依据持久化产物补齐前置状态。"""
        if current_state == next_state or self._validate_transition(current_state, next_state):
            return current_state

        try:
            config = self.storage.load_agent_config(client_path)
        except Exception as exc:
            logger.warning(
                "[FSM] 读取案件配置失败，无法修补前置状态: %s (case=%s)",
                exc,
                case_id,
            )
            return current_state

        inferred_state = infer_case_state_from_artifacts(self.storage.base_dir, config)
        if not inferred_state or inferred_state == current_state:
            return current_state

        if inferred_state != next_state and not self._validate_transition(inferred_state, next_state):
            return current_state

        self.storage.update_agent_field(client_path, "case_state", inferred_state)
        if inferred_state in SHARED_CASE_STATES:
            self._sync_shared_case_state(case_id, client_path, inferred_state)
        logger.warning(
            "[FSM] 修补案件前置状态: %s -> %s (case=%s, target=%s)",
            current_state,
            inferred_state,
            case_id,
            next_state,
        )
        return inferred_state

    def _infer_first_instance_precondition(self, case_id: str) -> Optional[str]:
        """根据共享状态推断一审开庭前置状态。"""
        case_key = case_id.replace("case_", "", 1)
        observed_states = []

        for role in ("plaintiff", "defendant"):
            agent_dir = self.storage.get_case_agent_path(case_key, role)
            config_file = agent_dir / "config.yaml"
            if not config_file.exists():
                continue

            try:
                observed_states.append(
                    self.storage.load_agent_config(agent_dir).get("case_state", CaseState.IDLE)
                )
            except Exception as exc:
                logger.warning(
                    "[FSM] 读取 %s %s 共享状态失败: %s",
                    case_id,
                    role,
                    exc,
                )

        if CaseState.TRIAL_FIRST_INSTANCE in observed_states:
            return CaseState.TRIAL_FIRST_INSTANCE

        if any(
            state in {CaseState.WAITING_FOR_FIRST_TRIAL, FIRST_TRIAL_WAITING_STATE}
            for state in observed_states
        ):
            return CaseState.WAITING_FOR_FIRST_TRIAL

        return None

    def _sync_shared_case_state(self, case_id: str, client_path: str, next_state: str) -> None:
        case_key = case_id.replace("case_", "", 1)
        for role in ("plaintiff", "defendant"):
            agent_dir = self.storage.get_case_agent_path(case_key, role)
            config_file = agent_dir / "config.yaml"
            if not config_file.exists() or str(agent_dir) == str(client_path):
                continue

            try:
                config = self.storage.load_agent_config(agent_dir)
                if config.get("case_state") != next_state:
                    self.storage.update_agent_field(agent_dir, "case_state", next_state)
            except Exception as exc:
                logger.warning(
                    "[FSM] Failed to sync shared state for %s (%s): %s",
                    case_id,
                    role,
                    exc,
                )

    @staticmethod
    def _normalize_case_id(case_id: str) -> str:
        case_key = str(case_id or "").strip()
        if case_key.startswith("case_"):
            return case_key
        return f"case_{case_key}" if case_key else ""

    @staticmethod
    def _resolve_party_role(payload: dict) -> str:
        payload_role = str(payload.get("party_role", "") or "").strip()
        if payload_role in {"plaintiff", "defendant"}:
            return payload_role

        client_path = Path(str(payload.get("client_path", "") or ""))
        if "defendant" in client_path.parts:
            return "defendant"
        if "plaintiff" in client_path.parts:
            return "plaintiff"
        return "plaintiff"

    def _load_case_runtime(self, case_id: str) -> dict[str, Any]:
        normalized_case_id = self._normalize_case_id(case_id)
        load_case_runtime = getattr(self.storage, "load_case_runtime", None)
        try:
            runtime = load_case_runtime(normalized_case_id) if callable(load_case_runtime) else {}
        except Exception:
            runtime = {}

        return {
            "case_id": normalized_case_id,
            "overall_state": str(runtime.get("overall_state", CaseState.IDLE) or CaseState.IDLE),
            "plaintiff_state": str(runtime.get("plaintiff_state", CaseState.IDLE) or CaseState.IDLE),
            "defendant_state": str(runtime.get("defendant_state", CaseState.IDLE) or CaseState.IDLE),
            "active_party_role": str(runtime.get("active_party_role", "plaintiff") or "plaintiff"),
        }

    def _resolve_runtime_state_owner(self, next_state: str, party_role: str) -> str:
        if next_state in SHARED_CASE_STATES:
            return "shared"
        if next_state in {
            CaseState.DEFENDANT_SUMMONED,
            CaseState.DEFENDANT_CONSULTATION,
            CaseState.DEFENSE_DRAFTING,
            CaseState.DEFENSE_FILED,
        }:
            return "defendant"
        if next_state in {
            CaseState.PLAINTIFF_CONSULTATION,
            CaseState.COMPLAINT_DRAFTING,
            CaseState.COMPLAINT_FILED,
            CaseState.DEFENDANT_WAITING,
        }:
            return "plaintiff"
        if next_state == CaseState.WAITING_FOR_RECEPTION:
            return party_role or "plaintiff"
        return party_role or "plaintiff"

    def _resolve_active_party_role(self, next_state: str, party_role: str) -> str:
        owner = self._resolve_runtime_state_owner(next_state, party_role)
        return "shared" if owner == "shared" else owner

    def _update_case_runtime(
        self,
        case_id: str,
        next_state: str,
        party_role: str,
    ) -> dict[str, Any]:
        runtime = self._load_case_runtime(case_id)
        owner = self._resolve_runtime_state_owner(next_state, party_role)

        if owner == "shared":
            runtime["plaintiff_state"] = next_state
            runtime["defendant_state"] = next_state
        elif owner == "defendant":
            runtime["defendant_state"] = next_state
        else:
            runtime["plaintiff_state"] = next_state

        runtime["overall_state"] = next_state
        runtime["active_party_role"] = self._resolve_active_party_role(next_state, party_role)
        save_case_runtime = getattr(self.storage, "save_case_runtime", None)
        if callable(save_case_runtime):
            save_case_runtime(self._normalize_case_id(case_id), runtime)
        return runtime


    @staticmethod
    def _validate_transition(current: str, next_state: str) -> bool:
        """校验状态迁移合法性。"""
        valid_next = VALID_TRANSITIONS.get(current, set())
        return next_state in valid_next

    @staticmethod
    def get_valid_next_states(current: str) -> Set[str]:
        """获取当前状态的所有合法后续状态。"""
        return VALID_TRANSITIONS.get(current, set())
