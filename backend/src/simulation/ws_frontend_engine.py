"""WebSocket 前端引擎 (WebSocketFrontendEngine)。

实现 TownAvatarInterface，通过 WebSocket 向前端发送移动/动画指令，
并等待前端确认完成。无前端连接时自动降级为 sleep 模拟。

新客户端连接时自动发送完整 Agent 状态快照，确保前端完全基于后端实时数据。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from typing import Any
from pathlib import Path

from fastapi import WebSocket

from .backend_pathfinder import BackendPathfinder
from .map_engine import TownAvatarInterface
from .location_registry import LocationRegistry
from . import ws_protocol as proto

logger = logging.getLogger(__name__)

FRONTEND_MODE_AUTO = "auto"
FRONTEND_MODE_LEGACY = "legacy"
FRONTEND_MODE_PLAYER_V2 = "player_v2"
VALID_FRONTEND_MODES = {
    FRONTEND_MODE_AUTO,
    FRONTEND_MODE_LEGACY,
    FRONTEND_MODE_PLAYER_V2,
}

TURN_MODE_AUTO = "auto"
TURN_MODE_DIALOGUE_GATE = "dialogue_gate"
TURN_MODE_STEP_GATE = "step_gate"
DIALOGUE_GATE_AUTO_TIMEOUT_SECONDS = 8.0
VALID_TURN_MODES = {
    TURN_MODE_AUTO,
    TURN_MODE_DIALOGUE_GATE,
    TURN_MODE_STEP_GATE,
}


def _normalize_frontend_mode(value: str | None) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    return normalized if normalized in VALID_FRONTEND_MODES else FRONTEND_MODE_AUTO


def _normalize_turn_mode(value: str | None) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    return normalized if normalized in VALID_TURN_MODES else TURN_MODE_AUTO


class WebSocketFrontendEngine(TownAvatarInterface):
    """通过 WebSocket 驱动前端渲染的地图引擎。"""

    def __init__(
        self,
        location_registry: LocationRegistry,
        fallback_speed: float = 1.0,
        backend_authoritative: bool = True,
        move_speed_px_per_second: float = 150.0,
        map_json_path: str | Path | None = None,
        frontend_mode: str = FRONTEND_MODE_AUTO,
        turn_mode: str = TURN_MODE_AUTO,
    ):
        self.registry = location_registry  # 地图位置注册表（用于查找坐标）
        self.loc_registry = location_registry  # 保留别名以兼容
        self.clients: set[WebSocket] = set()
        self._ack_events: dict[str, asyncio.Event] = {}
        self._fallback_speed = fallback_speed
        self._backend_authoritative = backend_authoritative
        self._move_speed_px_per_second = max(float(move_speed_px_per_second or 150.0), 1.0)
        self._min_move_duration = 0.15
        self._move_duration_padding = 0.25
        self._frontend_mode = _normalize_frontend_mode(frontend_mode)
        self._turn_mode = _normalize_turn_mode(turn_mode)
        self._pathfinder: BackendPathfinder | None = None
        if map_json_path:
            try:
                self._pathfinder = BackendPathfinder(map_json_path)
            except Exception as exc:
                logger.warning("Failed to initialize backend pathfinder: %s", exc)

        # 这些将在 ws_server.py 中初始化后注入
        self.agent_registry = None
        self.storage = None

        # ── 暂停控制 ──
        self._paused = False
        self._resumed_event = asyncio.Event()
        self._resumed_event.set()  # 初始为非暂停状态
        self._dialogue_turn_gate_enabled = False
        self._client_frontend_modes: dict[WebSocket, str] = {}
        self._dialogue_gate_supported_clients: set[WebSocket] = set()
        self._dialogue_gate_counter = 0
        self._dialogue_gate_events: dict[str, asyncio.Event] = {}
        self._active_dialogue_gate_id: str | None = None
        self._active_dialogue_gate_payload: dict[str, Any] | None = None
        self._dialogue_lookahead_started = False
        self._buffered_dialogue_message: dict[str, Any] | None = None

        # ── Agent 状态追踪（用于新客户端同步） ──
        # key = agent_id, value = 当前状态快照
        self._agent_states: dict[str, dict[str, Any]] = {}

    @property
    def _fallback_mode(self) -> bool:
        return len(self.clients) == 0

    # ── WebSocket 连接管理 ──

    async def add_client(self, ws: WebSocket) -> None:
        self.clients.add(ws)
        logger.info("Frontend connected (%d clients)", len(self.clients))

        # 向新客户端发送完整的 Agent 状态快照
        await self._send_snapshot(ws)

    async def _send_snapshot(self, ws: WebSocket) -> None:
        """向单个客户端发送所有 Agent 的当前状态快照。"""
        if not self._agent_states:
            return

        logger.info("Sending agent snapshot to new client (%d agents)", len(self._agent_states))

        for agent_id, state in self._agent_states.items():
            try:
                # 1. 发送 spawn 指令
                spawn_msg = proto.agent_spawn(
                    agent_id,
                    state["name"],
                    state["character_name"],
                    state["x"],
                    state["y"],
                    state.get("role", ""),
                )
                await ws.send_text(json.dumps(spawn_msg, ensure_ascii=False))

                # 2. 如果 Agent 正在坐着，发送 sit 指令
                if state.get("sitting"):
                    sit_info = state["sitting"]
                    sit_msg = proto.agent_sit(
                        agent_id,
                        sit_info["x"],
                        sit_info["y"],
                        sit_info.get("direction", "down"),
                    )
                    await ws.send_text(json.dumps(sit_msg, ensure_ascii=False))
                elif state.get("standing_direction"):
                    stand_msg = proto.agent_stand(
                        agent_id,
                        state.get("standing_direction", ""),
                    )
                    await ws.send_text(json.dumps(stand_msg, ensure_ascii=False))
            except Exception as e:
                logger.warning("Failed to send snapshot for %s: %s", agent_id, e)
                break

    async def remove_client(self, ws: WebSocket) -> None:
        self.clients.discard(ws)
        self._client_frontend_modes.pop(ws, None)
        self._dialogue_gate_supported_clients.discard(ws)
        logger.info("Frontend disconnected (%d clients)", len(self.clients))

    async def on_frontend_message(self, data: dict[str, Any], ws: WebSocket | None = None) -> None:
        """处理前端发来的消息。"""
        msg_type = data.get("type", "")

        if msg_type == proto.MSG_MOVE_COMPLETE:
            agent_id = data.get("agent_id", "")
            key = f"{agent_id}:move_complete"
            if key in self._ack_events:
                self._ack_events[key].set()

        elif msg_type == proto.MSG_ANIMATION_COMPLETE:
            agent_id = data.get("agent_id", "")
            key = f"{agent_id}:animation_complete"
            if key in self._ack_events:
                self._ack_events[key].set()

        elif msg_type == proto.MSG_CLIENT_READY:
            client_mode = self._resolve_client_frontend_mode(data)
            if ws is not None:
                self._client_frontend_modes[ws] = client_mode
            if self._client_supports_dialogue_gate(data, client_mode):
                if ws is not None:
                    self._dialogue_gate_supported_clients.add(ws)
                    await self._send_active_dialogue_gate_to_client(ws)
                else:
                    self._dialogue_turn_gate_enabled = True
                logger.info("Frontend supports dialogue turn gate (mode=%s, turn_mode=%s)", client_mode, self._turn_mode)
            logger.info("Frontend signaled ready (mode=%s)", client_mode)

        elif msg_type == proto.MSG_DIALOGUE_CONTINUE:
            gate_id = str(data.get("gate_id") or self._active_dialogue_gate_id or "")
            if gate_id and gate_id in self._dialogue_gate_events:
                self._dialogue_gate_events[gate_id].set()
                if ws is not None:
                    await self._send_to_client(ws, proto.dialogue_gate_accepted(self._case_id_from_gate(gate_id), gate_id))
            elif ws is not None:
                await self._send_to_client(
                    ws,
                    proto.dialogue_gate_error(
                        self._case_id_from_gate(gate_id),
                        gate_id,
                        "继续失败：后端当前没有等待这一步对话。请刷新状态或重新进入案件后再试。",
                        code="DIALOGUE_GATE_NOT_FOUND",
                    ),
                )

        elif msg_type == proto.MSG_SIMULATION_PAUSE:
            self._paused = True
            self._resumed_event.clear()
            logger.info("⏸  Simulation PAUSED by frontend")

        elif msg_type == proto.MSG_SIMULATION_RESUME:
            self._paused = False
            self._resumed_event.set()
            logger.info("▶  Simulation RESUMED by frontend")

    async def _sleep_with_pause(self, duration: float) -> None:
        """Sleep while respecting frontend pause/resume signals."""
        remaining = max(float(duration or 0.0), 0.0)
        while remaining > 0:
            if self._paused:
                await self._resumed_event.wait()
                continue
            step = min(0.1, remaining)
            try:
                await asyncio.sleep(step)
            except asyncio.CancelledError:
                break
            if not self._paused:
                remaining -= step

    async def _wait_until_resumed(self) -> None:
        """Block user-visible state pushes while the sandbox is paused."""
        if self._paused:
            await self._resumed_event.wait()

    async def _send_to_client(self, ws: WebSocket, message: dict[str, Any]) -> None:
        try:
            await ws.send_text(json.dumps(message, ensure_ascii=False))
        except Exception as exc:
            logger.warning("Failed to send frontend message %s: %s", message.get("type"), exc)

    def _message_supports_dialogue_gate(self, data: dict[str, Any]) -> bool:
        capabilities = data.get("capabilities")
        if isinstance(capabilities, list) and "dialogue_turn_gate" in capabilities:
            return True
        if isinstance(capabilities, dict) and capabilities.get("dialogue_turn_gate"):
            return True
        return bool(data.get("supports_dialogue_turn_gate"))

    def _message_supports_step_gate(self, data: dict[str, Any]) -> bool:
        capabilities = data.get("capabilities")
        if isinstance(capabilities, list) and "step_gate" in capabilities:
            return True
        if isinstance(capabilities, dict) and capabilities.get("step_gate"):
            return True
        return bool(data.get("supports_step_gate"))

    def _resolve_client_frontend_mode(self, data: dict[str, Any]) -> str:
        requested = _normalize_frontend_mode(str(data.get("frontend_mode") or data.get("mode") or ""))
        if requested != FRONTEND_MODE_AUTO:
            return requested
        if self._frontend_mode != FRONTEND_MODE_AUTO:
            return self._frontend_mode
        if (
            self._message_supports_dialogue_gate(data)
            or self._message_supports_step_gate(data)
            or bool(data.get("supports_player_mode"))
        ):
            return FRONTEND_MODE_PLAYER_V2
        return FRONTEND_MODE_LEGACY

    def _client_supports_dialogue_gate(self, data: dict[str, Any], client_mode: str) -> bool:
        if client_mode != FRONTEND_MODE_PLAYER_V2:
            return False
        if self._turn_mode == TURN_MODE_STEP_GATE:
            return False
        if self._turn_mode not in {TURN_MODE_AUTO, TURN_MODE_DIALOGUE_GATE}:
            return False
        return self._message_supports_dialogue_gate(data)

    def _should_gate_dialogue(self) -> bool:
        if not self.clients:
            return False
        if self._frontend_mode == FRONTEND_MODE_LEGACY:
            return False
        if self._turn_mode == TURN_MODE_STEP_GATE:
            return False
        return (
            self._dialogue_turn_gate_enabled
            or bool(self._dialogue_gate_supported_clients.intersection(self.clients))
        )

    def supports_player_v2_runtime(self) -> bool:
        """Whether this runtime is currently allowed to use player-v2 features."""
        if self._frontend_mode == FRONTEND_MODE_LEGACY:
            return False
        if self._frontend_mode == FRONTEND_MODE_PLAYER_V2:
            return True
        return (
            self._dialogue_turn_gate_enabled
            or bool(self._dialogue_gate_supported_clients.intersection(self.clients))
            or FRONTEND_MODE_PLAYER_V2 in set(self._client_frontend_modes.values())
        )

    def _next_dialogue_gate_id(self, case_id: str, turn: int) -> str:
        self._dialogue_gate_counter += 1
        return f"{case_id}:{turn}:{self._dialogue_gate_counter}"

    def _case_id_from_gate(self, gate_id: str) -> str:
        return gate_id.split(":", 1)[0] if gate_id else ""

    async def _send_active_dialogue_gate_to_client(self, ws: WebSocket) -> None:
        payload = self._active_dialogue_gate_payload
        if not payload:
            return
        case_id = str(payload.get("case_id") or "")
        gate_id = str(payload.get("gate_id") or "")
        speaker_name = str(payload.get("speaker_name") or "")
        turn = int(payload.get("turn") or 0)
        await self._send_to_client(
            ws,
            proto.runtime_progress(
                case_id,
                phase="next_ready",
                message="下一句已准备好",
                detail=speaker_name,
                blocking=True,
            ),
        )
        await self._send_to_client(ws, proto.dialogue_gate_waiting(case_id, gate_id, speaker_name, turn))

    async def _buffer_dialogue_until_continue(
        self,
        case_id: str,
        speaker_name: str,
        turn: int,
        message: dict[str, Any],
    ) -> None:
        gate_id = self._next_dialogue_gate_id(case_id, turn)
        gate_event = asyncio.Event()
        self._dialogue_gate_events[gate_id] = gate_event
        self._active_dialogue_gate_id = gate_id
        self._buffered_dialogue_message = message
        self._active_dialogue_gate_payload = {
            "case_id": case_id,
            "gate_id": gate_id,
            "speaker_name": speaker_name,
            "turn": turn,
        }
        try:
            await self.broadcast_runtime_progress(
                case_id,
                phase="next_ready",
                message="下一句已准备好",
                detail=speaker_name,
                blocking=True,
            )
            await self.broadcast(proto.dialogue_gate_waiting(case_id, gate_id, speaker_name, turn))
            auto_remaining = (
                self._dialogue_gate_auto_timeout_seconds()
                if self._turn_mode == TURN_MODE_AUTO
                else None
            )
            while True:
                if self._paused:
                    await self._resumed_event.wait()
                    continue
                wait_timeout = 1.0
                if auto_remaining is not None:
                    if auto_remaining <= 0:
                        logger.info(
                            "[DialogueGate] auto-continue timeout reached: case=%s gate=%s speaker=%s",
                            case_id,
                            gate_id,
                            speaker_name,
                        )
                        break
                    wait_timeout = min(wait_timeout, auto_remaining)
                try:
                    await asyncio.wait_for(gate_event.wait(), timeout=wait_timeout)
                    break
                except asyncio.TimeoutError:
                    if auto_remaining is not None and not self._paused:
                        auto_remaining -= wait_timeout
                    continue
            buffered = self._buffered_dialogue_message
            if buffered:
                await self.broadcast(buffered)
                await self.broadcast_runtime_progress(
                    case_id,
                    phase="generating_next",
                    message="正在准备下一句",
                    detail=speaker_name,
                    blocking=False,
                )
            return
        finally:
            self._dialogue_gate_events.pop(gate_id, None)
            if self._active_dialogue_gate_id == gate_id:
                self._active_dialogue_gate_id = None
                self._active_dialogue_gate_payload = None
                self._buffered_dialogue_message = None

    @staticmethod
    def _dialogue_gate_auto_timeout_seconds() -> float:
        try:
            return max(
                0.5,
                float(os.getenv("DIALOGUE_GATE_AUTO_TIMEOUT_SECONDS", DIALOGUE_GATE_AUTO_TIMEOUT_SECONDS)),
            )
        except (TypeError, ValueError):
            return DIALOGUE_GATE_AUTO_TIMEOUT_SECONDS

    def _estimate_move_duration(self, agent_id: str, dest_x: float, dest_y: float) -> float:
        """Estimate movement duration using backend-maintained coordinates."""
        state = self._agent_states.get(agent_id, {})
        start_x = float(state.get("x", dest_x))
        start_y = float(state.get("y", dest_y))
        if self._pathfinder:
            estimated = self._pathfinder.estimate_travel_duration(
                start_x,
                start_y,
                dest_x,
                dest_y,
                self._move_speed_px_per_second,
            )
            if estimated is not None:
                return max(estimated + self._move_duration_padding, self._min_move_duration)
        manhattan_distance = abs(dest_x - start_x) + abs(dest_y - start_y)
        if manhattan_distance <= 1.0:
            return 0.0
        duration = manhattan_distance / self._move_speed_px_per_second
        return max(duration + self._move_duration_padding, self._min_move_duration)

    async def _wait_for_frontend_ack(self, key: str, timeout: float) -> bool:
        """Wait for a frontend ack while respecting pause/resume state."""
        self._ack_events[key] = asyncio.Event()
        try:
            return await self._wait_for_existing_frontend_ack(key, timeout)
        finally:
            self._ack_events.pop(key, None)
        return False

    async def _wait_for_existing_frontend_ack(self, key: str, timeout: float) -> bool:
        """Wait for an ack event that has already been registered."""
        event = self._ack_events.get(key)
        if event is None:
            return False

        elapsed = 0.0
        while elapsed < timeout:
            if self._paused:
                await self._resumed_event.wait()
            try:
                await asyncio.wait_for(event.wait(), timeout=1.0)
                return True
            except asyncio.TimeoutError:
                if not self._paused:
                    elapsed += 1.0
        return False

    async def _move_to_location_with_frontend_ack(
        self,
        agent_id: str,
        dest_loc_id: str,
        loc,
    ) -> bool:
        estimated_duration = self._estimate_move_duration(agent_id, loc.x, loc.y)

        if self._fallback_mode:
            delay = estimated_duration * self._fallback_speed
            if delay > 0:
                logger.info("[Mock] %s → %s (%.2fs)", agent_id, dest_loc_id, delay)
                await self._sleep_with_pause(delay)
            if agent_id in self._agent_states:
                self._agent_states[agent_id]["x"] = loc.x
                self._agent_states[agent_id]["y"] = loc.y
                self._agent_states[agent_id]["sitting"] = None
            return True

        ack_key = f"{agent_id}:move_complete"
        ack_wait_task: asyncio.Task[bool] | None = None

        try:
            msg = proto.agent_move(agent_id, loc.x, loc.y, dest_loc_id)
            await self.broadcast(msg)

            if not self._backend_authoritative:
                self._ack_events[ack_key] = asyncio.Event()
                ack_wait_task = asyncio.create_task(
                    self._wait_for_existing_frontend_ack(ack_key, timeout=60.0)
                )
                logger.info(
                    "[Move] %s → %s (%.2f, %.2f), waiting for frontend completion...",
                    agent_id,
                    dest_loc_id,
                    loc.x,
                    loc.y,
                )
                acked = await ack_wait_task
                if not acked:
                    logger.warning(
                        "Move timeout: %s → %s (%.2f, %.2f) after %.1fs",
                        agent_id,
                        dest_loc_id,
                        loc.x,
                        loc.y,
                        60.0,
                    )
                    if agent_id in self._agent_states:
                        current_pos = self._agent_states[agent_id]
                        logger.warning(
                            "  Current position: (%.2f, %.2f)",
                            current_pos.get("x", 0.0),
                            current_pos.get("y", 0.0),
                        )
                    return False

                if agent_id in self._agent_states:
                    self._agent_states[agent_id]["x"] = loc.x
                    self._agent_states[agent_id]["y"] = loc.y
                    self._agent_states[agent_id]["sitting"] = None
                    self._save_agent_state(agent_id)
                return True

            logger.info(
                "[Move] %s → %s (%.2f, %.2f), backend authoritative duration=%.2fs",
                agent_id,
                dest_loc_id,
                loc.x,
                loc.y,
                estimated_duration,
            )

            if agent_id in self._agent_states:
                self._agent_states[agent_id]["x"] = loc.x
                self._agent_states[agent_id]["y"] = loc.y
                self._agent_states[agent_id]["sitting"] = None
                self._save_agent_state(agent_id)

            if estimated_duration > 0:
                await self._sleep_with_pause(estimated_duration)

            return True
        finally:
            if ack_wait_task is not None and not ack_wait_task.done():
                ack_wait_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await ack_wait_task
            if ack_wait_task is not None:
                self._ack_events.pop(ack_key, None)

    # ── 广播 ──

    async def broadcast(self, message: dict[str, Any]) -> None:
        """向所有连接的前端广播消息。"""
        if not self.clients:
            return
        payload = json.dumps(message, ensure_ascii=False)
        disconnected = []
        for ws in self.clients:
            try:
                await ws.send_text(payload)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self.clients.discard(ws)

    async def broadcast_case_runtime_issue(
        self,
        case_id: str,
        scenario_type: str,
        stage_label: str,
        code: str,
        message: str,
        retryable: bool,
        occurred_at: str,
    ) -> None:
        msg = proto.case_runtime_issue(
            case_id,
            scenario_type,
            stage_label,
            code,
            message,
            retryable,
            occurred_at,
        )
        await self.broadcast(msg)

    async def broadcast_runtime_progress(
        self,
        case_id: str,
        phase: str,
        message: str,
        detail: str = "",
        blocking: bool = False,
        metadata: dict[str, object] | None = None,
    ) -> None:
        msg = proto.runtime_progress(case_id, phase, message, detail=detail, blocking=blocking, metadata=metadata)
        await self.broadcast(msg)

    # ── 前台交互消息 ──

    async def send_goto_front_desk(
        self,
        agent_id: str,
        lawfirm: str,
        dialogue_text: str,
    ) -> None:
        """发送 Agent 移动到前台并显示对话的消息。"""
        await self._wait_until_resumed()
        msg = proto.agent_goto_front_desk(agent_id, lawfirm, dialogue_text)
        await self.broadcast(msg)
        logger.info(f"[WS] Agent {agent_id} 前往前台 {lawfirm}")

    async def send_update_dialogue(
        self,
        agent_id: str,
        dialogue_text: str,
        duration: float | None = None,
    ) -> None:
        """发送更新 Agent 对话内容的消息。"""
        await self._wait_until_resumed()
        msg = proto.agent_update_dialogue(agent_id, dialogue_text, duration)
        await self.broadcast(msg)
        logger.info(f"[WS] 更新 Agent {agent_id} 对话")

    async def send_end_interaction(self, agent_id: str) -> None:
        """发送结束 Agent 交互的消息。"""
        await self._wait_until_resumed()
        msg = proto.agent_end_interaction(agent_id)
        await self.broadcast(msg)
        logger.info(f"[WS] Agent {agent_id} 结束交互")

    # ── TownAvatarInterface 实现 ──

    async def move_to_location(self, agent_id: str, dest_loc_id: str) -> bool:
        loc = self.loc_registry.get(dest_loc_id)
        if not loc:
            logger.warning("Unknown location: %s", dest_loc_id)
            return False
        return await self._move_to_location_with_frontend_ack(agent_id, dest_loc_id, loc)

    async def play_animation(
        self, agent_id: str, animation_name: str, duration: float
    ) -> None:
        if self._fallback_mode:
            delay = duration * self._fallback_speed
            if delay > 0:
                logger.info("[Mock] %s anim '%s' (%.1fs)", agent_id, animation_name, delay)
                await self._sleep_with_pause(delay)
            return

        msg = {
            "type": "agent_animate",
            "agent_id": agent_id,
            "animation": animation_name,
            "duration": duration,
        }
        await self.broadcast(msg)
        if not self._backend_authoritative:
            await self._wait_for_frontend_ack(
                f"{agent_id}:animation_complete",
                timeout=max(duration, 0.0) + 10.0,
            )
            return

        if duration > 0:
            logger.info(
                "[Anim] %s plays '%s', backend authoritative duration=%.2fs",
                agent_id,
                animation_name,
                duration,
            )
            await self._sleep_with_pause(duration)

    async def show_bubble(
        self, agent_id: str, text: str, duration: float
    ) -> None:
        if self._fallback_mode:
            logger.info("[Mock] %s bubble: %s", agent_id, text[:30])
            await self._sleep_with_pause(duration * self._fallback_speed)
            return

        await self.send_update_dialogue(agent_id, text, duration)
        await self._sleep_with_pause(duration)

    # ── 扩展方法（超出 TownAvatarInterface） ──

    async def spawn_agent(
        self,
        agent_id: str,
        name: str,
        character_name: str,
        birth_loc_id: str = "birth_locationA",
        role: str = "",
    ) -> None:
        """在出生点生成 Agent 精灵。"""
        loc = self.loc_registry.birth_locations.get(birth_loc_id)
        if not loc:
            logger.warning("Unknown birth location: %s", birth_loc_id)
            return

        # 记录 Agent 状态（无论是否有前端连接）
        self._agent_states[agent_id] = {
            "name": name,
            "character_name": character_name,
            "x": loc.x,
            "y": loc.y,
            "role": role,
            "birth_loc_id": birth_loc_id,
            "sitting": None,
            "standing_direction": "",
        }

        if self._fallback_mode:
            logger.info("[Mock] Spawn %s at %s", agent_id, birth_loc_id)
            return

        msg = proto.agent_spawn(agent_id, name, character_name, loc.x, loc.y, role)
        await self.broadcast(msg)
        await asyncio.sleep(0.3)  # 短暂等待前端创建精灵
        self._save_agent_state(agent_id)

    async def sit_agent(
        self, agent_id: str, loc_id: str, direction_override: str | None = None
    ) -> None:
        """让 Agent 坐到指定位置。"""
        loc = self.loc_registry.get(loc_id)
        if not loc:
            logger.warning("Unknown sit location: %s", loc_id)
            return

        direction = direction_override or loc.direction or "down"

        # 更新状态
        if agent_id in self._agent_states:
            self._agent_states[agent_id]["sitting"] = {
                "x": loc.x,
                "y": loc.y,
                "direction": direction,
            }
            self._agent_states[agent_id]["x"] = loc.x
            self._agent_states[agent_id]["y"] = loc.y
            self._agent_states[agent_id]["standing_direction"] = ""

        if self._fallback_mode:
            logger.info("[Mock] %s sit at %s", agent_id, loc_id)
            return

        msg = proto.agent_sit(agent_id, loc.x, loc.y, direction)
        await self.broadcast(msg)
        self._save_agent_state(agent_id)

    async def stand_agent(
        self,
        agent_id: str,
        direction_override: str | None = None,
        x: float | None = None,
        y: float | None = None,
    ) -> None:
        """让 Agent 站起来。"""
        # 更新状态
        if agent_id in self._agent_states:
            self._agent_states[agent_id]["sitting"] = None
            if x is not None:
                self._agent_states[agent_id]["x"] = x
            if y is not None:
                self._agent_states[agent_id]["y"] = y
            if direction_override:
                self._agent_states[agent_id]["standing_direction"] = direction_override

        if self._fallback_mode:
            logger.info("[Mock] %s stand", agent_id)
            return

        msg = proto.agent_stand(agent_id, direction_override or "", x=x, y=y)
        await self.broadcast(msg)
        self._save_agent_state(agent_id)

    async def despawn_agent(self, agent_id: str) -> None:
        """移除 Agent 精灵。"""
        # 移除状态追踪
        self._agent_states.pop(agent_id, None)

        if self._fallback_mode:
            logger.info("[Mock] Despawn %s", agent_id)
            return

        msg = proto.agent_despawn(agent_id)
        await self.broadcast(msg)
        self._save_agent_state(agent_id)

    async def broadcast_dialogue(
        self,
        case_id: str,
        speaker_id: str,
        speaker_name: str,
        content: str,
        turn: int,
        scenario_type: str = "",
        generation_duration_seconds: float | None = None,
        generation_total_tokens: int | None = None,
        player_responsibility: bool | None = None,
        evaluation_marker_label: str = "",
        evaluation_marker_reason: str = "",
    ) -> None:
        """广播对话内容更新。"""
        await self._wait_until_resumed()
        msg = proto.dialogue_update(
            case_id,
            speaker_id,
            speaker_name,
            content,
            turn,
            scenario_type=scenario_type,
            generation_duration_seconds=generation_duration_seconds,
            generation_total_tokens=generation_total_tokens,
            player_responsibility=player_responsibility,
            evaluation_marker_label=evaluation_marker_label,
            evaluation_marker_reason=evaluation_marker_reason,
        )
        if self._should_gate_dialogue():
            if not self._dialogue_lookahead_started:
                self._dialogue_lookahead_started = True
                await self.broadcast(msg)
                await self.broadcast_runtime_progress(
                    case_id,
                    phase="generating_next",
                    message="正在准备下一句",
                    detail=speaker_name,
                    blocking=False,
                )
                return
            await self._buffer_dialogue_until_continue(case_id, speaker_name, turn, msg)
            return

        await self.broadcast(msg)
        await self._wait_for_dialogue_continue(case_id, speaker_name, turn)

    async def broadcast_state_change(
        self,
        case_id: str,
        event: str,
        from_state: str = "",
        to_state: str = "",
        party_role: str = "",
        overall_state: str = "",
    ) -> None:
        """广播案件状态变更。"""
        await self._wait_until_resumed()
        msg = proto.case_state_change(
            case_id,
            event,
            from_state,
            to_state,
            party_role=party_role,
            overall_state=overall_state,
        )
        await self.broadcast(msg)

    async def broadcast_scenario_start(
        self,
        case_id: str,
        scenario_type: str,
        participants: list[str],
    ) -> None:
        await self._wait_until_resumed()
        self._dialogue_lookahead_started = False
        await self.broadcast_runtime_progress(
            case_id,
            phase="scenario_start",
            message=f"当前阶段：{scenario_type}",
            detail=", ".join(participants),
            blocking=False,
        )
        msg = proto.scenario_start(case_id, scenario_type, participants)
        await self.broadcast(msg)

    async def broadcast_scenario_end(
        self,
        case_id: str,
        scenario_type: str,
    ) -> None:
        await self._wait_until_resumed()
        await self.broadcast_runtime_progress(
            case_id,
            phase="scenario_end",
            message=f"{scenario_type} 阶段已结束",
            blocking=False,
        )
        msg = proto.scenario_end(case_id, scenario_type)
        await self.broadcast(msg)

    def _save_agent_state(self, agent_id: str) -> None:
        """将单个 Agent 的地图状态保存到其 config.yaml 中。"""
        if not self.agent_registry or not self.storage:
            return
            
        agent = self.agent_registry.get_agent(agent_id)
        if not agent or not getattr(agent, "config_path", None):
            return
            
        state = self._agent_states.get(agent_id)
        # 如果 state 为 None (比如刚 despawn)，我们可以清理掉 map_state 字段
        self.storage.update_agent_field(agent.config_path, "map_state", state)

    def restore_state_from_configs(self) -> None:
        """从所有 Agent 的 config.yaml 恢复前端状态。"""
        if not self.agent_registry or not self.storage:
            return
            
        restored_count = 0
        for agent in self.agent_registry.get_all_agents():
            if not getattr(agent, "config_path", None):
                continue
                
            try:
                config = self.storage.load_agent_config(agent.config_path)
                map_state = config.get("map_state")
                if isinstance(map_state, dict) and map_state:
                    normalized_state = dict(map_state)
                    agent_name = str(getattr(agent, "name", "") or "").strip()
                    if agent_name and str(normalized_state.get("name", "") or "").strip() in {"", agent.agent_id}:
                        normalized_state["name"] = agent_name
                    self._agent_states[agent.agent_id] = normalized_state
                    if normalized_state != map_state:
                        self.storage.update_agent_field(agent.config_path, "map_state", normalized_state)
                    restored_count += 1
            except Exception as e:
                logger.warning(f"Failed to restore state for {agent.agent_id}: {e}")
                
        logger.info(f"Restored {restored_count} agents from config files")
