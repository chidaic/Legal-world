"""WebSocket message protocol helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


MSG_AGENT_SPAWN = "agent_spawn"
MSG_AGENT_MOVE = "agent_move"
MSG_AGENT_SIT = "agent_sit"
MSG_AGENT_STAND = "agent_stand"
MSG_AGENT_BUBBLE = "agent_bubble"
MSG_AGENT_DESPAWN = "agent_despawn"
MSG_DIALOGUE_UPDATE = "dialogue_update"
MSG_DIALOGUE_GATE_WAITING = "dialogue_gate_waiting"
MSG_DIALOGUE_GATE_ACCEPTED = "dialogue_gate_accepted"
MSG_DIALOGUE_GATE_ERROR = "dialogue_gate_error"
MSG_RUNTIME_PROGRESS = "runtime_progress"
MSG_STEP_GATE_WAITING = "step_gate_waiting"
MSG_STEP_GATE_ACCEPTED = "step_gate_accepted"
MSG_STEP_GATE_ERROR = "step_gate_error"
MSG_CASE_STATE_CHANGE = "case_state_change"
MSG_SCENARIO_START = "scenario_start"
MSG_SCENARIO_END = "scenario_end"
MSG_CASE_RUNTIME_ISSUE = "case_runtime_issue"
MSG_AGENT_GOTO_FRONT_DESK = "agent_goto_front_desk"
MSG_AGENT_UPDATE_DIALOGUE = "agent_update_dialogue"
MSG_AGENT_END_INTERACTION = "agent_end_interaction"

MSG_CLIENT_READY = "client_ready"
MSG_DIALOGUE_CONTINUE = "dialogue_continue"
MSG_MOVE_COMPLETE = "move_complete"
MSG_ANIMATION_COMPLETE = "animation_complete"
MSG_SIMULATION_PAUSE = "simulation_pause"
MSG_SIMULATION_RESUME = "simulation_resume"


def agent_spawn(
    agent_id: str,
    name: str,
    character_name: str,
    x: float,
    y: float,
    role: str = "",
) -> dict[str, Any]:
    return {
        "type": MSG_AGENT_SPAWN,
        "agent_id": agent_id,
        "name": name,
        "character_name": character_name,
        "x": x,
        "y": y,
        "role": role,
    }


def agent_move(
    agent_id: str,
    dest_x: float,
    dest_y: float,
    dest_loc_id: str = "",
) -> dict[str, Any]:
    return {
        "type": MSG_AGENT_MOVE,
        "agent_id": agent_id,
        "dest_x": dest_x,
        "dest_y": dest_y,
        "dest_loc_id": dest_loc_id,
    }


def agent_sit(
    agent_id: str,
    x: float,
    y: float,
    direction: str = "down",
) -> dict[str, Any]:
    return {
        "type": MSG_AGENT_SIT,
        "agent_id": agent_id,
        "x": x,
        "y": y,
        "direction": direction,
    }


def agent_stand(
    agent_id: str,
    direction: str = "",
    x: float | None = None,
    y: float | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": MSG_AGENT_STAND,
        "agent_id": agent_id,
    }
    if direction:
        payload["direction"] = direction
    if x is not None:
        payload["x"] = x
    if y is not None:
        payload["y"] = y
    return payload


def agent_bubble(
    agent_id: str,
    text: str,
    duration: float = 5.0,
) -> dict[str, Any]:
    return {
        "type": MSG_AGENT_BUBBLE,
        "agent_id": agent_id,
        "text": text,
        "duration": duration,
    }


def agent_despawn(agent_id: str) -> dict[str, Any]:
    return {"type": MSG_AGENT_DESPAWN, "agent_id": agent_id}


def dialogue_update(
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
) -> dict[str, Any]:
    payload = {
        "type": MSG_DIALOGUE_UPDATE,
        "case_id": case_id,
        "speaker_id": speaker_id,
        "speaker_name": speaker_name,
        "content": content,
        "turn": turn,
    }
    if scenario_type:
        payload["scenario_type"] = scenario_type
    try:
        duration = float(generation_duration_seconds) if generation_duration_seconds is not None else 0.0
    except (TypeError, ValueError):
        duration = 0.0
    if duration > 0:
        payload["generation_duration_seconds"] = round(duration, 4)
    try:
        total_tokens = int(generation_total_tokens) if generation_total_tokens is not None else 0
    except (TypeError, ValueError):
        total_tokens = 0
    if total_tokens > 0:
        payload["generation_total_tokens"] = total_tokens
    if player_responsibility:
        payload["player_responsibility"] = True
        label = str(evaluation_marker_label or "").strip()
        reason = str(evaluation_marker_reason or "").strip()
        if label:
            payload["evaluation_marker_label"] = label
        if reason:
            payload["evaluation_marker_reason"] = reason
    return payload


def dialogue_gate_waiting(
    case_id: str,
    gate_id: str,
    speaker_name: str,
    turn: int,
) -> dict[str, Any]:
    return {
        "type": MSG_DIALOGUE_GATE_WAITING,
        "case_id": case_id,
        "gate_id": gate_id,
        "speaker_name": speaker_name,
        "turn": turn,
    }


def dialogue_gate_accepted(case_id: str, gate_id: str) -> dict[str, Any]:
    return {
        "type": MSG_DIALOGUE_GATE_ACCEPTED,
        "case_id": case_id,
        "gate_id": gate_id,
    }


def dialogue_gate_error(
    case_id: str,
    gate_id: str,
    message: str,
    code: str = "DIALOGUE_GATE_ERROR",
) -> dict[str, Any]:
    return {
        "type": MSG_DIALOGUE_GATE_ERROR,
        "case_id": case_id,
        "gate_id": gate_id,
        "code": code,
        "message": message,
        "occurred_at": datetime.now(timezone.utc).isoformat(),
    }


def runtime_progress(
    case_id: str,
    phase: str,
    message: str,
    detail: str = "",
    blocking: bool = False,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "type": MSG_RUNTIME_PROGRESS,
        "case_id": case_id,
        "phase": phase,
        "message": message,
        "detail": detail,
        "blocking": blocking,
        "occurred_at": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        payload.update(metadata)
    return payload


def dialogue_gate_waiting(
    case_id: str,
    gate_id: str,
    speaker_name: str,
    turn: int,
) -> dict[str, Any]:
    return {
        "type": MSG_DIALOGUE_GATE_WAITING,
        "case_id": case_id,
        "gate_id": gate_id,
        "speaker_name": speaker_name,
        "turn": turn,
    }


def dialogue_gate_accepted(case_id: str, gate_id: str) -> dict[str, Any]:
    return {
        "type": MSG_DIALOGUE_GATE_ACCEPTED,
        "case_id": case_id,
        "gate_id": gate_id,
    }


def dialogue_gate_error(
    case_id: str,
    gate_id: str,
    message: str,
    code: str = "DIALOGUE_GATE_ERROR",
) -> dict[str, Any]:
    return {
        "type": MSG_DIALOGUE_GATE_ERROR,
        "case_id": case_id,
        "gate_id": gate_id,
        "code": code,
        "message": message,
        "occurred_at": datetime.now(timezone.utc).isoformat(),
    }


def case_state_change(
    case_id: str,
    event: str,
    from_state: str = "",
    to_state: str = "",
    party_role: str = "",
    overall_state: str = "",
) -> dict[str, Any]:
    payload = {
        "type": MSG_CASE_STATE_CHANGE,
        "case_id": case_id,
        "event": event,
        "from_state": from_state,
        "to_state": to_state,
    }
    if party_role:
        payload["party_role"] = party_role
    if overall_state:
        payload["overall_state"] = overall_state
    return payload


def scenario_start(
    case_id: str,
    scenario_type: str,
    participants: list[str],
) -> dict[str, Any]:
    return {
        "type": MSG_SCENARIO_START,
        "case_id": case_id,
        "scenario_type": scenario_type,
        "participants": participants,
    }


def scenario_end(case_id: str, scenario_type: str) -> dict[str, Any]:
    return {
        "type": MSG_SCENARIO_END,
        "case_id": case_id,
        "scenario_type": scenario_type,
    }


def case_runtime_issue(
    case_id: str,
    scenario_type: str,
    stage_label: str,
    code: str,
    message: str,
    retryable: bool,
    occurred_at: str,
) -> dict[str, Any]:
    return {
        "type": MSG_CASE_RUNTIME_ISSUE,
        "case_id": case_id,
        "scenario_type": scenario_type,
        "stage_label": stage_label,
        "code": code,
        "message": message,
        "retryable": retryable,
        "occurred_at": occurred_at,
    }


def agent_goto_front_desk(
    agent_id: str,
    lawfirm: str,
    dialogue_text: str,
) -> dict[str, Any]:
    return {
        "type": MSG_AGENT_GOTO_FRONT_DESK,
        "agent_id": agent_id,
        "lawfirm": lawfirm,
        "dialogue_text": dialogue_text,
    }


def agent_update_dialogue(
    agent_id: str,
    dialogue_text: str,
    duration: float | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": MSG_AGENT_UPDATE_DIALOGUE,
        "agent_id": agent_id,
        "dialogue_text": dialogue_text,
    }
    if duration is not None:
        payload["duration"] = duration
    return payload


def agent_end_interaction(agent_id: str) -> dict[str, Any]:
    return {
        "type": MSG_AGENT_END_INTERACTION,
        "agent_id": agent_id,
    }
