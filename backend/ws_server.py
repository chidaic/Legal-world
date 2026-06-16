"""法律AI小镇 WebSocket 服务器 (ws_server.py)。

替代 sandbox_main.py 作为主入口，通过 WebSocket 实时驱动前端渲染。
启动方式: python ws_server.py
"""

import asyncio
import contextlib
from collections import deque
from datetime import datetime, timezone
import json
import logging
import os
import random
import re
import shutil
import sys
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable
import yaml

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from pydantic import BaseModel

_backend_dir = Path(__file__).parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

ROOT_ENV_PATH = _backend_dir.parent / ".env"


def _should_override_dotenv() -> bool:
    return str(os.getenv("SIMLAW_DOTENV_OVERRIDE", "") or "").strip().lower() in {"1", "true", "yes", "on"}


load_dotenv(ROOT_ENV_PATH, override=_should_override_dotenv())

from src.core.event_bus import EventBus, EventType
from src.core.file_storage_manager import FileStorageManager
from src.core.checkpoint_manager import CheckpointManager
from src.core.auth import AuthError, create_access_token, decode_access_token, get_access_token_expires_at
from src.core.database import Base, create_database_engine, create_session_factory, get_db_session
from src.core.models import Sandbox, User
from src.core.sandbox_manager import SandboxManager, SandboxRuntimeContext
from src.core.sandbox_service import SandboxService
from src.core.user_service import (
    InvalidCredentialsError,
    InvalidAuthInputError,
    UserAlreadyExistsError,
    UserNotFoundError,
    authenticate_user,
    get_user_by_id,
    register_user,
)
from src.human_eval.routes import create_human_eval_router
from src.orchestration.case_fsm import CaseStateMachine
from src.orchestration.agent_registry import AgentRegistry
from src.orchestration.scenario_orchestrator import ScenarioOrchestrator
from src.pipeline.stage_tool_resolver import infer_stage_role_name, resolve_agent_type, resolve_configured_tool_names
from src.runtime_tech_catalog import build_runtime_tech_catalog
from src.simulation.location_registry import load_registry_from_map
from src.simulation.ws_frontend_engine import WebSocketFrontendEngine
from src.tools.common.skill_loader_tool import _FlatSkillToolkit
from src.utils.case_progress import infer_case_state_from_artifacts, normalize_case_state
from src.utils.runtime_flags import player_lawyer_mode_for_frontend, scenario_verbose_enabled
from src.utils.memory_initializer import initialize_client_memory, initialize_lawyer_memory
from src.version import BACKEND_VERSION, BACKEND_VERSION_LABEL, BACKEND_VERSION_TIME

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)

# 调整 CAMEL 库的日志级别，避免打印详细的 Messages 和 Response
logging.getLogger("camel.base_model").setLevel(logging.WARNING)
logging.getLogger("camel.camel.agents.chat_agent").setLevel(logging.WARNING)

logger = logging.getLogger("ws_server")
SCENARIO_VERBOSE = scenario_verbose_enabled()
_RUNTIME_ISSUE_LIMIT = 80
_runtime_issues: deque[dict[str, str]] = deque(maxlen=_RUNTIME_ISSUE_LIMIT)
_AGENT_DISCOVERY_ROOT_DIRS = ("cases", "law_firms", "court_system")
SIMLAW_FRONTEND_MODE = str(os.getenv("SIMLAW_FRONTEND_MODE", "auto") or "auto").strip().lower().replace("-", "_")
SIMLAW_TURN_MODE = str(os.getenv("SIMLAW_TURN_MODE", "auto") or "auto").strip().lower().replace("-", "_")


def _sanitize_log_message(message: str) -> str:
    sanitized = re.sub(r"(token=)[^&\\s]+", r"\\1***", str(message))
    sanitized = re.sub(r"(Bearer\\s+)[A-Za-z0-9._-]+", r"\\1***", sanitized)
    return sanitized


def _is_user_scoped_sandbox_root(storage_root: Path) -> bool:
    root = Path(storage_root)
    has_users_dir = (root / "users").is_dir()
    has_agent_roots = any((root / dirname).exists() for dirname in _AGENT_DISCOVERY_ROOT_DIRS)
    return has_users_dir and not has_agent_roots


class _RuntimeIssueHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno < logging.WARNING:
            return
        try:
            message = _sanitize_log_message(record.getMessage())
        except Exception:
            message = "日志解析失败"
        _runtime_issues.appendleft(
            {
                "level": record.levelname,
                "logger": record.name,
                "message": message,
                "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            }
        )


def _install_runtime_issue_handler() -> None:
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        if getattr(handler, "_simlaw_runtime_issue_handler", False):
            root_logger.removeHandler(handler)
    handler = _RuntimeIssueHandler(level=logging.WARNING)
    handler._simlaw_runtime_issue_handler = True
    root_logger.addHandler(handler)


_install_runtime_issue_handler()


_RUNTIME_STAGE_LABELS = {
    "RECEPTION": "前台导引",
    "PLC": "原告咨询",
    "LC": "法律咨询",
    "DLC": "被告咨询",
    "CD": "起诉状起草",
    "DD": "答辩状起草",
    "TIA": "一审庭前信息分析",
    "CI": "一审庭审",
    "AD": "上诉状起草",
    "AR": "上诉答辩状起草",
    "TIAA": "二审庭前信息分析",
    "CIA": "二审庭审",
}
_MODEL_UNAVAILABLE_HINTS = (
    "503",
    "model_not_found",
    "no available channel for model",
    "service unavailable",
    "service temporarily unavailable",
)

_CASE_DOCUMENT_SPECS: tuple[dict[str, str], ...] = (
    {
        "document_key": "CD",
        "stage": "CD",
        "document_type": "complaint",
        "title": "民事起诉状",
        "result_filename": "CD_result.json",
        "pdf_filename": "CD_document.pdf",
    },
    {
        "document_key": "DD",
        "stage": "DD",
        "document_type": "defense",
        "title": "民事答辩状",
        "result_filename": "DD_result.json",
        "pdf_filename": "DD_document.pdf",
    },
    {
        "document_key": "AD",
        "stage": "AD",
        "document_type": "appeal",
        "title": "民事上诉状",
        "result_filename": "AD_result.json",
        "pdf_filename": "AD_document.pdf",
    },
    {
        "document_key": "AR",
        "stage": "AR",
        "document_type": "appeal_response",
        "title": "民事上诉答辩状",
        "result_filename": "AR_result.json",
        "pdf_filename": "AR_document.pdf",
    },
    {
        "document_key": "CI",
        "stage": "CI",
        "document_type": "first_instance_judgment",
        "title": "一审民事判决书",
        "result_filename": "CI_result.json",
        "pdf_filename": "CI_document.pdf",
    },
    {
        "document_key": "CIA",
        "stage": "CIA",
        "document_type": "second_instance_judgment",
        "title": "二审民事判决书",
        "result_filename": "CIA_result.json",
        "pdf_filename": "CIA_document.pdf",
    },
)
_CASE_DOCUMENT_SPEC_BY_KEY = {
    spec["document_key"]: spec
    for spec in _CASE_DOCUMENT_SPECS
}


def _resolve_stage_label(scenario_type: str, fallback: str = "") -> str:
    normalized = str(scenario_type or "").strip().upper()
    return _RUNTIME_STAGE_LABELS.get(normalized) or fallback or normalized or "未知阶段"


def _build_runtime_issue_payload(
    *,
    case_id: str,
    scenario_type: str,
    code: str,
    message: str,
    retryable: bool,
    stage_label: str = "",
) -> dict[str, Any]:
    return {
        "scope": "sandbox",
        "case_id": str(case_id or "").strip(),
        "scenario_type": str(scenario_type or "").strip(),
        "stage_label": _resolve_stage_label(scenario_type, fallback=stage_label),
        "code": str(code or "").strip(),
        "message": str(message or "").strip(),
        "retryable": bool(retryable),
        "occurred_at": datetime.now(timezone.utc).isoformat(),
    }


def _normalize_runtime_issue_from_exception(
    *,
    case_id: str,
    scenario_type: str,
    exc: Exception,
    stage_label: str = "",
    event_type: str = "",
    handler_name: str = "",
) -> dict[str, Any] | None:
    raw_message = str(exc or "").strip()
    normalized = raw_message.lower()
    display_stage = _resolve_stage_label(scenario_type, fallback=stage_label)
    if not any(hint in normalized for hint in _MODEL_UNAVAILABLE_HINTS):
        detail = raw_message.rstrip("。.!? ") or f"{display_stage}发生未知运行异常"
        debug_parts = []
        exception_type = type(exc).__name__ if exc is not None else ""
        if exception_type:
            debug_parts.append(f"异常类型：{exception_type}")
        if event_type:
            debug_parts.append(f"事件：{event_type}")
        if handler_name:
            debug_parts.append(f"处理器：{handler_name}")
        if debug_parts:
            detail = f"{detail}（{'；'.join(debug_parts)}）"
        return _build_runtime_issue_payload(
            case_id=case_id,
            scenario_type=scenario_type,
            stage_label=display_stage,
            code="SCENARIO_RUNTIME_ERROR",
            message=f"{display_stage}运行失败：{detail}。已停止本轮模拟，请检查后端日志并修复后重新开始。",
            retryable=False,
        )

    return _build_runtime_issue_payload(
        case_id=case_id,
        scenario_type=scenario_type,
        stage_label=display_stage,
        code="MODEL_UNAVAILABLE",
        message=f"{display_stage}生成失败：当前模型不可用，已停止本轮模拟，请切换模型后重新开始。",
        retryable=True,
    )


def _reset_runtime_transient_state(context: SandboxRuntimeContext) -> None:
    _set_runtime_engine_paused(getattr(context, "engine", None), False)

    registry = getattr(context, "registry", None)
    storage = getattr(context, "storage_manager", None)
    event_bus = getattr(context, "event_bus", None)

    if registry is not None and storage is not None:
        for lawyer in registry.get_agents_by_type("lawyer"):
            if getattr(lawyer, "config_path", None):
                with contextlib.suppress(Exception):
                    storage.update_agent_field(lawyer.config_path, "current_handling_case", None)
                    storage.update_agent_field(lawyer.config_path, "case_queue", [])

        for judge in registry.get_agents_by_type("judge"):
            if getattr(judge, "config_path", None):
                with contextlib.suppress(Exception):
                    storage.update_agent_field(judge.config_path, "current_handling_case", None)

        for receptionist in registry.get_agents_by_type("receptionist"):
            front_desk_queue = getattr(receptionist, "_front_desk_queue", None)
            if hasattr(front_desk_queue, "clear"):
                front_desk_queue.clear()
            reserved_lawyers = getattr(receptionist, "_reserved_lawyers", None)
            if hasattr(reserved_lawyers, "clear"):
                reserved_lawyers.clear()
            queued_client_sofas = getattr(receptionist, "_queued_client_sofas", None)
            if hasattr(queued_client_sofas, "clear"):
                queued_client_sofas.clear()
            queued_client_wait_spots = getattr(receptionist, "_queued_client_wait_spots", None)
            if hasattr(queued_client_wait_spots, "clear"):
                queued_client_wait_spots.clear()
            setattr(receptionist, "_front_desk_busy", False)
            setattr(receptionist, "_last_assigned_lawyer_id", "")

    if event_bus is not None and hasattr(event_bus, "get_active_scenarios_snapshot"):
        active_scenarios = event_bus.get_active_scenarios_snapshot()
        for active_case_id in list(active_scenarios.keys()):
            with contextlib.suppress(Exception):
                event_bus.unregister_active_scenario(active_case_id)
        if context.checkpoint_mgr is not None and hasattr(context.checkpoint_mgr, "sync_active_scenarios_from_event_bus"):
            with contextlib.suppress(Exception):
                context.checkpoint_mgr.sync_active_scenarios_from_event_bus()

    orchestrator = getattr(context, "orchestrator", None)
    if orchestrator is not None:
        occupied_locations = getattr(orchestrator, "_occupied_locations", None)
        if hasattr(occupied_locations, "clear"):
            occupied_locations.clear()
        waiting_queues = getattr(orchestrator, "_waiting_queues", None)
        if hasattr(waiting_queues, "clear"):
            waiting_queues.clear()
        court_reservations = getattr(orchestrator, "_court_reservations", None)
        if hasattr(court_reservations, "clear"):
            court_reservations.clear()
        judge_reservations = getattr(orchestrator, "_judge_reservations", None)
        if hasattr(judge_reservations, "clear"):
            judge_reservations.clear()
        trial_queues = getattr(orchestrator, "_trial_queues", None)
        if isinstance(trial_queues, dict):
            for queue in trial_queues.values():
                if hasattr(queue, "clear"):
                    queue.clear()


async def _report_sandbox_runtime_issue(
    context: SandboxRuntimeContext,
    payload: dict[str, Any],
) -> bool:
    if not payload:
        return False

    context.last_error = dict(payload)
    _reset_runtime_transient_state(context)

    if context.checkpoint_mgr is not None and hasattr(context.checkpoint_mgr, "mark_session_paused"):
        with contextlib.suppress(Exception):
            context.checkpoint_mgr.mark_session_paused()

    runtime_engine = getattr(context, "engine", None)
    if runtime_engine is not None and hasattr(runtime_engine, "broadcast_case_runtime_issue"):
        await runtime_engine.broadcast_case_runtime_issue(
            case_id=payload.get("case_id", ""),
            scenario_type=payload.get("scenario_type", ""),
            stage_label=payload.get("stage_label", ""),
            code=payload.get("code", ""),
            message=payload.get("message", ""),
            retryable=bool(payload.get("retryable", False)),
            occurred_at=payload.get("occurred_at", ""),
        )

    task = context.simulation_task
    if task is not None and not task.done():
        task.cancel()

    return True


def _read_non_negative_int_env(name: str, default: int = 0) -> int:
    raw = str(os.getenv(name, "") or "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("Invalid %s=%r, fallback to %d", name, raw, default)
        return default

SANDBOX_DATA_DIR = _backend_dir / "sandbox_data"
SANDBOX_SEED_DIR = _backend_dir / "sandbox_seed_data"
CASE_PICKER_METADATA_PATH = SANDBOX_SEED_DIR / "case_picker_metadata.yaml"
DEBUG_UI_DIR = _backend_dir / "debug_ui"
RUNTIME_CONFIG_KEYS = (
    "SIMLAW_PROMPT_PROFILE",
    "OPENAI_API_KEY",
    "OPENAI_MODEL_NAME",
    "OPENAI_API_BASE_URL",
)
MAP_JSON_PATH = Path(__file__).parent.parent / "assets" / "map" / "new_ailaw_town.json"
CASE_SPAWN_INTERVAL_SECONDS = 15.0
MAX_CONCURRENT_CASES = _read_non_negative_int_env("MAX_CONCURRENT_CASES", 1)
CHARACTER_POOL = [
    "Adam",
    "Alex",
    "Amelia",
    "Ash",
    "Bob",
    "Bruce",
    "Conference_man",
    "Conference_woman",
    "Dan",
    "Edward",
    "Lucy",
    "Molly",
    "Pier",
    "Rob",
    "Roki",
    "Samuel",
]

app = FastAPI(title="SimLaw Town WebSocket Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/debug-assets", StaticFiles(directory=DEBUG_UI_DIR), name="debug-assets")

# ── Player-lawyer subsystem (feature-gated) ──
from src.player_lawyer.agent import is_player_plaintiff_mode as _is_player_plaintiff_mode
from src.player_lawyer.routes import (
    router as _player_lawyer_router,
    set_gateway_provider as _set_player_gw_provider,
    set_response_assist_provider as _set_player_response_assist_provider,
    set_status_provider as _set_player_status_provider,
)
from src.player_lawyer.input_gateway import PlayerInputGateway as _PlayerInputGateway

app.include_router(_player_lawyer_router)

# Per-sandbox gateway instances, keyed by sandbox_id
_player_gateways: dict[str, _PlayerInputGateway] = {}


def _player_lawyer_mode_for_engine(runtime_engine: Any | None) -> str:
    frontend_mode = getattr(runtime_engine, "_frontend_mode", None)
    supports_player_v2 = False
    supports_fn = getattr(runtime_engine, "supports_player_v2_runtime", None)
    if callable(supports_fn):
        supports_player_v2 = bool(supports_fn())
    return player_lawyer_mode_for_frontend(
        frontend_mode=frontend_mode,
        has_player_v2_client=supports_player_v2,
    )


def _player_lawyer_mode_for_context(context: Any | None) -> str:
    return _player_lawyer_mode_for_engine(getattr(context, "engine", None))


def _player_lawyer_status_for_request(request: Request) -> dict[str, Any]:
    if not _is_player_plaintiff_mode():
        return {"player_mode": "off", "enabled": False}

    token = _extract_bearer_token(request.headers.get("authorization"))
    try:
        with get_db_session(_get_session_factory()) as session:
            current_user = _get_user_from_access_token(token, session)
            sandbox = _require_user_sandbox(session, current_user)
    except (AuthError, UserNotFoundError):
        return {"player_mode": "off", "enabled": False}

    context = _get_sandbox_manager().get_or_create_context(sandbox)
    mode = _player_lawyer_mode_for_context(context)
    return {"player_mode": mode or "off", "enabled": mode == "plaintiff"}


def _get_player_gateway_for_request(request: Request) -> _PlayerInputGateway:
    """Resolve the player gateway for the authenticated user's sandbox."""
    token = _extract_bearer_token(request.headers.get("authorization"))
    try:
        with get_db_session(_get_session_factory()) as session:
            current_user = _get_user_from_access_token(token, session)
            sandbox = _require_user_sandbox(session, current_user)
    except (AuthError, UserNotFoundError) as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    context = _get_sandbox_manager().get_or_create_context(sandbox)
    if _player_lawyer_mode_for_context(context) != "plaintiff":
        raise HTTPException(status_code=403, detail="Player-lawyer mode is not enabled")

    gateway = getattr(context, "player_gateway", None)
    if gateway is None:
        gateway = get_or_create_player_gateway(sandbox.id, Path(sandbox.storage_root))
        setattr(context, "player_gateway", gateway)
    orchestrator = getattr(context, "orchestrator", None)
    if orchestrator is not None:
        setattr(orchestrator, "_player_gateway", gateway)
        setattr(orchestrator, "_sandbox_id", sandbox.id)
    return gateway


def _get_player_response_assist_for_request(request: Request):
    """Resolve the response assist service for the authenticated user's sandbox."""
    token = _extract_bearer_token(request.headers.get("authorization"))
    try:
        with get_db_session(_get_session_factory()) as session:
            current_user = _get_user_from_access_token(token, session)
            sandbox = _require_user_sandbox(session, current_user)
    except (AuthError, UserNotFoundError) as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    _ensure_player_lawyer_runtime(sandbox)
    from src.player_lawyer.response_assist import PlayerResponseAssistService

    return PlayerResponseAssistService(storage_root=Path(sandbox.storage_root))


async def _publish_player_document_completion_if_unmanaged(
    context: SandboxRuntimeContext,
    draft: Any,
) -> bool:
    """Publish the document completion event when no live scenario task can do it."""
    from src.core.event_bus import EventType

    completion_by_document_type = {
        "complaint": (EventType.COMPLAINT_DRAFTING_COMPLETED, "CD", "plaintiff"),
        "defense": (EventType.DEFENSE_DRAFTING_COMPLETED, "DD", "defendant"),
        "appeal": (EventType.APPEAL_DRAFTING_COMPLETED, "AD", "plaintiff"),
        "appeal_response": (EventType.APPEAL_RESPONSE_DRAFTING_COMPLETED, "AR", "defendant"),
    }
    event_info = completion_by_document_type.get(str(getattr(draft, "document_type", "") or "").strip())
    if not event_info:
        return False

    event_type, scenario_type, preferred_party_role = event_info
    case_id = _normalize_case_identifier(getattr(draft, "case_id", ""))
    if not case_id:
        return False

    event_bus = getattr(context, "event_bus", None)
    storage = getattr(context, "storage_manager", None)
    if event_bus is None or storage is None:
        return False

    active_snapshot = {}
    get_snapshot = getattr(event_bus, "get_active_scenarios_snapshot", None)
    if callable(get_snapshot):
        active_snapshot = get_snapshot() or {}
    active_scenario = active_snapshot.get(case_id) or active_snapshot.get(case_id.removeprefix("case_"))
    task = getattr(context, "simulation_task", None)
    task_running = bool(task is not None and not task.done())
    if task_running and str((active_scenario or {}).get("scenario_type") or "").upper() == scenario_type:
        return False

    client_path = storage.get_case_agent_path(case_id, preferred_party_role)
    if not (client_path / "config.yaml").exists() and preferred_party_role == "defendant":
        client_path = storage.get_case_agent_path(case_id, "plaintiff")
    if not (client_path / "config.yaml").exists():
        logger.warning("无法补发文书完成事件，缺少当事人配置: case=%s role=%s", case_id, preferred_party_role)
        return False

    config = storage.load_agent_config(client_path)
    await event_bus.publish(event_type, {
        "case_id": case_id,
        "client_path": str(client_path),
        "client_id": f"{case_id}_{config.get('party_role', preferred_party_role)}",
        "lawyer_id": config.get("assigned_lawyer_id", ""),
        "party_role": config.get("party_role", preferred_party_role),
        "firm_id": config.get("assigned_firm", ""),
    })
    return True


def get_or_create_player_gateway(sandbox_id: str | int, storage_root=None) -> _PlayerInputGateway:
    """Get or lazily create a player gateway for a specific sandbox."""
    key = str(sandbox_id)
    if key not in _player_gateways:
        from src.player_lawyer.input_gateway import make_json_persister, restore_json_requests
        from src.player_lawyer.run_ledger import PlayerRunLedger

        output_root = Path(storage_root) / "output" if storage_root else None
        ledger = PlayerRunLedger(storage_root=Path(storage_root)) if storage_root else None
        persist_fn = make_json_persister(output_root, ledger=ledger) if output_root else None
        gateway = _PlayerInputGateway(
            sandbox_id=int(sandbox_id) if str(sandbox_id).isdigit() else 0,
            persist_fn=persist_fn,
            ledger=ledger,
            storage_root=Path(storage_root) if storage_root else None,
        )
        if output_root is not None:
            restored = restore_json_requests(gateway, output_root)
            if restored:
                logger.info(
                    "[PlayerGateway] Restored %d pending request(s) for sandbox %s",
                    restored,
                    sandbox_id,
                )
        _player_gateways[key] = gateway
    return _player_gateways[key]


def reset_player_gateway(sandbox_id: str | int) -> int:
    """Cancel and drop the in-memory player gateway for a sandbox reset."""
    gateway = _player_gateways.pop(str(sandbox_id), None)
    if gateway is None:
        return 0
    try:
        cancelled = gateway.cancel_all_pending()
    except Exception as exc:
        logger.warning("[PlayerGateway] Failed to reset gateway for sandbox %s: %s", sandbox_id, exc)
        return 0
    return len(cancelled)


_set_player_gw_provider(_get_player_gateway_for_request)
_set_player_status_provider(_player_lawyer_status_for_request)
_set_player_response_assist_provider(_get_player_response_assist_for_request)


# ── 全局状态 ──

engine: WebSocketFrontendEngine | None = None
event_bus: EventBus | None = None
registry: AgentRegistry | None = None
checkpoint_mgr: CheckpointManager | None = None
storage_manager: FileStorageManager | None = None
case_fsm: CaseStateMachine | None = None
_simulation_task: asyncio.Task | None = None
_db_engine = None
_session_factory = None
sandbox_service = SandboxService(base_dir=SANDBOX_DATA_DIR, seed_source_dir=SANDBOX_SEED_DIR)
sandbox_manager: SandboxManager | None = None


class AuthRequest(BaseModel):
    email: str
    password: str


class RuntimeConfigRequest(BaseModel):
    prompt_profile: str
    api_key: str | None = None
    model_name: str
    api_base_url: str


class SandboxStartRequest(BaseModel):
    case_id: str


class PlayerDocumentAssistRequest(BaseModel):
    request_id: str | None = None
    case_id: str
    document_type: str
    skill_id: str
    player_prompt: str = ""
    player_draft: str = ""


class PlayerDocumentFollowupRequest(BaseModel):
    request_id: str
    message: str


class PlayerDocumentConfirmRequest(BaseModel):
    document_text: str


class PlayerDocumentManualConfirmRequest(BaseModel):
    request_id: str | None = None
    case_id: str
    document_type: str
    document_text: str


@dataclass(slots=True)
class _CaseLaunchRequest:
    case_id: str
    launch: Callable[[], Awaitable[bool | None]]
    post_launch_delay: float = 0.0


def _get_session_factory():
    global _db_engine, _session_factory
    if _session_factory is None:
        _db_engine = create_database_engine()
        Base.metadata.create_all(_db_engine)
        _session_factory = create_session_factory(_db_engine)
    return _session_factory


def _serialize_user(user: User) -> dict:
    return {
        "id": user.id,
        "email": user.email,
        "status": user.status,
        "token_version": user.token_version,
    }


def _build_auth_response(user: User) -> dict:
    return {
        "user": _serialize_user(user),
        "access_token": create_access_token(user_id=user.id, token_version=user.token_version),
        "token_type": "bearer",
        "expires_at": get_access_token_expires_at().isoformat(),
    }


def _normalize_runtime_config(payload: RuntimeConfigRequest) -> dict[str, str]:
    prompt_profile = str(payload.prompt_profile or "").strip().lower()
    if prompt_profile not in {"test", "prod"}:
        raise HTTPException(status_code=400, detail="prompt_profile 只能是 test 或 prod")

    api_key = str(payload.api_key or "").strip() or str(os.getenv("OPENAI_API_KEY", "") or "").strip()
    model_name = str(payload.model_name or "").strip()
    if not model_name:
        raise HTTPException(status_code=400, detail="model_name 不能为空")

    api_base_url = str(payload.api_base_url or "").strip()
    if not api_base_url:
        raise HTTPException(status_code=400, detail="api_base_url 不能为空")

    return {
        "SIMLAW_PROMPT_PROFILE": prompt_profile,
        "OPENAI_API_KEY": api_key,
        "OPENAI_MODEL_NAME": model_name,
        "OPENAI_API_BASE_URL": api_base_url,
    }


def _mask_runtime_secret(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if len(raw) <= 7:
        return "*" * len(raw)
    return f"{raw[:3]}{'*' * (len(raw) - 7)}{raw[-4:]}"


def _read_runtime_config() -> dict[str, str]:
    api_key = str(os.getenv("OPENAI_API_KEY", "") or "").strip()
    return {
        "prompt_profile": str(os.getenv("SIMLAW_PROMPT_PROFILE", "prod") or "").strip().lower() or "prod",
        "has_api_key": bool(api_key),
        "api_key_masked": _mask_runtime_secret(api_key),
        "model_name": str(os.getenv("OPENAI_MODEL_NAME", "") or "").strip(),
        "api_base_url": str(os.getenv("OPENAI_API_BASE_URL", "") or "").strip(),
    }


def _write_runtime_config_to_env_file(env_path: Path, config: dict[str, str]) -> None:
    env_path.parent.mkdir(parents=True, exist_ok=True)
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

    updated_keys: set[str] = set()
    rewritten_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            rewritten_lines.append(line)
            continue

        key, _value = line.split("=", 1)
        normalized_key = key.strip()
        if normalized_key in config:
            rewritten_lines.append(f"{normalized_key}={config[normalized_key]}")
            updated_keys.add(normalized_key)
            continue
        rewritten_lines.append(line)

    for key in RUNTIME_CONFIG_KEYS:
        if key not in updated_keys:
            rewritten_lines.append(f"{key}={config[key]}")

    env_path.write_text("\n".join(rewritten_lines).rstrip() + "\n", encoding="utf-8")


def _apply_runtime_config(config: dict[str, str]) -> None:
    for key, value in config.items():
        os.environ[key] = value

    _write_runtime_config_to_env_file(ROOT_ENV_PATH, config)


def _restart_backend_process() -> None:
    logger.warning("Debug runtime config requested backend restart; terminating current process for container restart.")
    os.kill(os.getpid(), signal.SIGTERM)


def _schedule_backend_restart(delay_seconds: float = 0.35) -> None:
    asyncio.get_running_loop().call_later(delay_seconds, _restart_backend_process)


def _extract_bearer_token(authorization: str | None) -> str:
    raw_value = str(authorization or "").strip()
    if not raw_value.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = raw_value[7:].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    return token


def _db_session_dependency():
    with get_db_session(_get_session_factory()) as session:
        yield session


def _get_current_user(
    authorization: str | None = Header(default=None),
    session=Depends(_db_session_dependency),
) -> User:
    token = _extract_bearer_token(authorization)
    try:
        claims = decode_access_token(token)
        user = get_user_by_id(session=session, user_id=claims["user_id"])
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except UserNotFoundError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    if user.token_version != claims["token_version"]:
        raise HTTPException(status_code=401, detail="Token version mismatch")
    return user


def _get_optional_current_user(
    authorization: str | None = Header(default=None),
    session=Depends(_db_session_dependency),
) -> User | None:
    if not str(authorization or "").strip():
        return None

    token = _extract_bearer_token(authorization)
    try:
        claims = decode_access_token(token)
        user = get_user_by_id(session=session, user_id=claims["user_id"])
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except UserNotFoundError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    if user.token_version != claims["token_version"]:
        raise HTTPException(status_code=401, detail="Token version mismatch")
    return user


app.include_router(
    create_human_eval_router(
        current_user_dependency=_get_current_user,
        session_dependency=_db_session_dependency,
    )
)


def _normalize_firm_building_type(firm_id: str) -> str:
    normalized = "".join(ch for ch in str(firm_id or "").lower() if ch.isalnum())
    if normalized in {"lawfirma", "lawfirma1"}:
        return "LawfirmA"
    if normalized in {"lawfirmb", "lawfirmb1"}:
        return "LawfirmB"
    return "LawfirmA"


def _build_agent_building_type(agent: Any) -> str:
    agent_type = str(getattr(agent, "agent_type", "") or "").strip().lower()
    if agent_type in {"lawyer", "receptionist"}:
        firm_id = str(getattr(agent, "firm_id", "") or "").strip()
        if not firm_id:
            config_path = Path(str(getattr(agent, "config_path", "") or ""))
            if config_path.parts:
                with contextlib.suppress(ValueError):
                    parts = list(config_path.parts)
                    idx = parts.index("law_firms")
                    firm_id = parts[idx + 1]
        return _normalize_firm_building_type(firm_id)
    if agent_type == "judge":
        return "Court"
    if agent_type == "client":
        return "community"
    return "community"


def _load_agent_shell_config(agent: Any, storage: FileStorageManager | None) -> dict[str, Any]:
    if storage is None:
        return {}

    config_path = str(getattr(agent, "config_path", "") or "").strip()
    if not config_path:
        return {}

    with contextlib.suppress(FileNotFoundError, OSError, ValueError, yaml.YAMLError):
        config = storage.load_agent_config(config_path)
        if isinstance(config, dict):
            return config
    return {}


def _build_agent_character_payload(agent: Any, storage: FileStorageManager | None) -> dict[str, Any]:
    config = _load_agent_shell_config(agent, storage)
    profile = config.get("profile", {})
    if not isinstance(profile, dict):
        profile = {}
    map_state = config.get("map_state", {})
    if not isinstance(map_state, dict):
        map_state = {}

    agent_type = str(getattr(agent, "agent_type", "") or "").strip().lower()
    building_type = _build_agent_building_type(agent)
    character_name = (
        str(profile.get("character_name", "") or "").strip()
        or str(config.get("character_name", "") or "").strip()
        or str(map_state.get("character_name", "") or "").strip()
    )
    law_firm = (
        str(profile.get("law_firm", "") or "").strip()
        or str(getattr(agent, "law_firm", "") or "").strip()
        or (
            "金杜律师事务所"
            if building_type == "LawfirmA"
            else "君合律师事务所" if building_type == "LawfirmB" else ""
        )
    )
    court_name = (
        str(profile.get("court_name", "") or "").strip()
        or str(getattr(agent, "court_name", "") or "").strip()
    )
    specialty_areas = profile.get("specialty") or getattr(agent, "specialty_areas", []) or []
    if not isinstance(specialty_areas, list):
        specialty_areas = []

    occupation = str(profile.get("occupation", "") or "").strip()
    if not occupation:
        if agent_type == "lawyer":
            occupation = "律师"
        elif agent_type == "judge":
            occupation = "法官"
        elif agent_type == "receptionist":
            occupation = "前台"
        else:
            occupation = "居民"

    character_info = {
        "name": str(profile.get("name", "") or getattr(agent, "name", "") or "").strip(),
        "gender": str(profile.get("gender", "") or getattr(agent, "gender", "") or "未知").strip() or "未知",
        "age": profile.get("age"),
        "occupation": occupation,
        "personality": str(profile.get("personality", "") or getattr(agent, "personality", "") or "").strip(),
        "speaking_style": str(profile.get("speaking_style", "") or getattr(agent, "speaking_style", "") or "").strip(),
        "background": str(profile.get("background", "") or "").strip(),
        "description": str(profile.get("description", "") or "").strip(),
        "character_name": character_name,
        "sprite_index": profile.get("sprite_index"),
        "law_firm": law_firm or None,
        "court": court_name or None,
        "specialty_areas": specialty_areas,
        "years_of_experience": (
            profile.get("years_of_experience")
            or getattr(agent, "years_of_experience", None)
        ),
        "legal_big_five": profile.get("legal_big_five"),
    }

    return {
        "npc_id": str(getattr(agent, "agent_id", "") or "").strip(),
        "npc_name": str(getattr(agent, "name", "") or "").strip(),
        "agent_type": agent_type,
        "building_type": building_type,
        "law_firm": law_firm or None,
        "court": court_name or None,
        "specialty_areas": specialty_areas,
        "years_of_experience": character_info["years_of_experience"],
        "character_name": character_name or None,
        "sprite_index": character_info["sprite_index"],
        "gender": character_info["gender"],
        "occupation": occupation,
        "personality": character_info["personality"],
        "speaking_style": character_info["speaking_style"],
        "background": character_info["background"],
        "description": character_info["description"],
        "character_info": character_info,
    }


def _serialize_registry_agents(
    agent_registry: AgentRegistry | None,
    storage: FileStorageManager | None,
) -> dict[str, Any]:
    if agent_registry is None:
        return {"agents": {}, "items": []}

    agent_map: dict[str, dict[str, Any]] = {}
    items: list[dict[str, Any]] = []
    for agent in agent_registry.get_all_agents():
        meta = _build_agent_character_payload(agent, storage)
        agent_id = str(meta.get("npc_id", "") or "").strip()
        if not agent_id:
            continue
        agent_map[agent_id] = meta
        items.append(
            {
                "id": agent_id,
                "name": meta.get("npc_name"),
                "type": meta.get("agent_type"),
                "law_firm": meta.get("law_firm"),
                "building_type": meta.get("building_type"),
            }
        )
    return {"agents": agent_map, "items": items}


def _build_sandbox_runtime_context(sandbox: Sandbox, storage_root: Path) -> SandboxRuntimeContext:
    runtime_engine = WebSocketFrontendEngine(
        load_registry_from_map(MAP_JSON_PATH),
        fallback_speed=0.5,
        backend_authoritative=True,
        move_speed_px_per_second=150.0,
        map_json_path=MAP_JSON_PATH,
        frontend_mode=SIMLAW_FRONTEND_MODE,
        turn_mode=SIMLAW_TURN_MODE,
    )
    runtime_event_bus, runtime_registry, runtime_checkpoint_mgr, runtime_storage, runtime_case_fsm = (
        _initialize_runtime_state(
            existing_engine=runtime_engine,
            sandbox_data_dir=storage_root,
            set_global_engine=False,
        )
    )
    context = SandboxRuntimeContext(
        sandbox_id=sandbox.id,
        user_id=sandbox.user_id,
        sandbox_key=sandbox.sandbox_key,
        storage_root=storage_root,
        engine=runtime_engine,
        event_bus=runtime_event_bus,
        registry=runtime_registry,
        checkpoint_mgr=runtime_checkpoint_mgr,
        storage_manager=runtime_storage,
        case_fsm=runtime_case_fsm,
    )
    context.orchestrator = getattr(runtime_engine, "orchestrator", None)
    if _is_player_plaintiff_mode():
        player_gateway = get_or_create_player_gateway(sandbox.id, storage_root)
        setattr(context, "player_gateway", player_gateway)

        try:
            player_event_loop = asyncio.get_running_loop()
        except RuntimeError:
            player_event_loop = None

        def _broadcast_player_lawyer_event(event_type: str, data: dict) -> None:
            payload = {"type": event_type, "event": event_type, "data": data}
            if player_event_loop is None or not player_event_loop.is_running():
                return
            asyncio.run_coroutine_threadsafe(
                _broadcast_sandbox_event(str(sandbox.id), payload),
                player_event_loop,
            )

        if context.orchestrator is not None:
            setattr(context.orchestrator, "_player_gateway", player_gateway)
            setattr(context.orchestrator, "_player_broadcast_fn", _broadcast_player_lawyer_event)
            setattr(context.orchestrator, "_sandbox_id", sandbox.id)

    async def _runtime_issue_reporter(
        *,
        case_id: str,
        scenario_type: str,
        exc: Exception,
        stage_label: str = "",
        event_type: str = "",
        handler_name: str = "",
    ) -> bool:
        payload = _normalize_runtime_issue_from_exception(
            case_id=case_id,
            scenario_type=scenario_type,
            exc=exc,
            stage_label=stage_label,
            event_type=event_type,
            handler_name=handler_name,
        )
        if payload is None:
            return False
        return await _report_sandbox_runtime_issue(context, payload)

    setattr(runtime_engine, "runtime_issue_reporter", _runtime_issue_reporter)
    setattr(runtime_event_bus, "runtime_issue_reporter", _runtime_issue_reporter)
    if context.orchestrator is not None:
        setattr(context.orchestrator, "runtime_issue_reporter", _runtime_issue_reporter)

    get_agents_by_type = getattr(runtime_registry, "get_agents_by_type", None)
    if callable(get_agents_by_type):
        for receptionist in get_agents_by_type("receptionist"):
            setattr(receptionist, "runtime_issue_reporter", _runtime_issue_reporter)

    return context


def _set_runtime_engine_paused(runtime_engine: WebSocketFrontendEngine | None, paused: bool) -> None:
    if runtime_engine is None:
        return
    runtime_engine._paused = paused
    if paused:
        runtime_engine._resumed_event.clear()
    else:
        runtime_engine._resumed_event.set()


def _tool_name_from_record(record: Any) -> str:
    if isinstance(record, dict):
        for key in ("name", "tool_name", "function_name"):
            value = str(record.get(key) or "").strip()
            if value:
                return value
        function_payload = record.get("function")
        if isinstance(function_payload, dict):
            return str(function_payload.get("name") or "").strip()

    for attr in ("name", "tool_name", "function_name"):
        value = str(getattr(record, attr, "") or "").strip()
        if value:
            return value
    function_payload = getattr(record, "function", None)
    return str(getattr(function_payload, "name", "") or "").strip()


def _tool_names_for_agent(agent: Any) -> list[str]:
    names: list[str] = []
    for tool in list(getattr(agent, "tools", []) or []):
        name = ""
        if hasattr(tool, "get_function_name"):
            with contextlib.suppress(Exception):
                name = str(tool.get_function_name() or "").strip()
        if not name:
            name = str(getattr(tool, "name", "") or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def _role_name_for_agent(agent: Any, stage_code: str) -> str:
    explicit = str(getattr(agent, "_simlaw_stage_role", "") or "").strip()
    if explicit:
        return explicit
    if not stage_code:
        return ""
    with contextlib.suppress(Exception):
        return infer_stage_role_name(stage_code, agent)
    return ""


def _agent_type_for_status(agent: Any) -> str:
    with contextlib.suppress(Exception):
        return str(resolve_agent_type(agent)).strip()
    class_name = type(agent).__name__
    if class_name == "PlayerPlaintiffLawyerAgent":
        return "player_lawyer"
    return class_name.replace("Agent", "").lower() or "agent"


def _configured_tool_names_for_agent(agent: Any, stage_code: str, role_name: str) -> list[str]:
    explicit = [
        str(name).strip()
        for name in list(getattr(agent, "_simlaw_configured_tool_names", []) or [])
        if str(name).strip()
    ]
    if explicit:
        return explicit
    if not stage_code or not role_name:
        return []
    with contextlib.suppress(Exception):
        return resolve_configured_tool_names(stage_code, role_name, resolve_agent_type(agent))
    return []


def _skill_usage_for_agent(agent: Any) -> dict[str, Any]:
    reporter = getattr(agent, "get_skill_usage_report", None)
    if not callable(reporter):
        return {}
    with contextlib.suppress(Exception):
        payload = reporter()
        if isinstance(payload, dict):
            return payload
    return {}


def _available_skill_names_for_agent(agent: Any, stage_code: str = "") -> list[str]:
    skill_dirs = list(getattr(agent, "skill_dirs", []) or [])
    if not skill_dirs:
        return []
    with contextlib.suppress(Exception):
        toolkit = _FlatSkillToolkit([str(item) for item in skill_dirs if str(item or "").strip()])
        names = [
            str(item.get("name") or "").strip()
            for item in toolkit.list_skills()
            if str(item.get("name") or "").strip()
        ]
        agent_type = _agent_type_for_status(agent)
        if agent_type == "client":
            return [name for name in names if name.startswith("client-")]
        if agent_type == "lawyer":
            stage_skill_map = {
                "CD": "lawyer-complaint-drafting",
                "DD": "lawyer-defense-drafting",
                "AD": "lawyer-appeal-drafting",
                "AR": "lawyer-appeal-response-drafting",
            }
            preferred = ["lawyer-memory-writing"]
            stage_skill = stage_skill_map.get(str(stage_code or "").strip().upper())
            if stage_skill:
                preferred.append(stage_skill)
            filtered = [name for name in names if name in preferred]
            return filtered or [name for name in names if name.startswith("lawyer-")]
        return names
    return []


def _stage_code_from_case_state(raw_state: Any) -> str:
    state = normalize_case_state(str(raw_state or "").strip())
    if not state or state in {"空闲", "已结案"}:
        return ""
    exact_map = {
        "原告咨询中": "PLC",
        "被告咨询中": "DLC",
        "起诉状起草中": "CD",
        "答辩状起草中": "DD",
        "上诉状起草中": "AD",
        "上诉答辩状起草中": "AR",
        "一审庭审中": "CI",
        "二审庭审中": "CIA",
    }
    if state in exact_map:
        return exact_map[state]
    if "起诉状" in state:
        return "CD"
    if "答辩状" in state and "上诉" not in state:
        return "DD"
    if "上诉答辩" in state:
        return "AR"
    if "上诉状" in state or "上诉决策" in state:
        return "AD"
    if "二审" in state:
        return "CIA"
    if "一审" in state or "庭前" in state:
        return "CI"
    return ""


def _load_status_agent_config(storage: Any, agent: Any) -> dict[str, Any]:
    config_path = getattr(agent, "config_path", None)
    if storage is None or not config_path:
        return {}
    with contextlib.suppress(Exception):
        config = storage.load_agent_config(config_path)
        if isinstance(config, dict):
            return config
    return {}


def _infer_case_stage_from_storage(storage: Any, case_id: str, fallback_config: dict[str, Any] | None = None) -> str:
    normalized_case_id = _normalize_case_identifier(case_id)
    if not normalized_case_id:
        return ""

    with contextlib.suppress(Exception):
        case_runtime = storage.load_case_runtime(normalized_case_id)
        stage_code = _stage_code_from_case_state(case_runtime.get("overall_state"))
        if stage_code:
            return stage_code

    config = fallback_config or {}
    stage_code = _stage_code_from_case_state(config.get("case_state"))
    if stage_code:
        return stage_code

    case_dir = Path(getattr(storage, "base_dir", "")) / "cases" / normalized_case_id
    for party_role in ("plaintiff", "defendant"):
        party_config = _load_yaml_mapping(case_dir / party_role / "config.yaml")
        stage_code = _stage_code_from_case_state(party_config.get("case_state"))
        if stage_code:
            return stage_code
    return ""


def _case_context_by_agent_id(context: SandboxRuntimeContext) -> dict[str, dict[str, str]]:
    storage = getattr(context, "storage_manager", None)
    registry = getattr(context, "registry", None)
    if storage is None or registry is None:
        return {}

    get_agents_by_type = getattr(registry, "get_agents_by_type", None)
    if not callable(get_agents_by_type):
        return {}

    context_by_agent: dict[str, dict[str, str]] = {}
    case_stage_by_id: dict[str, str] = {}

    for client in list(get_agents_by_type("client") or []):
        config = _load_status_agent_config(storage, client)
        case_id = _normalize_case_identifier(config.get("case_id"))
        if not case_id:
            continue
        stage_code = _infer_case_stage_from_storage(storage, case_id, config)
        if stage_code:
            case_stage_by_id[case_id] = stage_code
        client_stage_code = _stage_code_from_case_state(config.get("case_state"))
        agent_id = str(getattr(client, "agent_id", "") or "").strip()
        if agent_id:
            context_by_agent[agent_id] = {
                "case_id": case_id,
                "stage_code": client_stage_code,
                "party_role": str(config.get("party_role") or "").strip(),
            }
        assigned_lawyer_id = str(config.get("assigned_lawyer_id") or "").strip()
        if assigned_lawyer_id:
            context_by_agent.setdefault(
                assigned_lawyer_id,
                {
                    "case_id": case_id,
                    "stage_code": stage_code,
                    "party_role": str(config.get("party_role") or "").strip(),
                },
            )

    for lawyer in list(get_agents_by_type("lawyer") or []):
        agent_id = str(getattr(lawyer, "agent_id", "") or "").strip()
        if not agent_id:
            continue
        config = _load_status_agent_config(storage, lawyer)
        candidate_case_ids = [
            str(config.get("current_handling_case") or "").strip(),
            *[str(item or "").strip() for item in list(config.get("case_queue") or [])],
        ]
        for candidate_case_id in candidate_case_ids:
            case_id = _normalize_case_identifier(candidate_case_id)
            if not case_id:
                continue
            stage_code = case_stage_by_id.get(case_id) or _infer_case_stage_from_storage(storage, case_id)
            context_by_agent[agent_id] = {
                "case_id": case_id,
                "stage_code": stage_code,
                "party_role": context_by_agent.get(agent_id, {}).get("party_role", ""),
            }
            break

    return context_by_agent


def _active_agent_context(context: SandboxRuntimeContext) -> dict[str, dict[str, str]]:
    event_bus = getattr(context, "event_bus", None)
    if event_bus is None or not hasattr(event_bus, "get_active_scenarios_snapshot"):
        return {}

    active_by_agent: dict[str, dict[str, str]] = {}
    with contextlib.suppress(Exception):
        active_scenarios = event_bus.get_active_scenarios_snapshot()
        for case_id, scenario_info in dict(active_scenarios or {}).items():
            scenario_type = str(scenario_info.get("scenario_type") or "").strip()
            for agent_id in list(scenario_info.get("participants") or []):
                normalized_agent_id = str(agent_id or "").strip()
                if not normalized_agent_id:
                    continue
                active_by_agent[normalized_agent_id] = {
                    "case_id": str(case_id or "").strip(),
                    "scenario_type": scenario_type,
                }
    return active_by_agent


def _serialize_agent_capabilities(context: SandboxRuntimeContext) -> list[dict[str, Any]]:
    registry = getattr(context, "registry", None)
    if registry is None or not hasattr(registry, "get_all_agents"):
        return []

    active_by_agent = _active_agent_context(context)
    persisted_context_by_agent = _case_context_by_agent_id(context)
    capabilities: list[dict[str, Any]] = []
    for agent in list(registry.get_all_agents() or []):
        agent_id = str(getattr(agent, "agent_id", "") or "").strip()
        if not agent_id:
            continue

        active_context = active_by_agent.get(agent_id, {})
        persisted_context = persisted_context_by_agent.get(agent_id, {})
        stage_code = str(
            getattr(agent, "_simlaw_stage_code", "")
            or active_context.get("scenario_type")
            or persisted_context.get("stage_code")
            or ""
        ).strip().upper()
        role_name = _role_name_for_agent(agent, stage_code) or persisted_context.get("party_role", "")
        tool_names = _tool_names_for_agent(agent)
        available_tool_names = [
            str(name).strip()
            for name in list(getattr(agent, "_simlaw_available_tool_names", []) or tool_names)
            if str(name).strip()
        ]
        configured_tool_names = _configured_tool_names_for_agent(agent, stage_code, role_name)
        actual_tool_calls = []
        for record in list(getattr(agent, "_last_tool_call_records", []) or []):
            tool_name = _tool_name_from_record(record)
            if tool_name and tool_name not in actual_tool_calls:
                actual_tool_calls.append(tool_name)
        skill_usage = _skill_usage_for_agent(agent)
        skills = [
            {
                "name": str(item.get("name") or "").strip(),
                "load_count": int(item.get("load_count") or 0),
            }
            for item in list(skill_usage.get("skills") or [])
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        ]

        capabilities.append(
            {
                "agent_id": agent_id,
                "agent_name": str(getattr(agent, "name", "") or agent_id).strip(),
                "agent_type": _agent_type_for_status(agent),
                "agent_class": type(agent).__name__,
                "agent_role": role_name,
                "stage_code": stage_code,
                "case_id": active_context.get("case_id", "") or persisted_context.get("case_id", ""),
                "is_active": agent_id in active_by_agent or bool(getattr(agent, "is_active", False)),
                "configured_tool_names": configured_tool_names,
                "available_tool_names": available_tool_names,
                "actual_tool_calls": actual_tool_calls,
                "actual_tool_call_count": len(list(getattr(agent, "_last_tool_call_records", []) or [])),
                "skill_load_count": int(skill_usage.get("skill_load_count") or 0),
                "skill_names": [item["name"] for item in skills],
                "available_skill_names": _available_skill_names_for_agent(agent, stage_code),
                "has_skill_tool": "load_skill" in available_tool_names or bool(getattr(agent, "skill_dirs", []) or []),
                "is_player_agent": type(agent).__name__ == "PlayerPlaintiffLawyerAgent",
            }
        )

    capabilities.sort(
        key=lambda item: (
            0 if item.get("is_active") else 1,
            str(item.get("stage_code") or ""),
            str(item.get("agent_role") or ""),
            str(item.get("agent_id") or ""),
        )
    )
    return capabilities


def _build_sandbox_runtime_status(context: SandboxRuntimeContext) -> dict[str, Any]:
    task = context.simulation_task
    task_running = bool(task is not None and not task.done())
    paused = bool(context.engine and getattr(context.engine, "_paused", False))
    session_state = None
    if context.checkpoint_mgr is not None and hasattr(context.checkpoint_mgr, "load_session_state"):
        session_state = context.checkpoint_mgr.load_session_state()
    persisted_status = str((session_state or {}).get("simulation_status") or "").strip().lower()
    last_error = context.last_error

    if last_error:
        status = "error"
    elif task_running:
        status = "paused" if paused else "running"
    elif persisted_status in {"paused", "running"}:
        status = "paused"
    elif persisted_status == "completed" and getattr(context, "single_case_mode", False):
        status = "idle"
    elif persisted_status == "completed":
        status = "completed"
    else:
        status = "idle"

    clients_connected = len(context.connected_clients)
    if not clients_connected and getattr(context.engine, "clients", None) is not None:
        clients_connected = len(context.engine.clients)

    return {
        "status": status,
        "session_id": (session_state or {}).get("session_id"),
        "selected_case_id": _get_context_selected_case_id(context),
        "paused": status == "paused",
        "simulation_running": False if last_error else task_running,
        "clients_connected": clients_connected,
        "active_cases": 0 if last_error else _count_active_cases(context.storage_manager, context.registry),
        "last_error": last_error,
        "agent_capabilities": _serialize_agent_capabilities(context),
    }


def _start_or_resume_sandbox_context(
    context: SandboxRuntimeContext,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if (
        context.engine is None
        or context.event_bus is None
        or context.registry is None
        or context.checkpoint_mgr is None
        or context.storage_manager is None
        or context.case_fsm is None
    ):
        raise HTTPException(status_code=503, detail="Sandbox runtime not ready")

    payload = payload or {}
    requested_case_id = _normalize_case_identifier(payload.get("case_id"))
    existing_case_id = _get_context_selected_case_id(context)
    if requested_case_id:
        if existing_case_id and existing_case_id != requested_case_id and context.simulation_task is not None and not context.simulation_task.done():
            raise HTTPException(status_code=409, detail="当前已有其他案件在运行，请先等待其结束或重新开始")
        _set_context_case_selection(context, requested_case_id)
    elif not existing_case_id:
        raise HTTPException(status_code=400, detail="缺少 case_id")

    selected_case_id = _get_context_selected_case_id(context)
    context.last_error = None

    if context.simulation_task is not None and not context.simulation_task.done():
        _set_runtime_engine_paused(context.engine, False)
        context.checkpoint_mgr.mark_session_running()
        return _build_sandbox_runtime_status(context)

    session_state = context.checkpoint_mgr.load_session_state()
    _set_runtime_engine_paused(context.engine, False)

    if session_state and session_state.get("simulation_status") in {"running", "paused"}:
        context.checkpoint_mgr.mark_session_running()
        context.simulation_task = asyncio.create_task(
            resume_simulation(
                context.engine,
                context.event_bus,
                context.registry,
                context.storage_manager,
                context.case_fsm,
                context.checkpoint_mgr,
                selected_case_id=selected_case_id,
            )
        )
    else:
        context.checkpoint_mgr.create_new_session()
        _set_context_case_selection(context, selected_case_id)
        context.simulation_task = asyncio.create_task(
            run_simulation(
                context.engine,
                context.event_bus,
                context.registry,
                context.storage_manager,
                context.case_fsm,
                context.checkpoint_mgr,
                selected_case_id=selected_case_id,
            )
        )
    context.simulation_task.add_done_callback(_log_task_result)
    return _build_sandbox_runtime_status(context)


def _get_sandbox_manager() -> SandboxManager:
    global sandbox_manager
    if sandbox_manager is None:
        sandbox_manager = SandboxManager(
            base_dir=SANDBOX_DATA_DIR,
            runtime_factory=_build_sandbox_runtime_context,
            start_handler=_start_or_resume_sandbox_context,
            status_handler=_build_sandbox_runtime_status,
        )
    return sandbox_manager


def _serialize_sandbox_state(
    sandbox: Sandbox | None,
    *,
    runtime_status: dict | None = None,
) -> dict:
    if sandbox is None:
        return {
            "status": "not_created",
            "selected_case_id": "",
            "active_cases": 0,
            "clients_connected": 0,
            "can_start": True,
            "can_pause": False,
            "can_restart": False,
            "last_error": None,
            "agent_capabilities": [],
        }

    state = runtime_status or {}
    status = str(state.get("status") or sandbox.status)
    return {
        "id": sandbox.id,
        "user_id": sandbox.user_id,
        "sandbox_key": sandbox.sandbox_key,
        "storage_root": sandbox.storage_root,
        "status": status,
        "session_id": state.get("session_id"),
        "selected_case_id": state.get("selected_case_id") or "",
        "active_cases": int(state.get("active_cases", 0) or 0),
        "clients_connected": int(state.get("clients_connected", 0) or 0),
        "can_start": status in {"idle", "paused", "completed"},
        "can_pause": status == "running",
        "can_restart": sandbox is not None,
        "last_error": state.get("last_error"),
        "agent_capabilities": list(state.get("agent_capabilities") or []),
    }


def _update_sandbox_from_runtime_status(session, sandbox: Sandbox, runtime_status: dict) -> None:
    sandbox_service.update_sandbox_status(
        session=session,
        sandbox=sandbox,
        sandbox_status=str(runtime_status.get("status") or sandbox.status),
        simulation_status=str(runtime_status.get("status") or sandbox.status),
        active_cases=int(runtime_status.get("active_cases", 0) or 0),
        clients_connected=int(runtime_status.get("clients_connected", 0) or 0),
    )


def _require_user_sandbox(session, user: User) -> Sandbox:
    sandbox = sandbox_service.get_user_sandbox(session=session, user_id=user.id)
    if sandbox is None:
        raise HTTPException(status_code=404, detail="Sandbox not created")
    return sandbox


def _sandbox_storage_has_seed_data(storage_root: Path) -> bool:
    return (
        (storage_root / "case_data_extracted.json").exists()
        and any(storage_root.glob("cases/case_*/plaintiff/config.yaml"))
        and any(storage_root.glob("law_firms/*/lawyer_roster.yaml"))
        and any(storage_root.glob("court_system/*/judges/*/config.yaml"))
    )


def _sandbox_context_needs_rebuild(context: SandboxRuntimeContext, storage_root: Path) -> bool:
    task = getattr(context, "simulation_task", None)
    task_running = bool(task is not None and not task.done())
    registry = getattr(context, "registry", None)
    get_all_agents = getattr(registry, "get_all_agents", None)
    agents = get_all_agents() if callable(get_all_agents) else []
    return (not task_running) and not agents and _sandbox_storage_has_seed_data(storage_root)


def _normalize_case_identifier(case_id: str | None) -> str:
    raw_value = str(case_id or "").strip()
    if not raw_value:
        return ""
    return raw_value if raw_value.startswith("case_") else f"case_{raw_value}"


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else {}


def _load_case_picker_metadata(path: Path | None = None) -> dict[str, dict[str, str]]:
    target_path = path or CASE_PICKER_METADATA_PATH
    if not target_path.exists():
        return {}

    try:
        with target_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except Exception as exc:
        logger.warning("Failed to load case picker metadata from %s: %s", target_path, exc)
        return {}

    if not isinstance(payload, dict):
        return {}

    normalized: dict[str, dict[str, str]] = {}
    for case_id, raw_item in payload.items():
        if not isinstance(raw_item, dict):
            continue
        normalized_case_id = str(case_id or "").strip()
        if not normalized_case_id:
            continue
        normalized[normalized_case_id] = {
            "training_category": str(raw_item.get("training_category") or "").strip(),
            "difficulty": str(raw_item.get("difficulty") or "").strip(),
            "raw_case_cause": str(raw_item.get("raw_case_cause") or "").strip(),
        }
    return normalized


def _get_context_selected_case_id(context: SandboxRuntimeContext | None) -> str:
    if context is None:
        return ""

    selected_case_id = _normalize_case_identifier(getattr(context, "selected_case_id", ""))
    if selected_case_id:
        return selected_case_id

    checkpoint_mgr = getattr(context, "checkpoint_mgr", None)
    session_state = checkpoint_mgr.load_session_state() if checkpoint_mgr is not None and hasattr(checkpoint_mgr, "load_session_state") else None
    return _normalize_case_identifier((session_state or {}).get("selected_case_id"))


def _persist_context_case_selection(context: SandboxRuntimeContext) -> None:
    checkpoint_mgr = getattr(context, "checkpoint_mgr", None)
    session_state = getattr(checkpoint_mgr, "_session_state", None)
    if not isinstance(session_state, dict):
        return

    session_state["selected_case_id"] = getattr(context, "selected_case_id", "") or ""
    session_state["single_case_mode"] = bool(getattr(context, "single_case_mode", False))
    saver = getattr(checkpoint_mgr, "_save_session_state", None)
    if callable(saver):
        saver()


def _set_context_case_selection(context: SandboxRuntimeContext, case_id: str) -> None:
    context.selected_case_id = _normalize_case_identifier(case_id)
    context.single_case_mode = bool(context.selected_case_id)
    _persist_context_case_selection(context)


def _clear_context_case_selection(context: SandboxRuntimeContext) -> None:
    context.selected_case_id = ""
    context.single_case_mode = False
    _persist_context_case_selection(context)


def _resolve_case_progress_state(
    *,
    storage: FileStorageManager,
    case_id: str,
    plaintiff_config: dict[str, Any],
    defendant_config: dict[str, Any],
) -> str:
    try:
        case_runtime = storage.load_case_runtime(case_id)
    except FileNotFoundError:
        case_runtime = {}
    except Exception:
        case_runtime = {}

    overall_state = str(case_runtime.get("overall_state") or "").strip()
    if overall_state:
        return overall_state

    inferred_states: list[str] = []
    for config in (plaintiff_config, defendant_config):
        if not config:
            continue
        with contextlib.suppress(Exception):
            inferred_state = str(infer_case_state_from_artifacts(storage.base_dir, config) or "").strip()
            if inferred_state:
                inferred_states.append(inferred_state)

    config_states = [
        str(config.get("case_state") or "").strip()
        for config in (plaintiff_config, defendant_config)
        if isinstance(config, dict) and config
    ]

    for state in inferred_states + config_states:
        if state not in {"", "空闲", "已结案"}:
            return state

    all_states = [state for state in inferred_states + config_states if state]
    if "已结案" in config_states and all(state in {"空闲", "已结案"} for state in all_states):
        return "已结案"
    if all_states and all(state == "已结案" for state in all_states):
        return "已结案"
    return "空闲"


def _map_case_progress_to_picker_status(
    *,
    case_id: str,
    overall_state: str,
    runtime_status: dict[str, Any] | None,
    selected_case_id: str,
) -> str:
    runtime_state = str((runtime_status or {}).get("status") or "").strip().lower()
    if runtime_state in {"running", "paused", "error"} and case_id == selected_case_id:
        return "running"
    if overall_state == "已结案":
        return "closed"
    if overall_state and overall_state != "空闲":
        return "running"
    return "idle"


def _build_dataset_fallback_candidates(dataset_path: str) -> list[str]:
    current_path = str(dataset_path or "").strip()
    data_root = _backend_dir.parent / "data"
    if not data_root.exists():
        return []

    candidates: list[str] = []
    seen: set[str] = set()

    def _add(candidate: Path) -> None:
        resolved = str(candidate.resolve())
        if resolved not in seen:
            seen.add(resolved)
            candidates.append(resolved)

    if current_path:
        candidate_name = Path(current_path.replace("\\", "/")).name
        if candidate_name:
            same_name_candidate = data_root / candidate_name
            if same_name_candidate.exists():
                _add(same_name_candidate)

    for item in sorted(data_root.glob("*.json")):
        if item.is_file():
            _add(item)

    return candidates


def _resolve_case_picker_case_type(config: dict[str, Any], *, fallback_name: str = "") -> str:
    dataset_path = str(config.get("dataset_path") or "").strip()
    candidate_paths: list[str] = []
    seen: set[str] = set()

    def _add(path_value: str) -> None:
        normalized = str(path_value or "").strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            candidate_paths.append(normalized)

    _add(dataset_path)
    for fallback_path in _build_dataset_fallback_candidates(dataset_path):
        _add(fallback_path)

    if not candidate_paths:
        return ""

    from src.data.data_loader import DataLoader

    for candidate_path in candidate_paths:
        try:
            data_loader = DataLoader(candidate_path)
            case = data_loader.resolve_case_for_config(config, fallback_name=fallback_name)
            case_type = str(data_loader.extract_case_cause(case) or "").strip()
            if case_type:
                return case_type
        except Exception as exc:
            logger.warning("Failed to resolve case picker case_type from %s: %s", candidate_path, exc)

    return ""


def _build_case_picker_entries(
    *,
    storage_root: Path,
    runtime_status: dict[str, Any] | None = None,
    selected_case_id: str = "",
    metadata_path: Path | None = None,
) -> list[dict[str, str]]:
    try:
        storage: Any = FileStorageManager(base_dir=storage_root)
    except TypeError:
        class _FallbackStorage:
            def __init__(self, base_dir: Path) -> None:
                self.base_dir = base_dir

            def load_case_runtime(self, case_id: str) -> dict[str, Any]:
                raise FileNotFoundError(case_id)

        storage = _FallbackStorage(storage_root)
    entries: list[dict[str, str]] = []
    metadata_by_case_id = _load_case_picker_metadata(metadata_path)

    for plaintiff_path in sorted(storage_root.glob("cases/case_*/plaintiff/config.yaml")):
        case_dir = plaintiff_path.parent.parent
        case_id = case_dir.name
        defendant_path = case_dir / "defendant" / "config.yaml"
        plaintiff_config = _load_yaml_mapping(plaintiff_path)
        defendant_config = _load_yaml_mapping(defendant_path)

        plaintiff_name = str((plaintiff_config.get("profile") or {}).get("name") or "").strip() or "原告"
        defendant_name = str((defendant_config.get("profile") or {}).get("name") or "").strip() or "被告"
        raw_case_cause = str(
            plaintiff_config.get("case_type")
            or defendant_config.get("case_type")
            or ""
        ).strip()
        if not raw_case_cause:
            raw_case_cause = _resolve_case_picker_case_type(plaintiff_config, fallback_name=plaintiff_name)
        if not raw_case_cause:
            raw_case_cause = _resolve_case_picker_case_type(defendant_config, fallback_name=defendant_name)
        metadata = metadata_by_case_id.get(case_id, {})
        overall_state = _resolve_case_progress_state(
            storage=storage,
            case_id=case_id,
            plaintiff_config=plaintiff_config,
            defendant_config=defendant_config,
        )
        entries.append(
            {
                "case_id": case_id,
                "title": f"{plaintiff_name}诉{defendant_name}",
                "plaintiff_name": plaintiff_name,
                "defendant_name": defendant_name,
                "raw_case_cause": str(metadata.get("raw_case_cause") or raw_case_cause or "").strip(),
                "training_category": str(metadata.get("training_category") or "").strip(),
                "difficulty": str(metadata.get("difficulty") or "").strip(),
                "status": _map_case_progress_to_picker_status(
                    case_id=case_id,
                    overall_state=overall_state,
                    runtime_status=runtime_status,
                    selected_case_id=selected_case_id,
                ),
            }
        )

    entries.sort(key=lambda entry: int(re.sub(r"^case_", "", entry["case_id"]) or "0"))
    return entries


def _find_case_picker_entry(entries: list[dict[str, str]], case_id: str) -> dict[str, str] | None:
    normalized_case_id = _normalize_case_identifier(case_id)
    for entry in entries:
        if _normalize_case_identifier(entry.get("case_id")) == normalized_case_id:
            return entry
    return None


def _is_subpath(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _load_json_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle) or {}
    except Exception as exc:
        logger.warning("Failed to load JSON from %s: %s", path, exc)
        return {}
    return payload if isinstance(payload, dict) else {}


def _is_windows_absolute_path(path_value: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\\\/]", str(path_value or "").strip()))


def _resolve_case_output_dir_for_sandbox(sandbox: Sandbox, case_id: str) -> Path:
    storage_root = Path(sandbox.storage_root).resolve()
    return (storage_root / "output" / _normalize_case_identifier(case_id)).resolve()


def _resolve_case_document_pdf_path(
    *,
    case_output_dir: Path,
    raw_pdf_path: str,
    fallback_filename: str,
) -> Path | None:
    normalized_raw_pdf_path = str(raw_pdf_path or "").strip()
    if normalized_raw_pdf_path and not _is_windows_absolute_path(normalized_raw_pdf_path):
        candidate = Path(normalized_raw_pdf_path)
        candidate = candidate.resolve() if candidate.is_absolute() else (case_output_dir / candidate).resolve()
        if candidate.exists() and _is_subpath(candidate, case_output_dir):
            return candidate

    fallback_path = (case_output_dir / fallback_filename).resolve()
    if fallback_path.exists() and _is_subpath(fallback_path, case_output_dir):
        return fallback_path
    return None


def _require_sandbox_case_entry(sandbox: Sandbox, case_id: str) -> dict[str, str]:
    entries = _build_case_picker_entries(storage_root=Path(sandbox.storage_root))
    case_entry = _find_case_picker_entry(entries, case_id)
    if case_entry is None:
        raise HTTPException(status_code=404, detail="案件不存在")
    return case_entry


def _build_case_document_entry(
    *,
    sandbox: Sandbox,
    case_id: str,
    spec: dict[str, str],
) -> tuple[dict[str, Any], Path | None]:
    normalized_case_id = _normalize_case_identifier(case_id)
    case_output_dir = _resolve_case_output_dir_for_sandbox(sandbox, normalized_case_id)
    result_payload = _load_json_mapping(case_output_dir / spec["result_filename"])
    drafted_payload = result_payload.get("drafted_document_payload") or {}
    raw_pdf_path = str(
        result_payload.get("pdf_path")
        or drafted_payload.get("pdf_path")
        or ""
    ).strip()
    resolved_pdf_path = _resolve_case_document_pdf_path(
        case_output_dir=case_output_dir,
        raw_pdf_path=raw_pdf_path,
        fallback_filename=spec["pdf_filename"],
    )
    entry = {
        "document_key": spec["document_key"],
        "stage": spec["stage"],
        "document_type": spec["document_type"],
        "title": spec["title"],
        "file_name": resolved_pdf_path.name if resolved_pdf_path is not None else spec["pdf_filename"],
        "available": resolved_pdf_path is not None,
        "download_url": (
            f"/api/sandbox/cases/{normalized_case_id}/documents/{spec['document_key']}/download"
            if resolved_pdf_path is not None
            else ""
        ),
    }
    return entry, resolved_pdf_path


def _list_case_document_entries(sandbox: Sandbox, case_id: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for spec in _CASE_DOCUMENT_SPECS:
        entry, _ = _build_case_document_entry(
            sandbox=sandbox,
            case_id=case_id,
            spec=spec,
        )
        entries.append(entry)
    return entries


def _resolve_case_document_download_path(
    *,
    sandbox: Sandbox,
    case_id: str,
    document_key: str,
) -> tuple[dict[str, str], Path]:
    spec = _CASE_DOCUMENT_SPEC_BY_KEY.get(str(document_key or "").strip().upper())
    if spec is None:
        raise HTTPException(status_code=404, detail="文书类型不存在")

    _, resolved_pdf_path = _build_case_document_entry(
        sandbox=sandbox,
        case_id=case_id,
        spec=spec,
    )
    if resolved_pdf_path is None:
        raise HTTPException(status_code=404, detail="PDF 不存在")
    return spec, resolved_pdf_path


def _stage_for_player_document_type(document_type: str) -> str:
    mapping = {
        "complaint": "CD",
        "defense": "DD",
        "appeal": "AD",
        "appeal_response": "AR",
    }
    normalized = str(document_type or "").strip()
    return mapping.get(normalized, normalized.upper())


def _record_player_document_confirmation_to_ledger(
    *,
    storage_root: str | Path,
    draft: Any,
    document_payload: dict[str, Any],
) -> None:
    from src.player_lawyer.run_ledger import PlayerRunLedger

    stage = str(document_payload.get("scenario_type") or "").strip()
    if not stage:
        stage = _stage_for_player_document_type(getattr(draft, "document_type", ""))
    result_path = Path(storage_root) / "output" / draft.case_id / f"{stage}_result.json"
    PlayerRunLedger(storage_root=Path(storage_root)).record_document_confirmation(
        case_id=draft.case_id,
        request_id=draft.request_id,
        stage=stage,
        document_type=draft.document_type,
        document_text=draft.document_text,
        result_json_path=str(result_path),
        pdf_path=str(getattr(draft, "pdf_path", "") or document_payload.get("pdf_path", "") or ""),
        confirmed_at=str(getattr(draft, "confirmed_at", "") or ""),
    )


def _build_case_report_transcript(storage_root: Path, case_id: str) -> list[dict[str, Any]]:
    case_output = storage_root / "output" / case_id
    transcript: list[dict[str, Any]] = []
    for path in sorted(case_output.glob("*_result.json")):
        stage = path.stem.replace("_result", "")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        histories = []
        for key in ("dialog_history", "dialogue_history", "conversation", "dialogues"):
            value = payload.get(key)
            if isinstance(value, list):
                histories = value
                break
        for item in histories:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or item.get("message") or item.get("text") or "").strip()
            if not content:
                continue
            transcript.append({
                "stage": stage,
                "speaker": str(item.get("speaker") or item.get("role") or "").strip(),
                "content": content,
            })
    return transcript


def _ensure_player_lawyer_runtime(sandbox: Sandbox) -> tuple[SandboxRuntimeContext, _PlayerInputGateway]:
    context = _get_sandbox_manager().get_or_create_context(sandbox)
    if _player_lawyer_mode_for_context(context) != "plaintiff":
        raise HTTPException(status_code=403, detail="Player-lawyer mode is not enabled")

    gateway = getattr(context, "player_gateway", None)
    if gateway is None:
        gateway = get_or_create_player_gateway(sandbox.id, Path(sandbox.storage_root))
        setattr(context, "player_gateway", gateway)
    orchestrator = getattr(context, "orchestrator", None)
    if orchestrator is not None:
        setattr(orchestrator, "_player_gateway", gateway)
        setattr(orchestrator, "_sandbox_id", sandbox.id)
    return context, gateway


def _build_player_document_case_context(sandbox: Sandbox, case_id: str) -> dict[str, Any]:
    normalized_case_id = _normalize_case_identifier(case_id)
    output_dir = _resolve_case_output_dir_for_sandbox(sandbox, normalized_case_id)
    context: dict[str, Any] = {"case_id": normalized_case_id}
    lc_path = output_dir / "PLC_result.json"
    if not lc_path.exists():
        lc_path = output_dir / "LC_result.json"
    if lc_path.exists():
        try:
            with lc_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            dialog_history = payload.get("dialog_history", [])
            if isinstance(dialog_history, list):
                lines = []
                for entry in dialog_history[-10:]:
                    if not isinstance(entry, dict):
                        continue
                    role = str(entry.get("role", "") or "").strip()
                    content = str(entry.get("content", "") or "").strip()
                    if content:
                        lines.append(f"{role}: {content}")
                if lines:
                    context["consultation_history"] = "\n".join(lines)
        except Exception as exc:
            logger.warning("[PlayerLawyer] Failed to read LC context for %s: %s", normalized_case_id, exc)
    return context


def _get_user_from_access_token(token: str, session) -> User:
    claims = decode_access_token(token)
    user = get_user_by_id(session=session, user_id=claims["user_id"])
    if user.token_version != claims["token_version"]:
        raise AuthError("Token version mismatch")
    return user


def _resolve_sandbox_context_by_id(sandbox_id: str):
    manager = _get_sandbox_manager()
    if hasattr(manager, "_contexts"):
        return getattr(manager, "_contexts").get(sandbox_id)
    if hasattr(manager, "contexts"):
        return getattr(manager, "contexts").get(sandbox_id)
    return None


async def _broadcast_sandbox_event(sandbox_id: str, payload: dict) -> None:
    context = _resolve_sandbox_context_by_id(sandbox_id)
    if context is None:
        return

    disconnected_clients = []
    for client in list(getattr(context, "connected_clients", set())):
        try:
            await client.send_json(payload)
        except Exception:
            disconnected_clients.append(client)

    for client in disconnected_clients:
        context.connected_clients.discard(client)


async def _close_sandbox_realtime_clients(
    context: SandboxRuntimeContext,
    *,
    code: int = 1012,
    reason: str = "sandbox runtime reset",
) -> None:
    """Close stale WebSocket clients before replacing a sandbox runtime context."""
    runtime_engine = getattr(context, "engine", None)
    clients = set(getattr(context, "connected_clients", set()) or set())
    engine_clients = getattr(runtime_engine, "clients", None)
    if engine_clients is not None:
        clients.update(set(engine_clients))

    for client in list(clients):
        try:
            await client.close(code=code, reason=reason)
        except Exception:
            pass
        context.connected_clients.discard(client)
        if engine_clients is not None:
            engine_clients.discard(client)

    supported_clients = getattr(runtime_engine, "_dialogue_gate_supported_clients", None)
    if supported_clients is not None:
        for client in clients:
            supported_clients.discard(client)


def _reset_runtime_engine_state(runtime_engine: WebSocketFrontendEngine | None) -> None:
    if runtime_engine is None:
        return

    runtime_engine._paused = False
    resumed_event = getattr(runtime_engine, "_resumed_event", None)
    if resumed_event is not None and hasattr(resumed_event, "set"):
        resumed_event.set()
    ack_events = getattr(runtime_engine, "_ack_events", None)
    if hasattr(ack_events, "clear"):
        ack_events.clear()
    dialogue_gate_events = getattr(runtime_engine, "_dialogue_gate_events", None)
    if isinstance(dialogue_gate_events, dict):
        for event in list(dialogue_gate_events.values()):
            if hasattr(event, "set"):
                event.set()
        dialogue_gate_events.clear()
    agent_states = getattr(runtime_engine, "_agent_states", None)
    if hasattr(agent_states, "clear"):
        agent_states.clear()
    runtime_engine._active_dialogue_gate_id = None
    runtime_engine._active_dialogue_gate_payload = None
    runtime_engine._buffered_dialogue_message = None


async def _cancel_sandbox_simulation_task(
    context: SandboxRuntimeContext,
    timeout_seconds: float = 2.0,
) -> bool:
    task = context.simulation_task
    if task is not None and not task.done():
        task.cancel()
        try:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            return False

    context.simulation_task = None
    return True


def _initialize_runtime_state(
    *,
    existing_engine: WebSocketFrontendEngine | None = None,
    sandbox_data_dir: Path | None = None,
    set_global_engine: bool = True,
) -> tuple[EventBus, AgentRegistry, CheckpointManager, FileStorageManager, CaseStateMachine]:
    """(Re)build runtime state for startup and full simulation restarts."""
    global engine
    runtime_data_dir = sandbox_data_dir or SANDBOX_DATA_DIR

    loc_registry = load_registry_from_map(MAP_JSON_PATH)
    runtime_engine = existing_engine
    if runtime_engine is None:
        runtime_engine = WebSocketFrontendEngine(
            loc_registry,
            fallback_speed=0.5,
            backend_authoritative=True,
            move_speed_px_per_second=150.0,
            map_json_path=MAP_JSON_PATH,
            frontend_mode=SIMLAW_FRONTEND_MODE,
            turn_mode=SIMLAW_TURN_MODE,
        )
    else:
        runtime_engine.registry = loc_registry
        runtime_engine.loc_registry = loc_registry
        runtime_engine._ack_events.clear()
        runtime_engine._agent_states.clear()
        runtime_engine._paused = False
        runtime_engine._resumed_event.set()
    if set_global_engine:
        engine = runtime_engine

    storage = FileStorageManager(base_dir=runtime_data_dir)
    runtime_event_bus = EventBus()
    fsm = CaseStateMachine(
        runtime_event_bus,
        storage,
        state_change_notifier=getattr(runtime_engine, "broadcast_state_change", None),
    )

    runtime_registry = AgentRegistry(runtime_data_dir, runtime_event_bus, storage, map_engine=runtime_engine)
    skip_global_registry_discovery = sandbox_data_dir is None and _is_user_scoped_sandbox_root(runtime_data_dir)
    if skip_global_registry_discovery:
        logger.info(
            "[Registry] Global registry discovery skipped: user-scoped sandbox root detected at %s",
            runtime_data_dir,
        )
    else:
        runtime_registry.discover_all()

    runtime_engine.agent_registry = runtime_registry
    runtime_engine.storage = storage
    runtime_registry.map_engine = runtime_engine
    for receptionist in runtime_registry.get_agents_by_type("receptionist"):
        receptionist.map_engine = runtime_engine

    runtime_checkpoint_mgr = CheckpointManager(runtime_data_dir / "checkpoints")
    runtime_checkpoint_mgr.set_event_bus(runtime_event_bus)

    orchestrator = ScenarioOrchestrator(
        runtime_registry,
        runtime_event_bus,
        fsm,
        storage,
        runtime_data_dir,
        map_engine=runtime_engine,
        checkpoint_manager=runtime_checkpoint_mgr,
    )
    runtime_engine.orchestrator = orchestrator

    for evt in EventType:
        sim_engine = runtime_engine

        async def _broadcast_event(payload: dict, event_name: str = evt.value) -> None:
            await sim_engine.broadcast_state_change(
                case_id=payload.get("case_id", ""),
                event=event_name,
            )

        runtime_event_bus.subscribe(evt.value, _broadcast_event)

    runtime_engine.restore_state_from_configs()
    return runtime_event_bus, runtime_registry, runtime_checkpoint_mgr, storage, fsm


def _set_engine_paused(paused: bool) -> None:
    if engine is None:
        return
    engine._paused = paused
    if paused:
        engine._resumed_event.clear()
    else:
        engine._resumed_event.set()


async def _sleep_respecting_pause(sim_engine: WebSocketFrontendEngine, duration: float) -> None:
    """Sleep while honoring pause/resume even in tests with lightweight engine doubles."""
    sleep_with_pause = getattr(sim_engine, "_sleep_with_pause", None)
    if callable(sleep_with_pause):
        await sleep_with_pause(duration)
        return

    remaining = max(float(duration), 0.0)
    while remaining > 0:
        if getattr(sim_engine, "_paused", False):
            resumed_event = getattr(sim_engine, "_resumed_event", None)
            if resumed_event is not None:
                await resumed_event.wait()
                continue
        step = min(remaining, 0.01)
        await asyncio.sleep(step)
        remaining -= step


def _get_closed_case_count(sim_event_bus: EventBus) -> int:
    getter = getattr(sim_event_bus, "get_closed_case_count", None)
    if callable(getter):
        return int(getter())
    return len(getattr(sim_event_bus, "_closed_cases", set()))


async def _wait_for_case_close_since(sim_event_bus: EventBus, previous_count: int) -> int:
    waiter = getattr(sim_event_bus, "wait_for_case_close_since", None)
    if callable(waiter):
        return int(await waiter(previous_count))

    while _get_closed_case_count(sim_event_bus) <= previous_count:
        await asyncio.sleep(0.05)
    return _get_closed_case_count(sim_event_bus)


async def _dispatch_case_launch_requests(
    *,
    requests: list[_CaseLaunchRequest],
    sim_engine: WebSocketFrontendEngine,
    sim_event_bus: EventBus,
) -> None:
    if not requests:
        return

    max_concurrent_cases = max(int(MAX_CONCURRENT_CASES or 0), 0)
    if max_concurrent_cases > 0:
        logger.info("案件全局并发上限已启用: %d", max_concurrent_cases)

    launched_case_ids: set[str] = set()
    closed_case_count = _get_closed_case_count(sim_event_bus)

    for index, request in enumerate(requests):
        while max_concurrent_cases > 0:
            active_case_count = max(0, len(launched_case_ids) - _get_closed_case_count(sim_event_bus))
            if active_case_count < max_concurrent_cases:
                break

            logger.info(
                "案件并发已达上限 %d，等待已有案件结案后继续投放",
                max_concurrent_cases,
            )
            closed_case_count = await _wait_for_case_close_since(sim_event_bus, closed_case_count)

        launch_result = await request.launch()
        if launch_result is False:
            logger.warning("案件 %s 启动失败，本次不占用并发槽", request.case_id)
            continue

        launched_case_ids.add(request.case_id)
        closed_case_count = _get_closed_case_count(sim_event_bus)

        if request.post_launch_delay > 0 and index < len(requests) - 1:
            logger.info(
                "案件 %s 已进入待出生队列，%.0f 秒后投放下一案",
                request.case_id.removeprefix("case_"),
                request.post_launch_delay,
            )
            await _sleep_respecting_pause(sim_engine, request.post_launch_delay)
            closed_case_count = _get_closed_case_count(sim_event_bus)


def _count_active_cases(storage: FileStorageManager | None, sim_registry: AgentRegistry | None) -> int:
    if storage is None or sim_registry is None:
        return 0

    active_cases = 0
    counted_case_ids: set[str] = set()
    for client in sim_registry.get_agents_by_type("client"):
        if not client.config_path:
            continue
        try:
            config = storage.load_agent_config(client.config_path)
        except FileNotFoundError:
            continue
        case_id = str(config.get("case_id") or "").strip()
        if not case_id:
            continue
        normalized_case_id = case_id if case_id.startswith("case_") else f"case_{case_id}"
        if normalized_case_id in counted_case_ids:
            continue
        counted_case_ids.add(normalized_case_id)
        try:
            case_runtime = storage.load_case_runtime(normalized_case_id)
        except FileNotFoundError:
            if config.get("party_role") != "plaintiff":
                continue
            case_state = str(config.get("case_state") or "").strip()
        else:
            case_state = str(case_runtime.get("overall_state") or "").strip()
        if case_state not in {"", "空闲", "已结案"}:
            active_cases += 1
    return active_cases


def _build_simulation_status() -> dict:
    session_state = checkpoint_mgr.load_session_state() if checkpoint_mgr else None
    persisted_status = str((session_state or {}).get("simulation_status") or "").strip().lower()
    task_running = _simulation_task is not None and not _simulation_task.done()
    paused = bool(engine and engine._paused)

    if task_running:
        status = "paused" if paused else "running"
    elif persisted_status in {"paused", "running"}:
        status = "paused"
    elif persisted_status == "completed":
        status = "completed"
    else:
        status = "idle"

    return {
        "status": status,
        "session_status": persisted_status or "idle",
        "session_id": (session_state or {}).get("session_id"),
        "paused": status == "paused",
        "simulation_running": task_running,
        "clients_connected": len(engine.clients) if engine else 0,
        "active_cases": _count_active_cases(getattr(engine, "storage", None), registry),
        "can_start": status in {"idle", "paused", "completed"},
        "can_pause": status == "running",
        "can_restart": True,
    }


def _reset_case_client_config(storage: FileStorageManager, agent_dir: Path) -> None:
    config = storage.load_agent_config(agent_dir)
    config.pop("chat_history_summary", None)
    config.pop("long_term_memory", None)
    designated_lawyer_id = str(
        config.get("designated_lawyer_id", "") or config.get("assigned_lawyer_id", "") or ""
    ).strip()

    config["case_state"] = "空闲"
    config["map_state"] = None
    config["designated_lawyer_id"] = designated_lawyer_id
    config["assigned_lawyer_id"] = ""
    storage.save_agent_config(agent_dir, config)
    initialize_client_memory(storage, str(agent_dir))
def _reset_case_runtime(storage: FileStorageManager, case_dir: Path) -> None:
    storage.save_case_runtime(
        case_dir.name,
        {
            "case_id": case_dir.name,
            "overall_state": "空闲",
            "plaintiff_state": "空闲",
            "defendant_state": "空闲",
            "active_party_role": "plaintiff",
        },
    )


def _reset_closed_case_for_restart(storage_root: Path, case_id: str) -> None:
    """Clear closed-case state so it can be restarted without a full sandbox reset."""
    from src.core.file_storage_manager import FileStorageManager

    storage = FileStorageManager(storage_root)
    case_dir = storage_root / "cases" / case_id

    # Reset case-level runtime
    _reset_case_runtime(storage, case_dir)

    # Reset plaintiff / defendant config case_state
    for party in ("plaintiff", "defendant"):
        agent_dir = case_dir / party
        if agent_dir.exists():
            _reset_case_client_config(storage, agent_dir)

    # Remove stale output artifacts for this case
    output_dir = storage_root / "output" / case_id
    if output_dir.exists():
        shutil.rmtree(output_dir)

    logger.info("[Start] 已重置已结案案件 %s，允许重新启动", case_id)


def _reset_lawyer_config(storage: FileStorageManager, agent_dir: Path) -> None:
    config = storage.load_agent_config(agent_dir)
    config.pop("chat_history_summary", None)
    config.pop("long_term_memory", None)
    config["current_handling_case"] = None
    config["case_queue"] = []
    config["map_state"] = None
    storage.save_agent_config(agent_dir, config)
    initialize_lawyer_memory(storage, str(agent_dir))
def _reset_judge_config(storage: FileStorageManager, agent_dir: Path) -> None:
    config = storage.load_agent_config(agent_dir)
    config.pop("chat_history_summary", None)
    config["current_handling_case"] = None
    config["case_queue"] = []
    config["map_state"] = None
    storage.save_agent_config(agent_dir, config)


def _reset_receptionist_config(storage: FileStorageManager, firm_dir: Path) -> None:
    config = storage.load_agent_config(firm_dir)
    config["map_state"] = None
    storage.save_agent_config(firm_dir, config)


def _reset_simulation_storage(storage: FileStorageManager) -> None:
    cases_dir = SANDBOX_DATA_DIR / "cases"
    if cases_dir.exists():
        for case_dir in sorted(cases_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            for party_role in ("plaintiff", "defendant"):
                party_dir = case_dir / party_role
                if (party_dir / "config.yaml").exists():
                    _reset_case_client_config(storage, party_dir)
            _reset_case_runtime(storage, case_dir)

    firms_dir = SANDBOX_DATA_DIR / "law_firms"
    if firms_dir.exists():
        for firm_dir in sorted(firms_dir.iterdir()):
            if not firm_dir.is_dir():
                continue
            if (firm_dir / "config.yaml").exists():
                _reset_receptionist_config(storage, firm_dir)
            lawyers_dir = firm_dir / "lawyers"
            if not lawyers_dir.exists():
                continue
            for lawyer_dir in sorted(lawyers_dir.iterdir()):
                if (lawyer_dir / "config.yaml").exists():
                    _reset_lawyer_config(storage, lawyer_dir)

    court_dir = SANDBOX_DATA_DIR / "court_system"
    if court_dir.exists():
        for court_level_dir in sorted(court_dir.iterdir()):
            judges_dir = court_level_dir / "judges"
            if not judges_dir.exists():
                continue
            for judge_dir in sorted(judges_dir.iterdir()):
                if (judge_dir / "config.yaml").exists():
                    _reset_judge_config(storage, judge_dir)

    checkpoint_dir = SANDBOX_DATA_DIR / "checkpoints"
    if checkpoint_dir.exists():
        for checkpoint_file in checkpoint_dir.glob("*.yaml"):
            checkpoint_file.unlink()

    output_dir = SANDBOX_DATA_DIR / "output"
    if output_dir.exists():
        for child in output_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()


async def _cancel_simulation_task(timeout_seconds: float = 2.0) -> bool:
    global _simulation_task

    if _simulation_task and not _simulation_task.done():
        _simulation_task.cancel()
        try:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(_simulation_task, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            return False
    _simulation_task = None
    return True


async def _start_or_resume_simulation() -> dict:
    global _simulation_task

    if (
        engine is None
        or event_bus is None
        or registry is None
        or checkpoint_mgr is None
        or storage_manager is None
        or case_fsm is None
    ):
        raise HTTPException(status_code=503, detail="Simulation engine not ready")

    if _simulation_task and not _simulation_task.done():
        _set_engine_paused(False)
        checkpoint_mgr.mark_session_running()
        return _build_simulation_status()

    session_state = checkpoint_mgr.load_session_state()
    _set_engine_paused(False)

    if session_state and session_state.get("simulation_status") in {"running", "paused"}:
        checkpoint_mgr.mark_session_running()
        _simulation_task = asyncio.create_task(
            resume_simulation(engine, event_bus, registry, storage_manager, case_fsm, checkpoint_mgr)
        )
    else:
        checkpoint_mgr.create_new_session()
        _simulation_task = asyncio.create_task(
            run_simulation(engine, event_bus, registry, storage_manager, case_fsm, checkpoint_mgr)
        )
    _simulation_task.add_done_callback(_log_task_result)
    return _build_simulation_status()


# ── WebSocket 端点 ──

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    token = str(websocket.query_params.get("token") or "").strip()
    if not token:
        auth_header = str(websocket.headers.get("authorization") or "").strip()
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
    if not token:
        await websocket.close(code=4401)
        return

    try:
        with get_db_session(_get_session_factory()) as session:
            current_user = _get_user_from_access_token(token, session)
            sandbox = _require_user_sandbox(session, current_user)
    except (AuthError, UserNotFoundError, HTTPException):
        await websocket.close(code=4401)
        return

    context = _get_sandbox_manager().get_or_create_context(sandbox)
    runtime_engine = getattr(context, "engine", None)

    await websocket.accept()
    context.connected_clients.add(websocket)
    add_client = getattr(runtime_engine, "add_client", None)
    if callable(add_client):
        await add_client(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "client_logout":
                runtime_status = _get_sandbox_manager().pause_sandbox(sandbox)
                try:
                    with get_db_session(_get_session_factory()) as session:
                        current_sandbox = _require_user_sandbox(session, current_user)
                        _update_sandbox_from_runtime_status(session, current_sandbox, runtime_status)
                except Exception as exc:
                    logger.warning("Failed to persist sandbox pause on websocket logout: %s", exc)
                await websocket.send_json({
                    "type": "client_logout_ack",
                    "status": runtime_status.get("status", "paused"),
                })
                continue

            if data.get("type") == "player_lawyer_response":
                if _player_lawyer_mode_for_context(context) != "plaintiff":
                    await websocket.send_json({
                        "type": "player_lawyer_error",
                        "error": "Player-lawyer mode is not enabled",
                    })
                    continue

                body = data.get("data") if isinstance(data.get("data"), dict) else data
                request_id = str(body.get("request_id", "") or "").strip()
                message = str(body.get("message", "") or "").strip()
                if not request_id or not message:
                    await websocket.send_json({
                        "type": "player_lawyer_error",
                        "error": "request_id and message are required",
                    })
                    continue

                gateway = getattr(context, "player_gateway", None)
                if gateway is None:
                    gateway = get_or_create_player_gateway(sandbox.id, Path(sandbox.storage_root))
                    setattr(context, "player_gateway", gateway)
                try:
                    resolved = gateway.resolve(request_id, message)
                except ValueError as exc:
                    await websocket.send_json({"type": "player_lawyer_error", "error": str(exc)})
                    continue
                except RuntimeError as exc:
                    await websocket.send_json({"type": "player_lawyer_error", "error": str(exc)})
                    continue

                await _broadcast_sandbox_event(
                    str(sandbox.id),
                    {
                        "type": "player_lawyer_input_submitted",
                        "event": "player_lawyer_input_submitted",
                        "data": resolved.to_dict(),
                    },
                )
                continue

            on_frontend_message = getattr(runtime_engine, "on_frontend_message", None)
            if callable(on_frontend_message):
                await on_frontend_message(data, websocket)
    except WebSocketDisconnect:
        pass
    finally:
        context.connected_clients.discard(websocket)
        remove_client = getattr(runtime_engine, "remove_client", None)
        if callable(remove_client):
            await remove_client(websocket)


# ── REST 端点（少量查询用） ──

@app.post("/api/auth/register")
async def register_auth(payload: AuthRequest, session=Depends(_db_session_dependency)):
    try:
        user = register_user(session=session, email=payload.email, password=payload.password)
    except InvalidAuthInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UserAlreadyExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return _build_auth_response(user)


@app.post("/api/auth/login")
async def login_auth(payload: AuthRequest, session=Depends(_db_session_dependency)):
    try:
        user = authenticate_user(session=session, email=payload.email, password=payload.password)
    except InvalidAuthInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except InvalidCredentialsError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    return _build_auth_response(user)


@app.post("/api/auth/refresh")
async def refresh_auth(current_user: User = Depends(_get_current_user)):
    return _build_auth_response(current_user)


@app.get("/api/auth/me")
async def auth_me(current_user: User = Depends(_get_current_user)):
    return _serialize_user(current_user)


@app.get("/api/sandbox")
async def get_current_sandbox(
    current_user: User = Depends(_get_current_user),
    session=Depends(_db_session_dependency),
):
    sandbox = sandbox_service.get_user_sandbox(session=session, user_id=current_user.id)
    if sandbox is None:
        return _serialize_sandbox_state(None)

    runtime_status = _get_sandbox_manager().get_status(sandbox)
    return _serialize_sandbox_state(sandbox, runtime_status=runtime_status)


@app.get("/api/sandbox/cases")
async def get_sandbox_cases(
    current_user: User = Depends(_get_current_user),
    session=Depends(_db_session_dependency),
):
    sandbox = sandbox_service.get_or_create_user_sandbox(session=session, user_id=current_user.id)
    context = _resolve_sandbox_context_by_id(sandbox.id)
    runtime_status = _build_sandbox_runtime_status(context) if context is not None else None
    selected_case_id = _get_context_selected_case_id(context)
    entries = _build_case_picker_entries(
        storage_root=Path(sandbox.storage_root),
        runtime_status=runtime_status,
        selected_case_id=selected_case_id,
    )
    return {"cases": entries}


@app.get("/api/sandbox/cases/{case_id}/documents")
async def get_sandbox_case_documents(
    case_id: str,
    current_user: User = Depends(_get_current_user),
    session=Depends(_db_session_dependency),
):
    sandbox = _require_user_sandbox(session, current_user)
    normalized_case_id = _normalize_case_identifier(case_id)
    _require_sandbox_case_entry(sandbox, normalized_case_id)
    return {
        "case_id": normalized_case_id,
        "documents": _list_case_document_entries(sandbox, normalized_case_id),
    }


@app.get("/api/sandbox/cases/{case_id}/documents/{document_key}/download")
async def download_sandbox_case_document(
    case_id: str,
    document_key: str,
    current_user: User = Depends(_get_current_user),
    session=Depends(_db_session_dependency),
):
    sandbox = _require_user_sandbox(session, current_user)
    normalized_case_id = _normalize_case_identifier(case_id)
    _require_sandbox_case_entry(sandbox, normalized_case_id)
    _, resolved_pdf_path = _resolve_case_document_download_path(
        sandbox=sandbox,
        case_id=normalized_case_id,
        document_key=document_key,
    )
    return FileResponse(
        resolved_pdf_path,
        media_type="application/pdf",
        filename=resolved_pdf_path.name,
    )


@app.get("/api/sandbox/cases/{case_id}/closing-summary")
async def get_sandbox_case_closing_summary(
    case_id: str,
    current_user: User = Depends(_get_current_user),
    session=Depends(_db_session_dependency),
):
    from src.player_lawyer.closing_summary import PlayerClosingSummaryService

    sandbox = _require_user_sandbox(session, current_user)
    normalized_case_id = _normalize_case_identifier(case_id)
    case_entry = _require_sandbox_case_entry(sandbox, normalized_case_id)
    documents = _list_case_document_entries(sandbox, normalized_case_id)
    service = PlayerClosingSummaryService(storage_root=Path(sandbox.storage_root))
    return service.build_summary(
        case_id=normalized_case_id,
        case_entry=case_entry,
        documents=documents,
    )


@app.post("/api/sandbox/cases/{case_id}/closing-evaluation")
async def create_sandbox_case_closing_evaluation(
    case_id: str,
    current_user: User = Depends(_get_current_user),
    session=Depends(_db_session_dependency),
):
    from src.player_lawyer.closing_summary import PlayerClosingSummaryService

    sandbox = _require_user_sandbox(session, current_user)
    normalized_case_id = _normalize_case_identifier(case_id)
    case_entry = _require_sandbox_case_entry(sandbox, normalized_case_id)
    documents = _list_case_document_entries(sandbox, normalized_case_id)
    service = PlayerClosingSummaryService(storage_root=Path(sandbox.storage_root))
    try:
        evaluation = await asyncio.to_thread(
            service.generate_evaluation,
            case_id=normalized_case_id,
            case_entry=case_entry,
            documents=documents,
        )
    except Exception as exc:
        logger.warning("[ClosingEvaluation] Failed for %s: %s", normalized_case_id, exc, exc_info=True)
        raise HTTPException(status_code=502, detail="结案评价生成失败，请稍后重试") from exc
    return {"success": True, "evaluation": evaluation}


@app.get("/api/sandbox/cases/{case_id}/player-run-ledger")
async def get_sandbox_case_player_run_ledger(
    case_id: str,
    current_user: User = Depends(_get_current_user),
    session=Depends(_db_session_dependency),
):
    from src.player_lawyer.run_ledger import PlayerRunLedger

    sandbox = _require_user_sandbox(session, current_user)
    normalized_case_id = _normalize_case_identifier(case_id)
    _require_sandbox_case_entry(sandbox, normalized_case_id)
    return PlayerRunLedger(storage_root=Path(sandbox.storage_root)).load(normalized_case_id)


@app.get("/api/sandbox/cases/{case_id}/player-run-report.md")
async def download_sandbox_case_player_run_report(
    case_id: str,
    current_user: User = Depends(_get_current_user),
    session=Depends(_db_session_dependency),
):
    from fastapi.responses import Response
    from src.player_lawyer.run_ledger import PlayerRunLedger

    sandbox = _require_user_sandbox(session, current_user)
    normalized_case_id = _normalize_case_identifier(case_id)
    case_entry = _require_sandbox_case_entry(sandbox, normalized_case_id)
    documents = _list_case_document_entries(sandbox, normalized_case_id)
    ledger = PlayerRunLedger(storage_root=Path(sandbox.storage_root))
    markdown = ledger.build_markdown_report(
        case_id=normalized_case_id,
        case_entry=case_entry,
        documents=documents,
        transcript=_build_case_report_transcript(Path(sandbox.storage_root), normalized_case_id),
        evaluation=ledger.load(normalized_case_id).get("evaluation"),
    )
    return Response(
        content=markdown,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{normalized_case_id}-player-run-report.md"',
        },
    )


@app.get("/api/sandbox/player-lawyer/document-skills")
async def get_player_lawyer_document_skills(
    current_user: User = Depends(_get_current_user),
    session=Depends(_db_session_dependency),
):
    from src.player_lawyer.document_assist import PlayerDocumentAssistService

    sandbox = sandbox_service.get_or_create_user_sandbox(session=session, user_id=current_user.id)
    _ensure_player_lawyer_runtime(sandbox)
    service = PlayerDocumentAssistService(storage_root=Path(sandbox.storage_root))
    return {"skills": service.list_skills()}


@app.post("/api/sandbox/player-lawyer/document-followup")
async def create_player_lawyer_document_followup(
    payload: PlayerDocumentFollowupRequest,
    current_user: User = Depends(_get_current_user),
    session=Depends(_db_session_dependency),
):
    if not _is_player_plaintiff_mode():
        raise HTTPException(status_code=403, detail="Player-lawyer mode is not enabled")
    message = str(payload.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required.")

    sandbox = sandbox_service.get_or_create_user_sandbox(session=session, user_id=current_user.id)
    context, gateway = _ensure_player_lawyer_runtime(sandbox)
    req = gateway.get_request(payload.request_id)
    if req is None:
        raise HTTPException(status_code=404, detail=f"Unknown request_id: {payload.request_id}")
    if str(req.status.value if hasattr(req.status, "value") else req.status) != "pending":
        raise HTTPException(status_code=409, detail="当前文书任务已经结束，不能继续追问")
    if str(req.stage or "").upper() not in {"CD", "DD", "AD", "AR"}:
        raise HTTPException(status_code=409, detail="当前任务不是可追问的文书阶段")

    orchestrator = getattr(context, "orchestrator", None)
    followup_handler = getattr(orchestrator, "handle_player_document_followup", None)
    if not callable(followup_handler):
        raise HTTPException(status_code=409, detail="当前文书任务没有可追问的当事人会话")
    try:
        followup = await followup_handler(request_id=payload.request_id, message=message, request=req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    from src.player_lawyer.run_ledger import PlayerRunLedger

    PlayerRunLedger(storage_root=Path(sandbox.storage_root)).record_followup(
        case_id=req.case_id,
        request_id=req.request_id,
        stage=req.stage,
        question=followup.get("question", message),
        answer=followup.get("answer", ""),
    )
    return {
        "success": True,
        "request": req.to_dict(),
        "question": followup.get("question", message),
        "answer": followup.get("answer", ""),
    }


@app.post("/api/sandbox/player-lawyer/document-assist")
async def create_player_lawyer_document_draft(
    payload: PlayerDocumentAssistRequest,
    current_user: User = Depends(_get_current_user),
    session=Depends(_db_session_dependency),
):
    from src.player_lawyer.document_assist import PlayerDocumentAssistService

    if not _is_player_plaintiff_mode():
        raise HTTPException(status_code=403, detail="Player-lawyer mode is not enabled")
    sandbox = sandbox_service.get_or_create_user_sandbox(session=session, user_id=current_user.id)
    _ensure_player_lawyer_runtime(sandbox)
    normalized_case_id = _normalize_case_identifier(payload.case_id)
    _require_sandbox_case_entry(sandbox, normalized_case_id)

    service = PlayerDocumentAssistService(storage_root=Path(sandbox.storage_root))
    try:
        draft = service.create_draft(
            sandbox_id=int(sandbox.id) if str(sandbox.id).isdigit() else 0,
            request_id=str(payload.request_id or ""),
            case_id=normalized_case_id,
            document_type=payload.document_type,
            skill_id=payload.skill_id,
            player_prompt=payload.player_prompt,
            player_draft=payload.player_draft,
            case_context=_build_player_document_case_context(sandbox, normalized_case_id),
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    await _broadcast_sandbox_event(
        str(sandbox.id),
        {
            "type": "player_lawyer_document_draft_ready",
            "event": "player_lawyer_document_draft_ready",
            "data": draft.to_dict(),
        },
    )
    return {"success": True, "draft": draft.to_dict()}


@app.post("/api/sandbox/player-lawyer/documents/{draft_id}/confirm")
async def confirm_player_lawyer_document_draft(
    draft_id: str,
    payload: PlayerDocumentConfirmRequest,
    current_user: User = Depends(_get_current_user),
    session=Depends(_db_session_dependency),
):
    from src.player_lawyer.document_assist import PlayerDocumentAssistService

    if not _is_player_plaintiff_mode():
        raise HTTPException(status_code=403, detail="Player-lawyer mode is not enabled")
    sandbox = sandbox_service.get_or_create_user_sandbox(session=session, user_id=current_user.id)
    _context, gateway = _ensure_player_lawyer_runtime(sandbox)
    service = PlayerDocumentAssistService(storage_root=Path(sandbox.storage_root))
    try:
        draft, document_payload = service.confirm_draft(
            draft_id=draft_id,
            document_text=payload.document_text,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _record_player_document_confirmation_to_ledger(
        storage_root=sandbox.storage_root,
        draft=draft,
        document_payload=document_payload,
    )
    if draft.request_id:
        try:
            req = gateway.resolve(draft.request_id, draft.document_text)
        except (ValueError, RuntimeError):
            req = None
        if req is not None:
            await _broadcast_sandbox_event(
                str(sandbox.id),
                {
                    "type": "player_lawyer_input_submitted",
                    "event": "player_lawyer_input_submitted",
                    "data": req.to_dict(),
                },
            )

    await _broadcast_sandbox_event(
        str(sandbox.id),
        {
            "type": "player_lawyer_document_confirmed",
            "event": "player_lawyer_document_confirmed",
            "data": {
                "draft": draft.to_dict(),
                "document_payload": document_payload,
            },
        },
    )
    asyncio.create_task(_publish_player_document_completion_if_unmanaged(_context, draft))
    return {
        "success": True,
        "draft": draft.to_dict(),
        "document_payload": document_payload,
    }


@app.post("/api/sandbox/player-lawyer/documents/confirm-manual")
async def confirm_player_lawyer_manual_document(
    payload: PlayerDocumentManualConfirmRequest,
    current_user: User = Depends(_get_current_user),
    session=Depends(_db_session_dependency),
):
    from src.player_lawyer.document_assist import PlayerDocumentAssistService

    if not _is_player_plaintiff_mode():
        raise HTTPException(status_code=403, detail="Player-lawyer mode is not enabled")
    sandbox = sandbox_service.get_or_create_user_sandbox(session=session, user_id=current_user.id)
    _context, gateway = _ensure_player_lawyer_runtime(sandbox)
    normalized_case_id = _normalize_case_identifier(payload.case_id)
    _require_sandbox_case_entry(sandbox, normalized_case_id)

    service = PlayerDocumentAssistService(storage_root=Path(sandbox.storage_root))
    try:
        draft, document_payload = service.confirm_manual_document(
            sandbox_id=int(sandbox.id) if str(sandbox.id).isdigit() else 0,
            request_id=str(payload.request_id or ""),
            case_id=normalized_case_id,
            document_type=payload.document_type,
            document_text=payload.document_text,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _record_player_document_confirmation_to_ledger(
        storage_root=sandbox.storage_root,
        draft=draft,
        document_payload=document_payload,
    )
    if draft.request_id:
        try:
            req = gateway.resolve(draft.request_id, draft.document_text)
        except (ValueError, RuntimeError):
            req = None
        if req is not None:
            await _broadcast_sandbox_event(
                str(sandbox.id),
                {
                    "type": "player_lawyer_input_submitted",
                    "event": "player_lawyer_input_submitted",
                    "data": req.to_dict(),
                },
            )

    await _broadcast_sandbox_event(
        str(sandbox.id),
        {
            "type": "player_lawyer_document_confirmed",
            "event": "player_lawyer_document_confirmed",
            "data": {
                "draft": draft.to_dict(),
                "document_payload": document_payload,
            },
        },
    )
    asyncio.create_task(_publish_player_document_completion_if_unmanaged(_context, draft))
    return {
        "success": True,
        "draft": draft.to_dict(),
        "document_payload": document_payload,
    }


@app.post("/api/sandbox/start")
async def start_current_sandbox(
    payload: dict[str, Any] | None = None,
    current_user: User = Depends(_get_current_user),
    session=Depends(_db_session_dependency),
):
    normalized_case_id = _normalize_case_identifier((payload or {}).get("case_id"))
    if not normalized_case_id:
        raise HTTPException(status_code=400, detail="缺少 case_id")

    sandbox = sandbox_service.get_or_create_user_sandbox(session=session, user_id=current_user.id)
    context = _resolve_sandbox_context_by_id(sandbox.id)
    runtime_status = _build_sandbox_runtime_status(context) if context is not None else None
    selected_case_id = _get_context_selected_case_id(context)
    entries = _build_case_picker_entries(
        storage_root=Path(sandbox.storage_root),
        runtime_status=runtime_status,
        selected_case_id=selected_case_id,
    )
    case_entry = _find_case_picker_entry(entries, normalized_case_id)
    if case_entry is None:
        raise HTTPException(status_code=404, detail="案件不存在")
    if case_entry["status"] == "closed":
        _reset_closed_case_for_restart(Path(sandbox.storage_root), normalized_case_id)

    manager = _get_sandbox_manager()
    context = manager.get_or_create_context(sandbox)
    current_runtime_status = _build_sandbox_runtime_status(context)
    current_selected_case_id = _get_context_selected_case_id(context)
    if current_runtime_status["status"] == "error":
        raise HTTPException(status_code=409, detail="当前模拟处于运行异常，请先重新开始")
    if current_runtime_status["status"] in {"running", "paused"} and not current_selected_case_id:
        raise HTTPException(status_code=409, detail="当前已有案件在运行，请先等待其结束或重新开始")
    if current_runtime_status["status"] in {"running", "paused"} and current_selected_case_id and current_selected_case_id != normalized_case_id:
        raise HTTPException(status_code=409, detail="当前已有其他案件在运行，请先等待其结束或重新开始")
    if _sandbox_context_needs_rebuild(context, Path(sandbox.storage_root)):
        await _close_sandbox_realtime_clients(context)
        manager.reset_context(sandbox)
    runtime_status = manager.start_sandbox(sandbox, payload={"case_id": normalized_case_id})
    _update_sandbox_from_runtime_status(session, sandbox, runtime_status)
    return {"success": True, "sandbox": _serialize_sandbox_state(sandbox, runtime_status=runtime_status)}


@app.post("/api/sandbox/ensure")
async def ensure_current_sandbox(
    current_user: User = Depends(_get_current_user),
    session=Depends(_db_session_dependency),
):
    sandbox = sandbox_service.get_or_create_user_sandbox(session=session, user_id=current_user.id)
    runtime_status = None
    if _resolve_sandbox_context_by_id(sandbox.id) is not None:
        runtime_status = _get_sandbox_manager().get_status(sandbox)
    return {"success": True, "sandbox": _serialize_sandbox_state(sandbox, runtime_status=runtime_status)}


@app.post("/api/sandbox/pause")
async def pause_current_sandbox(
    current_user: User = Depends(_get_current_user),
    session=Depends(_db_session_dependency),
):
    sandbox = _require_user_sandbox(session, current_user)
    runtime_status = _get_sandbox_manager().pause_sandbox(sandbox)
    _update_sandbox_from_runtime_status(session, sandbox, runtime_status)
    return {"success": True, "sandbox": _serialize_sandbox_state(sandbox, runtime_status=runtime_status)}


@app.post("/api/sandbox/restart")
async def restart_current_sandbox(
    current_user: User = Depends(_get_current_user),
    session=Depends(_db_session_dependency),
):
    sandbox = _require_user_sandbox(session, current_user)
    manager = _get_sandbox_manager()
    context = manager.get_or_create_context(sandbox)
    reset_player_gateway(sandbox.id)
    cancelled = await _cancel_sandbox_simulation_task(context)
    if not cancelled:
        raise HTTPException(
            status_code=409,
            detail="当前仍有模型调用未退出，请先暂停并等待几秒后再重跑",
        )

    _reset_runtime_engine_state(getattr(context, "engine", None))
    await _close_sandbox_realtime_clients(context)
    sandbox_service.reset_sandbox_storage(Path(sandbox.storage_root))
    manager.reset_context(sandbox)
    runtime_status = {
        "status": "idle",
        "paused": False,
        "simulation_running": False,
        "clients_connected": 0,
        "active_cases": 0,
    }
    _update_sandbox_from_runtime_status(session, sandbox, runtime_status)
    return {
        "success": True,
        "reload_required": True,
        "sandbox": _serialize_sandbox_state(sandbox, runtime_status=runtime_status),
    }


@app.get("/api/agents")
async def list_agents(
    current_user: User | None = Depends(_get_optional_current_user),
    session=Depends(_db_session_dependency),
):
    """返回 Agent 列表，优先使用当前用户 sandbox 的 runtime registry。"""
    runtime_registry = registry
    runtime_storage = storage_manager

    if current_user is not None:
        sandbox = sandbox_service.get_or_create_user_sandbox(session=session, user_id=current_user.id)
        context = _get_sandbox_manager().get_or_create_context(sandbox)
        runtime_registry = getattr(context, "registry", None) or runtime_registry
        runtime_storage = getattr(context, "storage_manager", None) or runtime_storage

    return _serialize_registry_agents(runtime_registry, runtime_storage)


@app.get("/api/status")
async def server_status():
    """服务器状态。"""
    return {
        "status": "running",
        "backend_version": BACKEND_VERSION,
        "backend_version_time": BACKEND_VERSION_TIME,
        "backend_version_label": BACKEND_VERSION_LABEL,
        "clients_connected": len(engine.clients) if engine else 0,
        "simulation_running": _simulation_task is not None and not _simulation_task.done(),
    }


@app.get("/api/runtime-tech-catalog")
async def runtime_tech_catalog():
    """返回前端展示用的运行时 Tool / Skill 能力目录。"""
    return build_runtime_tech_catalog()


@app.get("/api/debug/runtime-issues")
async def debug_runtime_issues():
    latest = _runtime_issues[0] if _runtime_issues else None
    return {
        "issues": list(_runtime_issues),
        "latest": latest,
        "has_errors": any(issue["level"] == "ERROR" for issue in _runtime_issues),
    }


@app.get("/api/debug/runtime-config")
async def debug_runtime_config(current_user: User = Depends(_get_current_user)):
    _ = current_user
    return {
        "success": True,
        "config": _read_runtime_config(),
        "restart_required": False,
        "restart_hint": "修改后如需整套后端环境按新配置重启，请点“保存并重启后端”。",
    }


@app.post("/api/debug/runtime-config")
async def update_debug_runtime_config(
    payload: RuntimeConfigRequest,
    current_user: User = Depends(_get_current_user),
):
    _ = current_user
    config = _normalize_runtime_config(payload)
    _apply_runtime_config(config)
    return {
        "success": True,
        "config": _read_runtime_config(),
        "restart_required": False,
        "restart_hint": "当前后端进程和 .env 已同步；如果你想按 Docker 环境重启一遍，请点“保存并重启后端”。",
    }


@app.post("/api/debug/runtime-config/restart")
async def update_debug_runtime_config_and_restart(
    payload: RuntimeConfigRequest,
    current_user: User = Depends(_get_current_user),
):
    _ = current_user
    config = _normalize_runtime_config(payload)
    _apply_runtime_config(config)
    _schedule_backend_restart()
    return {
        "success": True,
        "config": _read_runtime_config(),
        "restart_required": True,
        "restart_pending": True,
        "message": "配置已保存，后端正在重启。页面会短暂断开，几秒后刷新即可。",
    }


@app.get("/debug")
async def debug_console_page():
    return FileResponse(DEBUG_UI_DIR / "index.html")


@app.get("/api/simulation/status")
async def simulation_status():
    """模拟状态。"""
    return _build_simulation_status()


@app.post("/api/simulation/start")
async def start_simulation():
    """开始或恢复模拟。"""
    status = await _start_or_resume_simulation()
    return {"success": True, "simulation": status}


@app.post("/api/simulation/pause")
async def pause_simulation():
    """暂停模拟。"""
    if checkpoint_mgr is None:
        raise HTTPException(status_code=503, detail="Simulation engine not ready")

    _set_engine_paused(True)
    checkpoint_mgr.mark_session_paused()
    return {"success": True, "simulation": _build_simulation_status()}


@app.post("/api/simulation/restart")
async def restart_simulation():
    """重置模拟进度并回到待启动状态。"""
    global event_bus, registry, checkpoint_mgr, storage_manager, case_fsm

    if storage_manager is None or checkpoint_mgr is None:
        raise HTTPException(status_code=503, detail="Simulation engine not ready")

    cancelled = await _cancel_simulation_task()
    if not cancelled:
        raise HTTPException(
            status_code=409,
            detail="当前仍有模型调用未退出，请先暂停并等待几秒后再重跑",
        )
    _set_engine_paused(False)
    _reset_simulation_storage(storage_manager)
    if engine is None:
        raise HTTPException(status_code=503, detail="Simulation engine not ready")

    (
        event_bus,
        registry,
        checkpoint_mgr,
        storage_manager,
        case_fsm,
    ) = _initialize_runtime_state(existing_engine=engine)
    engine.agent_registry = registry
    engine.storage = storage_manager
    engine._agent_states.clear()
    engine._ack_events.clear()

    return {
        "success": True,
        "reload_required": True,
        "simulation": _build_simulation_status(),
    }


# ── 模拟流程 ──

def _log_task_result(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        return
    if exc is not None:
        logger.error("Simulation task crashed: %s", exc, exc_info=exc)


def _find_case_client_agent(sim_registry, storage, case_id: str, party_role: str):
    for client in sim_registry.get_agents_by_type("client"):
        if not client.config_path:
            continue
        try:
            config = storage.load_agent_config(client.config_path)
        except Exception:
            continue
        if f"case_{config.get('case_id', '')}" != case_id:
            continue
        if config.get("party_role", "plaintiff") != party_role:
            continue
        return client, client.config_path, config
    return None, "", {}


def _build_first_instance_resume_payload(case_id: str, anchor_client, sim_registry, storage):
    case_dir = Path(anchor_client.config_path).parent
    plaintiff_path = case_dir / "plaintiff"
    defendant_path = case_dir / "defendant"
    plaintiff_agent, _, plaintiff_config = _find_case_client_agent(sim_registry, storage, case_id, "plaintiff")
    defendant_agent, _, defendant_config = _find_case_client_agent(sim_registry, storage, case_id, "defendant")
    judges = sim_registry.get_agents_by_type("judge")
    judge = next((j for j in judges if getattr(j, "court_level", "") == "basic"), None)
    plaintiff_lawyer_id = plaintiff_config.get("assigned_lawyer_id", "")
    defendant_lawyer_id = defendant_config.get("assigned_lawyer_id", "")

    if not plaintiff_agent or not defendant_agent or not judge:
        logger.error(
            "Case %s missing first-instance participants: plaintiff=%s, defendant=%s, judge=%s",
            case_id,
            getattr(plaintiff_agent, "agent_id", ""),
            getattr(defendant_agent, "agent_id", ""),
            getattr(judge, "agent_id", ""),
        )
        return None

    if not plaintiff_lawyer_id or not defendant_lawyer_id:
        logger.error(
            "Case %s missing first-instance lawyers: plaintiff_lawyer=%s, defendant_lawyer=%s",
            case_id,
            plaintiff_lawyer_id,
            defendant_lawyer_id,
        )
        return None

    return {
        "case_id": case_id,
        "client_path": str(plaintiff_path),
        "plaintiff_id": plaintiff_agent.agent_id,
        "defendant_id": defendant_agent.agent_id,
        "plaintiff_lawyer_id": plaintiff_lawyer_id,
        "defendant_lawyer_id": defendant_lawyer_id,
        "judge_id": judge.agent_id,
    }


def _build_second_instance_resume_payload(case_id: str, anchor_client, sim_registry, storage):
    case_dir = Path(anchor_client.config_path).parent
    plaintiff_path = case_dir / "plaintiff"
    plaintiff_agent, _, plaintiff_config = _find_case_client_agent(sim_registry, storage, case_id, "plaintiff")
    defendant_agent, _, defendant_config = _find_case_client_agent(sim_registry, storage, case_id, "defendant")
    judges = sim_registry.get_agents_by_type("judge")
    judge = next((j for j in judges if getattr(j, "court_level", "") == "intermediate"), None)
    plaintiff_lawyer_id = plaintiff_config.get("assigned_lawyer_id", "")
    defendant_lawyer_id = defendant_config.get("assigned_lawyer_id", "")

    if not plaintiff_agent or not defendant_agent or not judge:
        logger.error(
            "Case %s missing second-instance participants: plaintiff=%s, defendant=%s, judge=%s",
            case_id,
            getattr(plaintiff_agent, "agent_id", ""),
            getattr(defendant_agent, "agent_id", ""),
            getattr(judge, "agent_id", ""),
        )
        return None

    if not plaintiff_lawyer_id or not defendant_lawyer_id:
        logger.error(
            "Case %s missing second-instance lawyers: plaintiff_lawyer=%s, defendant_lawyer=%s",
            case_id,
            plaintiff_lawyer_id,
            defendant_lawyer_id,
        )
        return None

    return {
        "case_id": case_id,
        "client_path": str(plaintiff_path),
        "plaintiff_id": plaintiff_agent.agent_id,
        "defendant_id": defendant_agent.agent_id,
        "plaintiff_lawyer_id": plaintiff_lawyer_id,
        "defendant_lawyer_id": defendant_lawyer_id,
        "judge_id": judge.agent_id,
    }


def _get_available_firms(sim_registry) -> list[str]:
    firms = [str(firm_id) for firm_id in sim_registry._firms.keys() if str(firm_id)]
    return firms or ["law_firm_A", "law_firm_B"]


def _resolve_map_prefix_from_firm(firm_id: str) -> str:
    key = str(firm_id or "").strip().lower()
    if key in {"law_firm_b", "lawfirmb"}:
        return "lawfirmB"
    return "lawfirmA"


def _resolve_birth_location_for_firm(firm_id: str) -> str:
    return "birth_locationB" if _resolve_map_prefix_from_firm(firm_id).lower().endswith("b") else "birth_locationA"


def _find_case_party_client(
    sim_registry: AgentRegistry,
    storage: FileStorageManager,
    case_id: str,
    party_role: str,
):
    normalized_case_id = str(case_id or "").removeprefix("case_")
    target_role = str(party_role or "").strip().lower()
    if not normalized_case_id or not target_role:
        return None

    for client in sim_registry.get_agents_by_type("client"):
        config_path = getattr(client, "config_path", None)
        if not config_path:
            continue

        try:
            config = storage.load_agent_config(config_path)
        except Exception:
            continue

        if str(config.get("case_id", "") or "").strip() != normalized_case_id:
            continue
        if str(config.get("party_role", "") or "").strip().lower() != target_role:
            continue
        return client

    return None


def _get_or_assign_character_name(agent, storage: FileStorageManager | None = None) -> str:
    configured = str(getattr(agent, "character_name", "") or "").strip()
    if configured:
        return configured

    config_path = getattr(agent, "config_path", None)
    if storage and config_path:
        try:
            config = storage.load_agent_config(config_path)
            configured = str(config.get("character_name", "") or "").strip()
            if configured:
                setattr(agent, "character_name", configured)
                return configured
        except Exception:
            pass

    configured = random.choice(CHARACTER_POOL)
    setattr(agent, "character_name", configured)
    if storage and config_path:
        try:
            storage.update_agent_field(config_path, "character_name", configured)
        except Exception as exc:
            logger.debug("Failed to persist character name for %s: %s", getattr(agent, "agent_id", ""), exc)
    return configured


def _choose_initial_target_firm(sim_registry, case_state: str, config: dict) -> str:
    firms = _get_available_firms(sim_registry)
    preferred_firm = str(config.get("assigned_firm", "") or "").strip()
    if preferred_firm in firms:
        return preferred_firm

    if case_state in ("空闲", "等待前台接待"):
        return random.choice(firms)

    return firms[0]


async def _launch_plaintiff_case(
    *,
    client,
    config: dict,
    case_state: str,
    sim_engine: WebSocketFrontendEngine,
    sim_event_bus: EventBus,
    sim_registry: AgentRegistry,
    storage: FileStorageManager,
) -> None:
    case_id = f"case_{config.get('case_id', '1')}"
    target_firm = _choose_initial_target_firm(sim_registry, case_state, config)
    map_prefix = _resolve_map_prefix_from_firm(target_firm)
    birth_loc_id = _resolve_birth_location_for_firm(target_firm)

    if client.config_path:
        try:
            storage.update_agent_field(client.config_path, "assigned_firm", target_firm)
            config["assigned_firm"] = target_firm
        except Exception as exc:
            logger.warning("Failed to persist assigned firm for %s: %s", case_id, exc)

    char_name = _get_or_assign_character_name(client, storage)

    await sim_engine.spawn_agent(
        agent_id=client.agent_id,
        name=client.name,
        character_name=char_name,
        birth_loc_id=birth_loc_id,
        role="plaintiff",
    )

    if case_state in ("空闲", "等待前台接待"):
        await sim_event_bus.publish(EventType.PLAINTIFF_ARRIVED, {
            "client_id": client.agent_id,
            "case_id": case_id,
            "target_firm": target_firm,
            "map_prefix": map_prefix,
            "party_role": "plaintiff",
            "client_path": client.config_path,
        })
        return

    if case_state == "原告咨询中":
        logger.info("案件 %s 状态为 '原告咨询中'，恢复咨询流程...", case_id)
        lawyer_id = config.get("assigned_lawyer_id")
        if not lawyer_id:
            lawyers = sim_registry.get_agents_by_type("lawyer")
            lawyer_id = lawyers[0].agent_id if lawyers else ""
            logger.info("未找到 assigned_lawyer_id，使用默认律师 %s", lawyer_id)

        if lawyer_id:
            lawyer = sim_registry.get_agent(lawyer_id)
            if lawyer and lawyer.config_path:
                try:
                    storage.update_agent_field(lawyer.config_path, "current_handling_case", None)
                    logger.info("已清空律师 %s 的当前案件，准备恢复咨询", lawyer_id)
                except Exception as exc:
                    logger.error("清空律师状态失败: %s", exc)

        await sim_event_bus.publish(EventType.CASE_ASSIGNED, {
            "client_id": client.agent_id,
            "case_id": case_id,
            "target_firm": target_firm,
            "map_prefix": map_prefix,
            "party_role": "plaintiff",
            "client_path": client.config_path,
            "lawyer_id": lawyer_id,
        })


async def run_simulation(
    sim_engine: WebSocketFrontendEngine,
    sim_event_bus: EventBus,
    sim_registry: AgentRegistry,
    storage: FileStorageManager,
    fsm: CaseStateMachine,
    checkpoint_mgr: CheckpointManager,
    *,
    selected_case_id: str = "",
):
    """运行法律全流程模拟（与 sandbox_main.py 逻辑一致）。"""
    normalized_selected_case_id = _normalize_case_identifier(selected_case_id)

    # 等待前端连接（最多 60 秒，超时则以 fallback 模式运行）
    logger.info("Waiting for frontend connection...")
    for _ in range(120):
        if not sim_engine._fallback_mode:
            break
        await _sleep_respecting_pause(sim_engine, 0.5)

    if sim_engine._fallback_mode:
        logger.info("No frontend connected, running in fallback (mock) mode")
    else:
        logger.info("Frontend connected, running in real-time mode")

    # 重置状态 & 统计活跃案件
    active_cases = []
    for client in sim_registry.get_agents_by_type("client"):
        if not client.config_path:
            continue
        config = storage.load_agent_config(client.config_path)
        stored_case_state = config.get("case_state", "空闲")
        case_state = normalize_case_state(stored_case_state)
        if case_state != stored_case_state:
            storage.update_agent_field(client.config_path, "case_state", case_state)
            config["case_state"] = case_state
        case_id = _normalize_case_identifier(config.get("case_id"))
        if normalized_selected_case_id and case_id != normalized_selected_case_id:
            continue
        if config.get("party_role") == "plaintiff" and case_state != "已结案":
            active_cases.append((client, config, case_state))

    if normalized_selected_case_id and not active_cases:
        logger.warning("未找到可启动的案件 %s，本轮模拟直接结束", normalized_selected_case_id)
        if checkpoint_mgr:
            checkpoint_mgr.mark_session_completed()
        return

    # Reset lawyer queues
    for lawyer in sim_registry.get_agents_by_type("lawyer"):
        if not lawyer.config_path:
            continue
        try:
            storage.update_agent_field(lawyer.config_path, "current_handling_case", None)
            storage.update_agent_field(lawyer.config_path, "case_queue", [])
        except FileNotFoundError:
            pass

    sim_event_bus.set_expected_cases(len(active_cases))

    logger.info("=" * 60)
    logger.info("法律AI小镇模拟启动")
    logger.info("发现 %d 个活跃案件", len(active_cases))
    logger.info("=" * 60)

    idle_case_entries = [
        (client, config, case_state)
        for client, config, case_state in active_cases
        if case_state in ("空闲", "等待前台接待")
    ]
    resumed_case_entries = [
        (client, config, case_state)
        for client, config, case_state in active_cases
        if case_state not in ("空闲", "等待前台接待")
    ]

    launch_requests: list[_CaseLaunchRequest] = []

    for client, config, case_state in resumed_case_entries:
        case_id = f"case_{config.get('case_id', '1')}"

        async def _launch_resumed_case(
            client=client,
            config=config,
            case_state=case_state,
        ) -> None:
            await _launch_plaintiff_case(
                client=client,
                config=config,
                case_state=case_state,
                sim_engine=sim_engine,
                sim_event_bus=sim_event_bus,
                sim_registry=sim_registry,
                storage=storage,
            )

        launch_requests.append(_CaseLaunchRequest(case_id=case_id, launch=_launch_resumed_case))

    for index, (client, config, case_state) in enumerate(idle_case_entries):
        case_id = f"case_{config.get('case_id', '1')}"
        post_launch_delay = CASE_SPAWN_INTERVAL_SECONDS if index < len(idle_case_entries) - 1 else 0.0

        async def _launch_idle_case(
            client=client,
            config=config,
            case_state=case_state,
        ) -> None:
            await _launch_plaintiff_case(
                client=client,
                config=config,
                case_state=case_state,
                sim_engine=sim_engine,
                sim_event_bus=sim_event_bus,
                sim_registry=sim_registry,
                storage=storage,
            )

        launch_requests.append(
            _CaseLaunchRequest(
                case_id=case_id,
                launch=_launch_idle_case,
                post_launch_delay=post_launch_delay,
            )
        )

    await _dispatch_case_launch_requests(
        requests=launch_requests,
        sim_engine=sim_engine,
        sim_event_bus=sim_event_bus,
    )

    # 等待所有案件结案（带防卡死检查）
    await sim_event_bus.spin_until_all_closed(
        storage_manager=storage,
        agent_registry=sim_registry,
        check_interval=15.0,
    )

    # 标记会话完成
    if checkpoint_mgr:
        checkpoint_mgr.mark_session_completed()

    logger.info("=" * 60)
    logger.info("模拟结束")
    logger.info("=" * 60)


async def resume_simulation(
    sim_engine: WebSocketFrontendEngine,
    sim_event_bus: EventBus,
    sim_registry: AgentRegistry,
    storage: FileStorageManager,
    fsm: CaseStateMachine,
    checkpoint_mgr: CheckpointManager,
    *,
    selected_case_id: str = "",
):
    """从检查点恢复模拟，扫描沙盒数据恢复所有未结案案件。"""
    normalized_selected_case_id = _normalize_case_identifier(selected_case_id)

    logger.info("=" * 60)
    logger.info("从检查点恢复模拟")
    logger.info("=" * 60)

    # 等待前端连接
    logger.info("Waiting for frontend connection...")
    for _ in range(120):
        if not sim_engine._fallback_mode:
            break
        await _sleep_respecting_pause(sim_engine, 0.5)

    if sim_engine._fallback_mode:
        logger.info("No frontend connected, running in fallback (mock) mode")
    else:
        logger.info("Frontend connected, running in real-time mode")

    # 加载会话状态
    session_state = checkpoint_mgr.load_session_state()
    if not session_state:
        logger.warning("无法加载会话状态，启动新模拟")
        return await run_simulation(
            sim_engine,
            sim_event_bus,
            sim_registry,
            storage,
            fsm,
            checkpoint_mgr,
            selected_case_id=normalized_selected_case_id,
        )

    # 恢复活跃场景状态到 EventBus
    active_scenario_details = session_state.get("active_scenario_details", {})
    if active_scenario_details:
        if max(int(MAX_CONCURRENT_CASES or 0), 0) == 1:
            logger.info(
                "串行模式启用，跳过预恢复 %d 个活跃场景，改由案件调度顺序恢复",
                len(active_scenario_details),
            )
        else:
            logger.info(f"从检查点恢复 {len(active_scenario_details)} 个活跃场景状态")
            sim_event_bus.restore_active_scenarios(active_scenario_details)
    else:
        logger.info("检查点中没有活跃场景详情，可能是旧版本检查点")

    # 扫描沙盒数据，收集所有未结案的案件
    logger.info("扫描沙盒数据，收集所有未结案案件...")
    active_cases = []
    active_case_ids = set()  # 用于去重，按案件 ID 统计

    clients = sim_registry.get_agents_by_type("client")
    logger.info(f"找到 {len(clients)} 个当事人 Agent")

    for client in clients:
        logger.info(f"检查当事人: {client.name} (ID: {client.agent_id})")
        if not client.config_path:
            logger.info(f"  跳过: 没有 config_path")
            continue

        try:
            config = storage.load_agent_config(client.config_path)
            if not isinstance(config, dict):
                logger.warning(f"  Invalid config shape, skipping resume: {client.config_path}")
                continue

            case_state = infer_case_state_from_artifacts(storage.base_dir, config)
            if case_state != config.get("case_state", "空闲"):
                storage.update_agent_field(client.config_path, "case_state", case_state)
                config["case_state"] = case_state
            normalized_case_id = _normalize_case_identifier(config.get("case_id"))
            if normalized_selected_case_id and normalized_case_id != normalized_selected_case_id:
                logger.info("  跳过: 不在本轮选定案件内 (%s)", normalized_selected_case_id)
                continue
            party_role = config.get("party_role", "plaintiff")  # 从配置中读取 party_role
            map_state = config.get("map_state") or {}
            is_seated = map_state.get("sitting") is not None
            logger.info(f"  状态: {case_state}, 角色: {party_role}, 就座: {is_seated}")

            # 将所有非空闲且非结案的当事人都视为活跃（无论原告被告）
            # 或者如果是原告且状态为空闲（代表尚未开始的新案件）
            # 或者已经在座位上（代表可能处于咨询中，即使状态被错误复位）
            should_resume = (case_state not in ["空闲", "已结案"]) or (party_role == "plaintiff" and case_state == "空闲") or is_seated

            if should_resume:
                active_cases.append((client, config, case_state))
                case_id = config.get('case_id', '')
                if case_id:
                    active_case_ids.add(f"case_{case_id}")
                logger.info(f"  ✓ 发现未结案案件角色: case_id={case_id}, role={party_role}, state={case_state}, client={client.name}")
            else:
                logger.info(f"  跳过: 不需要恢复")
        except Exception as e:
            logger.error(f"  加载配置失败: {e}", exc_info=True)
            continue

    # 关键修复：按案件数而不是角色数统计
    num_cases = len(active_case_ids)
    sim_event_bus.set_expected_cases(num_cases)
    logger.info(f"共发现 {len(active_cases)} 个未结案角色，对应 {num_cases} 个案件")
    if normalized_selected_case_id and num_cases == 0:
        logger.warning("未找到可恢复的案件 %s，本轮恢复直接结束", normalized_selected_case_id)
        checkpoint_mgr.mark_session_completed()
        return

    # 获取检查点中未完成的场景，并先清理缺文件的脏检查点
    raw_incomplete_scenarios = checkpoint_mgr.get_incomplete_scenarios()
    incomplete_scenarios: list[dict] = []
    incomplete_case_ids: set[str] = set()
    for scenario_info in raw_incomplete_scenarios:
        scenario_id = scenario_info["scenario_id"]
        checkpoint_file = scenario_info["checkpoint_file"]
        checkpoint_data = checkpoint_mgr.load_scenario_checkpoint(checkpoint_file)
        if not checkpoint_data:
            checkpoint_data = _build_player_lawyer_lc_checkpoint_from_request(storage, scenario_info)
            if not checkpoint_data:
                logger.warning(
                    "检查点场景 %s 缺少可用检查点文件 %s，按脏检查点清理并回落到案件状态恢复",
                    scenario_id,
                    checkpoint_file,
                )
                checkpoint_mgr.mark_scenario_completed(scenario_id)
                continue

        enriched_scenario_info = dict(scenario_info)
        enriched_scenario_info["_checkpoint_data"] = checkpoint_data
        checkpoint_case_id = str(enriched_scenario_info.get("case_id", "") or "")
        if normalized_selected_case_id and _normalize_case_identifier(checkpoint_case_id) != normalized_selected_case_id:
            continue
        incomplete_scenarios.append(enriched_scenario_info)
        if checkpoint_case_id:
            incomplete_case_ids.add(checkpoint_case_id)

    logger.info(f"检查点中有 {len(incomplete_scenarios)} 个未完成场景: {incomplete_case_ids}")

    from src.scenarios.legal_consultation import LegalConsultationScenario
    from src.prompts.prompt_assembler import PromptAssembler

    launch_requests: list[_CaseLaunchRequest] = []

    # 1. 先恢复检查点中未完成的场景
    for scenario_info in incomplete_scenarios:
        scenario_id = scenario_info["scenario_id"]
        checkpoint_file = scenario_info["checkpoint_file"]
        party_role = scenario_info.get("party_role", "plaintiff")
        checkpoint_case_id = str(scenario_info.get("case_id", "") or "")
        checkpoint_data = scenario_info["_checkpoint_data"]
        if not checkpoint_case_id:
            logger.warning("检查点场景 %s 缺少 case_id，跳过", scenario_id)
            continue

        async def _resume_incomplete_scenario(
            scenario_id=scenario_id,
            checkpoint_file=checkpoint_file,
            party_role=party_role,
            checkpoint_case_id=checkpoint_case_id,
            checkpoint_data=checkpoint_data,
        ) -> None:
            logger.info(f"恢复检查点场景: {scenario_id}")

            case_id = _normalize_case_identifier(checkpoint_data.get("case_id", "") or checkpoint_case_id)
            client_id = checkpoint_data.get("client_id") or scenario_info.get("client_id")
            lawyer_id = checkpoint_data.get("lawyer_id") or scenario_info.get("lawyer_id")

            client = sim_registry.get_agent(client_id)
            lawyer = sim_registry.get_agent(lawyer_id)

            if not client or not lawyer:
                logger.error(f"无法找到 agent: client={client_id}, lawyer={lawyer_id}")
                return False

            if client_id not in sim_engine._agent_states:
                birth_loc_id = "birth_locationB" if party_role == "defendant" else "birth_locationA"
                await sim_engine.spawn_agent(
                    agent_id=client_id,
                    name=client.name,
                    character_name=_get_or_assign_character_name(client, storage),
                    birth_loc_id=birth_loc_id,
                    role=party_role,
                )

            if lawyer_id not in sim_engine._agent_states:
                lawyer_birth = "birth_locationB" if getattr(lawyer, "firm_id", "") == "law_firm_B" else "birth_locationA"
                await sim_engine.spawn_agent(
                    agent_id=lawyer_id,
                    name=lawyer.name,
                    character_name=_get_or_assign_character_name(lawyer, storage),
                    birth_loc_id=lawyer_birth,
                    role="lawyer",
                )

            data_loader, case, client_config = _load_case_data_for_resume(client.config_path, storage)
            scenario_data = checkpoint_data.get("scenario_data", {})
            extracted_profile = (
                data_loader.extract_plaintiff_profile(case)
                if party_role == "plaintiff"
                else data_loader.extract_defendant_profile(case)
            )

            lawyer_scenario = PromptAssembler.build_scenario_prompt("lawyer", "LC", scenario_data)
            lawyer_config = storage.load_agent_config(lawyer.config_path) if lawyer.config_path else {}
            lawyer_prompt = PromptAssembler.build(
                profile={"name": lawyer.name, "law_firm": lawyer.law_firm, "specialty": lawyer.specialty_areas},
                scenario_prompt=lawyer_scenario,
            )

            client_scenario = PromptAssembler.build_scenario_prompt("client", "LC", scenario_data)
            client_prompt = PromptAssembler.build(
                profile=ScenarioOrchestrator._build_client_prompt_profile(client, extracted_profile),
                scenario_prompt=client_scenario,
            )

            lawyer.activate(lawyer_prompt)
            client.activate(client_prompt)

            try:
                display_stage_code = ScenarioOrchestrator._consultation_display_stage_code(party_role)
                sim_event_bus.register_active_scenario(
                    case_id=case_id,
                    scenario_type="LC",
                    participant_ids=[client_id, lawyer_id],
                )

                output_path = str(Path(storage.base_dir) / "output" / case_id / f"{display_stage_code}_result.json")
                scenario = LegalConsultationScenario(
                    client_agent=client,
                    lawyer_agent=lawyer,
                    max_turns=ScenarioOrchestrator._resolve_lc_max_turns(
                        len(extracted_profile.get("questions") or []),
                        player_lawyer_enabled=_player_lawyer_mode_for_engine(sim_engine) == "plaintiff",
                    ),
                    output_path=output_path,
                    verbose=SCENARIO_VERBOSE,
                    map_engine=sim_engine,
                    checkpoint_manager=checkpoint_mgr,
                    scenario_id=scenario_id,
                    trace_stage_code=display_stage_code,
                    trace_stage_key=f"{display_stage_code}_{party_role}".upper(),
                )

                result = await scenario.resume_from_checkpoint(checkpoint_data)

                output_dir = Path(storage.base_dir) / "output" / case_id
                output_dir.mkdir(parents=True, exist_ok=True)
                result_file = output_dir / f"{display_stage_code}_result.json"
                with open(result_file, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(result, f, ensure_ascii=False, indent=2)
                if display_stage_code == "PLC":
                    compat_file = output_dir / "LC_result.json"
                    with open(compat_file, 'w', encoding='utf-8') as f:
                        import json
                        json.dump(result, f, ensure_ascii=False, indent=2)

                checkpoint_mgr.mark_scenario_completed(scenario_id)
                sim_event_bus.unregister_active_scenario(case_id)
                checkpoint_mgr.sync_active_scenarios_from_event_bus()

                completion_event = (
                    EventType.PLAINTIFF_CONSULTATION_COMPLETED
                    if party_role == "plaintiff"
                    else EventType.DEFENDANT_CONSULTATION_COMPLETED
                )
                await sim_event_bus.publish(completion_event, {
                    "case_id": case_id,
                    "client_path": client.config_path,
                    "client_id": client.agent_id,
                    "lawyer_id": lawyer.agent_id,
                    "party_role": party_role,
                    "firm_id": getattr(lawyer, "firm_id", "law_firm_A"),
                })
                return True

            except Exception as e:
                logger.error(f"恢复场景失败: {e}", exc_info=True)
                sim_event_bus.unregister_active_scenario(case_id)
                checkpoint_mgr.sync_active_scenarios_from_event_bus()
                lawyer.recover_from_error()
                client.recover_from_error()
                return False
            finally:
                if lawyer.is_active:
                    lawyer.deactivate()
                if client.is_active:
                    client.deactivate()

        launch_requests.append(
            _CaseLaunchRequest(case_id=checkpoint_case_id, launch=_resume_incomplete_scenario)
        )

    # 2. 启动检查点中没有的案件（根据状态恢复到相应流程）
    # 状态到恢复事件的映射
    from src.core.event_bus import EventType
    STATE_TO_EVENT_MAP = {
        "空闲": EventType.PLAINTIFF_ARRIVED,
        "等待前台接待": EventType.PLAINTIFF_ARRIVED,
        "原告咨询中": EventType.CASE_ASSIGNED,
        "起诉状起草中": EventType.ENTER_COMPLAINT_DRAFTING,
        "起诉状已递交": EventType.LAWSUIT_FILED,     # -> 激活被告
        "被告已传唤": EventType.DEFENDANT_SERVED,   # -> 被告咨询
        "等待被告": EventType.LAWSUIT_FILED,
        "被告咨询中": EventType.CASE_ASSIGNED,
        "答辩状起草中": EventType.ENTER_DEFENSE_DRAFTING,
        "答辩状已递交": EventType.DEFENSE_FILED,      # -> 庭前分析
        "一审庭前信息分析中": EventType.DEFENSE_FILED,
        "等待一审开庭": EventType.ENTER_TRIAL_FIRST_INSTANCE,
        "一审庭审中": EventType.TRIAL_FIRST_INSTANCE_READY,
        "一审判决": EventType.FIRST_INSTANCE_VERDICT_ISSUED,
        "上诉决策中": EventType.FIRST_INSTANCE_VERDICT_ISSUED,
        "上诉状起草中": EventType.APPEAL_DECISION_MADE,
        "上诉状已递交": EventType.APPEAL_FILED,
        "等待上诉答辩": EventType.APPEAL_FILED,
        "上诉答辩状起草中": EventType.ENTER_DEFENSE_DRAFTING,
        "上诉答辩状已递交": EventType.APPEAL_RESPONSE_FILED,
        "二审庭前信息分析中": EventType.APPEAL_RESPONSE_FILED,
        "等待二审开庭": EventType.ENTER_TRIAL_SECOND_INSTANCE,
        "二审庭审中": EventType.TRIAL_SECOND_INSTANCE_READY,
        "终审判决": EventType.FINAL_VERDICT_ISSUED,
    }

    shared_case_resumed = set()
    for client, config, case_state in active_cases:
        case_id = f"case_{config.get('case_id', '1')}"
        party_role = config.get("party_role", "plaintiff")
        map_state = config.get("map_state") or {}
        is_seated = map_state.get("sitting") is not None

        # 如果已经在检查点中恢复过，跳过
        if case_id in incomplete_case_ids:
            logger.info(f"案件 {case_id} 已从检查点恢复，跳过")
            continue

        if case_state == "空闲" and party_role == "defendant" and is_seated:
            case_state = "被告咨询中"
            logger.info(f"案件 {case_id} ({party_role}) 启发式修补状态: 空闲 -> {case_state}")

        # 跨角色协同修复：如果当前是原告且处于早期，但发现被告已激活，原告转入等待
        if party_role == "plaintiff" and case_state in ["空闲", "起诉状已递交", "被告已传唤"]:
             case_dir = Path(client.config_path).parent
             def_path = case_dir / "defendant" / "config.yaml"
             if def_path.exists():
                 try:
                     def_conf = storage.load_yaml(def_path)
                     # 如果被告已经不是空闲，或者被告也已经坐下了
                     def_map_state = (def_conf or {}).get("map_state") or {}
                     if def_conf.get("case_state") != "空闲" or def_map_state.get("sitting"):
                         case_state = "等待被告"
                         logger.info(f"案件 {case_id} 原告协同修补: 被告已激活，原告状态转为 {case_state}")
                 except: pass

        # 共享阶段修复：如果同案另一方已经进入一审共享阶段，则当前角色直接对齐。
        shared_resume_states = ["一审庭前信息分析中", "等待一审开庭", "一审庭审中", "一审判决", "上诉决策中"]
        counterpart_role = "defendant" if party_role == "plaintiff" else "plaintiff"
        counterpart_path = Path(client.config_path).parent / counterpart_role / "config.yaml"
        if counterpart_path.exists():
            try:
                counterpart_conf = storage.load_yaml(counterpart_path)
                counterpart_state = counterpart_conf.get("case_state", "空闲")
                if counterpart_state in shared_resume_states and case_state not in shared_resume_states:
                    case_state = counterpart_state
                    logger.info(
                        f"案件 {case_id} ({party_role}) 共享阶段修复: 对齐另一方状态 -> {case_state}"
                    )
            except Exception as e:
                logger.warning(f"案件 {case_id} 共享阶段修复失败: {e}")

        # 获取恢复事件
        if case_state in ["\u7b49\u5f85\u4e00\u5ba1\u5f00\u5ead", "\u4e00\u5ba1\u5ead\u5ba1\u4e2d"]:
            if case_id in shared_case_resumed:
                logger.info(f"Case {case_id} first-instance resume already triggered, skip role {party_role}")
                continue
            shared_case_resumed.add(case_id)
            trial_payload = _build_first_instance_resume_payload(case_id, client, sim_registry, storage)
            if not trial_payload:
                continue
            logger.info(
                f"Resume shared case: {case_id}, shared_state={case_state} "
                f"-> event={EventType.ENTER_TRIAL_FIRST_INSTANCE}"
            )
            await sim_event_bus.publish(EventType.ENTER_TRIAL_FIRST_INSTANCE, trial_payload)
            continue

        if case_state in ["\u4e00\u5ba1\u5224\u51b3", "\u4e0a\u8bc9\u51b3\u7b56\u4e2d"]:
            if case_id in shared_case_resumed:
                logger.info(f"Case {case_id} post-verdict resume already triggered, skip role {party_role}")
                continue
            shared_case_resumed.add(case_id)
            verdict_payload = {
                "case_id": case_id,
                "client_path": str(Path(client.config_path).parent / "plaintiff"),
            }
            logger.info(
                f"Resume shared case: {case_id}, shared_state={case_state} "
                f"-> event={EventType.FIRST_INSTANCE_VERDICT_ISSUED}"
            )
            await sim_event_bus.publish(EventType.FIRST_INSTANCE_VERDICT_ISSUED, verdict_payload)
            continue

        if case_state in ["\u7b49\u5f85\u4e8c\u5ba1\u5f00\u5ead", "\u4e8c\u5ba1\u5ead\u5ba1\u4e2d"]:
            if case_id in shared_case_resumed:
                logger.info(f"Case {case_id} second-instance resume already triggered, skip role {party_role}")
                continue
            shared_case_resumed.add(case_id)
            trial_payload = _build_second_instance_resume_payload(case_id, client, sim_registry, storage)
            if not trial_payload:
                continue
            logger.info(
                f"Resume shared case: {case_id}, shared_state={case_state} "
                f"-> event={EventType.ENTER_TRIAL_SECOND_INSTANCE}"
            )
            await sim_event_bus.publish(EventType.ENTER_TRIAL_SECOND_INSTANCE, trial_payload)
            continue

        recovery_event = STATE_TO_EVENT_MAP.get(case_state)
        
        # 如果是“等待被告”且已经是原告，则跳过特定事件（由被告端逻辑激活）
        if case_state == "等待被告" and party_role == "plaintiff":
            recovery_event = None

        logger.info(f"恢复角色执行: {case_id}, client={client.name}, role={party_role}, state={case_state} -> event={recovery_event}")

        async def _resume_case_action(
            client=client,
            config=config,
            case_id=case_id,
            case_state=case_state,
            party_role=party_role,
            recovery_event=recovery_event,
        ) -> None:
            firms = list(sim_registry._firms.keys())
            target_firm = config.get("assigned_firm", firms[0] if firms else "law_firm_A")
            map_prefix = _resolve_map_prefix_from_firm(target_firm)
            birth_loc_id = _resolve_birth_location_for_firm(target_firm)
            char_name = _get_or_assign_character_name(client, storage)

            await sim_engine.spawn_agent(
                agent_id=client.agent_id,
                name=client.name,
                character_name=char_name,
                birth_loc_id=birth_loc_id,
                role=party_role,
            )

            if recovery_event in {EventType.PLAINTIFF_ARRIVED, EventType.DEFENDANT_ARRIVED}:
                await sim_event_bus.publish(recovery_event, {
                    "client_id": client.agent_id,
                    "client_path": client.config_path,
                    "case_id": case_id,
                    "target_firm": target_firm,
                    "map_prefix": map_prefix,
                    "party_role": party_role,
                })
                return

            if party_role == "plaintiff" and case_state in {"等待被告", "被告已传唤"}:
                defendant = _find_case_party_client(sim_registry, storage, case_id, "defendant")
                if not defendant or not getattr(defendant, "config_path", None):
                    logger.warning("案件 %s 缺少被告配置，回退到 LAWSUIT_FILED 重新激活被告", case_id)
                    await sim_event_bus.publish(EventType.LAWSUIT_FILED, {
                        "case_id": case_id,
                        "client_path": client.config_path,
                        "client_id": client.agent_id,
                        "party_role": party_role,
                        "firm_id": target_firm,
                        "target_firm": target_firm,
                        "map_prefix": map_prefix,
                    })
                    return

                defendant_config = storage.load_agent_config(defendant.config_path)
                defendant_firm = defendant_config.get("assigned_firm", target_firm)
                defendant_map_prefix = _resolve_map_prefix_from_firm(defendant_firm)
                defendant_birth_loc_id = _resolve_birth_location_for_firm(defendant_firm)

                if defendant.agent_id not in getattr(sim_engine, "_agent_states", {}):
                    await sim_engine.spawn_agent(
                        agent_id=defendant.agent_id,
                        name=defendant.name,
                        character_name=_get_or_assign_character_name(defendant, storage),
                        birth_loc_id=defendant_birth_loc_id,
                        role="defendant",
                    )

                await sim_event_bus.publish(EventType.DEFENDANT_ARRIVED, {
                    "client_id": defendant.agent_id,
                    "case_id": case_id,
                    "target_firm": defendant_firm,
                    "map_prefix": defendant_map_prefix,
                    "party_role": "defendant",
                    "client_path": defendant.config_path,
                })
                return

            if recovery_event == EventType.CASE_ASSIGNED:
                lawyer_id = config.get("assigned_lawyer_id", "")
                if lawyer_id:
                    lawyer = sim_registry.get_agent(lawyer_id)
                    if lawyer and lawyer.config_path:
                        try:
                            storage.update_agent_field(lawyer.config_path, "current_handling_case", None)
                            logger.info(f"已清空律师 {lawyer_id} 的当前案件，准备恢复咨询")
                        except Exception as e:
                            logger.error(f"清空律师状态失败: {e}")

                    await sim_event_bus.publish(recovery_event, {
                        "client_id": client.agent_id,
                        "client_path": client.config_path,
                        "case_id": case_id,
                        "lawyer_id": lawyer_id,
                        "target_firm": target_firm,
                        "map_prefix": map_prefix,
                        "party_role": party_role,
                    })
                else:
                    logger.warning(f"案件 {case_id} 没有分配律师，无法恢复咨询")
                return

            if recovery_event:
                await sim_event_bus.publish(recovery_event, {
                    "case_id": case_id,
                    "client_path": client.config_path,
                    "client_id": client.agent_id,
                    "lawyer_id": config.get("assigned_lawyer_id", ""),
                    "party_role": party_role,
                    "firm_id": target_firm,
                    "target_firm": target_firm,
                    "map_prefix": map_prefix,
                })
            else:
                logger.info(f"案件 {case_id} ({party_role}) 状态 {case_state} 无需恢复事件，跳过")

        launch_requests.append(_CaseLaunchRequest(case_id=case_id, launch=_resume_case_action))

    await _dispatch_case_launch_requests(
        requests=launch_requests,
        sim_engine=sim_engine,
        sim_event_bus=sim_event_bus,
    )

    # 等待所有案件结案（带防卡死检查）
    await sim_event_bus.spin_until_all_closed(
        storage_manager=storage,
        agent_registry=sim_registry,
        check_interval=15.0,
    )

    # 标记会话完成
    checkpoint_mgr.mark_session_completed()

    logger.info("=" * 60)
    logger.info("模拟恢复完成")
    logger.info("=" * 60)


def _build_player_lawyer_lc_checkpoint_from_request(
    storage: FileStorageManager,
    scenario_info: dict[str, Any],
) -> dict[str, Any] | None:
    """Build a minimal LC checkpoint from a persisted player-lawyer turn."""
    if str(scenario_info.get("scenario_type") or "").upper() != "LC":
        return None
    case_id = str(scenario_info.get("case_id") or "").strip()
    if not case_id:
        return None

    request_dir = Path(storage.base_dir) / "output" / case_id / "_player_lawyer"
    if not request_dir.exists():
        return None

    candidates: list[dict[str, Any]] = []
    for path in sorted(request_dir.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if str(payload.get("case_id") or "") != case_id:
            continue
        if str(payload.get("stage") or "").upper() != "LC":
            continue
        if str(payload.get("status") or "") not in {"pending", "submitted"}:
            continue
        if not str(payload.get("prompt") or "").strip():
            continue
        candidates.append(payload)

    if not candidates:
        return None

    candidates.sort(key=lambda item: str(item.get("resolved_at") or item.get("created_at") or ""))
    request_payload = candidates[-1]
    created_at = str(request_payload.get("created_at") or datetime.now().isoformat())
    logger.info(
        "使用玩家输入请求恢复 LC 检查点: scenario=%s request=%s status=%s",
        scenario_info.get("scenario_id"),
        request_payload.get("request_id"),
        request_payload.get("status"),
    )
    return {
        "scenario_type": "LC",
        "case_id": case_id,
        "client_id": scenario_info.get("client_id"),
        "lawyer_id": scenario_info.get("lawyer_id"),
        "dialog_history": [
            {
                "turn": 0,
                "role": "client",
                "content": str(request_payload.get("prompt") or ""),
                "timestamp": created_at,
            }
        ],
        "turn_count": 0,
        "completed": False,
        "finish_reason": "max_turns",
    }


def _load_case_data_for_resume(client_config_path: str, storage):
    """加载案件数据用于恢复。"""
    from src.data.data_loader import DataLoader

    config = storage.load_agent_config(client_config_path)
    dataset_path = config.get("dataset_path", "")

    data_loader = DataLoader(dataset_path)
    case = data_loader.resolve_case_for_config(config)
    return data_loader, case, config


# ── 启动事件 ──

@app.on_event("startup")
async def startup():
    global engine, event_bus, registry, checkpoint_mgr, storage_manager, case_fsm, _simulation_task

    (
        event_bus,
        registry,
        checkpoint_mgr,
        storage_manager,
        case_fsm,
    ) = _initialize_runtime_state()

    # 默认不自动开始模拟，等待前端显式控制
    session_state = checkpoint_mgr.load_session_state()
    _simulation_task = None
    _set_engine_paused(False)

    if session_state and session_state.get("simulation_status") == "running":
        checkpoint_mgr.mark_session_paused()
        logger.info("检测到上次会话为 running，已切换为 paused，等待前端手动开始")
    elif session_state and session_state.get("simulation_status") == "paused":
        logger.info("检测到未完成会话，等待前端手动恢复")
    else:
        logger.info("当前为待启动状态，等待前端手动开始模拟")

    logger.info("WebSocket server started on ws://localhost:8000/ws")


@app.on_event("shutdown")
async def shutdown():
    """优雅关闭处理器。"""
    global checkpoint_mgr, _simulation_task

    logger.info("Shutting down server...")

    # 标记会话为暂停状态（而非完成）
    if checkpoint_mgr:
        checkpoint_mgr.mark_session_paused()
        logger.info("Session marked as paused for recovery")

    # 取消模拟任务
    if _simulation_task and not _simulation_task.done():
        _simulation_task.cancel()
        try:
            await _simulation_task
        except asyncio.CancelledError:
            pass

    logger.info("Server shutdown complete")


def handle_sigterm(signum, frame):
    """处理 SIGTERM 信号（优雅关闭）。"""
    logger.info("Received SIGTERM, initiating graceful shutdown...")
    # FastAPI 会自动调用 shutdown 事件处理器


def handle_sigint(signum, frame):
    """处理 SIGINT 信号（Ctrl+C）。"""
    logger.info("Received SIGINT (Ctrl+C), initiating graceful shutdown...")
    # FastAPI 会自动调用 shutdown 事件处理器
    sys.exit(0)


# 注册信号处理器
signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigint)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
