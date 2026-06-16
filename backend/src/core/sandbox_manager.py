from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class SandboxRuntimeContext:
    sandbox_id: str
    user_id: str
    sandbox_key: str
    storage_root: Path
    engine: Any
    event_bus: Any
    registry: Any
    checkpoint_mgr: Any
    storage_manager: Any
    case_fsm: Any
    orchestrator: Any = None
    simulation_task: Any = None
    last_error: dict[str, Any] | None = None
    selected_case_id: str = ""
    single_case_mode: bool = False
    connected_clients: set[Any] = field(default_factory=set)


RuntimeFactory = Callable[[Any, Path], SandboxRuntimeContext]
RuntimeHandler = Callable[..., dict[str, Any] | None]
StatusHandler = Callable[[SandboxRuntimeContext], dict[str, Any]]


class SandboxManager:
    def __init__(
        self,
        *,
        base_dir: Path,
        runtime_factory: RuntimeFactory,
        start_handler: RuntimeHandler | None = None,
        pause_handler: RuntimeHandler | None = None,
        restart_handler: RuntimeHandler | None = None,
        status_handler: StatusHandler | None = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.runtime_factory = runtime_factory
        self.start_handler = start_handler or self._default_start_handler
        self.pause_handler = pause_handler or self._default_pause_handler
        self.restart_handler = restart_handler or self._default_restart_handler
        self.status_handler = status_handler or self._default_status_handler
        self._contexts: dict[str, SandboxRuntimeContext] = {}

    def compute_storage_root(self, *, user_id: str, sandbox_key: str) -> Path:
        return self.base_dir / "users" / str(user_id) / str(sandbox_key)

    def get_or_create_context(self, sandbox: Any) -> SandboxRuntimeContext:
        sandbox_id = str(getattr(sandbox, "id"))
        context = self._contexts.get(sandbox_id)
        if context is not None:
            return context

        storage_root = self.compute_storage_root(
            user_id=str(getattr(sandbox, "user_id")),
            sandbox_key=str(getattr(sandbox, "sandbox_key")),
        )
        context = self.runtime_factory(sandbox, storage_root)
        if context.connected_clients is None:
            context.connected_clients = set()
        self._contexts[sandbox_id] = context
        return context

    def start_sandbox(self, sandbox: Any, *, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        context = self.get_or_create_context(sandbox)
        result = self.start_handler(context, payload or {})
        return result or self.status_handler(context)

    def pause_sandbox(self, sandbox: Any) -> dict[str, Any]:
        context = self.get_or_create_context(sandbox)
        result = self.pause_handler(context)
        return result or self.status_handler(context)

    def restart_sandbox(self, sandbox: Any) -> dict[str, Any]:
        context = self.get_or_create_context(sandbox)
        result = self.restart_handler(context)
        return result or self.status_handler(context)

    def get_status(self, sandbox: Any) -> dict[str, Any]:
        context = self.get_or_create_context(sandbox)
        return self.status_handler(context)

    def reset_context(self, sandbox: Any) -> None:
        sandbox_id = str(getattr(sandbox, "id"))
        self._contexts.pop(sandbox_id, None)

    @staticmethod
    def _default_start_handler(
        context: SandboxRuntimeContext,
        payload: dict[str, Any] | None = None,
    ) -> None:
        del payload
        engine = context.engine
        if engine is not None:
            setattr(engine, "_paused", False)
            resumed_event = getattr(engine, "_resumed_event", None)
            if resumed_event is not None and hasattr(resumed_event, "set"):
                resumed_event.set()

        checkpoint_mgr = context.checkpoint_mgr
        if checkpoint_mgr is not None and hasattr(checkpoint_mgr, "mark_session_running"):
            checkpoint_mgr.mark_session_running()
        return None

    @staticmethod
    def _default_pause_handler(context: SandboxRuntimeContext) -> None:
        engine = context.engine
        if engine is not None:
            setattr(engine, "_paused", True)
            resumed_event = getattr(engine, "_resumed_event", None)
            if resumed_event is not None and hasattr(resumed_event, "clear"):
                resumed_event.clear()

        checkpoint_mgr = context.checkpoint_mgr
        if checkpoint_mgr is not None and hasattr(checkpoint_mgr, "mark_session_paused"):
            checkpoint_mgr.mark_session_paused()
        return None

    @staticmethod
    def _default_restart_handler(context: SandboxRuntimeContext) -> None:
        task = context.simulation_task
        if task is not None and hasattr(task, "cancel") and not getattr(task, "done", lambda: False)():
            task.cancel()
            context.simulation_task = None
        context.last_error = None
        context.selected_case_id = ""
        context.single_case_mode = False

        engine = context.engine
        if engine is not None:
            setattr(engine, "_paused", False)
            ack_events = getattr(engine, "_ack_events", None)
            if hasattr(ack_events, "clear"):
                ack_events.clear()
            agent_states = getattr(engine, "_agent_states", None)
            if hasattr(agent_states, "clear"):
                agent_states.clear()
        return None

    @staticmethod
    def _default_status_handler(context: SandboxRuntimeContext) -> dict[str, Any]:
        task = context.simulation_task
        task_running = bool(task is not None and not getattr(task, "done", lambda: False)())
        paused = bool(getattr(context.engine, "_paused", False))
        session_state = None
        if context.checkpoint_mgr is not None and hasattr(context.checkpoint_mgr, "load_session_state"):
            session_state = context.checkpoint_mgr.load_session_state()
        persisted_status = str((session_state or {}).get("simulation_status") or "").strip().lower()
        single_case_mode = bool(
            getattr(context, "single_case_mode", False)
            or (session_state or {}).get("single_case_mode")
            or (session_state or {}).get("selected_case_id")
        )
        last_error = context.last_error

        if last_error:
            status = "error"
        elif task_running:
            status = "paused" if paused else "running"
        elif persisted_status in {"paused", "running"}:
            status = "paused"
        elif persisted_status == "completed" and single_case_mode:
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
            "paused": status == "paused",
            "simulation_running": task_running,
            "clients_connected": clients_connected,
            "last_error": last_error,
        }
