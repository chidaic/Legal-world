from __future__ import annotations

from pathlib import Path
import shutil
from uuid import uuid4
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session
import yaml

from ..data.data_loader import DataLoader
from ..utils.memory_initializer import initialize_client_memory

from .models import Sandbox, SandboxRuntimeSnapshot
class SandboxService:
    def __init__(self, *, base_dir: Path, seed_source_dir: Path | None = None) -> None:
        self.base_dir = Path(base_dir)
        self.seed_source_dir = Path(seed_source_dir) if seed_source_dir is not None else None

    def compute_storage_root(self, *, user_id: str, sandbox_key: str) -> str:
        return str(self.base_dir / "users" / str(user_id) / str(sandbox_key))

    def get_user_sandbox(self, *, session: Session, user_id: str) -> Sandbox | None:
        return session.execute(select(Sandbox).where(Sandbox.user_id == user_id)).scalar_one_or_none()

    def get_or_create_user_sandbox(self, *, session: Session, user_id: str) -> Sandbox:
        sandbox = self.get_user_sandbox(session=session, user_id=user_id)
        if sandbox is not None:
            self._initialize_seed_storage(Path(sandbox.storage_root))
            return sandbox

        sandbox_key = uuid4().hex
        sandbox = Sandbox(
            user_id=user_id,
            sandbox_key=sandbox_key,
            status="idle",
            storage_root=self.compute_storage_root(user_id=user_id, sandbox_key=sandbox_key),
        )
        session.add(sandbox)
        session.flush()
        session.refresh(sandbox)
        self._initialize_seed_storage(Path(sandbox.storage_root))
        return sandbox

    def _initialize_seed_storage(self, storage_root: Path) -> None:
        if not self._storage_needs_seed(storage_root):
            return

        storage_root.mkdir(parents=True, exist_ok=True)
        if self.seed_source_dir is None or not self.seed_source_dir.exists():
            return

        for directory_name in ("cases", "law_firms", "court_system"):
            source_dir = self.seed_source_dir / directory_name
            if source_dir.exists():
                shutil.copytree(source_dir, storage_root / directory_name, dirs_exist_ok=True)

        dataset_source = self.seed_source_dir / "case_data_extracted.json"
        if dataset_source.exists():
            shutil.copy2(dataset_source, storage_root / "case_data_extracted.json")

        self._reset_seed_configs(storage_root)

    def reset_sandbox_storage(self, storage_root: Path) -> None:
        if storage_root.exists():
            shutil.rmtree(storage_root)
        self._initialize_seed_storage(storage_root)
        if not storage_root.exists():
            storage_root.mkdir(parents=True, exist_ok=True)

    def _reset_seed_configs(self, storage_root: Path) -> None:
        fallback_dataset_path = str(storage_root / "case_data_extracted.json")
        lawyer_roster_cache: dict[Path, dict[str, dict[str, Any]]] = {}

        for config_path in sorted(storage_root.glob("cases/case_*/plaintiff/config.yaml")) + sorted(
            storage_root.glob("cases/case_*/defendant/config.yaml")
        ):
            config = self._load_yaml(config_path)
            config.pop("chat_history_summary", None)
            designated_lawyer_id = str(
                config.get("designated_lawyer_id", "") or config.get("assigned_lawyer_id", "") or ""
            ).strip()
            config["case_state"] = "空闲"
            resolved_dataset_path = str(config.get("dataset_path", "") or "").strip()
            if (storage_root / "case_data_extracted.json").exists():
                resolved_dataset_path = self._resolve_seed_dataset_path(
                    config=config,
                    fallback_dataset_path=fallback_dataset_path,
                )
                config["dataset_path"] = resolved_dataset_path
            resolved_profile = self._resolve_initial_client_profile(
                config=config,
                dataset_path=resolved_dataset_path,
            )
            profile = config.get("profile", {})
            if not isinstance(profile, dict):
                profile = {}
            for key in (
                "name",
                "type",
                "party_type",
                "gender",
                "ethnicity",
                "birth_date",
                "address",
                "representative",
            ):
                value = str(resolved_profile.get(key, "") or "").strip()
                if value:
                    profile[key] = value
            legal_persona_profile = resolved_profile.get("legal_persona_profile", {})
            if isinstance(legal_persona_profile, dict) and legal_persona_profile:
                profile["legal_persona_profile"] = legal_persona_profile
            else:
                profile.pop("legal_persona_profile", None)
            config["profile"] = profile
            initial_case_stage = self._resolve_initial_case_stage(
                config=config,
                dataset_path=resolved_dataset_path,
            )
            config.pop("long_term_memory", None)
            config["map_state"] = None
            config["designated_lawyer_id"] = designated_lawyer_id
            config["assigned_lawyer_id"] = ""
            self._save_yaml(config_path, config)

        for config_path in sorted(storage_root.glob("law_firms/*/lawyers/*/config.yaml")):
            config = self._load_yaml(config_path)
            config["profile"] = self._resolve_initial_lawyer_profile(
                config_path=config_path,
                config=config,
                lawyer_roster_cache=lawyer_roster_cache,
            )
            config.pop("chat_history_summary", None)
            config["current_handling_case"] = None
            config["case_queue"] = []
            config.pop("long_term_memory", None)
            config["map_state"] = None
            self._save_yaml(config_path, config)

        for config_path in sorted(storage_root.glob("court_system/*/judges/*/config.yaml")):
            config = self._load_yaml(config_path)
            config.pop("chat_history_summary", None)
            config["current_handling_case"] = None
            config["case_queue"] = []
            config["map_state"] = None
            self._save_yaml(config_path, config)

        for transient_dir in ("output", "checkpoints"):
            target_dir = storage_root / transient_dir
            if target_dir.exists():
                shutil.rmtree(target_dir)

        from .file_storage_manager import FileStorageManager

        storage = FileStorageManager(storage_root)
        for case_agent_dir in sorted(storage_root.glob("cases/case_*/plaintiff")) + sorted(
            storage_root.glob("cases/case_*/defendant")
        ):
            try:
                initialize_client_memory(storage, str(case_agent_dir))
            except Exception:
                continue

    @staticmethod
    def _storage_needs_seed(storage_root: Path) -> bool:
        if not storage_root.exists():
            return True
        has_case_configs = any(storage_root.glob("cases/case_*/plaintiff/config.yaml"))
        has_law_firm_roster = any(storage_root.glob("law_firms/*/lawyer_roster.yaml"))
        has_judge_configs = any(storage_root.glob("court_system/*/judges/*/config.yaml"))
        has_dataset = (storage_root / "case_data_extracted.json").exists()
        return not (has_case_configs and has_law_firm_roster and has_judge_configs and has_dataset)

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _save_yaml(path: Path, payload: dict) -> None:
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)

    def _resolve_seed_dataset_path(self, *, config: dict[str, Any], fallback_dataset_path: str) -> str:
        project_root = Path(__file__).resolve().parents[3]
        candidate_paths: list[str] = []
        seen_paths: set[str] = set()

        def append_candidate(raw_path: str) -> None:
            normalized = str(raw_path or "").strip()
            if not normalized:
                return

            candidate = Path(normalized.replace("\\", "/")).expanduser()
            if candidate.exists():
                resolved = str(candidate.resolve())
            else:
                candidate_name = Path(normalized.replace("\\", "/")).name
                if not candidate_name:
                    return
                repo_candidate = project_root / "data" / candidate_name
                if not repo_candidate.exists():
                    return
                resolved = str(repo_candidate.resolve())

            if resolved in seen_paths:
                return
            seen_paths.add(resolved)
            candidate_paths.append(resolved)

        append_candidate(fallback_dataset_path)
        append_candidate(str(config.get("dataset_path", "") or "").strip())
        for candidate in self._project_dataset_candidates():
            append_candidate(candidate)

        if not candidate_paths:
            return fallback_dataset_path

        profile = config.get("profile", {})
        profile_name = str(profile.get("name", "") or "").strip() if isinstance(profile, dict) else ""
        party_role = str(config.get("party_role", "plaintiff") or "plaintiff").strip() or "plaintiff"
        case_id = config.get("case_id", "0")
        if not profile_name:
            return candidate_paths[0]

        lookup_config = {
            "case_id": case_id,
            "party_role": party_role,
            "profile": {"name": profile_name},
        }
        for candidate_path in candidate_paths:
            try:
                loader = DataLoader(candidate_path)
                if loader.resolve_case_for_config(lookup_config):
                    return candidate_path
            except Exception:
                continue

        return candidate_paths[0]

    def _resolve_initial_client_profile(self, *, config: dict[str, Any], dataset_path: str) -> dict[str, Any]:
        normalized_dataset_path = str(dataset_path or "").strip()
        if not normalized_dataset_path:
            return {}

        try:
            loader = DataLoader(normalized_dataset_path)
            case = loader.resolve_case_for_config(config)
            party_role = str(config.get("party_role", "plaintiff") or "plaintiff").strip() or "plaintiff"
            if party_role == "defendant":
                return loader.extract_defendant_profile(case)
            return loader.extract_plaintiff_profile(case)
        except Exception:
            return {}

    def _resolve_initial_case_stage(self, *, config: dict[str, Any], dataset_path: str) -> str:
        normalized_dataset_path = str(dataset_path or "").strip()
        if not normalized_dataset_path:
            return ""

        try:
            loader = DataLoader(normalized_dataset_path)
            case = loader.resolve_case_for_config(config)
            return str(loader.extract_case_background(case) or "").strip()
        except Exception:
            return ""

    def _resolve_initial_lawyer_profile(
        self,
        *,
        config_path: Path,
        config: dict[str, Any],
        lawyer_roster_cache: dict[Path, dict[str, dict[str, Any]]],
    ) -> dict[str, Any]:
        profile = config.get("profile", {})
        if not isinstance(profile, dict):
            profile = {}
        profile = dict(profile)

        lawyer_dir = config_path.parent
        firm_dir = lawyer_dir.parent.parent
        roster_file = firm_dir / "lawyer_roster.yaml"
        roster_profiles = lawyer_roster_cache.get(roster_file)
        if roster_profiles is None:
            roster_profiles = {}
            roster_payload = self._load_yaml(roster_file) if roster_file.exists() else {}
            lawyers = roster_payload.get("lawyers", [])
            if isinstance(lawyers, list):
                for item in lawyers:
                    if not isinstance(item, dict):
                        continue
                    lawyer_id = str(item.get("id", "") or "").strip()
                    if lawyer_id:
                        roster_profiles[lawyer_id] = item
            lawyer_roster_cache[roster_file] = roster_profiles

        lawyer_id = str(profile.get("lawyer_id", "") or lawyer_dir.name).strip() or lawyer_dir.name
        roster_entry = roster_profiles.get(lawyer_id, {})
        roster_payload = self._load_yaml(roster_file) if roster_file.exists() else {}

        specialty = profile.get("specialty")
        if not isinstance(specialty, list) or not specialty:
            specialty = roster_entry.get("specialty", [])
        if not isinstance(specialty, list):
            specialty = []

        profile["lawyer_id"] = lawyer_id
        profile["name"] = str(profile.get("name", "") or roster_entry.get("name", "") or lawyer_id).strip()
        profile["firm_id"] = str(profile.get("firm_id", "") or roster_payload.get("firm_id", "") or firm_dir.name).strip()
        profile["law_firm"] = str(profile.get("law_firm", "") or roster_payload.get("firm_name", "") or "").strip()
        profile["specialty"] = specialty
        profile["seniority"] = str(profile.get("seniority", "") or roster_entry.get("seniority", "") or "Partner").strip()
        return profile

    @staticmethod
    def _project_dataset_candidates() -> list[str]:
        project_root = Path(__file__).resolve().parents[3]
        data_dir = project_root / "data"
        if not data_dir.exists():
            return []

        preferred: list[Path] = sorted(data_dir.glob("*question*.json"))
        fallback: list[Path] = sorted(data_dir.glob("*.json"))
        ordered: list[str] = []
        seen: set[str] = set()
        for candidate in preferred + fallback:
            resolved = str(candidate.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            ordered.append(resolved)
        return ordered

    @staticmethod
    def _resolve_seed_dataset_path_legacy(*, config_dataset_path: str, fallback_dataset_path: str) -> str:
        normalized_config_path = str(config_dataset_path or "").strip()
        if normalized_config_path:
            raw_path = Path(normalized_config_path.replace("\\", "/")).expanduser()
            if raw_path.exists():
                return str(raw_path.resolve())

            candidate_name = Path(normalized_config_path.replace("\\", "/")).name
            if candidate_name:
                project_root = Path(__file__).resolve().parents[3]
                repo_candidate = project_root / "data" / candidate_name
                if repo_candidate.exists():
                    return str(repo_candidate.resolve())

        return fallback_dataset_path

    def update_sandbox_status(
        self,
        *,
        session: Session,
        sandbox: Sandbox,
        sandbox_status: str,
        simulation_status: str,
        active_cases: int,
        clients_connected: int,
    ) -> SandboxRuntimeSnapshot:
        sandbox.status = sandbox_status
        snapshot = session.get(SandboxRuntimeSnapshot, sandbox.id)
        if snapshot is None:
            snapshot = SandboxRuntimeSnapshot(sandbox_id=sandbox.id)
            session.add(snapshot)

        snapshot.simulation_status = simulation_status
        snapshot.active_cases = active_cases
        snapshot.clients_connected = clients_connected
        session.flush()
        session.refresh(snapshot)
        return snapshot

