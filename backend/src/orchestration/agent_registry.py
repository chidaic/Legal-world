"""配置驱动的 Agent 发现与注册中心 (AgentRegistry)。

扫描 sandbox_data/ 目录结构，自动发现并实例化所有 Agent（shell 模式）。
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.event_bus import EventBus
    from ..core.file_storage_manager import FileStorageManager
    from ..agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Discovers and manages all sandbox agents from sandbox_data/ configs."""

    def __init__(
        self,
        sandbox_data_dir: Path,
        event_bus: "EventBus",
        storage: "FileStorageManager",
        map_engine: Optional[Any] = None,
    ):
        self.sandbox_data_dir = Path(sandbox_data_dir)
        self.event_bus = event_bus
        self.storage = storage
        self.map_engine = map_engine
        self._agents: Dict[str, "BaseAgent"] = {}
        self._firms: Dict[str, dict] = {}

    def discover_all(self) -> None:
        """Scan sandbox_data/ and instantiate all agents as shells."""
        self._discover_clients()
        self._discover_law_firms()
        self._discover_judges()

        # 让 ReceptionistAgent 能通过 event_bus 查找其他 Agent
        self.event_bus._registry = self

        logger.info(
            f"[Registry] Discovered {len(self._agents)} agents: "
            f"{[a.name for a in self._agents.values()]}"
        )

    def get_all_agents(self) -> List["BaseAgent"]:
        return list(self._agents.values())

    def get_agent(self, agent_id: str) -> Optional["BaseAgent"]:
        return self._agents.get(agent_id)

    def get_agents_by_type(self, agent_type: str) -> List["BaseAgent"]:
        """Get all agents of a given type: 'client', 'lawyer', 'judge', 'receptionist'."""
        type_map = {
            "client": "ClientAgent",
            "lawyer": "LawyerAgent",
            "judge": "JudgeAgent",
            "receptionist": "ReceptionistAgent",
        }
        cls_name = type_map.get(agent_type, agent_type)
        return [a for a in self._agents.values() if a.__class__.__name__ == cls_name]

    def get_firm_roster(self, firm_id: str) -> dict:
        return self._firms.get(firm_id, {})

    # ── Discovery Methods ──

    def _discover_clients(self) -> None:
        """Scan both new case-based structure and legacy clients directory."""
        from ..agents.client_agent import ClientAgent

        # Priority 1: Scan new case-based structure (cases/case_*/plaintiff|defendant/)
        cases_dir = self.sandbox_data_dir / "cases"
        if cases_dir.exists():
            for case_dir in sorted(cases_dir.iterdir()):
                if not case_dir.is_dir() or not case_dir.name.startswith("case_"):
                    continue

                case_id = case_dir.name.replace("case_", "")

                for party_role in ["plaintiff", "defendant"]:
                    party_dir = case_dir / party_role
                    config_file = party_dir / "config.yaml"
                    if not config_file.exists():
                        continue

                    config = self.storage.load_agent_config(str(party_dir))
                    profile = config.get("profile", {})
                    agent_id = f"{case_dir.name}_{party_role}"

                    agent = ClientAgent(
                        agent_id=agent_id,
                        name=profile.get("name", agent_id),
                        gender=profile.get("gender", ""),
                        role=party_role,
                        event_bus=self.event_bus,
                        storage=self.storage,
                        config_path=str(party_dir),
                    )
                    agent.register_sandbox_events()
                    self._agents[agent_id] = agent
                    logger.info(f"[Registry] Discovered client (case-based): {agent.name} ({agent_id})")

        # Priority 2: Scan legacy structure (agents/clients/*/)
        clients_dir = self.sandbox_data_dir / "agents" / "clients"
        if clients_dir.exists():
            for client_dir in sorted(clients_dir.iterdir()):
                config_file = client_dir / "config.yaml"
                if not config_file.exists():
                    continue

                config = self.storage.load_agent_config(str(client_dir))
                profile = config.get("profile", {})
                agent_id = client_dir.name  # e.g. "client_001"

                # Skip if already registered from case-based structure
                if agent_id in self._agents:
                    continue

                agent = ClientAgent(
                    agent_id=agent_id,
                    name=profile.get("name", agent_id),
                    gender=profile.get("gender", ""),
                    role=config.get("party_role", "plaintiff"),
                    event_bus=self.event_bus,
                    storage=self.storage,
                    config_path=str(client_dir),
                )
                agent.register_sandbox_events()
                self._agents[agent_id] = agent
                logger.info(f"[Registry] Discovered client (legacy): {agent.name} ({agent_id})")

    def _discover_law_firms(self) -> None:
        """Scan sandbox_data/law_firms/*/lawyer_roster.yaml and lawyers/*/config.yaml"""
        from ..agents.lawyer_agent import LawyerAgent
        from ..agents.receptionist_agent import ReceptionistAgent

        firms_dir = self.sandbox_data_dir / "law_firms"
        if not firms_dir.exists():
            return

        for firm_dir in sorted(firms_dir.iterdir()):
            roster_file = firm_dir / "lawyer_roster.yaml"
            if not roster_file.exists():
                continue

            roster = self.storage.load_yaml(roster_file)
            firm_id = roster.get("firm_id", firm_dir.name)
            self._firms[firm_id] = roster

            # Create receptionist for this firm
            receptionist = ReceptionistAgent(
                firm_id=firm_id,
                event_bus=self.event_bus,
                storage=self.storage,
                config_path=str(firm_dir),
                map_engine=self.map_engine,
            )
            receptionist._firm_dir = str(firm_dir)
            self._agents[receptionist.agent_id] = receptionist

            # Discover lawyers
            lawyers_dir = firm_dir / "lawyers"
            if not lawyers_dir.exists():
                continue

            for lawyer_dir in sorted(lawyers_dir.iterdir()):
                config_file = lawyer_dir / "config.yaml"
                if not config_file.exists():
                    continue

                config = self.storage.load_agent_config(str(lawyer_dir))
                profile = self._resolve_lawyer_profile(config=config, roster=roster, lawyer_dir=lawyer_dir)
                agent_id = profile.get("lawyer_id", lawyer_dir.name)

                agent = LawyerAgent(
                    agent_id=agent_id,
                    name=profile.get("name", agent_id),
                    law_firm=roster.get("firm_name", ""),
                    firm_id=firm_id,
                    event_bus=self.event_bus,
                    storage=self.storage,
                    config_path=str(lawyer_dir),
                )
                agent._firm_dir = str(firm_dir)
                agent.register_sandbox_events()
                self._agents[agent_id] = agent
                logger.info(f"[Registry] Discovered lawyer: {agent.name} ({agent_id}) @ {roster.get('firm_name', '')}")

    @staticmethod
    def _resolve_lawyer_profile(config: dict, roster: dict, lawyer_dir: Path) -> dict:
        profile = config.get("profile", {})
        if not isinstance(profile, dict):
            profile = {}
        profile = dict(profile)

        roster_by_id: dict[str, dict] = {}
        lawyers = roster.get("lawyers", [])
        if isinstance(lawyers, list):
            for item in lawyers:
                if not isinstance(item, dict):
                    continue
                lawyer_id = str(item.get("id", "") or "").strip()
                if lawyer_id:
                    roster_by_id[lawyer_id] = item

        lawyer_id = str(profile.get("lawyer_id", "") or lawyer_dir.name).strip() or lawyer_dir.name
        roster_entry = roster_by_id.get(lawyer_id, {})

        specialty = profile.get("specialty")
        if not isinstance(specialty, list) or not specialty:
            specialty = roster_entry.get("specialty", [])
        if not isinstance(specialty, list):
            specialty = []

        profile["lawyer_id"] = lawyer_id
        profile["name"] = str(profile.get("name", "") or roster_entry.get("name", "") or lawyer_id).strip()
        profile["firm_id"] = str(profile.get("firm_id", "") or roster.get("firm_id", "") or lawyer_dir.parent.parent.name).strip()
        profile["law_firm"] = str(profile.get("law_firm", "") or roster.get("firm_name", "") or "").strip()
        profile["specialty"] = specialty
        profile["seniority"] = str(profile.get("seniority", "") or roster_entry.get("seniority", "") or "Partner").strip()
        return profile

    def _discover_judges(self) -> None:
        """Scan sandbox_data/court_system/*/judges/*/config.yaml"""
        from ..agents.judge_agent import JudgeAgent

        court_dir = self.sandbox_data_dir / "court_system"
        if not court_dir.exists():
            return

        for court_level_dir in sorted(court_dir.iterdir()):
            if not court_level_dir.is_dir():
                continue

            # Infer court_level from directory name
            dir_name = court_level_dir.name  # e.g. "basic_court", "intermediate_court"
            if "basic" in dir_name:
                court_level = "basic"
            elif "intermediate" in dir_name:
                court_level = "intermediate"
            else:
                court_level = dir_name.replace("_court", "")

            judges_dir = court_level_dir / "judges"
            if not judges_dir.exists():
                continue

            for judge_dir in sorted(judges_dir.iterdir()):
                config_file = judge_dir / "config.yaml"
                if not config_file.exists():
                    continue

                config = self.storage.load_agent_config(str(judge_dir))
                profile = config.get("profile", {})
                agent_id = judge_dir.name  # e.g. "judge_basic_01"

                agent = JudgeAgent(
                    agent_id=agent_id,
                    name=profile.get("name", agent_id),
                    court_name=profile.get("court_name", "人民法院"),
                    court_level=profile.get("court_level", court_level),
                    years_of_experience=profile.get("years_of_experience"),
                    event_bus=self.event_bus,
                    storage=self.storage,
                    config_path=str(judge_dir),
                )
                agent.register_sandbox_events()
                self._agents[agent_id] = agent
                logger.info(f"[Registry] Discovered judge: {agent.name} ({agent_id}) @ {agent.court_name}")
