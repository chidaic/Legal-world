"""Checkpoint Manager for simulation state persistence and recovery.

Manages checkpoint files in YAML format for human readability and debugging.
Handles session state, scenario checkpoints, and recovery logic.
"""

import logging
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages simulation checkpoints for crash recovery and pause/resume."""

    def __init__(self, checkpoint_dir: Path):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.session_state_file = self.checkpoint_dir / "session_state.yaml"
        self._session_state: Optional[Dict[str, Any]] = None
        self._event_bus: Optional[Any] = None  # Reference to EventBus for syncing

    def create_new_session(self) -> str:
        """Create a new simulation session.

        Returns:
            Session ID
        """
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._session_state = {
            "session_id": session_id,
            "last_checkpoint_time": datetime.now().isoformat(),
            "simulation_status": "running",
            "active_scenarios": [],
            "active_scenario_details": {},  # New field for real-time scenario tracking
        }
        self._save_session_state()
        logger.info(f"Created new session: {session_id}")
        return session_id

    def load_session_state(self) -> Optional[Dict[str, Any]]:
        """Load session state from disk.

        Returns:
            Session state dict or None if no session exists
        """
        if not self.session_state_file.exists():
            return None

        try:
            with open(self.session_state_file, 'r', encoding='utf-8') as f:
                self._session_state = yaml.safe_load(f)
            logger.info(f"Loaded session state: {self._session_state.get('session_id')}")
            return self._session_state
        except Exception as e:
            logger.error(f"Failed to load session state: {e}")
            return None

    def _save_session_state(self) -> None:
        """Save session state to disk."""
        if not self._session_state:
            return

        self._session_state["last_checkpoint_time"] = datetime.now().isoformat()

        try:
            with open(self.session_state_file, 'w', encoding='utf-8') as f:
                yaml.dump(self._session_state, f, allow_unicode=True, default_flow_style=False)
        except Exception as e:
            logger.error(f"Failed to save session state: {e}")

    def register_scenario(
        self,
        scenario_id: str,
        case_id: str,
        scenario_type: str,
        party_role: str,
        client_id: str,
        lawyer_id: str,
    ) -> None:
        """Register a new scenario in the session.

        Args:
            scenario_id: Unique scenario identifier
            case_id: Case ID
            scenario_type: Type of scenario (e.g., "LC")
            party_role: Party role (e.g., "plaintiff")
            client_id: Client agent ID
            lawyer_id: Lawyer agent ID
        """
        if not self._session_state:
            self.create_new_session()

        checkpoint_file = f"{scenario_id}_checkpoint.yaml"

        scenario_info = {
            "scenario_id": scenario_id,
            "case_id": case_id,
            "scenario_type": scenario_type,
            "party_role": party_role,
            "client_id": client_id,
            "lawyer_id": lawyer_id,
            "status": "in_progress",
            "checkpoint_file": checkpoint_file,
        }

        # Remove existing entry if present
        self._session_state["active_scenarios"] = [
            s for s in self._session_state.get("active_scenarios", [])
            if s["scenario_id"] != scenario_id
        ]

        self._session_state["active_scenarios"].append(scenario_info)
        self._save_session_state()
        logger.info(f"Registered scenario: {scenario_id}")

    def mark_scenario_completed(self, scenario_id: str) -> None:
        """Mark a scenario as completed.

        Args:
            scenario_id: Scenario identifier
        """
        if not self._session_state:
            return

        for scenario in self._session_state.get("active_scenarios", []):
            if scenario["scenario_id"] == scenario_id:
                scenario["status"] = "completed"
                break

        self._save_session_state()
        logger.info(f"Marked scenario completed: {scenario_id}")

    def save_scenario_checkpoint(
        self,
        scenario_id: str,
        checkpoint_data: Dict[str, Any],
    ) -> None:
        """Save scenario checkpoint to disk.

        Args:
            scenario_id: Scenario identifier
            checkpoint_data: Checkpoint data to save
        """
        # Find checkpoint file name
        checkpoint_file = None
        if self._session_state:
            for scenario in self._session_state.get("active_scenarios", []):
                if scenario["scenario_id"] == scenario_id:
                    checkpoint_file = scenario["checkpoint_file"]
                    break

        if not checkpoint_file:
            checkpoint_file = f"{scenario_id}_checkpoint.yaml"

        checkpoint_path = self.checkpoint_dir / checkpoint_file

        # Add timestamp
        checkpoint_data["timestamp"] = datetime.now().isoformat()

        try:
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                yaml.dump(checkpoint_data, f, allow_unicode=True, default_flow_style=False)
            logger.info(f"Saved checkpoint: {checkpoint_file}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint {checkpoint_file}: {e}")

    def load_scenario_checkpoint(self, checkpoint_file: str) -> Optional[Dict[str, Any]]:
        """Load scenario checkpoint from disk.

        Args:
            checkpoint_file: Checkpoint file name

        Returns:
            Checkpoint data or None if not found
        """
        checkpoint_path = self.checkpoint_dir / checkpoint_file

        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint file not found: {checkpoint_file}")
            return None

        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            logger.info(f"Loaded checkpoint: {checkpoint_file}")
            return data
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_file}: {e}")
            return None

    def get_incomplete_scenarios(self) -> List[Dict[str, Any]]:
        """Get list of incomplete scenarios from session state.

        Returns:
            List of scenario info dicts with status "in_progress"
        """
        if not self._session_state:
            return []

        return [
            s for s in self._session_state.get("active_scenarios", [])
            if s["status"] == "in_progress"
        ]

    def mark_session_completed(self) -> None:
        """Mark the current session as completed."""
        if not self._session_state:
            return

        self._session_state["simulation_status"] = "completed"
        self._save_session_state()
        logger.info("Marked session as completed")

    def mark_session_paused(self) -> None:
        """Mark the current session as paused."""
        if not self._session_state:
            return

        self._session_state["simulation_status"] = "paused"
        self._save_session_state()
        logger.info("Marked session as paused")

    def mark_session_running(self) -> None:
        """Mark the current session as running."""
        if not self._session_state:
            return

        self._session_state["simulation_status"] = "running"
        self._save_session_state()
        logger.info("Marked session as running")

    def cleanup_completed_checkpoints(self) -> None:
        """Remove checkpoint files for completed scenarios."""
        if not self._session_state:
            return

        for scenario in self._session_state.get("active_scenarios", []):
            if scenario["status"] == "completed":
                checkpoint_file = scenario["checkpoint_file"]
                checkpoint_path = self.checkpoint_dir / checkpoint_file
                if checkpoint_path.exists():
                    try:
                        checkpoint_path.unlink()
                        logger.info(f"Cleaned up checkpoint: {checkpoint_file}")
                    except Exception as e:
                        logger.warning(f"Failed to cleanup checkpoint {checkpoint_file}: {e}")

    def set_event_bus(self, event_bus: Any) -> None:
        """Set reference to EventBus for syncing active scenarios.

        Args:
            event_bus: EventBus instance
        """
        self._event_bus = event_bus

    def sync_active_scenarios_from_event_bus(self) -> None:
        """Sync active scenarios from EventBus to session state.

        This should be called periodically or after scenario registration/unregistration.
        """
        if not self._event_bus or not self._session_state:
            return

        # Get snapshot from EventBus
        active_scenario_details = self._event_bus.get_active_scenarios_snapshot()

        # Update session state
        self._session_state["active_scenario_details"] = active_scenario_details
        self._save_session_state()

        logger.debug(
            f"[CheckpointManager] Synced {len(active_scenario_details)} active scenarios from EventBus"
        )
