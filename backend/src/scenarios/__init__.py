"""Scenario module for SimLawFirm framework.

This module provides scenario classes for legal simulation.
"""

from .base_scenario import BaseScenario
from .legal_consultation import LegalConsultationScenario
from .complaint_drafting import ComplaintDraftingScenario
from .defense_drafting import DefenseDraftingScenario
from .court_investigation import CourtInvestigationScenario
from .appeal_drafting import AppealDraftingScenario
from .appeal_response_drafting import AppealResponseDraftingScenario
from .appeal_court_investigation import AppealCourtInvestigationScenario

__all__ = [
    "BaseScenario",
    "LegalConsultationScenario",
    "ComplaintDraftingScenario",
    "DefenseDraftingScenario",
    "CourtInvestigationScenario",
    "AppealDraftingScenario",
    "AppealResponseDraftingScenario",
    "AppealCourtInvestigationScenario",
]

