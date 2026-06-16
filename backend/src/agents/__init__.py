"""Agent module for SimLawFirm framework.

This module provides agent classes for legal simulation scenarios.
"""

from .base_agent import BaseAgent
from .client_agent import ClientAgent
from .lawyer_agent import LawyerAgent
from .judge_agent import JudgeAgent

__all__ = ["BaseAgent", "ClientAgent", "LawyerAgent", "JudgeAgent"]
