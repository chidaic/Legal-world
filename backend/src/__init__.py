"""SimLawFirm - A flexible multi-agent framework for legal scenarios."""

from .version import BACKEND_VERSION

__version__ = BACKEND_VERSION

__all__ = [
    "BaseAgent",
    "LawyerAgent",
    "ClientAgent",
]


def __getattr__(name: str):
    if name in {"BaseAgent", "LawyerAgent", "ClientAgent"}:
        from .agents import BaseAgent, ClientAgent, LawyerAgent

        return {
            "BaseAgent": BaseAgent,
            "LawyerAgent": LawyerAgent,
            "ClientAgent": ClientAgent,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
