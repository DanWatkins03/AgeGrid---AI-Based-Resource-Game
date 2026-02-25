from __future__ import annotations
from typing import Protocol

from src.agegrid.env.agegrid_env import AgeGridEnv


class Agent(Protocol):
    def act(self, env: AgeGridEnv) -> tuple | None:
        """Return an action tuple (e.g. ('gather', id)) or None to stop early."""
        