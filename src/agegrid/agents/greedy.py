from __future__ import annotations

from src.agegrid.agents.heuristic import HEURISTIC_PROFILES, HeuristicAgent


class GreedyAgent(HeuristicAgent):
    """Compatibility wrapper around the shared heuristic engine using the greedy profile."""

    def __init__(self, desired_workers: int | None = None):
        super().__init__(desired_workers=desired_workers, profile=HEURISTIC_PROFILES["greedy"])
