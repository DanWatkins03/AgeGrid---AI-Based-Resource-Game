from __future__ import annotations

from dataclasses import dataclass

from src.agegrid.agents.greedy import GreedyAgent
from src.agegrid.agents.heuristic import HEURISTIC_PROFILES, HeuristicAgent
from src.agegrid.agents.random import RandomAgent


@dataclass(frozen=True)
class AgentSpec:
    key: str
    label: str
    description: str


AGENT_SPECS: tuple[AgentSpec, ...] = (
    AgentSpec("heuristic", "Heuristic", "Balanced planner that develops, defends, and pushes with combined arms."),
    AgentSpec("greedy", "Greedy", "Economic-biased heuristic that develops efficiently before committing."),
    AgentSpec("aggressive", "Aggressive", "Pushes pressure earlier with leaner home defense and more cavalry."),
    AgentSpec("defensive", "Defensive", "Stabilizes home threats sooner and keeps a safer defensive core."),
    AgentSpec("random", "Random", "Picks from legal actions at random."),
)


def create_agent(key: str, *, seed: int = 0):
    if key == "heuristic":
        return HeuristicAgent(profile=HEURISTIC_PROFILES["balanced"])
    if key == "greedy":
        return GreedyAgent()
    if key == "aggressive":
        return HeuristicAgent(profile=HEURISTIC_PROFILES["aggressive"])
    if key == "defensive":
        return HeuristicAgent(profile=HEURISTIC_PROFILES["defensive"])
    if key == "random":
        return RandomAgent(seed=seed)
    raise ValueError(f"Unknown agent key: {key}")
