from __future__ import annotations

from dataclasses import dataclass

from src.agegrid.agents.greedy import GreedyAgent
from src.agegrid.agents.heuristic import HeuristicAgent
from src.agegrid.agents.random import RandomAgent


@dataclass(frozen=True)
class AgentSpec:
    key: str
    label: str
    description: str


AGENT_SPECS: tuple[AgentSpec, ...] = (
    AgentSpec("heuristic", "Heuristic", "Techs up, builds economy/military, then attacks."),
    AgentSpec("greedy", "Greedy", "Economy-first baseline with simple progression priorities."),
    AgentSpec("random", "Random", "Picks from legal actions at random."),
)


def create_agent(key: str, *, seed: int = 0):
    if key == "heuristic":
        return HeuristicAgent(desired_workers=3)
    if key == "greedy":
        return GreedyAgent(desired_workers=2)
    if key == "random":
        return RandomAgent(seed=seed)
    raise ValueError(f"Unknown agent key: {key}")

