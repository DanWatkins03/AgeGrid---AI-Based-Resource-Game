from __future__ import annotations
import random

from src.agegrid.env.actions import Action
from src.agegrid.env.agegrid_env import AgeGridEnv


class RandomAgent:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, env: AgeGridEnv) -> Action | None:
        faction = env.factions[env.current_player]
        legal = env.legal_actions(faction)
        if not legal:
            return None
        return self.rng.choice(legal)
