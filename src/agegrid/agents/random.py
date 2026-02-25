from __future__ import annotations
import random

from src.agegrid.env.agegrid_env import AgeGridEnv


class RandomAgent:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, env: AgeGridEnv) -> tuple | None:
        faction = env.factions[env.current_player]
        workers = [u for u in env.units if u.faction == faction and u.unit_type == "worker"]
        if not workers:
            return None

        # sometimes try spawn
        if env.bank[faction] >= env.config.worker_spawn_cost and self.rng.random() < 0.2:
            return ("spawn_worker",)

        w = self.rng.choice(workers)

        # random action
        r = self.rng.random()
        if r < 0.4:
            return ("gather", w.id)

        # random move target (still uses move_towards)
        tx = self.rng.randrange(env.config.width)
        ty = self.rng.randrange(env.config.height)
        return ("move_towards", w.id, (tx, ty))