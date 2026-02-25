from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.agegrid.env.agegrid_env import AgeGridEnv
from src.agegrid.agents.greedy import GreedyAgent
from src.agegrid.agents.random import RandomAgent


@dataclass
class EpisodeResult:
    winner: Optional[str]
    turns: int
    red_bank: int
    blue_bank: int
    ended_by: str  # "target_bank" or "max_turns"


def run_episode(env: AgeGridEnv, red_agent, blue_agent) -> EpisodeResult:
    """
    Runs one episode until:
      - someone reaches env.config.target_bank, OR
      - env.config.max_turns is reached
    """
    while env.turn < env.config.max_turns:
        # --- Red phase ---
        env.step_faction(lambda e: red_agent.act(e))
        env.step_end_turn()

        w = env.winner()
        if w is not None:
            return EpisodeResult(
                winner=w,
                turns=env.turn,
                red_bank=env.bank["Red"],
                blue_bank=env.bank["Blue"],
                ended_by="target_bank",
            )

        # --- Blue phase ---
        env.step_faction(lambda e: blue_agent.act(e))
        env.step_end_turn()

        w = env.winner()
        if w is not None:
            return EpisodeResult(
                winner=w,
                turns=env.turn,
                red_bank=env.bank["Red"],
                blue_bank=env.bank["Blue"],
                ended_by="target_bank",
            )

    # If we hit max turns, call it by bank or draw
    winner = None
    if env.bank["Red"] > env.bank["Blue"]:
        winner = "Red"
    elif env.bank["Blue"] > env.bank["Red"]:
        winner = "Blue"

    return EpisodeResult(
        winner=winner,
        turns=env.turn,
        red_bank=env.bank["Red"],
        blue_bank=env.bank["Blue"],
        ended_by="max_turns",
    )


def main() -> None:
    episodes = 50

    red_wins = 0
    blue_wins = 0
    draws = 0

    ended_target = 0
    ended_max = 0

    total_turns = 0

    for i in range(episodes):
        env = AgeGridEnv()

        # Baseline comparison
        red = GreedyAgent(desired_workers=2)
        blue = RandomAgent(seed=i)

        result = run_episode(env, red, blue)
        total_turns += result.turns

        if result.ended_by == "target_bank":
            ended_target += 1
        else:
            ended_max += 1

        if result.winner == "Red":
            red_wins += 1
        elif result.winner == "Blue":
            blue_wins += 1
        else:
            draws += 1

    print(f"Episodes: {episodes}")
    print(f"Win condition: first to target_bank={AgeGridEnv().config.target_bank} (else max_turns)")
    print(f"Red wins: {red_wins} | Blue wins: {blue_wins} | Draws: {draws}")
    print(f"Ended by target_bank: {ended_target} | Ended by max_turns: {ended_max}")
    print(f"Avg turns: {total_turns / episodes:.1f}")


if __name__ == "__main__":
    main()