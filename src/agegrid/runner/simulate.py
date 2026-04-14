from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.agegrid.agents.registry import create_agent
from src.agegrid.env.agegrid_env import AgeGridEnv


@dataclass
class EpisodeResult:
    winner: Optional[str]
    turns: int
    red_bank: int
    blue_bank: int
    ended_by: str  # "winner" or "max_turns"


@dataclass
class MatchupSummary:
    red_agent: str
    blue_agent: str
    episodes: int
    red_wins: int
    blue_wins: int
    draws: int
    ended_target: int
    ended_max: int
    avg_turns: float
    avg_red_bank: float
    avg_blue_bank: float


def run_episode(env: AgeGridEnv, red_agent, blue_agent) -> EpisodeResult:
    """
    Runs one episode until:
      - someone meets a win condition, OR
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
                ended_by="winner",
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
                ended_by="winner",
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


def run_matchup(red_key: str, blue_key: str, episodes: int = 20) -> MatchupSummary:
    red_wins = 0
    blue_wins = 0
    draws = 0
    ended_target = 0
    ended_max = 0
    total_turns = 0
    total_red_bank = 0
    total_blue_bank = 0

    for i in range(episodes):
        env = AgeGridEnv()
        red_agent = create_agent(red_key, seed=i * 2)
        blue_agent = create_agent(blue_key, seed=i * 2 + 1)

        result = run_episode(env, red_agent, blue_agent)
        total_turns += result.turns
        total_red_bank += result.red_bank
        total_blue_bank += result.blue_bank

        if result.ended_by == "winner":
            ended_target += 1
        else:
            ended_max += 1

        if result.winner == "Red":
            red_wins += 1
        elif result.winner == "Blue":
            blue_wins += 1
        else:
            draws += 1

    return MatchupSummary(
        red_agent=red_key,
        blue_agent=blue_key,
        episodes=episodes,
        red_wins=red_wins,
        blue_wins=blue_wins,
        draws=draws,
        ended_target=ended_target,
        ended_max=ended_max,
        avg_turns=total_turns / episodes,
        avg_red_bank=total_red_bank / episodes,
        avg_blue_bank=total_blue_bank / episodes,
    )


def main() -> None:
    matchups = [
        ("heuristic", "random"),
        ("heuristic", "greedy"),
        ("greedy", "random"),
    ]

    config = AgeGridEnv().config
    bank_target = config.target_bank if config.target_bank is not None else "disabled"
    print(f"AgeGrid benchmark | target_bank={bank_target} | max_turns={config.max_turns}")

    for red_key, blue_key in matchups:
        summary = run_matchup(red_key, blue_key, episodes=20)
        print(
            f"{summary.red_agent} vs {summary.blue_agent} | "
            f"wins {summary.red_wins}-{summary.blue_wins}-{summary.draws} | "
            f"ended target/max {summary.ended_target}/{summary.ended_max} | "
            f"avg turns {summary.avg_turns:.1f} | "
            f"avg bank {summary.avg_red_bank:.1f}/{summary.avg_blue_bank:.1f}"
        )


if __name__ == "__main__":
    main()
