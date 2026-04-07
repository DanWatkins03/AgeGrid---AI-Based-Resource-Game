# AgeGrid - AI Strategy Sandbox

AgeGrid is a lightweight turn-based strategy simulation built to explore AI decision-making in a Civilization-inspired environment.

The project focuses on resource management, tech progression, production, combat, and agent benchmarking inside a deterministic grid world. It is designed as both a portfolio project and a sandbox for experimenting with heuristic and, later, learning-based agents.

## Current Features

- Symmetric grid map generation
- Two factions with bases, workers, and military units
- Per-turn action limits for both factions
- Resource gathering and passive economic scaling through buildings
- Tech progression with unlock requirements
- Building and unit production systems
- Unit combat and base destruction victory condition
- Pluggable agent system with `Random`, `Greedy`, and `Heuristic` agents
- Pygame viewer with faction agent selection and turn-by-turn stepping
- Headless simulation runner for agent benchmarking
- Automated tests covering progression, movement, combat, and victory flow

## Project Structure

```text
src/agegrid/
  agents/
    base.py
    greedy.py
    heuristic.py
    random.py
    registry.py
  env/
    actions.py
    agegrid_env.py
    entities.py
    state.py
    systems/
      combat.py
      economy.py
      mapgen.py
      movement.py
      production.py
      tech.py
      victory.py
  runner/
    simulate.py
  ui/
    pygame_viewer.py
```

The environment acts as the game orchestrator, while rule logic lives in focused systems modules so new mechanics can be added without turning the main environment into one large file.

## Running the Project

From the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Launch the viewer:

```powershell
python -m src.agegrid.main
```

Run the headless simulation benchmark:

```powershell
python -m src.agegrid.runner.simulate
```

Run the test suite:

```powershell
python -m unittest discover -s tests -v
```

## Viewer Controls

Setup screen:

- Click each faction card to cycle agents
- `A` / `D` changes the Red faction agent
- `J` / `L` changes the Blue faction agent
- `Enter` or `Space` starts the match

In match:

- `Space` or `Enter` advances one full turn
- Click `Next Turn` to advance one full turn
- `P` writes a debug snapshot to `agegrid_debug_snapshot.txt`
- `R` returns to the setup screen
- `Esc` closes the viewer

## Agent Overview

- `Random`: chooses randomly from currently legal actions
- `Greedy`: follows a simple economy-first progression strategy
- `Heuristic`: researches, builds, trains, and attacks with a more structured priority system

## Why This Project Exists

AgeGrid is meant to show more than a toy game loop. The main goal is to build a modular strategy environment where agents must make tradeoffs under action, resource, and positioning constraints.

That makes it useful for:

- benchmarking hand-written policies
- comparing agent behavior under identical rules
- testing game-system design for strategic depth
- creating a base for future search or reinforcement learning experiments

## Roadmap

Planned improvements include:

- stronger military and economic balance
- richer unit and building types
- better tactical decision-making and pathfinding
- cleaner observation interfaces for agents
- more benchmark scenarios and evaluation metrics
- reinforcement learning or self-play experiments

## Status

Active prototype with working simulation, viewer, agent baselines, and tests.
