# AgeGrid - AI Strategy Sandbox

AgeGrid is a lightweight turn-based strategy sandbox for exploring AI decision-making in a Civilization-inspired ruleset.

The project combines economy, tech progression, production, combat, diplomacy, and agent benchmarking inside a deterministic strategy environment. It is designed both as a portfolio project and as a practical sandbox for building and testing heuristic agents before moving into more advanced search or learning-based approaches.

## Current Features

- Staggered hex-grid board with centered tile, object, and character rendering
- Symmetric map generation with resources, bases, and production buildings
- Two playable factions with workers, soldiers, archers, and horsemen
- Resource gathering, passive income, and economy scaling through buildings
- Tech progression with unlock chains, military upgrades, and era tracking
- Building and unit production systems with terrain/resource requirements
- Combat, base destruction, collapse handling, and recovery behavior
- Lightweight diplomacy layer with `Peace`, `War`, and `Truce`
- Heuristic AI with rally, push, siege, defense, rebuild, and recovery behaviors
- Pygame viewer with agent selection, inspect panels, zoom, pan, and debug snapshot export
- Headless simulation runner for agent benchmarking
- Automated regression tests covering progression, movement, combat, diplomacy, and heuristic behavior

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
- Left click selects units, buildings, bases, and resources to open inspect panels
- Mouse wheel zooms the board in and out
- Middle mouse drag pans the camera
- Middle click resets the camera to the default centered view
- Use the on-board `Reset` button to reset the camera
- `P` writes a debug snapshot to `agegrid_debug_snapshot.txt`
- `R` returns to the setup screen
- `Esc` closes the viewer

## Agent Overview

- `Random`: chooses randomly from currently legal actions
- `Greedy`: follows a simple economy-first progression strategy
- `Heuristic`: researches, builds, declares war, defends, rallies, sieges, and recovers using a structured priority system

## Visuals

The current viewer is built around a presentable prototype-style strategy board rather than pure debug rendering. It includes:

- hex-tile terrain rendering
- object and building sprites
- character sprites for units with tech-based visual upgrades
- hover and selection highlights
- inspect panels for units, resources, buildings, and bases
- camera zoom and panning for browsing the board

## Why This Project Exists

AgeGrid is meant to show more than a toy game loop. The main goal is to build a modular strategy environment where agents must make tradeoffs under action, resource, and positioning constraints.

That makes it useful for:

- benchmarking hand-written policies
- comparing agent behavior under identical rules
- testing game-system design for strategic depth
- creating a base for future search or reinforcement learning experiments

## Roadmap

Planned improvements include:

- stronger military, diplomacy, and economy balance
- more factions and broader multi-team support
- richer unit, building, and world-object types
- better tactical coordination and long-match pacing
- cleaner observation interfaces for agents
- more benchmark scenarios and evaluation metrics
- reinforcement learning or self-play experiments

## Asset Credits

AgeGrid uses Kenney-style game assets from local project asset packs for board tiles, objects, UI, and character visuals.

- Kenney
- Public-domain / free game art asset packs used through the local `assets/` folders in this project
- Website: [https://kenney.nl/](https://kenney.nl/)

## Status

Active prototype with a working simulation, hex-based viewer, diplomacy-enabled heuristic agents, and a growing regression test suite.
