# AgeGrid – AI-Based Resource Game

AgeGrid is a turn-based grid simulation built for experimenting with AI decision-making, agent benchmarking, and resource strategy dynamics.

The project is designed as a lightweight research playground for testing heuristic and (eventually) learning-based agents in a controlled strategy environment.

This project is still in very early stages, core mechanics and code will have massive changes in the future.

## Current features
- Symmetric grid map generation
- Two factions with bases and workers
- Worker movement with collision rules
- Resource gathering with depletion
- Simple deterministic environment API
- Basic UI implementation with pygame

## Architecture

```
src/agegrid/
  env/
    agegrid_env.py       # Core environment + turn engine
    entities.py          # Base, Unit, ResourceNode
    systems/
      mapgen.py          # Symmetric resource placement
      movement.py        # Movement rules
      economy.py         # Gathering + spawning
  ui/
    pygame_viewer.py     # Visualisation layer
```

The environment acts as an orchestrator, while rule logic is separated into systems modules for scalability.

---

## Running the Project

Clone the repo:

```bash
git clone https://github.com/DanWatkins03/AgeGrid---AI-Based-Resource-Game.git
cd AgeGrid---AI-Based-Resource-Game
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python -m src.agegrid.main
```

---

## Roadmap

Planned expansions:

- Independent multi-worker policies 
- Combat system 
- Faction age progression / tech scaling 
- Building mechanics 
- Agent abstraction layer (separate policy modules) 
- Automated headless benchmarking mode 
- Reinforcement learning integration 

---

## Project Motivation

The goal of AgeGrid is to explore how structured environments can be designed for:

- Strategy modelling 
- Agent comparison 
- Economic scaling analysis 
- Decision efficiency under constraints 

This project also serves as a sandbox for experimenting with AI systems architecture and clean simulation design.

---

## 🛠 Status

Active development. 
Core simulation loop and modular architecture established. 
Mechanics are being expanded incrementally.
