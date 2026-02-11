from src.agegrid.env.agegrid_env import AgeGridEnv
"""


def main():
    env = AgeGridEnv()
    red_worker_id = next(u.id for u in env.units if u.faction == "Red")

    # pick the closest resource node to Red worker
    worker = next(u for u in env.units if u.id == red_worker_id)
    closest = min(
        env.resources,
        key=lambda r: abs(r.position[0] - worker.position[0]) + abs(r.position[1] - worker.position[1]),
    )
    target = closest.position

    print("Initial state:")
    print(env.summary())
    print("\nClosest resource:", target)

    # Walk to resource
    steps = 0
    while worker.position != target and steps < 50:
        env.move_towards(red_worker_id, target)
        steps += 1

    print("\nAfter walking:")
    print(env.summary())

    # Gather a few times
    print("\nGathering 3 times...")
    for i in range(3):
        ok = env.gather(red_worker_id)
        print(f"Gather {i+1}: {ok} | Bank Red={env.bank['Red']} | Remaining={env._resource_at(worker.position).remaining}")

    print("\nFinal:")
    print(env.summary())


if __name__ == "__main__":
    main()
"""
from src.agegrid.ui.pygame_viewer import run_viewer


def main():
    run_viewer()


if __name__ == "__main__":
    main()
