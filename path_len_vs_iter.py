#!/usr/bin/env python3
"""
Plot best path length vs. iterations for RRT* and plain RRT.

Runs both planners from rrt_planner1.py on the same world / start / goal
over several random seeds, recording the best start-to-goal path cost
through the tree after every iteration:

  - RRT* keeps rewiring, so its best path keeps getting shorter.
  - Plain RRT stops at the first path it finds, so its line goes flat
    from that point on (it never improves).

The A* grid path is drawn as a dashed reference line. All lengths are the
raw tree / grid path, before simplify_path() removes redundant waypoints.

Example:
  python3 path_len_vs_iter.py generated_world.sdf --start 9 9 --goal -9.5 -9.5
"""

import argparse
import math
import os
import statistics

from rrt_planner1 import astar, build_occupancy_grid_from_world, random_free_point, rrt

COLORS = {"rrtstar": "#2a78d6", "rrt": "#eb6834"}  # categorical slots 1 and 2
LABELS = {"rrtstar": "RRT*", "rrt": "RRT"}
TEXT, MUTED = "#2f3e46", "#6b7780"


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Plot best RRT / RRT* path length against iteration count.")
    parser.add_argument("world_file", help="Path to the PX4/Gazebo .world (SDF) file")
    parser.add_argument("--start", type=float, nargs=2, metavar=("X", "Y"), default=None,
                        help="Start point in meters (default: random, from --seed-base)")
    parser.add_argument("--goal", type=float, nargs=2, metavar=("X", "Y"), default=None,
                        help="Goal point in meters (default: random, from --seed-base)")
    parser.add_argument("--iters", type=int, default=8000,
                        help="Iterations per run (default: 8000)")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of random seeds per method (default: 5)")
    parser.add_argument("--seed-base", type=int, default=1,
                        help="First seed; runs use seed-base .. seed-base+seeds-1 (default: 1)")
    parser.add_argument("--methods", nargs="+", choices=["rrtstar", "rrt"],
                        default=["rrtstar", "rrt"], help="Planners to compare (default: both)")
    parser.add_argument("--rrt-step", type=float, default=1.0,
                        help="RRT/RRT* max extension step in meters (default: 1.0)")
    parser.add_argument("--no-astar", action="store_true", help="Skip the A* reference line")
    parser.add_argument("--resolution", type=float, default=0.5,
                        help="Grid cell size in meters (default: 0.5)")
    parser.add_argument("--inflate", type=float, default=0.5,
                        help="Obstacle inflation radius in meters (default: 0.5)")
    parser.add_argument("--margin", type=float, default=5.0,
                        help="Free-space margin around obstacles (default: 5.0)")
    parser.add_argument("--altitude", type=float, default=None,
                        help="Flight altitude in meters (default: consider all obstacles)")
    parser.add_argument("--plot", default="path_len_vs_iter.png",
                        help="Output plot path (default: path_len_vs_iter.png)")
    return parser


def _path_length(path):
    return sum(math.dist(a, b) for a, b in zip(path[:-1], path[1:]))


def main():
    args = _build_arg_parser().parse_args()

    grid, _ = build_occupancy_grid_from_world(
        args.world_file, resolution_m=args.resolution, inflate_m=args.inflate,
        margin_m=args.margin, flight_altitude_m=args.altitude, verbose=False)

    import random
    random.seed(args.seed_base)
    start = tuple(args.start) if args.start else random_free_point(grid)
    goal = tuple(args.goal) if args.goal else random_free_point(grid)
    print(f"Start: {start}  Goal: {goal}")

    seeds = list(range(args.seed_base, args.seed_base + args.seeds))
    runs = {}  # method -> list of per-iteration best-cost histories
    for method in args.methods:
        runs[method] = []
        for seed in seeds:
            history = []
            rrt(grid, start, goal, star=(method == "rrtstar"), max_iter=args.iters,
                step_m=args.rrt_step, seed=seed, history=history)
            # Plain RRT stops at the first path; it keeps that path afterwards
            history += [history[-1]] * (args.iters + 1 - len(history))
            runs[method].append(history)
            first = next((k for k, c in enumerate(history) if c < math.inf), None)
            print(f"{LABELS[method]:5s} seed {seed}: first path at iteration "
                  f"{first if first is not None else '-'}, final length "
                  f"{history[-1]:.2f} m")

    astar_len = None
    if not args.no_astar:
        a_path = astar(grid, start, goal)
        if a_path:
            astar_len = _path_length(a_path)
            print(f"A*    grid path: {astar_len:.2f} m")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor("#f4f6f8")
    ax.grid(True, which="major", color="white", linewidth=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    iters = list(range(args.iters + 1))
    y_values = []
    end_labels = []  # (y, text, color) for direct labels at the right edge
    for method, histories in runs.items():
        color = COLORS[method]
        # One thin line per seed
        for h in histories:
            ks = [k for k in iters if h[k] < math.inf]
            if ks:
                ax.plot(ks, [h[k] for k in ks], color=color, linewidth=0.8, alpha=0.3)
        # Median across seeds, from the iteration where every seed has a path
        median = []
        for k in iters:
            vals = [h[k] for h in histories]
            median.append(statistics.median(vals) if max(vals) < math.inf else None)
        ks = [k for k in iters if median[k] is not None]
        if not ks:
            continue
        ax.plot(ks, [median[k] for k in ks], color=color, linewidth=2.0,
                label=f"{LABELS[method]} median of {len(histories)} seeds")
        ax.plot(ks[0], median[ks[0]], "o", color=color, markersize=8,
                markeredgecolor="white", markeredgewidth=2.0, zorder=4)
        y_values += [median[k] for k in ks]
        end_labels.append((median[-1], f"{LABELS[method]}  {median[-1]:.1f} m", color))

    if astar_len is not None:
        ax.axhline(astar_len, color=MUTED, linewidth=1.2, linestyle="--",
                   label="A* grid path")
        y_values.append(astar_len)
        end_labels.append((astar_len, f"A*  {astar_len:.1f} m", MUTED))

    # Direct labels at the right end of each line, nudged apart if they collide
    end_labels.sort()
    if y_values:
        spread = max(y_values) - min(y_values) or 1.0
        min_gap = spread * 0.05
        placed = []
        for y, text, color in end_labels:
            if placed and y - placed[-1] < min_gap:
                y = placed[-1] + min_gap
            placed.append(y)
            ax.annotate(text, xy=(args.iters, y), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=10, fontweight="bold", color=TEXT,
                        annotation_clip=False)
            ax.plot([args.iters * 1.03], [y], marker="s", color=color, markersize=6,
                    clip_on=False)

    first_found = [next((k for k, c in enumerate(h) if c < math.inf), args.iters)
                   for hs in runs.values() for h in hs]
    ax.set_xscale("log")
    ax.set_xlim(10 ** math.floor(math.log10(max(1, min(first_found)))), args.iters)
    ax.set_xlabel("iterations  (log scale)", color=TEXT)
    ax.set_ylabel("best path length  [m]", color=TEXT)
    ax.tick_params(colors=TEXT)
    ax.set_title(f"Best path length vs iterations — {os.path.basename(args.world_file)}\n"
                 f"start ({start[0]:g}, {start[1]:g}) → goal ({goal[0]:g}, {goal[1]:g})",
                 fontsize=12, fontweight="bold", color=TEXT)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.text(0.0, -0.13, "Thin lines: individual seeds.  Dot: first iteration where every "
            "seed has a path.  Lengths are raw tree / grid paths, before simplification.",
            transform=ax.transAxes, fontsize=8.5, color=MUTED)

    fig.tight_layout()
    fig.subplots_adjust(right=0.85)
    plt.savefig(args.plot, dpi=150)
    print(f"Saved plot to {args.plot}")


if __name__ == "__main__":
    main()
