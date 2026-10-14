#!/usr/bin/env python3
"""
Plot the A*, RRT and RRT* paths for the same world / start / goal on one map.

Uses the planners from rrt_planner1.py. Each path is simplified with
simplify_path() (as the drone would fly it) and drawn in its own colour,
with East on x and North on y.

Example:
  python3 plot_all_planners.py generated_world.sdf --start 9 9 --goal -9.5 -9.5 --seed 1
"""

import argparse
import math
import os
import random

from rrt_planner1 import (astar, build_occupancy_grid_from_world, random_free_point, rrt,
                          simplify_path)

# Categorical slots 1-3, in fixed order
METHODS = [
    ("astar", "A*", "#1baf7a"),
    ("rrt", "RRT", "#eb6834"),
    ("rrtstar", "RRT*", "#2a78d6"),
]
TEXT, MUTED = "#2f3e46", "#6b7780"


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Plot A*, RRT and RRT* paths on the same map.")
    parser.add_argument("world_file", help="Path to the PX4/Gazebo .world (SDF) file")
    parser.add_argument("--start", type=float, nargs=2, metavar=("X", "Y"), default=None,
                        help="Start point in meters, North East (default: random)")
    parser.add_argument("--goal", type=float, nargs=2, metavar=("X", "Y"), default=None,
                        help="Goal point in meters, North East (default: random)")
    parser.add_argument("--rrt-iters", type=int, default=4000,
                        help="RRT/RRT* iteration budget (default: 4000)")
    parser.add_argument("--rrt-step", type=float, default=1.0,
                        help="RRT/RRT* max extension step in meters (default: 1.0)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for repeatable RRT runs and random start/goal")
    parser.add_argument("--resolution", type=float, default=0.5,
                        help="Grid cell size in meters (default: 0.5)")
    parser.add_argument("--inflate", type=float, default=0.5,
                        help="Obstacle inflation radius in meters (default: 0.5)")
    parser.add_argument("--margin", type=float, default=5.0,
                        help="Free-space margin around obstacles (default: 5.0)")
    parser.add_argument("--altitude", type=float, default=None,
                        help="Flight altitude in meters (default: consider all obstacles)")
    parser.add_argument("--plot", default="all_planners.png",
                        help="Output plot path (default: all_planners.png)")
    return parser


def _path_length(path):
    return sum(math.dist(a, b) for a, b in zip(path[:-1], path[1:]))


def main():
    args = _build_arg_parser().parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    grid, footprints = build_occupancy_grid_from_world(
        args.world_file, resolution_m=args.resolution, inflate_m=args.inflate,
        margin_m=args.margin, flight_altitude_m=args.altitude, verbose=False)

    start = tuple(args.start) if args.start else random_free_point(grid)
    goal = tuple(args.goal) if args.goal else random_free_point(grid)
    print(f"Start: {start}  Goal: {goal}")

    results = []  # (label, color, path or None)
    for method, label, color in METHODS:
        if method == "astar":
            path = astar(grid, start, goal)
        else:
            path, _ = rrt(grid, start, goal, star=(method == "rrtstar"),
                          max_iter=args.rrt_iters, step_m=args.rrt_step, seed=args.seed)
        if path is None:
            print(f"{label:5s}: no path found")
        else:
            path = simplify_path(grid, path)
            print(f"{label:5s}: {_path_length(path):.2f} m, {len(path)} waypoints")
        results.append((label, color, path))

    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon, Rectangle

    # East on the horizontal axis, North on the vertical axis.
    # Grid-frame points are (North, East), so every plotted point is (p[1], p[0]).
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_facecolor("#f4f6f8")
    ax.grid(True, color="white", linewidth=1.0)
    ax.set_axisbelow(True)

    x0, y0 = grid.origin[1], grid.origin[0]
    area_w, area_h = grid.rows * grid.resolution, grid.cols * grid.resolution
    ax.add_patch(Rectangle((x0, y0), area_w, area_h, fill=False, linestyle="--",
                           edgecolor="#8a9ba8", linewidth=1.0, zorder=1))

    for fp in footprints:
        if fp["type"] == "polygon":
            patch = Polygon([(p[1], p[0]) for p in fp["points"]], closed=True)
        else:
            patch = Circle((fp["cy"], fp["cx"]), fp["radius"])
        patch.set(facecolor="#3d4a55", edgecolor="#1a1a1a", linewidth=0.6, zorder=2)
        ax.add_patch(patch)

    # Paths, with a thin surface-coloured ring so overlapping lines stay distinct
    ring = [pe.Stroke(linewidth=5.0, foreground="#f4f6f8"), pe.Normal()]
    for i, (label, color, path) in enumerate(results):
        if path is None:
            continue
        ax.plot([p[1] for p in path], [p[0] for p in path], "-", color=color,
                linewidth=3.0, solid_capstyle="round", solid_joinstyle="round",
                path_effects=ring, zorder=3 + i,
                label=f"{label}  {_path_length(path):.1f} m, {len(path)} waypoints")
        ax.plot([p[1] for p in path[1:-1]], [p[0] for p in path[1:-1]], "o",
                color=color, markersize=6, markeredgecolor="white",
                markeredgewidth=1.2, zorder=3 + i)

    def fmt(v):
        return f"{v:g}" if float(v).is_integer() else f"{v:.1f}"

    ax.scatter(start[1], start[0], s=260, color="#00c853", edgecolor="black",
               linewidth=1.3, zorder=10, label=f"start ({fmt(start[0])}, {fmt(start[1])})")
    ax.scatter(goal[1], goal[0], s=300, marker="*", color="#e00000", edgecolor="black",
               linewidth=1.0, zorder=10, label=f"goal ({fmt(goal[0])}, {fmt(goal[1])})")

    pad = 0.5
    ax.set_xlim(x0 - pad, x0 + area_w + pad)
    ax.set_ylim(y0 - pad, y0 + area_h + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x  [m]  (East)", color=TEXT)
    ax.set_ylabel("y  [m]  (North)", color=TEXT)
    ax.tick_params(colors=TEXT)
    iters_note = f"RRT / RRT*: {args.rrt_iters} iterations"
    if args.seed is not None:
        iters_note += f", seed {args.seed}"
    ax.set_title(f"A*, RRT and RRT* paths — {os.path.basename(args.world_file)}\n{iters_note}",
                 fontweight="bold", fontsize=13, color=TEXT)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    plt.savefig(args.plot, dpi=150)
    print(f"Saved plot to {args.plot}")


if __name__ == "__main__":
    main()
