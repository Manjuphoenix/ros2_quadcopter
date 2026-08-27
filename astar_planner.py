#!/usr/bin/env python3
"""
Grid-based A* path planner for 2D quadcopter navigation.

This module is standalone (no ROS dependency) so you can test/tune it
on its own before wiring it into a ROS 2 node. It works in a local
metric frame (meters), matching PX4's local NED frame (x=North, y=East),
at a fixed flight altitude.

Core pieces:
  - OccupancyGrid: metric <-> grid-cell conversion + obstacle storage
  - astar(): the search itself
  - simplify_path(): removes redundant waypoints via line-of-sight checks,
    so the drone doesn't have to stop at every single grid cell
  - random_free_point(): picks a random collision-free point, used to
    generate random start/goal points
"""

import heapq
import math
import random


class OccupancyGrid:
    def __init__(self, width_m, height_m, resolution_m, origin=(0.0, 0.0)):
        """
        width_m, height_m: size of the flight area in meters
        resolution_m: size of one grid cell in meters (e.g. 0.5)
        origin: (x, y) world coordinate of the grid's (0,0) cell corner
        """
        self.resolution = resolution_m
        self.origin = origin
        self.cols = int(math.ceil(width_m / resolution_m))
        self.rows = int(math.ceil(height_m / resolution_m))
        # 0 = free, 1 = occupied
        self.grid = [[0] * self.cols for _ in range(self.rows)]

    def world_to_cell(self, x, y):
        col = int((x - self.origin[0]) / self.resolution)
        row = int((y - self.origin[1]) / self.resolution)
        return row, col

    def cell_to_world(self, row, col):
        x = self.origin[0] + (col + 0.5) * self.resolution
        y = self.origin[1] + (row + 0.5) * self.resolution
        return x, y

    def in_bounds(self, row, col):
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_free(self, row, col):
        return self.in_bounds(row, col) and self.grid[row][col] == 0

    def add_rect_obstacle(self, x_min, y_min, x_max, y_max, inflate_m=0.0):
        """Mark a rectangular obstacle (in world meters), optionally
        inflated by inflate_m to keep the vehicle a safe distance away
        (accounts for vehicle size / control error)."""
        x_min -= inflate_m
        y_min -= inflate_m
        x_max += inflate_m
        y_max += inflate_m
        r0, c0 = self.world_to_cell(x_min, y_min)
        r1, c1 = self.world_to_cell(x_max, y_max)
        for row in range(max(0, r0), min(self.rows, r1 + 1)):
            for col in range(max(0, c0), min(self.cols, c1 + 1)):
                self.grid[row][col] = 1

    def add_circle_obstacle(self, cx, cy, radius_m, inflate_m=0.0):
        r = radius_m + inflate_m
        r0, c0 = self.world_to_cell(cx - r, cy - r)
        r1, c1 = self.world_to_cell(cx + r, cy + r)
        for row in range(max(0, r0), min(self.rows, r1 + 1)):
            for col in range(max(0, c0), min(self.cols, c1 + 1)):
                x, y = self.cell_to_world(row, col)
                if math.hypot(x - cx, y - cy) <= r:
                    self.grid[row][col] = 1


# 8-connected neighbor offsets (dr, dc, cost)
_NEIGHBORS = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
]


def _octile_heuristic(a, b):
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return (dr + dc) + (math.sqrt(2) - 2) * min(dr, dc)


def astar(grid: OccupancyGrid, start_world, goal_world):
    """
    start_world, goal_world: (x, y) tuples in meters.
    Returns a list of (x, y) world-coordinate waypoints from start to goal,
    or None if no path exists.
    """
    start = grid.world_to_cell(*start_world)
    goal = grid.world_to_cell(*goal_world)

    if not grid.is_free(*start):
        raise ValueError("Start point is inside an obstacle or out of bounds")
    if not grid.is_free(*goal):
        raise ValueError("Goal point is inside an obstacle or out of bounds")

    open_set = [(0.0, start)]
    came_from = {}
    g_score = {start: 0.0}
    visited = set()

    while open_set:
        _, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            # reconstruct path
            path_cells = [current]
            while current in came_from:
                current = came_from[current]
                path_cells.append(current)
            path_cells.reverse()
            return [grid.cell_to_world(r, c) for r, c in path_cells]

        for dr, dc, step_cost in _NEIGHBORS:
            neighbor = (current[0] + dr, current[1] + dc)
            if not grid.is_free(*neighbor):
                continue
            tentative_g = g_score[current] + step_cost
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + _octile_heuristic(neighbor, goal)
                heapq.heappush(open_set, (f, neighbor))

    return None  # no path found


def _line_of_sight(grid: OccupancyGrid, p1, p2, samples=50):
    """Check whether a straight line between two world points stays
    entirely in free cells (used to simplify/prune the A* path)."""
    for i in range(samples + 1):
        t = i / samples
        x = p1[0] + t * (p2[0] - p1[0])
        y = p1[1] + t * (p2[1] - p1[1])
        row, col = grid.world_to_cell(x, y)
        if not grid.is_free(row, col):
            return False
    return True


def simplify_path(grid: OccupancyGrid, path):
    """Greedy waypoint reduction: keep only waypoints where a straight
    line to the next kept waypoint would cross an obstacle otherwise.
    Cuts a long grid-cell staircase path down to a handful of turns."""
    if len(path) <= 2:
        return path
    simplified = [path[0]]
    anchor_idx = 0
    i = 2
    while i < len(path):
        if not _line_of_sight(grid, path[anchor_idx], path[i]):
            simplified.append(path[i - 1])
            anchor_idx = i - 1
        i += 1
    simplified.append(path[-1])
    return simplified


def random_free_point(grid: OccupancyGrid, margin_m=1.0, max_tries=1000):
    """Pick a random collision-free (x, y) point, at least margin_m away
    from the grid boundary (keeps the drone from spawning right at an edge)."""
    x_min = grid.origin[0] + margin_m
    x_max = grid.origin[0] + grid.cols * grid.resolution - margin_m
    y_min = grid.origin[1] + margin_m
    y_max = grid.origin[1] + grid.rows * grid.resolution - margin_m

    for _ in range(max_tries):
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        row, col = grid.world_to_cell(x, y)
        if grid.is_free(row, col):
            return (x, y)
    raise RuntimeError("Could not find a free random point — grid too cluttered?")


if __name__ == "__main__":
    # Quick standalone test / visualization (requires matplotlib).
    # Run: python3 astar_planner.py
    import matplotlib.pyplot as plt

    grid = OccupancyGrid(width_m=20, height_m=20, resolution_m=0.5, origin=(-10, -10))
    # Example obstacles — replace with ones matching your actual Gazebo world
    grid.add_rect_obstacle(-2, -2, 2, 2, inflate_m=0.5)
    grid.add_circle_obstacle(5, -5, 2.0, inflate_m=0.5)
    grid.add_rect_obstacle(-8, 3, -4, 8, inflate_m=0.5)

    random.seed()  # remove/seed for reproducibility while testing
    start = random_free_point(grid)
    goal = random_free_point(grid)
    print(f"Start: {start}  Goal: {goal}")

    path = astar(grid, start, goal)
    if path is None:
        print("No path found")
    else:
        path = simplify_path(grid, path)
        print(f"Path ({len(path)} waypoints):")
        for wp in path:
            print(f"  ({wp[0]:.2f}, {wp[1]:.2f})")

        # visualize
        fig, ax = plt.subplots(figsize=(7, 7))
        obs = [[c for c in row] for row in grid.grid]
        ax.imshow(obs, origin="lower", cmap="Greys",
                  extent=[grid.origin[0], grid.origin[0] + grid.cols * grid.resolution,
                          grid.origin[1], grid.origin[1] + grid.rows * grid.resolution])
        xs, ys = zip(*path)
        ax.plot(xs, ys, "-o", color="tab:blue")
        ax.plot(*start, "go", markersize=12, label="start")
        ax.plot(*goal, "ro", markersize=12, label="goal")
        ax.legend()
        ax.set_title("A* path")
        plt.savefig("astar_test.png", dpi=120)
        print("Saved visualization to astar_test.png")
