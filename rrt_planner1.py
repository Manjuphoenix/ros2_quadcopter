#!/usr/bin/env python3
"""
Grid-based RRT / RRT* / A* path planner for 2D quadcopter navigation.

This module is standalone (no ROS dependency) so you can test/tune it
on its own before wiring it into a ROS 2 node. It works in a local
metric frame (meters), matching PX4's local NED frame (x=North, y=East),
at a fixed flight altitude.

Core pieces:
  - OccupancyGrid: metric <-> grid-cell conversion + obstacle storage,
    including a generic rotated-polygon obstacle (used for boxes at any yaw)
  - WorldFileParser / build_occupancy_grid_from_world(): parses a PX4 /
    Gazebo SDF ".world" file and dynamically builds an OccupancyGrid from
    whatever obstacles it contains — no hardcoded obstacle list needed
  - rrt(): sampling-based RRT / RRT* search (the default planner)
  - astar(): grid A* search, kept for comparison (--method astar)
  - simplify_path(): removes redundant waypoints via line-of-sight checks,
    so the drone doesn't have to stop at every single grid cell
  - random_free_point(): picks a random collision-free point, used to
    generate random start/goal points

World-file parsing notes
-------------------------
PX4 SITL worlds (both classic Gazebo and Gazebo Sim / "gz") are SDF XML.
This parser walks <world> -> <model>/<include> -> <link> -> <collision> ->
<geometry> and extracts box/cylinder/sphere obstacles, composing poses
along the way. Known limitations (documented rather than silently wrong):
  - <pose relative_to="..."> frame graphs (SDF >=1.7) are not resolved;
    poses are treated as relative to their immediate parent, which covers
    the vast majority of hand-authored PX4 worlds.
  - <mesh> geometry has no parseable collision dimensions, so mesh
    obstacles are skipped by default (with a warning). Pass
    treat_mesh_as_circle_m to approximate them as a circle instead.
  - <plane> geometry (almost always the ground) is always skipped — a
    flying vehicle doesn't collide with the ground plane in a 2D slice.
  - <include> models (uri="model://name") are resolved by searching the
    directories in `model_search_paths` (or $GZ_SIM_RESOURCE_PATH /
    $GAZEBO_MODEL_PATH) for `<name>/model.sdf` or `<name>/model.config`.
    If a model can't be resolved, it's skipped with a warning rather than
    aborting the whole parse.
"""

import argparse
import heapq
import math
import os
import random
import sys
import xml.etree.ElementTree as ET


# ---------------------------------------------------------------------------
# Occupancy grid
# ---------------------------------------------------------------------------

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
        """Mark an axis-aligned rectangular obstacle (in world meters),
        optionally inflated by inflate_m (vehicle size / control error
        margin)."""
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

    def add_polygon_obstacle(self, points, inflate_m=0.0):
        """Mark an arbitrary convex/concave polygon obstacle, given as a
        list of (x, y) world-meter vertices in order. Used for boxes at
        any yaw (axis-aligned rect is just a 4-point special case).

        Inflation is a true Minkowski-sum-with-a-disk approximation: a
        cell is occupied if its center is inside the polygon OR within
        inflate_m of the polygon boundary.
        """
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_min, x_max = min(xs) - inflate_m, max(xs) + inflate_m
        y_min, y_max = min(ys) - inflate_m, max(ys) + inflate_m
        r0, c0 = self.world_to_cell(x_min, y_min)
        r1, c1 = self.world_to_cell(x_max, y_max)
        for row in range(max(0, r0), min(self.rows, r1 + 1)):
            for col in range(max(0, c0), min(self.cols, c1 + 1)):
                x, y = self.cell_to_world(row, col)
                if _point_in_polygon(x, y, points):
                    self.grid[row][col] = 1
                elif inflate_m > 0 and _point_to_polygon_dist(x, y, points) <= inflate_m:
                    self.grid[row][col] = 1


def _point_in_polygon(x, y, poly):
    """Standard ray-casting point-in-polygon test."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_intersect = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def _point_to_segment_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def _point_to_polygon_dist(x, y, poly):
    n = len(poly)
    return min(
        _point_to_segment_dist(x, y, poly[i][0], poly[i][1],
                                poly[(i + 1) % n][0], poly[(i + 1) % n][1])
        for i in range(n)
    )


# ---------------------------------------------------------------------------
# A* search
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# RRT / RRT* search
# ---------------------------------------------------------------------------

def _segment_free(grid: OccupancyGrid, p1, p2):
    """Collision-check a straight segment by sampling it at half the grid
    resolution against the (inflated) occupancy grid."""
    samples = max(1, int(math.ceil(math.dist(p1, p2) / (grid.resolution * 0.5))))
    return _line_of_sight(grid, p1, p2, samples=samples)


def rrt(grid: OccupancyGrid, start_world, goal_world, star=True, max_iter=4000,
        step_m=1.0, goal_bias=0.1, rewire_radius_m=2.5, seed=None, history=None):
    """
    Sampling-based planner on the occupancy grid.

    star=False: plain RRT, returns as soon as the goal is connected.
    star=True:  RRT* — chooses the cheapest parent among nearby nodes and
                rewires neighbours through each new node, running all
                max_iter iterations to keep shortening the path.

    Returns (path, tree_edges): path is a list of (x, y) waypoints from
    start to goal (or None), tree_edges a list of ((x, y), (x, y)) segments
    of the explored tree, for plotting.

    history: optional list; if given, history[k] is set to the best
    start-to-goal path cost through the tree after k iterations (inf
    before the goal is first connected), e.g. for convergence plots.
    """
    if not grid.is_free(*grid.world_to_cell(*start_world)):
        raise ValueError("Start point is inside an obstacle or out of bounds")
    if not grid.is_free(*grid.world_to_cell(*goal_world)):
        raise ValueError("Goal point is inside an obstacle or out of bounds")

    rng = random.Random(seed)
    x_min, y_min = grid.origin
    x_max = x_min + grid.cols * grid.resolution
    y_max = y_min + grid.rows * grid.resolution

    nodes = [tuple(start_world)]
    parent = [None]
    cost = [0.0]
    children = [[]]

    # Spatial hash so neighbour queries don't scan the whole tree
    bucket_m = max(rewire_radius_m, step_m)
    buckets = {}

    def bucket_key(p):
        return int(math.floor(p[0] / bucket_m)), int(math.floor(p[1] / bucket_m))

    def insert(i):
        buckets.setdefault(bucket_key(nodes[i]), []).append(i)

    def near(p, radius):
        bx, by = bucket_key(p)
        reach = int(math.ceil(radius / bucket_m))
        found = []
        for dx in range(-reach, reach + 1):
            for dy in range(-reach, reach + 1):
                for i in buckets.get((bx + dx, by + dy), ()):
                    if math.dist(nodes[i], p) <= radius:
                        found.append(i)
        return found

    def nearest(p):
        radius = bucket_m
        while True:
            cands = near(p, radius)
            if cands:
                return min(cands, key=lambda i: math.dist(nodes[i], p))
            radius *= 2.0

    def propagate_cost(i):
        stack = [i]
        while stack:
            j = stack.pop()
            for c in children[j]:
                cost[c] = cost[j] + math.dist(nodes[j], nodes[c])
                stack.append(c)

    insert(0)
    goal = tuple(goal_world)
    best_goal_parent, best_goal_cost = None, float("inf")
    goal_links = []  # (node index, distance to goal) for nodes that can see the goal

    def record_history():
        if history is not None:
            history.append(min((cost[j] + d for j, d in goal_links), default=float("inf")))

    for _ in range(max_iter):
        record_history()
        if rng.random() < goal_bias:
            sample = goal
        else:
            sample = (rng.uniform(x_min, x_max), rng.uniform(y_min, y_max))

        i_near = nearest(sample)
        p_near = nodes[i_near]
        d = math.dist(p_near, sample)
        if d < 1e-9:
            continue
        if d > step_m:
            t = step_m / d
            new = (p_near[0] + t * (sample[0] - p_near[0]),
                   p_near[1] + t * (sample[1] - p_near[1]))
        else:
            new = sample
        if not _segment_free(grid, p_near, new):
            continue

        # Choose parent (RRT*: cheapest collision-free neighbour)
        best_parent, best_cost = i_near, cost[i_near] + math.dist(p_near, new)
        neighbours = near(new, rewire_radius_m) if star else []
        for j in neighbours:
            c = cost[j] + math.dist(nodes[j], new)
            if c < best_cost and _segment_free(grid, nodes[j], new):
                best_parent, best_cost = j, c

        i_new = len(nodes)
        nodes.append(new)
        parent.append(best_parent)
        cost.append(best_cost)
        children.append([])
        children[best_parent].append(i_new)
        insert(i_new)

        # Rewire neighbours through the new node (RRT* only)
        for j in neighbours:
            c = best_cost + math.dist(new, nodes[j])
            if c < cost[j] and _segment_free(grid, new, nodes[j]):
                children[parent[j]].remove(j)
                parent[j] = i_new
                children[i_new].append(j)
                cost[j] = c
                propagate_cost(j)

        # Try to connect to the goal
        d_goal = math.dist(new, goal)
        if d_goal <= step_m and _segment_free(grid, new, goal):
            goal_links.append((i_new, d_goal))
            if best_cost + d_goal < best_goal_cost:
                best_goal_parent, best_goal_cost = i_new, best_cost + d_goal
            if not star:
                break

    record_history()
    tree_edges = [(nodes[parent[i]], nodes[i]) for i in range(1, len(nodes))]

    if star:
        # Rewiring may have made another goal-adjacent node cheaper
        for j in near(goal, step_m):
            c = cost[j] + math.dist(nodes[j], goal)
            if c < best_goal_cost and _segment_free(grid, nodes[j], goal):
                best_goal_parent, best_goal_cost = j, c

    if best_goal_parent is None:
        return None, tree_edges

    path = [goal]
    i = best_goal_parent
    while i is not None:
        path.append(nodes[i])
        i = parent[i]
    path.reverse()
    return path, tree_edges


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


# ---------------------------------------------------------------------------
# PX4 / Gazebo SDF world-file parsing
# ---------------------------------------------------------------------------

# Model names commonly present in PX4 worlds that are never flight obstacles
# (vehicles, lighting, world furniture). Extend/override with --exclude.
_DEFAULT_EXCLUDE_SUBSTRINGS = [
    "sun", "iris", "x500", "standard_vtol", "typhoon", "solo", "plane",
    "rover", "tailsitter", "gimbal", "camera", "runway", "asphalt_plane",
    "ground_plane",
]


def _parse_pose(pose_text):
    """SDF <pose> is 'x y z roll pitch yaw' (meters / radians)."""
    if pose_text is None or not pose_text.strip():
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    vals = [float(v) for v in pose_text.split()]
    while len(vals) < 6:
        vals.append(0.0)
    return tuple(vals[:6])


def _compose_pose_2d(parent, local):
    """Compose a local pose into its parent's frame, in 2D (x, y, yaw)
    plus a simple additive z. Roll/pitch are ignored — fine for the
    near-flat worlds PX4 SITL typically uses."""
    px, py, pz, _, _, pyaw = parent
    lx, ly, lz, _, _, lyaw = local
    cos_y, sin_y = math.cos(pyaw), math.sin(pyaw)
    wx = px + lx * cos_y - ly * sin_y
    wy = py + lx * sin_y + ly * cos_y
    wz = pz + lz
    wyaw = pyaw + lyaw
    return (wx, wy, wz, 0.0, 0.0, wyaw)


def _enu_to_grid(x_east, y_north):
    """Gazebo/SDF world frame is ENU (x=East, y=North). This module's
    grid frame matches PX4 local NED (x=North, y=East) at fixed altitude.
    Swap axes accordingly. Pass convert_enu_to_ned=False upstream to skip
    this and keep raw Gazebo (East, North) coordinates instead."""
    return y_north, x_east


def enu_to_grid(x_east, y_north):
    """Public alias of _enu_to_grid — e.g. for converting an RViz-clicked
    goal (published in a REP-105 ENU "map" frame) into this module's
    (North, East) grid frame."""
    return _enu_to_grid(x_east, y_north)


def grid_to_enu(x_north, y_east):
    """Inverse of enu_to_grid — this module's (North, East) grid frame to
    a REP-105 ENU "map" frame (East, North), e.g. for publishing TF or
    nav_msgs/Path so RViz displays them correctly."""
    return y_east, x_north


class _RawObstacle:
    """An obstacle in its native SDF (ENU) frame, before grid conversion."""

    __slots__ = ("kind", "name", "pose", "size", "radius", "length")

    def __init__(self, kind, name, pose, size=None, radius=None, length=None):
        self.kind = kind          # 'box' | 'cylinder' | 'sphere'
        self.name = name
        self.pose = pose          # (x, y, z, roll, pitch, yaw) in ENU
        self.size = size          # (sx, sy, sz) for box
        self.radius = radius      # cylinder / sphere
        self.length = length      # cylinder height


def _find_model_sdf(model_name, search_paths):
    for base in search_paths:
        candidate_dir = os.path.join(base, model_name)
        for fname in ("model.sdf",):
            candidate = os.path.join(candidate_dir, fname)
            if os.path.isfile(candidate):
                return candidate
    return None


def _default_model_search_paths(extra_paths=None):
    paths = list(extra_paths or [])
    for env_var in ("GZ_SIM_RESOURCE_PATH", "GAZEBO_MODEL_PATH"):
        env_val = os.environ.get(env_var)
        if env_val:
            paths.extend(p for p in env_val.split(os.pathsep) if p)
    return paths


def _model_name_from_uri(uri):
    # uris look like "model://my_obstacle" (possibly with a sub-path)
    if uri.startswith("model://"):
        rest = uri[len("model://"):]
        return rest.split("/")[0]
    return os.path.basename(uri.rstrip("/"))


def _extract_geometry(geometry_el, name, pose, warn):
    """geometry_el is a <geometry> element. Returns a _RawObstacle or None
    (skipped: plane, unrecognized, or unresolvable mesh)."""
    box_el = geometry_el.find("box")
    if box_el is not None:
        size_text = box_el.findtext("size", default="1 1 1")
        sx, sy, sz = (float(v) for v in size_text.split())
        return _RawObstacle("box", name, pose, size=(sx, sy, sz))

    cyl_el = geometry_el.find("cylinder")
    if cyl_el is not None:
        radius = float(cyl_el.findtext("radius", default="0.5"))
        length = float(cyl_el.findtext("length", default="1.0"))
        return _RawObstacle("cylinder", name, pose, radius=radius, length=length)

    sphere_el = geometry_el.find("sphere")
    if sphere_el is not None:
        radius = float(sphere_el.findtext("radius", default="0.5"))
        return _RawObstacle("sphere", name, pose, radius=radius, length=2 * radius)

    if geometry_el.find("plane") is not None:
        return None  # ground plane — never an obstacle for a flying vehicle

    if geometry_el.find("mesh") is not None:
        warn(f"'{name}': mesh geometry has no parseable collision size — skipping "
             f"(pass treat_mesh_as_circle_m to approximate it as a circle)")
        return None

    warn(f"'{name}': unrecognized <geometry> child — skipping")
    return None


def _collect_from_model_element(model_el, model_world_pose, name_prefix, warn):
    """Walk a <model>'s <link>/<collision>/<geometry> and return a list of
    _RawObstacle, with poses composed up to model_world_pose."""
    obstacles = []
    model_local_pose = _parse_pose(model_el.findtext("pose"))
    model_pose = _compose_pose_2d(model_world_pose, model_local_pose)

    for link_el in model_el.findall("link"):
        link_pose = _compose_pose_2d(model_pose, _parse_pose(link_el.findtext("pose")))
        for coll_el in link_el.findall("collision"):
            coll_pose = _compose_pose_2d(link_pose, _parse_pose(coll_el.findtext("pose")))
            geometry_el = coll_el.find("geometry")
            if geometry_el is None:
                continue
            coll_name = coll_el.get("name", "collision")
            obstacle = _extract_geometry(
                geometry_el, f"{name_prefix}/{link_el.get('name', 'link')}/{coll_name}",
                coll_pose, warn)
            if obstacle is not None:
                obstacles.append(obstacle)
    return obstacles


def parse_world_obstacles(world_file, model_search_paths=None, exclude_substrings=None,
                           verbose=True):
    """Parse a PX4/Gazebo SDF world file and return a list of _RawObstacle
    (still in the file's native ENU frame — see _enu_to_grid for the
    conversion used to build the OccupancyGrid)."""
    exclude_substrings = ([s.lower() for s in exclude_substrings]
                           if exclude_substrings is not None
                           else _DEFAULT_EXCLUDE_SUBSTRINGS)
    search_paths = _default_model_search_paths(model_search_paths)
    search_paths.append(os.path.dirname(os.path.abspath(world_file)))

    def warn(msg):
        if verbose:
            print(f"[world-parser] warning: {msg}", file=sys.stderr)

    tree = ET.parse(world_file)
    root = tree.getroot()
    world_el = root.find("world")
    if world_el is None:
        # Some files hand you the <world> element as root directly
        world_el = root if root.tag == "world" else root

    identity_pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    obstacles = []

    def excluded(name):
        lname = name.lower()
        return any(sub in lname for sub in exclude_substrings)

    # Inline <model> elements directly under <world>
    for model_el in world_el.findall("model"):
        name = model_el.get("name", "model")
        if excluded(name):
            continue
        obstacles.extend(_collect_from_model_element(model_el, identity_pose, name, warn))

    # <include> elements referencing external model.sdf files
    for include_el in world_el.findall("include"):
        uri = include_el.findtext("uri", default="")
        model_name = _model_name_from_uri(uri)
        override_name = include_el.findtext("name")
        display_name = override_name or model_name
        if excluded(display_name) or excluded(model_name):
            continue

        include_pose = _parse_pose(include_el.findtext("pose"))

        resolved_path = _find_model_sdf(model_name, search_paths)
        if resolved_path is None:
            warn(f"could not resolve include '{uri}' (model '{model_name}') in any of "
                 f"{search_paths or ['<no search paths configured>']} — skipping. "
                 f"Pass model_search_paths=[...] or set GZ_SIM_RESOURCE_PATH/"
                 f"GAZEBO_MODEL_PATH.")
            continue

        try:
            inner_tree = ET.parse(resolved_path)
        except ET.ParseError as exc:
            warn(f"failed to parse included model '{resolved_path}': {exc} — skipping")
            continue

        inner_root = inner_tree.getroot()
        inner_model_el = inner_root if inner_root.tag == "model" else inner_root.find("model")
        if inner_model_el is None:
            warn(f"'{resolved_path}' has no <model> element — skipping")
            continue

        obstacles.extend(
            _collect_from_model_element(inner_model_el, include_pose, display_name, warn))

    return obstacles


def build_occupancy_grid_from_world(world_file, resolution_m=0.5, inflate_m=0.5,
                                     margin_m=5.0, flight_altitude_m=None,
                                     altitude_tolerance_m=0.25,
                                     width_m=None, height_m=None, origin=None,
                                     model_search_paths=None, exclude_substrings=None,
                                     convert_enu_to_ned=True,
                                     treat_mesh_as_circle_m=None, verbose=True):
    """
    Parse `world_file` (a PX4/Gazebo SDF ".world" file) and build an
    OccupancyGrid populated with every obstacle it contains.

    resolution_m, inflate_m: as in OccupancyGrid / add_*_obstacle.
    margin_m: extra free space added around the obstacles' bounding box
        when width_m/height_m/origin aren't given explicitly.
    flight_altitude_m: if given, obstacles whose z-extent doesn't cover
        this altitude (+/- altitude_tolerance_m) are skipped — e.g. a curb
        on the ground won't block a drone cruising at 3 m.
    width_m, height_m, origin: override auto-sizing entirely.
    model_search_paths: extra directories to search for <include> models,
        in addition to $GZ_SIM_RESOURCE_PATH / $GAZEBO_MODEL_PATH and the
        world file's own directory.
    convert_enu_to_ned: convert Gazebo's ENU world frame to this module's
        North/East grid frame (matches PX4 local NED at fixed altitude).
        Set False to keep raw Gazebo (East, North) coordinates.
    treat_mesh_as_circle_m: if given, mesh-geometry obstacles that would
        otherwise be skipped are approximated as a circle of this radius.

    Returns (grid, obstacle_footprints) where obstacle_footprints is a
    list of {'type': 'polygon'|'circle', ...} dicts in grid-frame
    coordinates, handy for plotting the true shapes rather than just the
    rasterized grid.
    """
    def warn(msg):
        if verbose:
            print(f"[world-parser] warning: {msg}", file=sys.stderr)

    raw_obstacles = parse_world_obstacles(
        world_file, model_search_paths=model_search_paths,
        exclude_substrings=exclude_substrings, verbose=verbose)

    def convert(x, y):
        return _enu_to_grid(x, y) if convert_enu_to_ned else (x, y)

    footprints = []  # grid-frame shapes, built before we know grid bounds
    for obs in raw_obstacles:
        x, y, z, _, _, yaw = obs.pose

        if flight_altitude_m is not None:
            if obs.kind == "box":
                half_h = obs.size[2] / 2.0
            else:
                half_h = (obs.length or 0.0) / 2.0
            z_min, z_max = z - half_h, z + half_h
            if not (z_min - altitude_tolerance_m <= flight_altitude_m
                    <= z_max + altitude_tolerance_m):
                continue

        if obs.kind == "box":
            sx, sy, _ = obs.size
            hx, hy = sx / 2.0, sy / 2.0
            local_corners = [(hx, hy), (hx, -hy), (-hx, -hy), (-hx, hy)]
            cos_y, sin_y = math.cos(yaw), math.sin(yaw)
            corners = []
            for lx, ly in local_corners:
                wx = x + lx * cos_y - ly * sin_y
                wy = y + lx * sin_y + ly * cos_y
                corners.append(convert(wx, wy))
            footprints.append({"type": "polygon", "points": corners, "name": obs.name})
        else:  # cylinder or sphere -> circle footprint
            cx, cy = convert(x, y)
            footprints.append({"type": "circle", "cx": cx, "cy": cy,
                                "radius": obs.radius, "name": obs.name})

    if not footprints and treat_mesh_as_circle_m is None:
        warn("no obstacles were extracted from the world file — the resulting grid "
             "will be entirely free space. Check --exclude / mesh handling if that's "
             "unexpected.")

    # Determine grid bounds
    if width_m is not None and height_m is not None and origin is not None:
        grid_width, grid_height, grid_origin = width_m, height_m, origin
    else:
        xs, ys = [], []
        for fp in footprints:
            if fp["type"] == "polygon":
                xs.extend(p[0] for p in fp["points"])
                ys.extend(p[1] for p in fp["points"])
            else:
                xs.extend([fp["cx"] - fp["radius"], fp["cx"] + fp["radius"]])
                ys.extend([fp["cy"] - fp["radius"], fp["cy"] + fp["radius"]])
        if xs and ys:
            x_min, x_max = min(xs) - margin_m, max(xs) + margin_m
            y_min, y_max = min(ys) - margin_m, max(ys) + margin_m
        else:
            # No obstacles found — fall back to a reasonable default arena
            x_min, x_max, y_min, y_max = -10.0, 10.0, -10.0, 10.0
        grid_origin = (x_min, y_min) if origin is None else origin
        grid_width = width_m if width_m is not None else (x_max - x_min)
        grid_height = height_m if height_m is not None else (y_max - y_min)

    grid = OccupancyGrid(grid_width, grid_height, resolution_m, origin=grid_origin)

    for fp in footprints:
        if fp["type"] == "polygon":
            grid.add_polygon_obstacle(fp["points"], inflate_m=inflate_m)
        else:
            grid.add_circle_obstacle(fp["cx"], fp["cy"], fp["radius"], inflate_m=inflate_m)

    return grid, footprints


# ---------------------------------------------------------------------------
# CLI / standalone test
# ---------------------------------------------------------------------------

def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Build an occupancy grid from a PX4/Gazebo world file and "
                    "run RRT* (or RRT / A*) between two (random, by default) points.")
    parser.add_argument("world_file", help="Path to the PX4/Gazebo .world (SDF) file")
    parser.add_argument("--resolution", type=float, default=0.5,
                        help="Grid cell size in meters (default: 0.5)")
    parser.add_argument("--inflate", type=float, default=0.5,
                        help="Obstacle inflation radius in meters (default: 0.5)")
    parser.add_argument("--margin", type=float, default=5.0,
                        help="Free-space margin added around obstacles when "
                             "auto-sizing the grid (default: 5.0)")
    parser.add_argument("--altitude", type=float, default=None,
                        help="Flight altitude in meters; obstacles that don't "
                             "span this altitude are ignored (default: consider all)")
    parser.add_argument("--width", type=float, default=None,
                        help="Override grid width in meters (default: auto)")
    parser.add_argument("--height", type=float, default=None,
                        help="Override grid height in meters (default: auto)")
    parser.add_argument("--origin-x", type=float, default=None,
                        help="Override grid origin x (requires --origin-y)")
    parser.add_argument("--origin-y", type=float, default=None,
                        help="Override grid origin y (requires --origin-x)")
    parser.add_argument("--model-path", action="append", default=[],
                        help="Extra directory to search for <include> models "
                             "(repeatable)")
    parser.add_argument("--exclude", action="append", default=None,
                        help="Substring of model names to always exclude "
                             "(repeatable; replaces the built-in default list)")
    parser.add_argument("--no-frame-conversion", action="store_true",
                        help="Keep raw Gazebo ENU coordinates instead of "
                             "converting to the North/East grid frame")
    parser.add_argument("--start", type=float, nargs=2, metavar=("X", "Y"),
                        default=None, help="Start point in meters (default: random)")
    parser.add_argument("--goal", type=float, nargs=2, metavar=("X", "Y"),
                        default=None, help="Goal point in meters (default: random)")
    parser.add_argument("--method", choices=["rrtstar", "rrt", "astar"], default="rrtstar",
                        help="Planning algorithm (default: rrtstar)")
    parser.add_argument("--rrt-iters", type=int, default=4000,
                        help="RRT/RRT* iteration budget (default: 4000)")
    parser.add_argument("--rrt-step", type=float, default=1.0,
                        help="RRT/RRT* max extension step in meters (default: 1.0)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for repeatable RRT runs and random start/goal")
    parser.add_argument("--plot", default="rrt_test.png",
                        help="Output plot path (default: astar_test.png)")
    parser.add_argument("--quiet", action="store_true", help="Suppress parser warnings")
    return parser


def main():
    args = _build_arg_parser().parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    origin = None
    if args.origin_x is not None and args.origin_y is not None:
        origin = (args.origin_x, args.origin_y)

    grid, footprints = build_occupancy_grid_from_world(
        args.world_file,
        resolution_m=args.resolution,
        inflate_m=args.inflate,
        margin_m=args.margin,
        flight_altitude_m=args.altitude,
        width_m=args.width,
        height_m=args.height,
        origin=origin,
        model_search_paths=args.model_path,
        exclude_substrings=args.exclude,
        convert_enu_to_ned=not args.no_frame_conversion,
        verbose=not args.quiet,
    )

    print(f"Built {grid.rows}x{grid.cols} grid "
          f"({grid.rows * grid.resolution:.1f}m x {grid.cols * grid.resolution:.1f}m) "
          f"from {len(footprints)} obstacle(s), origin={grid.origin}")

    start = tuple(args.start) if args.start else random_free_point(grid)
    goal = tuple(args.goal) if args.goal else random_free_point(grid)
    print(f"Start: {start}  Goal: {goal}")

    method_label = {"rrtstar": "RRT*", "rrt": "RRT", "astar": "A*"}[args.method]
    tree_edges = []
    if args.method == "astar":
        path = astar(grid, start, goal)
    else:
        path, tree_edges = rrt(grid, start, goal, star=(args.method == "rrtstar"),
                               max_iter=args.rrt_iters, step_m=args.rrt_step,
                               seed=args.seed)
        print(f"{method_label} tree: {len(tree_edges) + 1} nodes")
    have_path = path is not None
    if not have_path:
        print("No path found")
        if not tree_edges:
            return
        # Still plot the partial RRT tree, to show how far exploration got
        path = []
    else:
        raw_waypoints = len(path)
        path = simplify_path(grid, path)
        print(f"Simplified {raw_waypoints} -> {len(path)} waypoints")
        print(f"Path ({len(path)} waypoints):")
        for wp in path:
            print(f"  ({wp[0]:.2f}, {wp[1]:.2f})")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot")
        return

    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Circle, Polygon, Rectangle

    # Plot East on the horizontal axis and North on the vertical axis.
    # Grid-frame points are (North, East), so every plotted point is (p[1], p[0]).
    obstacle_face, obstacle_edge = "#3d4a55", "#1a1a1a"
    cmap = LinearSegmentedColormap.from_list(
        "path_blues", plt.cm.Blues([i / 255.0 for i in range(77, 256)]))

    # Densely resample the path so the colour gradient looks continuous
    dense = path[:1]
    for (n0, e0), (n1, e1) in zip(path[:-1], path[1:]):
        steps = max(1, int(math.hypot(n1 - n0, e1 - e0) / 0.05))
        dense.extend((n0 + (n1 - n0) * k / steps, e0 + (e1 - e0) * k / steps)
                     for k in range(1, steps + 1))
    dist_along = [0.0]
    for (n0, e0), (n1, e1) in zip(dense[:-1], dense[1:]):
        dist_along.append(dist_along[-1] + math.hypot(n1 - n0, e1 - e0))
    path_length = dist_along[-1] if have_path else 0.0

    # Clearance from each path point to the nearest true obstacle surface
    def clearance(n, e):
        best = float("inf")
        for fp in footprints:
            if fp["type"] == "polygon":
                d = (0.0 if _point_in_polygon(n, e, fp["points"])
                     else _point_to_polygon_dist(n, e, fp["points"]))
            else:
                d = max(0.0, math.hypot(n - fp["cx"], e - fp["cy"]) - fp["radius"])
            best = min(best, d)
        return best

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.set_facecolor("#f4f6f8")
    ax.grid(True, color="white", linewidth=1.0)
    ax.set_axisbelow(True)

    # Dashed boundary of the planning area
    x0, y0 = grid.origin[1], grid.origin[0]
    area_w, area_h = grid.rows * grid.resolution, grid.cols * grid.resolution
    ax.add_patch(Rectangle((x0, y0), area_w, area_h, fill=False, linestyle="--",
                           edgecolor="#8a9ba8", linewidth=1.0, zorder=1))

    # True (pre-rasterization) obstacle shapes
    for fp in footprints:
        if fp["type"] == "polygon":
            patch = Polygon([(p[1], p[0]) for p in fp["points"]], closed=True)
        else:
            patch = Circle((fp["cy"], fp["cx"]), fp["radius"])
        patch.set(facecolor=obstacle_face, edgecolor=obstacle_edge,
                  linewidth=0.6, zorder=2)
        ax.add_patch(patch)

    # Explored RRT tree, drawn faintly underneath the obstacles and path
    if tree_edges:
        tree_lc = LineCollection([[(a[1], a[0]), (b[1], b[0])] for a, b in tree_edges],
                                 colors="#9fb0bf", linewidths=0.5, zorder=1.5,
                                 alpha=0.7 if len(tree_edges) < 1000 else 0.45,
                                 label=f"{method_label} tree ({len(tree_edges) + 1} nodes)")
        ax.add_collection(tree_lc)

    # Path, coloured by distance along it
    pts = [(e, n) for n, e in dense]
    segments = [[pts[i], pts[i + 1]] for i in range(len(pts) - 1)]
    lc = LineCollection(segments, cmap=cmap, linewidth=3.5, capstyle="round", zorder=3)
    lc.set_array(dist_along[1:])
    lc.set_clim(0.0, path_length if have_path else 1.0)
    ax.add_collection(lc)

    def fmt(v):
        return f"{v:g}" if float(v).is_integer() else f"{v:.1f}"

    ax.scatter(start[1], start[0], s=260, color="#00c853", edgecolor="black",
               linewidth=1.3, zorder=5, label=f"start ({fmt(start[0])}, {fmt(start[1])})")
    ax.scatter(goal[1], goal[0], s=300, marker="*", color="#e00000", edgecolor="black",
               linewidth=1.0, zorder=5, label=f"goal ({fmt(goal[0])}, {fmt(goal[1])})")

    # Callout at the tightest clearance point
    if footprints and have_path:
        clear = [clearance(n, e) for n, e in dense]
        i_min = min(range(len(clear)), key=clear.__getitem__)
        cn, ce = dense[i_min]
        # Put the label in the most open spot ~4 m away: clear of obstacles,
        # the path and the plot edges
        def label_score(e, n):
            edge = min(e - x0 - 2.0, x0 + area_w - e - 2.0, n - y0 - 1.0, y0 + area_h - n - 1.0)
            to_path = min(math.hypot(n - pn, e - pe) for pn, pe in dense[::10])
            return min(edge, clearance(n, e), to_path)
        candidates = [(ce + 4.0 * math.cos(k * math.pi / 8), cn + 4.0 * math.sin(k * math.pi / 8))
                      for k in range(16)]
        label_xy = max(candidates, key=lambda c: label_score(*c))
        ax.annotate(f"tightest clearance\n{clear[i_min]:.2f} m",
                    xy=(ce, cn), xytext=label_xy, textcoords="data",
                    ha="center", fontsize=10, fontweight="bold", color="#2f3e46",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="#cccccc"),
                    arrowprops=dict(arrowstyle="-", color="#2f3e46", linewidth=1.0),
                    zorder=6)

    pad = 0.5
    ax.set_xlim(x0 - pad, x0 + area_w + pad)
    ax.set_ylim(y0 - pad, y0 + area_h + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x  [m]  (East)")
    ax.set_ylabel("y  [m]  (North)")
    if have_path:
        title = (f"{method_label} planned path, coloured by distance along path\n"
                 f"{path_length:.1f} m in {len(path)} waypoints")
    else:
        title = (f"{method_label} exploration, goal not reached yet\n"
                 f"{len(tree_edges) + 1} nodes after {args.rrt_iters} iterations")
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    cbar = fig.colorbar(lc, ax=ax, fraction=0.06, pad=0.03)
    cbar.set_label("distance along path  [m]")
    cbar.outline.set_visible(False)
    if not have_path:
        cbar.ax.set_visible(False)  # keep the layout identical to the with-path plots

    fig.tight_layout()
    plt.savefig(args.plot, dpi=150)
    print(f"Saved visualization to {args.plot}")


if __name__ == "__main__":
    main()
