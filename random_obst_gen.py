#!/usr/bin/env python3
# generate_obstacles.py -- generates a complete, valid PX4/Gazebo (gz-sim) world
# SDF file with N randomly placed, non-overlapping obstacles.

import argparse
import math
import random

WORLD_TEMPLATE_HEADER = """<?xml version="1.0" encoding="UTF-8"?>
<sdf version="1.9">
  <world name="lawn">

    <!-- Required gz-sim system plugins -->
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"></plugin>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"></plugin>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"></plugin>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"></plugin>
    <plugin filename="gz-sim-air-pressure-system" name="gz::sim::systems::AirPressure"></plugin>
    <plugin filename="gz-sim-magnetometer-system" name="gz::sim::systems::Magnetometer"></plugin>
    <plugin filename="gz-sim-navsat-system" name="gz::sim::systems::NavSat"></plugin>

    <physics type="ode">
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>250</real_time_update_rate>
    </physics>
    <gravity>0 0 -9.8</gravity>
    <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>
    <atmosphere type="adiabatic"/>
    <scene>
      <sky>
        <clouds>
          <speed>12</speed>
        </clouds>
      </sky>
      <grid>true</grid>
      <ambient>0.5 0.5 0.5 0.3</ambient>
      <background>0.3 0.3 0.3 0.3</background>
      <shadows>true</shadows>
    </scene>

    <include>
      <uri>model://ground_plane</uri>
    </include>

"""

WORLD_TEMPLATE_FOOTER = """
    <light name="sunUTC" type="directional">
      <pose>0 0 500 0 -0 0</pose>
      <cast_shadows>true</cast_shadows>
      <intensity>1</intensity>
      <direction>0.001 0.625 -0.78</direction>
      <diffuse>0.904 0.904 0.904 1</diffuse>
      <specular>0.271 0.271 0.271 1</specular>
      <attenuation>
        <range>2000</range>
        <linear>0</linear>
        <constant>1</constant>
        <quadratic>0</quadratic>
      </attenuation>
      <spot>
        <inner_angle>0</inner_angle>
        <outer_angle>0</outer_angle>
        <falloff>0</falloff>
      </spot>
    </light>
    <light type="directional" name="sun_2">
      <cast_shadows>true</cast_shadows>
      <pose>200 200 200 0 0 0</pose>
      <diffuse>0.5 0.5 0.5 0.3</diffuse>
      <specular>0.2 0.2 0.2 0.3</specular>
      <attenuation>
        <range>2000</range>
        <constant>1</constant>
        <linear>0</linear>
        <quadratic>0</quadratic>
      </attenuation>
      <direction>0.01 0.01 -0.9</direction>
    </light>

    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <world_frame_orientation>ENU</world_frame_orientation>
      <latitude_deg>47.397971057728974</latitude_deg>
      <longitude_deg>8.546163739800146</longitude_deg>
      <elevation>0</elevation>
    </spherical_coordinates>
  </world>
</sdf>
"""


def make_box(name, x, y, z, sx, sy, sz):
    return f"""
  <model name="{name}">
    <static>true</static>
    <pose>{x:.6f} {y:.6f} {z:.6f} 0 0 0</pose>
    <link name="link">
      <collision name="collision"><geometry><box><size>{sx:.6f} {sy:.6f} {sz:.6f}</size></box></geometry></collision>
      <visual name="visual"><geometry><box><size>{sx:.6f} {sy:.6f} {sz:.6f}</size></box></geometry>
        <material><ambient>0.7 0.3 0.1 1</ambient><diffuse>0.7 0.3 0.1 1</diffuse></material></visual>
    </link>
  </model>"""


def make_cylinder(name, x, y, z, r, h):
    return f"""
  <model name="{name}">
    <static>true</static>
    <pose>{x:.6f} {y:.6f} {z:.6f} 0 0 0</pose>
    <link name="link">
      <collision name="collision"><geometry><cylinder><radius>{r:.6f}</radius><length>{h:.6f}</length></cylinder></geometry></collision>
      <visual name="visual"><geometry><cylinder><radius>{r:.6f}</radius><length>{h:.6f}</length></cylinder></geometry>
        <material><ambient>0.1 0.4 0.7 1</ambient><diffuse>0.1 0.4 0.7 1</diffuse></material></visual>
    </link>
  </model>"""


def bounding_radius_xy(kind, dims):
    """Conservative 2D (x,y) bounding radius used for overlap/no-spawn checks."""
    if kind == 'box':
        sx, sy, _sz = dims
        return math.hypot(sx / 2, sy / 2)
    else:  # cylinder
        r, _h = dims
        return r


def generate_obstacles(n, area, seed, no_spawn_radius, no_spawn_center,
                        min_gap, max_attempts_per_obstacle=200):
    """
    Returns list of (name, kind, x, y, z, dims) with no overlaps and
    none inside the no-spawn zone (e.g., drone launch point).
    """
    rng = random.Random(seed)
    placed = []  # list of (x, y, bounding_radius)
    snippets = []

    for i in range(n):
        placed_ok = False
        for _attempt in range(max_attempts_per_obstacle):
            x = rng.uniform(-area / 2, area / 2)
            y = rng.uniform(-area / 2, area / 2)
            is_box = rng.random() < 0.5

            if is_box:
                h = rng.uniform(1, 4)
                sx, sy, sz = rng.uniform(0.5, 1.5), rng.uniform(0.5, 1.5), h
                dims = (sx, sy, sz)
                kind = 'box'
            else:
                h = rng.uniform(1, 4)
                r = rng.uniform(0.3, 0.8)
                dims = (r, h)
                kind = 'cylinder'

            b_rad = bounding_radius_xy(kind, dims)

            # reject if inside no-spawn zone (e.g., around drone launch point)
            dx = x - no_spawn_center[0]
            dy = y - no_spawn_center[1]
            if math.hypot(dx, dy) < (no_spawn_radius + b_rad):
                continue

            # reject if overlapping (with margin) any already-placed obstacle
            overlap = False
            for (px, py, p_rad) in placed:
                if math.hypot(x - px, y - py) < (b_rad + p_rad + min_gap):
                    overlap = True
                    break
            if overlap:
                continue

            # accepted
            placed.append((x, y, b_rad))
            z = h / 2  # sits flush on ground plane (z=0 base)
            name = f"obstacle_box_{i}" if kind == 'box' else f"obstacle_cyl_{i}"
            if kind == 'box':
                snippets.append(make_box(name, x, y, z, *dims))
            else:
                snippets.append(make_cylinder(name, x, y, z, *dims))
            placed_ok = True
            break

        if not placed_ok:
            print(f"Warning: could not place obstacle {i} after "
                  f"{max_attempts_per_obstacle} attempts (area too crowded) -- skipping it.")

    return snippets


def main():
    parser = argparse.ArgumentParser(description="Generate a PX4/gz-sim world with random obstacles.")
    parser.add_argument('-n', '--num-obstacles', type=int, default=15)
    parser.add_argument('-a', '--area', type=float, default=20.0,
                         help="Obstacles are placed within [-area/2, area/2] on x and y.")
    parser.add_argument('-s', '--seed', type=int, default=None,
                         help="Random seed for reproducible layouts. Omit for a random layout.")
    parser.add_argument('--no-spawn-radius', type=float, default=2.5,
                         help="Radius around the launch point kept clear of obstacles.")
    parser.add_argument('--no-spawn-center', type=float, nargs=2, default=(0.0, 0.0),
                         metavar=('X', 'Y'))
    parser.add_argument('--min-gap', type=float, default=0.5,
                         help="Minimum clear gap enforced between obstacles.")
    parser.add_argument('-o', '--output', default="generated_world.sdf")
    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.randrange(2**31)
    print(f"Using seed: {args.seed}  (record this to reproduce this exact layout)")

    obstacle_snippets = generate_obstacles(
        n=args.num_obstacles,
        area=args.area,
        seed=args.seed,
        no_spawn_radius=args.no_spawn_radius,
        no_spawn_center=tuple(args.no_spawn_center),
        min_gap=args.min_gap,
    )

    world_sdf = WORLD_TEMPLATE_HEADER + "\n".join(obstacle_snippets) + WORLD_TEMPLATE_FOOTER

    with open(args.output, "w") as f:
        f.write(world_sdf)

    print(f"Wrote complete world file to {args.output} "
          f"({len(obstacle_snippets)} obstacles placed)")


if __name__ == "__main__":
    main()
