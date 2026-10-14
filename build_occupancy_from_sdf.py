#!/usr/bin/env python3
# build_occupancy_from_sdf.py
import re
import numpy as np
import octomap
import math


def parse_sdf_obstacles(sdf_path):
    with open(sdf_path) as f:
        content = f.read()

    obstacles = []
    for model_match in re.finditer(r'<model name="(obstacle_\w+)">(.*?)</model>', content, re.DOTALL):
        name, body = model_match.groups()
        pose_match = re.search(r'<pose>\s*([\d\.\-e\s]+?)\s*</pose>', body)
        if not pose_match:
            continue
        pose = [float(x) for x in pose_match.group(1).split()]
        x, y, z = pose[0], pose[1], pose[2]

        box_match = re.search(r'<box><size>\s*([\d\.\-e\s]+?)\s*</size></box>', body)
        cyl_match = re.search(
            r'<cylinder><radius>([\d\.eE\-]+)</radius><length>([\d\.eE\-]+)</length></cylinder>', body
        )

        if box_match:
            sx, sy, sz = [float(v) for v in box_match.group(1).split()]
            obstacles.append(('box', x, y, z, sx, sy, sz))
            print(f"  {name}: box at ({x:.2f},{y:.2f},{z:.2f}) size ({sx:.2f},{sy:.2f},{sz:.2f})")
        elif cyl_match:
            r, h = float(cyl_match.group(1)), float(cyl_match.group(2))
            obstacles.append(('cyl', x, y, z, r, h))
            print(f"  {name}: cylinder at ({x:.2f},{y:.2f},{z:.2f}) r={r:.2f} h={h:.2f}")
        else:
            print(f"  WARNING: could not parse geometry for {name}")

    return obstacles

# def build_octree(obstacles, resolution=0.2):
#     tree = octomap.OcTree(resolution)
#     for obs in obstacles:
#         if obs[0] == 'box':
#             _, x, y, z, sx, sy, sz = obs
#             for xi in np.arange(x - sx/2, x + sx/2, resolution):
#                 for yi in np.arange(y - sy/2, y + sy/2, resolution):
#                     for zi in np.arange(max(0, z - sz/2), z + sz/2, resolution):
#                         tree.updateNode(np.array([xi, yi, zi]), True)
#         elif obs[0] == 'cyl':
#             _, x, y, z, r, h = obs
#             for xi in np.arange(x - r, x + r, resolution):
#                 for yi in np.arange(y - r, y + r, resolution):
#                     if (xi - x)**2 + (yi - y)**2 <= r**2:
#                         for zi in np.arange(max(0, z - h/2), z + h/2, resolution):
#                             tree.updateNode(np.array([xi, yi, zi]), True)
#     tree.updateInnerOccupancy()
#     return tree


def build_octree(obstacles, resolution=0.2):
    tree = octomap.OcTree(resolution)
    r_ = resolution

    def centers(lo, hi):
        # centres of every voxel whose cell [c - r/2, c + r/2] overlaps [lo, hi]
        k0, k1 = math.floor(lo / r_), math.ceil(hi / r_)
        return (np.arange(k0, k1 + 1) + 0.5) * r_

    for obs in obstacles:
        if obs[0] == 'box':
            _, x, y, z, sx, sy, sz = obs
            lo = np.array([x - sx/2, y - sy/2, max(0, z - sz/2)])
            hi = np.array([x + sx/2, y + sy/2, z + sz/2])
            for xi in centers(lo[0], hi[0]):
                for yi in centers(lo[1], hi[1]):
                    for zi in centers(lo[2], hi[2]):
                        if all(lo - r_/2 < (xi, yi, zi)) and all((xi, yi, zi) < hi + r_/2):
                            tree.updateNode(np.array([xi, yi, zi]), True)
        elif obs[0] == 'cyl':
            _, x, y, z, r, h = obs
            for xi in centers(x - r, x + r):
                for yi in centers(y - r, y + r):
                    # nearest point of the voxel's square to the cylinder axis
                    dx = max(abs(xi - x) - r_/2, 0)
                    dy = max(abs(yi - y) - r_/2, 0)
                    if dx*dx + dy*dy <= r*r:
                        for zi in centers(max(0, z - h/2), z + h/2):
                            tree.updateNode(np.array([xi, yi, zi]), True)
    tree.updateInnerOccupancy()
    return tree

if __name__ == "__main__":
    import sys
    sdf_path = sys.argv[1] if len(sys.argv) > 1 else "rrt_world.sdf"
    print(f"Parsing {sdf_path} ...")
    obstacles = parse_sdf_obstacles(sdf_path)
    print(f"\nParsed {len(obstacles)} obstacles total.")
    if not obstacles:
        print("ERROR: no obstacles found — check the file path and <model name=\"obstacle_...\"> naming.")
        sys.exit(1)
    tree = build_octree(obstacles)
    out_path = "static_map.bt"
    tree.writeBinary(out_path.encode())
    print(f"Wrote {out_path}")
