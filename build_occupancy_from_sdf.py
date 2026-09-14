#!/usr/bin/env python3
# build_occupancy_from_sdf.py
import re
import numpy as np
import octomap

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

def build_octree(obstacles, resolution=0.2):
    tree = octomap.OcTree(resolution)
    for obs in obstacles:
        if obs[0] == 'box':
            _, x, y, z, sx, sy, sz = obs
            for xi in np.arange(x - sx/2, x + sx/2, resolution):
                for yi in np.arange(y - sy/2, y + sy/2, resolution):
                    for zi in np.arange(max(0, z - sz/2), z + sz/2, resolution):
                        tree.updateNode(np.array([xi, yi, zi]), True)
        elif obs[0] == 'cyl':
            _, x, y, z, r, h = obs
            for xi in np.arange(x - r, x + r, resolution):
                for yi in np.arange(y - r, y + r, resolution):
                    if (xi - x)**2 + (yi - y)**2 <= r**2:
                        for zi in np.arange(max(0, z - h/2), z + h/2, resolution):
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
