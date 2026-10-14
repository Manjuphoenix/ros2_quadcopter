#!/usr/bin/env python3
"""
A* offboard navigator for the video recorder (vid_rec_astar.py).

Same planning as astar_offboard_navigator.py (occupancy grid built from the
world SDF, A*, path simplification, PX4 offboard waypoint following), plus:
  - spawn offset handled: the grid is in WORLD coordinates, PX4's local frame
    starts at the spawn point, so start/setpoints are converted both ways
  - start = the drone's real position, goal = ROS parameters (Gazebo world ENU)
  - publishes, for the recorder (both latched / transient_local):
      /astar_path  geometry_msgs/PoseArray  simplified path, world ENU
      /astar_tree  geometry_msgs/PoseArray  A* search tree, consecutive pairs
                                            of poses = one (parent, child) edge

Frames:
  grid      (x=North, y=East) of the Gazebo WORLD   (astar_planner, convert_enu_to_ned=True)
  PX4 local (x=North, y=East, z=Down), origin = spawn point
  world ENU (x=East,  y=North, z=Up)  -> Gazebo / recorder

Requires astar_planner.py next to this file:  git restore astar_planner.py

Run (after sourcing ROS 2 + the px4_ros2_ws workspace):
  python3 astar_nav_video.py --ros-args \
    -p world_file:=$HOME/Week1/PX4-Autopilot/Tools/simulation/gz/worlds/rrt_world.sdf \
    -p spawn_x:=9.0 -p spawn_y:=-9.0 -p goal_x:=-10.0 -p goal_y:=10.0
spawn_x/spawn_y must match PX4_GZ_MODEL_POSE.
"""

import heapq
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import Pose, PoseArray
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleStatus,
    VehicleLocalPosition,
)

import astar_planner as ap

WAYPOINT_TOLERANCE_M = 0.3   # how close counts as "reached" a waypoint
CRUISE_ALTITUDE_M = 2.0      # constant flight altitude (NED, so setpoint z = -CRUISE_ALTITUDE_M)
GRID_RESOLUTION_M = 0.3
GRID_INFLATE_M = 0.6         # vehicle footprint + margin
GRID_MARGIN_M = 5.0          # free-space padding around the world's obstacle bounding box


def astar_with_tree(grid, start_world, goal_world):
    """astar_planner.astar(), but also returns the search tree.

    Returns (path, came_from): path is a list of grid (x, y) waypoints or None,
    came_from maps every reached cell to its parent cell.
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
            path_cells = [current]
            while current in came_from:
                current = came_from[current]
                path_cells.append(current)
            path_cells.reverse()
            return [grid.cell_to_world(r, c) for r, c in path_cells], came_from

        for dr, dc, step_cost in ap._NEIGHBORS:
            neighbor = (current[0] + dr, current[1] + dc)
            if not grid.is_free(*neighbor):
                continue
            tentative_g = g_score[current] + step_cost
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + ap._octile_heuristic(neighbor, goal)
                heapq.heappush(open_set, (f, neighbor))

    return None, came_from


class AstarNavVideo(Node):

    def __init__(self):
        super().__init__('astar_nav_video')

        self.declare_parameter('world_file', '')
        self.declare_parameter('grid_resolution_m', GRID_RESOLUTION_M)
        self.declare_parameter('grid_inflate_m', GRID_INFLATE_M)
        self.declare_parameter('grid_margin_m', GRID_MARGIN_M)
        self.declare_parameter('cruise_altitude_m', CRUISE_ALTITUDE_M)
        self.declare_parameter('model_search_paths', [''])
        self.declare_parameter('spawn_x', 0.0)   # Gazebo world ENU, same as PX4_GZ_MODEL_POSE
        self.declare_parameter('spawn_y', 0.0)
        self.declare_parameter('goal_x', -9.0)   # Gazebo world ENU
        self.declare_parameter('goal_y', -9.5)

        world_file = self.get_parameter('world_file').value
        if not world_file:
            raise RuntimeError(
                "No world file configured. Launch with "
                "--ros-args -p world_file:=/path/to/your_world.sdf")

        search_paths = [p for p in self.get_parameter('model_search_paths').value if p]
        self.cruise_altitude_m = self.get_parameter('cruise_altitude_m').value
        # PX4 local NED origin expressed in the grid's (North, East) world frame
        self.off_n = self.get_parameter('spawn_y').value
        self.off_e = self.get_parameter('spawn_x').value
        # goal: world ENU -> grid (North, East)
        self.goal = (self.get_parameter('goal_y').value, self.get_parameter('goal_x').value)

        # PX4 uXRCE-DDS topics: best effort, volatile
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', px4_qos)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', px4_qos)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', px4_qos)
        self.status_sub = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v1', self._status_cb, px4_qos)
        self.position_sub = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self._position_cb, px4_qos)

        # path + tree for the recorder: latched, so a recorder started later still gets them
        latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.path_pub = self.create_publisher(PoseArray, '/astar_path', latched)
        self.tree_pub = self.create_publisher(PoseArray, '/astar_tree', latched)

        self.grid, footprints = ap.build_occupancy_grid_from_world(
            world_file,
            resolution_m=self.get_parameter('grid_resolution_m').value,
            inflate_m=self.get_parameter('grid_inflate_m').value,
            margin_m=self.get_parameter('grid_margin_m').value,
            flight_altitude_m=self.cruise_altitude_m,
            model_search_paths=search_paths,
            convert_enu_to_ned=True,  # grid x=North, y=East
            verbose=False,
        )
        self.get_logger().info(
            f'Loaded {len(footprints)} obstacle(s): {self.grid.rows}x{self.grid.cols} grid, '
            f'origin={self.grid.origin}')

        self.vehicle_status = VehicleStatus()
        self.current_pos = None       # (x, y, z) PX4 local NED
        self.have_position = False
        self.waypoints = None         # list of grid (x, y) once planned
        self.wp_index = 0
        self.planned = False
        self.setpoint_counter = 0
        self._goal_logged = False

        self.timer = self.create_timer(0.1, self._timer_cb)  # 10 Hz

    # ---- subscriptions -----------------------------------------------

    def _status_cb(self, msg):
        self.vehicle_status = msg

    def _position_cb(self, msg):
        if not (msg.xy_valid and msg.z_valid):
            return
        self.current_pos = (msg.x, msg.y, msg.z)
        self.have_position = True

    # ---- planning -------------------------------------------------------

    def _grid_to_pose(self, gx, gy):
        p = Pose()
        p.position.x, p.position.y = gy, gx     # grid (N, E) -> world ENU (E, N)
        p.position.z = self.cruise_altitude_m
        return p

    def _plan_path(self):
        lx, ly, _ = self.current_pos
        start = (lx + self.off_n, ly + self.off_e)   # PX4 local -> grid (world N, E)
        self.get_logger().info(
            f'Planning path: start={start} goal={self.goal} (grid N,E) | '
            f'world ENU start=({start[1]:.2f}, {start[0]:.2f}) goal=({self.goal[1]:.2f}, {self.goal[0]:.2f})')

        try:
            raw_path, came_from = astar_with_tree(self.grid, start, self.goal)
        except ValueError as e:
            self.get_logger().error(f'Planning failed: {e}')
            return False
        if raw_path is None:
            self.get_logger().error('No path found between start and goal')
            return False

        self.waypoints = ap.simplify_path(self.grid, raw_path)
        self.wp_index = 0
        self.planned = True

        stamp = self.get_clock().now().to_msg()
        tree_msg = PoseArray()
        tree_msg.header.frame_id = 'map'
        tree_msg.header.stamp = stamp
        for child, parent in came_from.items():
            tree_msg.poses.append(self._grid_to_pose(*self.grid.cell_to_world(*parent)))
            tree_msg.poses.append(self._grid_to_pose(*self.grid.cell_to_world(*child)))
        self.tree_pub.publish(tree_msg)

        path_msg = PoseArray()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = stamp
        path_msg.poses = [self._grid_to_pose(gx, gy) for gx, gy in self.waypoints]
        self.path_pub.publish(path_msg)

        self.get_logger().info(
            f'Path planned with {len(self.waypoints)} waypoints, '
            f'tree has {len(came_from)} edges. Waypoints (world ENU):')
        for gx, gy in self.waypoints:
            self.get_logger().info(f'  ({gy:.2f}, {gx:.2f})')
        return True

    # ---- PX4 command helpers --------------------------------------------

    def _arm(self):
        self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def _engage_offboard_mode(self):
        self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info('Offboard mode command sent')

    def _publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def _publish_trajectory_setpoint(self, x, y, z):
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.yaw = 0.0
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)

    def _publish_vehicle_command(self, command, **params):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get('param1', 0.0)
        msg.param2 = params.get('param2', 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    # ---- main loop --------------------------------------------------------

    def _timer_cb(self):
        if not self.have_position:
            return  # wait for a position fix before doing anything

        if not self.planned:
            self._plan_path()
            return  # retry next tick on failure

        # PX4 requires a short warm-up stream of setpoints before it will
        # accept the Offboard mode switch.
        if self.setpoint_counter == 10:
            self._engage_offboard_mode()
            self._arm()

        self._publish_offboard_control_mode()

        # Current target waypoint: grid (world N, E) -> PX4 local
        gx, gy = self.waypoints[self.wp_index]
        target_x, target_y = gx - self.off_n, gy - self.off_e
        target_z = -self.cruise_altitude_m
        self._publish_trajectory_setpoint(target_x, target_y, target_z)

        # Advance once close enough (horizontal + vertical, so it climbs before leaving)
        cx, cy, cz = self.current_pos
        dist = math.sqrt((cx - target_x) ** 2 + (cy - target_y) ** 2 + (cz - target_z) ** 2)
        if dist < WAYPOINT_TOLERANCE_M and self.wp_index < len(self.waypoints) - 1:
            self.wp_index += 1
            self.get_logger().info(
                f'Waypoint reached, advancing to {self.wp_index + 1}/{len(self.waypoints)}')
        elif dist < WAYPOINT_TOLERANCE_M and not self._goal_logged:
            self.get_logger().info('Final goal reached — holding position.')
            self._goal_logged = True

        if self.setpoint_counter < 11:
            self.setpoint_counter += 1


def main(args=None):
    rclpy.init(args=args)
    node = AstarNavVideo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
