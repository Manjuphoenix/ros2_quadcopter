#!/usr/bin/env python3
"""
A* waypoint navigator for PX4 Offboard mode over ROS 2 / uXRCE-DDS.

Flow:
  1. Wait for a live position fix from PX4 (/fmu/out/vehicle_local_position).
  2. Build the occupancy grid by parsing your actual Gazebo/PX4 world SDF
     file (see build_occupancy_grid() / the `world_file` ROS parameter) —
     no more hand-copying obstacle coordinates into this script.
  3. Generate a random goal point (point A = current position, point B =
     random free point).
  4. Run A* to get a waypoint path, simplify it.
  5. Stream Offboard setpoints, advancing to the next waypoint once the
     vehicle is within WAYPOINT_TOLERANCE_M of the current one.

Requires: px4_msgs in your workspace, PX4 SITL + MicroXRCEAgent running.
Put this file in a ROS 2 package's scripts, or run directly after sourcing
your workspace, alongside astar_planner.py on the PYTHONPATH.

World file:
  Pass the path to your world's SDF file via the `world_file` ROS
  parameter (see run instructions at the bottom of this file). The grid
  is built once, at node startup, by parsing that file — so if you edit
  the world in Gazebo, restart this node to pick up the change.
"""

import math
import random

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleStatus,
    VehicleLocalPosition,
)

from astar_planner import (
    astar,
    simplify_path,
    random_free_point,
    build_occupancy_grid_from_world,
)

WAYPOINT_TOLERANCE_M = 0.5   # how close counts as "reached" a waypoint
CRUISE_ALTITUDE_M = 2.0      # constant flight altitude (NED, so setpoint z = -CRUISE_ALTITUDE_M)

# Grid-building defaults — overridable via ROS parameters at launch (see
# bottom of file for the CLI/launch syntax).
DEFAULT_WORLD_FILE = ''      # required: set via the `world_file` parameter
GRID_RESOLUTION_M = 0.3
GRID_INFLATE_M = 0.6         # keep in sync with your vehicle's real footprint + margin
GRID_MARGIN_M = 5.0          # free-space padding around the world's obstacle bounding box


def build_occupancy_grid(world_file, resolution_m=GRID_RESOLUTION_M,
                          inflate_m=GRID_INFLATE_M, margin_m=GRID_MARGIN_M,
                          flight_altitude_m=CRUISE_ALTITUDE_M,
                          model_search_paths=None, logger=None):
    """Parse `world_file` (a PX4/Gazebo SDF world) and build the
    OccupancyGrid the planner will use, instead of hand-maintained
    obstacle coordinates.

    flight_altitude_m filters out obstacles that don't actually span the
    vehicle's cruise altitude (e.g. a curb won't block a drone at 5 m).
    Set to None to keep every obstacle regardless of height.
    """
    grid, footprints = build_occupancy_grid_from_world(
        world_file,
        resolution_m=resolution_m,
        inflate_m=inflate_m,
        margin_m=margin_m,
        flight_altitude_m=flight_altitude_m,
        model_search_paths=model_search_paths,
        convert_enu_to_ned=True,  # matches PX4's local NED (x=North, y=East) frame
        verbose=logger is None,
    )
    if logger is not None:
        logger.info(
            f'Loaded {len(footprints)} obstacle(s) from {world_file}: '
            f'{grid.rows}x{grid.cols} grid '
            f'({grid.rows * grid.resolution:.1f}m x {grid.cols * grid.resolution:.1f}m), '
            f'origin={grid.origin}')
    return grid


class AstarOffboardNavigator(Node):

    def __init__(self):
        super().__init__('astar_offboard_navigator')

        # ROS parameters — set these from your launch file or
        # `--ros-args -p world_file:=/path/to/your.sdf`
        self.declare_parameter('world_file', DEFAULT_WORLD_FILE)
        self.declare_parameter('grid_resolution_m', GRID_RESOLUTION_M)
        self.declare_parameter('grid_inflate_m', GRID_INFLATE_M)
        self.declare_parameter('grid_margin_m', GRID_MARGIN_M)
        self.declare_parameter('cruise_altitude_m', CRUISE_ALTITUDE_M)
        self.declare_parameter('model_search_paths', [''])

        world_file = self.get_parameter('world_file').get_parameter_value().string_value
        if not world_file:
            raise RuntimeError(
                "No world file configured. Launch with "
                "--ros-args -p world_file:=/path/to/your_world.sdf")

        search_paths = [p for p in
                         self.get_parameter('model_search_paths').get_parameter_value()
                         .string_array_value if p]
        cruise_altitude_m = self.get_parameter('cruise_altitude_m').value

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        self.status_sub = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v1', self._status_cb, qos_profile)
        self.position_sub = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self._position_cb, qos_profile)

        self.vehicle_status = VehicleStatus()
        self.current_pos = None       # (x, y, z) in NED, local frame
        self.have_position = False
        self.cruise_altitude_m = cruise_altitude_m

        try:
            self.grid = build_occupancy_grid(
                world_file,
                resolution_m=self.get_parameter('grid_resolution_m').value,
                inflate_m=self.get_parameter('grid_inflate_m').value,
                margin_m=self.get_parameter('grid_margin_m').value,
                flight_altitude_m=cruise_altitude_m,
                model_search_paths=search_paths,
                logger=self.get_logger(),
            )
        except (OSError, ValueError) as exc:
            # Fail fast and loud — flying with a grid that failed to load
            # is worse than not flying at all.
            self.get_logger().fatal(f'Failed to build occupancy grid from '
                                     f'"{world_file}": {exc}')
            raise

        self.waypoints = None         # list of (x, y) once planned
        self.wp_index = 0
        self.planned = False
        self.setpoint_counter = 0

        self.timer = self.create_timer(0.1, self._timer_cb)  # 10 Hz

    # ---- subscriptions -----------------------------------------------

    def _status_cb(self, msg):
        self.vehicle_status = msg

    def _position_cb(self, msg):
        self.current_pos = (msg.x, msg.y, msg.z)
        self.have_position = True

    # ---- planning -------------------------------------------------------

    def _plan_path(self):
        start_x, start_y, _ = self.current_pos
        # start = (start_x, start_y)
        # start = (-8.880973443144867, 7.83713586807413)
        start = (0.00, 0.00)

        # If start doesn't land on a free cell (e.g. the vehicle's actual
        # spawn point sits inside the inflation margin of an obstacle,
        # or outside the grid entirely), planning can't proceed.
        if not self.grid.is_free(*self.grid.world_to_cell(*start)):
            self.get_logger().error(
                f'Current position {start} is not inside a free grid cell — '
                f'check that grid_margin_m covers your spawn point.')
            return False

        # goal = random_free_point(self.grid, margin_m=1.0)
        # goal = (5.0, 8.0)
        # goal = (-9.69933971131896, 1.5983551104030571)
        goal = (-10.000, 6.000)
        self.get_logger().info(f'Planning path: start={start}, goal={goal}')

        try:
            raw_path = astar(self.grid, start, goal)
        except ValueError as e:
            self.get_logger().error(f'Planning failed: {e}')
            return False

        if raw_path is None:
            self.get_logger().error('No path found between start and goal')
            return False

        self.waypoints = simplify_path(self.grid, raw_path)
        self.wp_index = 0
        self.planned = True
        self.get_logger().info(f'Path planned with {len(self.waypoints)} waypoints:')
        for wp in self.waypoints:
            self.get_logger().info(f'  ({wp[0]:.2f}, {wp[1]:.2f})')
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
            if not self._plan_path():
                return  # retry next tick
            return

        # PX4 requires a short warm-up stream of setpoints before it will
        # accept the Offboard mode switch.
        if self.setpoint_counter == 10:
            self._engage_offboard_mode()
            self._arm()

        self._publish_offboard_control_mode()

        # Current target waypoint (hold last one once path is complete)
        target_x, target_y = self.waypoints[self.wp_index]
        target_z = -self.cruise_altitude_m
        self._publish_trajectory_setpoint(target_x, target_y, target_z)

        # Check if we've reached the current waypoint; advance if so
        cx, cy, cz = self.current_pos
        dist = math.sqrt((cx - target_x) ** 2 + (cy - target_y) ** 2)
        if dist < WAYPOINT_TOLERANCE_M and self.wp_index < len(self.waypoints) - 1:
            self.wp_index += 1
            self.get_logger().info(
                f'Waypoint reached, advancing to {self.wp_index + 1}/{len(self.waypoints)}')
        elif dist < WAYPOINT_TOLERANCE_M and self.wp_index == len(self.waypoints) - 1:
            # reached final goal — log once, then keep holding position
            if not getattr(self, '_goal_logged', False):
                self.get_logger().info('Final goal reached — holding position.')
                self._goal_logged = True

        if self.setpoint_counter < 11:
            self.setpoint_counter += 1


def main(args=None):
    rclpy.init(args=args)
    node = AstarOffboardNavigator()
    rclpy.spin(node)
    #import ipdb; ipdb.set_trace()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

# Run example (after sourcing your ROS 2 workspace, with PX4 SITL +
# MicroXRCEAgent already running):
#
#   python3 astar_offboard_navigator.py --ros-args \
#     -p world_file:=/path/to/rrt_world.sdf \
#     -p cruise_altitude_m:=5.0 \
#     -p grid_inflate_m:=0.6
#
# Or set these as defaults in a launch file's `parameters=[{...}]` block.
