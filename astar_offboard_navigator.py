#!/usr/bin/env python3
"""
A* waypoint navigator for PX4 Offboard mode over ROS 2 / uXRCE-DDS.

Flow:
  1. Wait for a live position fix from PX4 (/fmu/out/vehicle_local_position).
  2. Build/define an occupancy grid representing your flight area (edit
     build_occupancy_grid() below to match your actual Gazebo world's
     obstacles for a meaningful test).
  3. Generate a random goal point (point A = current position, point B =
     random free point).
  4. Run A* to get a waypoint path, simplify it.
  5. Stream Offboard setpoints, advancing to the next waypoint once the
     vehicle is within WAYPOINT_TOLERANCE_M of the current one.

Requires: px4_msgs in your workspace, PX4 SITL + MicroXRCEAgent running.
Put this file in a ROS 2 package's scripts, or run directly after sourcing
your workspace, alongside astar_planner.py on the PYTHONPATH.
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

from astar_planner import OccupancyGrid, astar, simplify_path, random_free_point

WAYPOINT_TOLERANCE_M = 0.5   # how close counts as "reached" a waypoint
CRUISE_ALTITUDE_M = 5.0      # constant flight altitude (NED, so setpoint z = -CRUISE_ALTITUDE_M)


def build_occupancy_grid():
    """Define the flight-area grid and obstacles here.

    IMPORTANT: for the planner to be meaningful (not just flying through
    obstacles that only exist on paper), these obstacle coordinates should
    match real object positions in your Gazebo world file. If you're using
    the default empty x500 world, there are no obstacles to match yet —
    add some <model> boxes/cylinders to your world SDF and mirror their
    x/y/size here.
    """
    grid = OccupancyGrid(width_m=20, height_m=20, resolution_m=0.5, origin=(-10, -10))
    # Example obstacles (edit/remove to match your world):
    #grid.add_rect_obstacle(-2, -2, 2, 2, inflate_m=0.6)
    #grid.add_circle_obstacle(5, -5, 2.0, inflate_m=0.6)
    #grid.add_rect_obstacle(-8, 3, -4, 8, inflate_m=0.6)
    #grid.add_rect_obstacle(2, -3, 4, -1, inflate_m=0.6)
    grid.add_rect_obstacle(3, 3, 7, 7, inflate_m=0.6)
    grid.add_circle_obstacle(5, -5, 2.0, inflate_m=0.6)
    grid.add_rect_obstacle(-8, 3, -4, 8, inflate_m=0.6)
    return grid


class AstarOffboardNavigator(Node):

    def __init__(self):
        super().__init__('astar_offboard_navigator')

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

        self.grid = build_occupancy_grid()
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
        start = (start_x, start_y)

        goal = random_free_point(self.grid, margin_m=1.0)
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
        target_z = -CRUISE_ALTITUDE_M
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
