#!/usr/bin/env python3
# px4_waypoint_follower.py
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseArray
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleStatus,
    VehicleLocalPosition,
)
import math


class PX4WaypointFollower(Node):
    def __init__(self):
        super().__init__('px4_waypoint_follower')

        # QoS matching PX4's uXRCE-DDS agent (best effort, volatile) for control topics
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        # QoS matching rrt_planner_node's transient_local path publisher
        path_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,    # VOLATILE was previously TRANSIENT_LOCAL
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', px4_qos)
        self.setpoint_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', px4_qos)
        self.command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', px4_qos)

        self.status_sub = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v4', self.status_cb, px4_qos)
        self.position_sub = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self.position_cb, px4_qos)
        self.path_sub = self.create_subscription(
            PoseArray, '/rrt_path', self.path_cb, path_qos)
            
        self.declare_parameter('spawn_x', 0.0)
        self.declare_parameter('spawn_y', 0.0)
        self.spawn_x = self.get_parameter('spawn_x').value
        self.spawn_y = self.get_parameter('spawn_y').value

        # State
        self.waypoints = []          # list of (x, y, z) in NED, from /rrt_path
        self.last_raw_waypoints = None
        self.idx = 0
        self.offboard_setpoint_counter = 0
        self.armed = False
        self.nav_state = None
        self.have_position = False
        self.cur_x = 0.0
        self.cur_y = 0.0
        self.cur_z = 0.0
        

        # Tuning
        self.waypoint_tolerance = 0.5   # meters — distance to consider a waypoint "reached"
        self.hold_setpoint = (0.0, 0.0, -2.0)  # NED hover point used before a path arrives

        self.timer = self.create_timer(0.1, self.timer_cb)  # 10Hz

        self.get_logger().info('px4_waypoint_follower started, waiting for /rrt_path ...')

    # --- Callbacks ---------------------------------------------------

    """
    def path_cb(self, msg):
        # Planner publishes ENU-style, Z-up poses -> convert to PX4 NED (negate Z)
        self.waypoints = [(p.position.x, p.position.y, -p.position.z) for p in msg.poses]
        self.idx = 0
        self.get_logger().info(f"Received {len(self.waypoints)} waypoints")
    """
        
    """
    def path_cb(self, msg):
        self.waypoints = [(p.position.x, p.position.y, -p.position.z) for p in msg.poses]
        if self.have_position:
            takeoff_wp = (self.cur_x, self.cur_y, -2.0)  # climb straight up first
            self.waypoints.insert(0, takeoff_wp)
        self.idx = 0
        self.get_logger().info(f"Received {len(self.waypoints)} waypoints (with takeoff prepended)")
        """
        
    """
    def path_cb(self, msg):
        new_waypoints = [(p.position.x, p.position.y, -p.position.z) for p in msg.poses]

        # Ignore repeated republishes of the same path — only reset on a genuinely new path
        if new_waypoints == self.waypoints:
            return

        self.waypoints = new_waypoints
        if self.have_position:
            takeoff_wp = (self.cur_x, self.cur_y, -2.0)
            self.waypoints.insert(0, takeoff_wp)
        self.idx = 0
        self.get_logger().info(f"Received {len(self.waypoints)} NEW waypoints (with takeoff prepended)")
        """
        
    def path_cb(self, msg):
        # new_waypoints = [(p.position.x, p.position.y, -p.position.z) for p in msg.poses]
        # /rrt_path is ENU (x=East, y=North, z=Up) -> PX4 NED (x=North, y=East, z=Down)
        # new_waypoints = [(p.position.y, p.position.x, -p.position.z) for p in msg.poses]
        
        new_waypoints = [(p.position.y - self.spawn_y,    # North = world y - spawn y
                  p.position.x - self.spawn_x,    # East  = world x - spawn x
                  -p.position.z) for p in msg.poses]

        # Compare against the last RAW path we received, not the takeoff-modified execution list
        if new_waypoints == getattr(self, 'last_raw_waypoints', None):
            return

        self.last_raw_waypoints = new_waypoints
        self.waypoints = list(new_waypoints)  # copy, so takeoff insert doesn't mutate the raw list

        if self.have_position:
            takeoff_wp = (self.cur_x, self.cur_y, -2.0)
            self.waypoints.insert(0, takeoff_wp)

        self.idx = 0
        self.get_logger().info(f"Received {len(self.waypoints)} NEW waypoints (with takeoff prepended)")

    
    """
    def status_cb(self, msg):
        self.nav_state = msg.nav_state
        self.armed = (msg.arming_state == VehicleStatus.ARMING_STATE_ARMED)
        """
        
        
    def status_cb(self, msg):
        if msg.nav_state != self.nav_state or (msg.arming_state == VehicleStatus.ARMING_STATE_ARMED) != self.armed:
            self.get_logger().info(
            f"Status change: nav_state {self.nav_state} -> {msg.nav_state}, "
            f"armed {self.armed} -> {msg.arming_state == VehicleStatus.ARMING_STATE_ARMED}"
            )
        self.nav_state = msg.nav_state
        self.armed = (msg.arming_state == VehicleStatus.ARMING_STATE_ARMED)
    

    def position_cb(self, msg):
        if not (msg.xy_valid and msg.z_valid):
            return
        self.cur_x = msg.x
        self.cur_y = msg.y
        self.cur_z = msg.z
        self.have_position = True

    # --- Publishing helpers -------------------------------------------

    def publish_offboard_mode(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_pub.publish(msg)

    def publish_setpoint(self, x, y, z):
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.velocity = [math.nan, math.nan, math.nan]
        msg.acceleration = [math.nan, math.nan, math.nan]
        msg.jerk = [math.nan, math.nan, math.nan]
        msg.yaw = float('nan')
        msg.yawspeed = 0.0
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.setpoint_pub.publish(msg)

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.command_pub.publish(msg)

    def arm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)

    def engage_offboard(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)

    # --- Main loop ------------------------------------------------------

    def distance_to(self, x, y, z):
        return math.sqrt(
            (self.cur_x - x) ** 2 +
            (self.cur_y - y) ** 2 +
            (self.cur_z - z) ** 2
        )

    """ def timer_cb(self):
        # PX4 requires OffboardControlMode streamed continuously, always publish this
        self.publish_offboard_mode()

        if not self.waypoints:
            # No path yet -- hold position so PX4 doesn't reject the offboard stream
            self.publish_setpoint(*self.hold_setpoint)
        else:
            # Advance through waypoints based on real distance, using live position feedback
            if self.have_position and self.idx < len(self.waypoints) - 1:
                tx, ty, tz = self.waypoints[self.idx]
                if self.distance_to(tx, ty, tz) < self.waypoint_tolerance:
                    self.idx += 1
                    self.get_logger().info(f"Reached waypoint {self.idx - 1}, advancing to {self.idx}")

            x, y, z = self.waypoints[self.idx]
            self.publish_setpoint(x, y, z)

        # Arm + switch to offboard only after enough setpoints have been streamed
        if self.offboard_setpoint_counter == 10:
            self.engage_offboard()
            self.arm()
            self.get_logger().info('Sent offboard + arm commands')

        if self.offboard_setpoint_counter <= 10:
            self.offboard_setpoint_counter += 1 """
            
    def timer_cb(self):
        self.publish_offboard_mode()

        if not self.waypoints:
            self.publish_setpoint(*self.hold_setpoint)
        else:
            if self.have_position and self.idx < len(self.waypoints) - 1:
                tx, ty, tz = self.waypoints[self.idx]
                if self.distance_to(tx, ty, tz) < self.waypoint_tolerance:
                    self.idx += 1
                    self.get_logger().info(f"Reached waypoint {self.idx - 1}, advancing to {self.idx}")
            x, y, z = self.waypoints[self.idx]
            self.publish_setpoint(x, y, z)

        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1
            return  # keep streaming setpoints before attempting anything else

        # Once warmed up, actively drive toward offboard+armed using real feedback
        if self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.engage_offboard()
            self.get_logger().info(f'Requesting OFFBOARD (nav_state={self.nav_state})')
        elif not self.armed:
            self.arm()
            self.get_logger().info('OFFBOARD confirmed, requesting ARM')            



def main():
    rclpy.init()
    node = PX4WaypointFollower()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
