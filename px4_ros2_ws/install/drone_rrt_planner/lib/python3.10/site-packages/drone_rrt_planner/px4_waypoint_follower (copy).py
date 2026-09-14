#!/usr/bin/env python3
# px4_waypoint_follower.py
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseArray
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleStatus
import math

class PX4WaypointFollower(Node):
    def __init__(self):
        super().__init__('px4_waypoint_follower')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.get_logger().info('px4_waypoint_follower started')
        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.setpoint_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos)

        self.status_sub = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v4', self.status_cb, qos)
        self.path_sub = self.create_subscription(
            PoseArray, '/rrt_path', self.path_cb, qos)
            
        self.waypoints = []
        self.idx = 0
        self.offboard_setpoint_counter = 0
        self.armed = False
        self.nav_state = None

        self.timer = self.create_timer(0.1, self.timer_cb)  # 10Hz
        
        if self.offboard_setpoint_counter == 10:
            self.engage_offboard()
            self.arm()
            self.get_logger().info('Sent offboard + arm commands')

    def path_cb(self, msg):
        self.waypoints = [(p.position.x, p.position.y, -p.position.z) for p in msg.poses]  # NED: negate Z
        #self.waypoints = [50.0, 30.0, -5.0]  # NED: negate Z
        self.idx = 0
        self.get_logger().info(f"Received {len(self.waypoints)} waypoints")

    def status_cb(self, msg):
        self.nav_state = msg.nav_state
        self.armed = (msg.arming_state == VehicleStatus.ARMING_STATE_ARMED)

    def publish_offboard_mode(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_pub.publish(msg)

    """ def publish_setpoint(self, x, y, z):
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.setpoint_pub.publish(msg) """
        
    def publish_setpoint(self, x, y, z):
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.velocity = [math.nan, math.nan, math.nan]
        msg.acceleration = [math.nan, math.nan, math.nan]
        msg.jerk = [math.nan, math.nan, math.nan]
        msg.yaw = float('nan')  # let PX4 hold current heading, or set a real value if you want yaw control
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

    def timer_cb(self):
        self.publish_offboard_mode()

        if not self.waypoints:
            self.publish_setpoint(0.0, 0.0, -1.0)  # hold if no path yet
            return

        # x, y, z = self.waypoints[self.idx]
        x, y, z = 80.0, 10, 4.0
        self.publish_setpoint(x, y, z)

        # PX4 requires ~10 setpoints streamed before it accepts the mode switch
        if self.offboard_setpoint_counter == 10:
            self.engage_offboard()
            self.arm()

        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1

        # advance waypoint on proximity (placeholder — ideally use actual vehicle position feedback)
        # left simple here; refine with /fmu/out/vehicle_local_position_v1 distance check

def main():
    rclpy.init()
    node = PX4WaypointFollower()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
