#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped
from px4_msgs.msg import VehicleLocalPosition

from tf2_ros import TransformBroadcaster


class PX4TFBroadcaster(Node):

    def __init__(self):
        super().__init__('px4_tf_broadcaster')

        self.tf_broadcaster = TransformBroadcaster(self)

        self.subscription = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position_v1',
            self.vehicle_position_callback,
            10
        )

        self.get_logger().info(
            'PX4 TF broadcaster started: map -> base_link'
        )

    def vehicle_position_callback(self, msg):

        if not msg.xy_valid or not msg.z_valid:
            return

        transform = TransformStamped()

        transform.header.stamp = self.get_clock().now().to_msg()

        # Parent frame
        transform.header.frame_id = 'map'

        # Child frame
        transform.child_frame_id = 'base_link'

        # ------------------------------------------------
        # PX4 local position
        #
        # For the first test we keep the coordinates
        # exactly as published by PX4.
        # ------------------------------------------------

        transform.transform.translation.x = msg.x
        transform.transform.translation.y = msg.y
        transform.transform.translation.z = msg.z

        # Heading around Z
        yaw = msg.heading

        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = math.sin(yaw / 2.0)
        transform.transform.rotation.w = math.cos(yaw / 2.0)

        self.tf_broadcaster.sendTransform(transform)


def main(args=None):

    rclpy.init(args=args)

    node = PX4TFBroadcaster()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
