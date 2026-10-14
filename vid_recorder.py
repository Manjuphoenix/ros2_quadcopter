#!/usr/bin/env python3
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge

FPS = 30
HEIGHT = 720   # both halves scaled to this height


class SideBySideRecorder(Node):
    def __init__(self):
        super().__init__('side_by_side_recorder')
        self.declare_parameter('top_topic', '/top_camera')
        self.declare_parameter('fpv_topic', '/fpv_camera')
        self.declare_parameter('out', 'flight.mp4')
        self.bridge = CvBridge()
        self.writer = None
        self.last_t = None

        top = Subscriber(self, Image, self.get_parameter('top_topic').value,
                         qos_profile=qos_profile_sensor_data)
        fpv = Subscriber(self, Image, self.get_parameter('fpv_topic').value,
                         qos_profile=qos_profile_sensor_data)
        # stamps from the same render tick are identical, so the slop can be tight
        self.sync = ApproximateTimeSynchronizer([top, fpv], queue_size=30, slop=0.5 / FPS)
        self.sync.registerCallback(self.cb)
        self.get_logger().info('Recording... Ctrl+C to stop')

    def _fit(self, img, label):
        img = cv2.resize(img, (int(img.shape[1] * HEIGHT / img.shape[0]), HEIGHT))
        cv2.putText(img, label, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
        return img

    def cb(self, top_msg, fpv_msg):
            top = self._fit(self.bridge.imgmsg_to_cv2(top_msg, 'bgr8'), 'Top view')
            fpv = self._fit(self.bridge.imgmsg_to_cv2(fpv_msg, 'bgr8'), 'FPV')
            frame = np.hstack([top, fpv])

            t = top_msg.header.stamp.sec + top_msg.header.stamp.nanosec * 1e-9
            cv2.putText(frame, f'sim t = {t:7.2f} s', (20, HEIGHT - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            if self.writer is None:
                h, w = frame.shape[:2]
                self.writer = cv2.VideoWriter(self.get_parameter('out').value,
                                            cv2.VideoWriter_fourcc(*'mp4v'), FPS, (w, h))
            # if the bridge dropped frames, repeat this one so video time == sim time
            n = 1 if self.last_t is None else max(1, round((t - self.last_t) * FPS))
            for _ in range(n):
                self.writer.write(frame)
            self.last_t = t


def main():
    rclpy.init()
    node = SideBySideRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.writer is not None:
            node.writer.release()   # finalises the MP4; without this the file is unplayable
        node.destroy_node()


if __name__ == '__main__':
    main()
