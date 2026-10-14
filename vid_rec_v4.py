#!/usr/bin/python3
# Run with the SYSTEM python (not conda), after sourcing ROS 2:
#   source /opt/ros/humble/setup.bash && /usr/bin/python3 vid_rec_v3.py
#
# v3 additions over vid_rec_v2.py:
#   - subscribes to the planner's /rrt_path (ROS 2) and draws the path itself:
#       * as a green 3D marker in the camera-sensor scene (seen by top view + FPV)
#       * as a thick green 2D overlay on the top view (1 px marker lines are thin from 30 m)
#   - no planner change needed; the RRT tree is NOT drawn (it is never published)
#
# v2 features kept: 1 m world grid + X/Y axes as 3D markers, top-view tick labels,
# colour legend bottom-left of both halves.
# The top-view overlays assume the top camera pose <pose>0 0 30 0 1.5708 1.5708</pose>
# (+X right, +Y up) and the TOP_* values below matching its SDF.
import math
import signal
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node as RosNode
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import PoseArray
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
from gz.msgs10.marker_pb2 import Marker
from gz.msgs10.empty_pb2 import Empty

TOP = '/top_camera'
FPV = '/world/rrt_world/model/x500_mono_cam_0/link/camera_link/sensor/camera/image'
FPS = 30
HEIGHT = 720
OUT = 'flight.mp4'
PATH_TOPIC = '/rrt_path'       # geometry_msgs/PoseArray in Gazebo world ENU

TREE_TOPIC = '/rrt_tree'       # PoseArray, consecutive pairs = edges, world ENU

tree_segs = []                 # [((x1, y1, z1), (x2, y2, z2))], latest /rrt_tree


LEGEND = [
    ((0, 255, 0),   'Path followed'),
    ((0, 255, 255), 'Tree / neighbour edges'),
    ((0, 0, 255),   'World X axis'),
    ((255, 102, 0), 'World Y axis'),
    ((64, 64, 64),  'Grid cell (1 m)'),
]

# World grid drawn in the sensor scene (match your arena)
GRID_MIN, GRID_MAX, GRID_STEP = -10.0, 10.0, 1.0
GRID_Z = 0.02                  # just above the ground so it doesn't z-fight

# Top camera, must match rrt_world.sdf
# Tip: at 30 m the 20 m grid fills most of the frame and the bottom-left legend
# covers its corner; raising the camera to ~36 m (and TOP_Z here) leaves room.
TOP_W, TOP_H, TOP_HFOV, TOP_Z = 960, 720, 1.0, 30.0
TOP_F = (TOP_W / 2) / math.tan(TOP_HFOV / 2)   # focal length in pixels

# OpenCV colours are BGR
LEGEND = [
    ((0, 255, 0),   'Path followed'),
    ((0, 0, 255),   'World X axis'),
    ((255, 102, 0), 'World Y axis'),
    ((64, 64, 64),  'Grid cell (1 m)'),
]

lock = threading.Lock()
top_q, fpv_q = [], []          # (sim_time_s, msg), filled by gz callback threads
path_pts = []                  # [(x, y, z)] world ENU, latest /rrt_path
running = True


def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9


def to_bgr(msg, label):
    img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width * 3]
    img = cv2.cvtColor(img.reshape(msg.height, msg.width, 3), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (int(msg.width * HEIGHT / msg.height), HEIGHT))
    cv2.putText(img, label, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    return img


# ---- 3D markers (grid + axes) ------------------------------------------------

def _color(m, r, g, b):
    for c in (m.material.ambient, m.material.diffuse, m.material.emissive):
        c.r, c.g, c.b, c.a = r, g, b, 1.0


def _line_marker(ns, mid, rgb, segments):
    m = Marker()
    m.ns, m.id = ns, mid
    m.action = Marker.ADD_MODIFY
    m.type = Marker.LINE_LIST
    m.visibility = Marker.ALL
    _color(m, *rgb)
    for (x1, y1), (x2, y2) in segments:
        for x, y in ((x1, y1), (x2, y2)):
            p = m.point.add()
            p.x, p.y, p.z = x, y, GRID_Z
    return m


def send_marker(node, m, services=('/marker', '/sensors/marker')):
    # one-way services, so a "timeout" result is expected; the marker is still applied
    for srv in services:
        node.request(srv, m, Marker, Empty, 50)


def draw_path_marker(node, pts):
    m = Marker()
    m.ns, m.id = 'video_path', 0
    m.action = Marker.ADD_MODIFY
    m.type = Marker.LINE_STRIP
    m.visibility = Marker.ALL
    _color(m, 0.0, 1.0, 0.0)
    for x, y, z in pts:
        p = m.point.add()
        p.x, p.y, p.z = x, y, z
    # cameras only: the planner already draws its own path in the GUI
    send_marker(node, m, services=('/sensors/marker',))


def draw_world_grid(node):
    n = int(round((GRID_MAX - GRID_MIN) / GRID_STEP))
    cells = []
    for i in range(n + 1):
        k = GRID_MIN + i * GRID_STEP
        cells.append(((k, GRID_MIN), (k, GRID_MAX)))   # lines of constant x
        cells.append(((GRID_MIN, k), (GRID_MAX, k)))   # lines of constant y
    send_marker(node, _line_marker('world_grid', 0, (0.25, 0.25, 0.25), cells))
    send_marker(node, _line_marker('world_axes', 0, (1.0, 0.0, 0.0), [((0, 0), (GRID_MAX, 0))]))  # +X red
    send_marker(node, _line_marker('world_axes', 1, (0.0, 0.4, 1.0), [((0, 0), (0, GRID_MAX))]))  # +Y blue



def draw_tree_marker(node, segs):
    m = Marker()
    m.ns, m.id = 'video_tree', 0
    m.action = Marker.ADD_MODIFY            # same ns/id -> a new tree replaces the old one
    m.type = Marker.LINE_LIST
    m.visibility = Marker.ALL
    _color(m, 1.0, 1.0, 0.0)                # yellow
    for a, b in segs:
        for x, y, z in (a, b):
            p = m.point.add()
            p.x, p.y, p.z = x, y, z
    send_marker(node, m, services=('/sensors/marker',))

# ---- 2D overlays ---------------------------------------------------------------

def world_to_top_px(x, y, z=0.0):
    s = TOP_F / (TOP_Z - z)
    k = HEIGHT / TOP_H                     # to_bgr() rescales the frame to HEIGHT
    return int((TOP_W / 2 + s * x) * k), int((TOP_H / 2 - s * y) * k)

    


def draw_top_axes(img, step=2):
    font, white = cv2.FONT_HERSHEY_SIMPLEX, (255, 255, 255)
    for k in range(int(GRID_MIN), int(GRID_MAX) + 1, step):
        u, v = world_to_top_px(k, GRID_MIN)          # x ticks along the bottom edge of the grid
        cv2.putText(img, f'{k}', (u - 10, v + 22), font, 0.5, white, 1, cv2.LINE_AA)
        u, v = world_to_top_px(GRID_MIN, k)          # y ticks along the left edge of the grid
        cv2.putText(img, f'{k}', (u - 34, v + 5), font, 0.5, white, 1, cv2.LINE_AA)
    u, v = world_to_top_px(GRID_MAX, 0)
    cv2.putText(img, 'X (m)', (u + 8, v + 5), font, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    u, v = world_to_top_px(0, GRID_MAX)
    cv2.putText(img, 'Y (m)', (u - 25, v - 10), font, 0.6, (255, 102, 0), 2, cv2.LINE_AA)


def draw_top_path(img, pts):
    if len(pts) < 2:
        return
    px = np.array([world_to_top_px(x, y, z) for x, y, z in pts], np.int32)
    cv2.polylines(img, [px], False, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.circle(img, tuple(px[0]), 7, (0, 255, 0), -1)          # start
    cv2.circle(img, tuple(px[-1]), 7, (0, 0, 255), -1)         # goal


def draw_legend(img):
    pad, row, width = 8, 20, 180
    h = 2 * pad + row * len(LEGEND)
    x0, y0 = 8, img.shape[0] - h - 8
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)      # semi-transparent background
    for i, (col, label) in enumerate(LEGEND):
        y = y0 + pad + row * i + row // 2
        cv2.line(img, (x0 + pad, y), (x0 + pad + 28, y), col, 3)
        cv2.putText(img, label, (x0 + pad + 36, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


# def compose(cur_top, cur_fpv, t, pts):
#     top = to_bgr(cur_top, 'Top view')
#     draw_top_axes(top)
#     draw_top_path(top, pts)
#     draw_legend(top)
#     fpv = to_bgr(cur_fpv, 'FPV')
#     draw_legend(fpv)
#     frame = np.hstack([top, fpv])
#     # sim-time stamp top-right, since the legends sit bottom-left
#     cv2.putText(frame, f'sim t = {t:7.2f} s', (frame.shape[1] - 260, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
#     return frame


def compose(cur_top, cur_fpv, t, pts, segs):
    top = to_bgr(cur_top, 'Top view')
    draw_top_axes(top)
    draw_top_tree(top, segs)      # tree first...
    draw_top_path(top, pts)       # ...so the path is drawn on top
    draw_legend(top)
    fpv = to_bgr(cur_fpv, 'FPV')
    draw_legend(fpv)
    frame = np.hstack([top, fpv])
    # sim-time stamp top-right, since the legends sit bottom-left
    cv2.putText(frame, f'sim t = {t:7.2f} s', (frame.shape[1] - 260, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return frame


def draw_top_tree(img, segs):
    if not segs:
        return
    lines = [np.array([world_to_top_px(*a), world_to_top_px(*b)], np.int32) for a, b in segs]
    cv2.polylines(img, lines, False, (0, 255, 255), 1, cv2.LINE_AA)   # thin, so the path stands out


# def compose(cur_top, cur_fpv, t, pts, segs):
#     top = to_bgr(cur_top, 'Top view')
#     draw_top_axes(top)
#     draw_top_tree(top, segs)
#     draw_top_path(top, pts)


# ---- ROS 2 path subscriber -----------------------------------------------------

class PathListener(RosNode):
    def __init__(self, gz_node):
        super().__init__('video_path_listener')
        self.gz_node = gz_node
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(PoseArray, PATH_TOPIC, self.on_path, qos)

        tree_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL,   # must match the planner
                              history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(PoseArray, TREE_TOPIC, self.on_tree, tree_qos)



    def on_path(self, msg):
        global path_pts
        pts = [(p.position.x, p.position.y, p.position.z) for p in msg.poses]
        with lock:
            if pts == path_pts:
                return            # planner republishes every 200 ms; only redraw on change
            path_pts = pts
        draw_path_marker(self.gz_node, pts)
        self.get_logger().info(f'Path received: {len(pts)} waypoints')

    def on_tree(self, msg):
        global tree_segs
        p = [(q.position.x, q.position.y, q.position.z) for q in msg.poses]
        segs = list(zip(p[0::2], p[1::2]))
        with lock:
            tree_segs = segs
        draw_tree_marker(self.gz_node, segs)
        self.get_logger().info(f'Tree received: {len(segs)} edges')


# ---- gz callbacks / main -------------------------------------------------------

def on_top(msg):
    with lock:
        top_q.append((stamp(msg), msg))


def on_fpv(msg):
    with lock:
        fpv_q.append((stamp(msg), msg))


def stop(*_):
    global running
    running = False


def main():
    # keep our own Ctrl+C handler so the MP4 is finalised properly
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, stop)
    node = Node()
    draw_world_grid(node)

    executor = SingleThreadedExecutor()
    executor.add_node(PathListener(node))
    threading.Thread(target=executor.spin, daemon=True).start()
    assert node.subscribe(Image, TOP, on_top), f'cannot subscribe {TOP}'
    assert node.subscribe(Image, FPV, on_fpv), f'cannot subscribe {FPV}'
    print('Recording... Ctrl+C to stop')

    writer, t_next, cur_top, cur_fpv, n = None, None, None, None, 0
    while running:
        with lock:
            if t_next is None and top_q and fpv_q:
                t_next = max(top_q[0][0], fpv_q[0][0])
            # emit output frames while both streams have reached t_next
            while t_next is not None and top_q and fpv_q \
                    and top_q[-1][0] >= t_next and fpv_q[-1][0] >= t_next:
                while top_q and top_q[0][0] <= t_next:
                    cur_top = top_q.pop(0)[1]
                while fpv_q and fpv_q[0][0] <= t_next:
                    cur_fpv = fpv_q.pop(0)[1]
                if cur_top is not None and cur_fpv is not None:
                    # frame = compose(cur_top, cur_fpv, t_next, path_pts)
                    frame = compose(cur_top, cur_fpv, t_next, path_pts, tree_segs)
                    if writer is None:
                        h, w = frame.shape[:2]
                        writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (w, h))
                    writer.write(frame)
                    n += 1
                t_next += 1.0 / FPS
        time.sleep(0.005)

    if writer is not None:
        writer.release()
        print(f'Saved {n} frames ({n / FPS:.1f} s) to {OUT}')
    executor.shutdown()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
