#!/usr/bin/python3
# Side-by-side (top view | FPV) video of an A* flight, synced on sim time.
#
# Run with the SYSTEM python (not conda), after sourcing ROS 2:
#   source /opt/ros/humble/setup.bash && /usr/bin/python3 vid_rec_astar.py
#
# Same video as vid_rec_v4.py (RRT), but for the A* navigator astar_nav_video.py:
#   - /astar_tree (PoseArray, pairs = edges): A* search tree, yellow
#   - /astar_path (PoseArray): simplified A* path, green
#   both drawn as 3D markers (GUI + camera-sensor scene) and as a top-view overlay,
#   plus the 1 m world grid, X/Y axes, top-view tick labels and a legend.
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
OUT = 'astar_flight.mp4'
PATH_TOPIC = '/astar_path'     # PoseArray, world ENU
TREE_TOPIC = '/astar_tree'     # PoseArray, consecutive pairs = edges, world ENU

# World grid drawn in the sensor scene (match your arena)
GRID_MIN, GRID_MAX, GRID_STEP = -10.0, 10.0, 1.0
GRID_Z = 0.02                  # just above the ground so it doesn't z-fight

# Top camera, must match rrt_world.sdf
TOP_W, TOP_H, TOP_HFOV, TOP_Z = 960, 720, 1.0, 30.0
TOP_F = (TOP_W / 2) / math.tan(TOP_HFOV / 2)   # focal length in pixels

# OpenCV colours are BGR
LEGEND = [
    ((0, 255, 0),   'A* path followed'),
    ((0, 255, 255), 'A* explored edges'),
    ((0, 0, 255),   'World X axis'),
    ((255, 102, 0), 'World Y axis'),
    ((64, 64, 64),  'Grid cell (1 m)'),
]

lock = threading.Lock()
top_q, fpv_q = [], []          # (sim_time_s, msg), filled by gz callback threads
path_pts = []                  # [(x, y, z)] world ENU, latest /astar_path
tree_px = []                   # top-view pixel segments of the latest /astar_tree (cached)
record_from = None             # sim time to start writing (once the path is drawn)
running = True


def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9


def to_bgr(msg, label):
    img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width * 3]
    img = cv2.cvtColor(img.reshape(msg.height, msg.width, 3), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (int(msg.width * HEIGHT / msg.height), HEIGHT))
    cv2.putText(img, label, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    return img


# ---- 3D markers ----------------------------------------------------------------

def _color(m, r, g, b):
    for c in (m.material.ambient, m.material.diffuse, m.material.emissive):
        c.r, c.g, c.b, c.a = r, g, b, 1.0


def _marker(ns, mid, mtype, rgb, pts):
    m = Marker()
    m.ns, m.id = ns, mid
    m.action = Marker.ADD_MODIFY          # same ns/id -> replaces the previous one
    m.type = mtype
    m.visibility = Marker.ALL
    _color(m, *rgb)
    for x, y, z in pts:
        p = m.point.add()
        p.x, p.y, p.z = x, y, z
    return m


def send_marker(node, m, services=('/marker', '/sensors/marker')):
    # one-way services, so a "timeout" result is expected; the marker is still applied
    for srv in services:
        node.request(srv, m, Marker, Empty, 50)


def draw_world_grid(node):
    n = int(round((GRID_MAX - GRID_MIN) / GRID_STEP))
    pts = []
    for i in range(n + 1):
        k = GRID_MIN + i * GRID_STEP
        pts += [(k, GRID_MIN, GRID_Z), (k, GRID_MAX, GRID_Z)]   # lines of constant x
        pts += [(GRID_MIN, k, GRID_Z), (GRID_MAX, k, GRID_Z)]   # lines of constant y
    send_marker(node, _marker('world_grid', 0, Marker.LINE_LIST, (0.25, 0.25, 0.25), pts))
    send_marker(node, _marker('world_axes', 0, Marker.LINE_LIST, (1.0, 0.0, 0.0),
                              [(0, 0, GRID_Z), (GRID_MAX, 0, GRID_Z)]))   # +X red
    send_marker(node, _marker('world_axes', 1, Marker.LINE_LIST, (0.0, 0.4, 1.0),
                              [(0, 0, GRID_Z), (0, GRID_MAX, GRID_Z)]))   # +Y blue


# ---- 2D overlays (top view) ----------------------------------------------------

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


def draw_top_tree(img, segs_px):
    if segs_px:
        cv2.polylines(img, segs_px, False, (0, 255, 255), 1, cv2.LINE_AA)   # thin, path stays on top


def draw_top_path(img, pts):
    if len(pts) < 2:
        return
    px = np.array([world_to_top_px(x, y, z) for x, y, z in pts], np.int32)
    cv2.polylines(img, [px], False, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.circle(img, tuple(int(v) for v in px[0]), 7, (0, 255, 0), -1)    # start
    cv2.circle(img, tuple(int(v) for v in px[-1]), 7, (0, 0, 255), -1)   # goal


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


def compose(cur_top, cur_fpv, t, pts, segs_px):
    top = to_bgr(cur_top, 'Top view (A*)')
    draw_top_axes(top)
    draw_top_tree(top, segs_px)
    draw_top_path(top, pts)
    draw_legend(top)
    fpv = to_bgr(cur_fpv, 'FPV')
    draw_legend(fpv)
    frame = np.hstack([top, fpv])
    # sim-time stamp top-right, since the legends sit bottom-left
    cv2.putText(frame, f'sim t = {t:7.2f} s', (frame.shape[1] - 260, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return frame


# ---- ROS 2 path / tree subscriber ----------------------------------------------

class AstarListener(RosNode):
    def __init__(self, gz_node):
        super().__init__('video_astar_listener')
        self.gz_node = gz_node
        # the navigator publishes both latched, so the recorder can start at any time
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(PoseArray, PATH_TOPIC, self.on_path, qos)
        self.create_subscription(PoseArray, TREE_TOPIC, self.on_tree, qos)

    def on_path(self, msg):
        global path_pts
        pts = [(p.position.x, p.position.y, p.position.z) for p in msg.poses]
        with lock:
            if pts == path_pts:
                return
            path_pts = pts
        send_marker(self.gz_node, _marker('astar_path', 0, Marker.LINE_STRIP, (0.0, 1.0, 0.0), pts))
        self.get_logger().info(f'Path received: {len(pts)} waypoints')

    def on_tree(self, msg):
        global tree_px
        pts = [(q.position.x, q.position.y, q.position.z) for q in msg.poses]
        segs_px = [np.array([world_to_top_px(*a), world_to_top_px(*b)], np.int32)
                   for a, b in zip(pts[0::2], pts[1::2])]
        with lock:
            tree_px = segs_px
        send_marker(self.gz_node, _marker('astar_tree', 0, Marker.LINE_LIST, (1.0, 1.0, 0.0), pts))
        self.get_logger().info(f'Tree received: {len(segs_px)} edges')


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
    global record_from
    # keep our own Ctrl+C handler so the MP4 is finalised properly
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, stop)
    node = Node()
    draw_world_grid(node)
    assert node.subscribe(Image, TOP, on_top), f'cannot subscribe {TOP}'
    assert node.subscribe(Image, FPV, on_fpv), f'cannot subscribe {FPV}'

    executor = SingleThreadedExecutor()
    executor.add_node(AstarListener(node))
    threading.Thread(target=executor.spin, daemon=True).start()
    print('Waiting for /astar_path, then recording... Ctrl+C to stop')

    writer, t_next, cur_top, cur_fpv, n = None, None, None, None, 0
    while running:
        with lock:
            if t_next is None and top_q and fpv_q:
                t_next = max(top_q[0][0], fpv_q[0][0])
            if record_from is None and path_pts and t_next is not None:
                record_from = t_next + 1.0   # give the sensor scene ~1 s to show the markers
            # emit output frames while both streams have reached t_next
            while t_next is not None and top_q and fpv_q \
                    and top_q[-1][0] >= t_next and fpv_q[-1][0] >= t_next:
                while top_q and top_q[0][0] <= t_next:
                    cur_top = top_q.pop(0)[1]
                while fpv_q and fpv_q[0][0] <= t_next:
                    cur_fpv = fpv_q.pop(0)[1]
                if record_from is not None and t_next >= record_from \
                        and cur_top is not None and cur_fpv is not None:
                    frame = compose(cur_top, cur_fpv, t_next, path_pts, tree_px)
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
    else:
        print('No frames written (no /astar_path received?)')
    executor.shutdown()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
