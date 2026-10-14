#!/usr/bin/python3
# Run with the SYSTEM python (not conda):  /usr/bin/python3 vid_recorder.py
import signal
import threading
import time

import cv2
import numpy as np
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image

TOP = '/top_camera'
FPV = '/world/rrt_world/model/x500_mono_cam_0/link/camera_link/sensor/camera/image'
FPS = 30
HEIGHT = 720
OUT = 'flight.mp4'

lock = threading.Lock()
top_q, fpv_q = [], []          # (sim_time_s, msg), filled by gz callback threads
running = True


def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9


def to_bgr(msg, label):
    img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width * 3]
    img = cv2.cvtColor(img.reshape(msg.height, msg.width, 3), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (int(msg.width * HEIGHT / msg.height), HEIGHT))
    cv2.putText(img, label, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    return img


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
    signal.signal(signal.SIGINT, stop)
    node = Node()
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
                    frame = np.hstack([to_bgr(cur_top, 'Top view'), to_bgr(cur_fpv, 'FPV')])
                    cv2.putText(frame, f'sim t = {t_next:7.2f} s', (20, HEIGHT - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
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


if __name__ == '__main__':
    main()

