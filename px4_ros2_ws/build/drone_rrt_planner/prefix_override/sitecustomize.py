import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/jarvis/Week1/px4_ros2_ws/install/drone_rrt_planner'
