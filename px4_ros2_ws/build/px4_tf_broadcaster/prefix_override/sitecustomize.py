import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/jarvis/Week1/px4_ros2_ws/install/px4_tf_broadcaster'
