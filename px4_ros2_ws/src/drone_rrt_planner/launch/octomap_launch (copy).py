from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='octomap_server',
            executable='octomap_server_node',
            name='octomap_server',
            output='screen',
            parameters=[{
                'resolution': 0.2,
                'frame_id': 'map',
                'base_frame_id': 'base_link',
                'sensor_model.max_range': 8.0,
                'filter_ground': False,
                'occupancy_min_z': -0.1,
                'occupancy_max_z': 5.0,
            }],
            remappings=[
                ('cloud_in', '/depth_camera/points')
            ]
        )
    ])
