import os
from glob import glob
from setuptools import setup

package_name = 'drone_rrt_planner'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='you@example.com',
    description='RRT* path planning for PX4 X500 using octomap',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
    'console_scripts': [
        'px4_waypoint_follower = drone_rrt_planner.px4_waypoint_follower:main',
    ],
},
)
