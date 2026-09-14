from setuptools import find_packages, setup

package_name = 'px4_tf_broadcaster'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jarvis',
    maintainer_email='manjunath1055@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'tf_broadcaster = px4_tf_broadcaster.tf_broadcaster:main',
            'px4_waypoint_follower = drone_rrt_planner.px4_waypoint_follower:main',
        ],
    },
)
