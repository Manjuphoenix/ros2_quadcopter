from setuptools import find_packages, setup

package_name = 'px4_tf_broadcaster'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    install_requires=['setuptools'],
    zip_safe=True,
    description='PX4 to ROS TF broadcaster',
    entry_points={
        'console_scripts': [
            'tf_broadcaster = px4_tf_broadcaster.tf_broadcaster:main',
        ],
    },
)
