from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    joy_node = Node(
        package="joy",
        executable="joy_node",
        name="joystick",
        output="screen",
        parameters=[
            os.path.join(
                get_package_share_directory("marid_controller"),
                "config",
                "joy_config.yaml",
            )
        ],
    )

    joy_teleop = Node(
        package="joy_teleop",
        executable="joy_teleop",
        parameters=[
            os.path.join(
                get_package_share_directory("marid_controller"),
                "config",
                "joy_teleop.yaml",
            )
        ],
    )

    cmd_vel_to_wings = Node(
        package="marid_controller",
        executable="cmd_vel_to_wings.py",
        output="screen",
        parameters=[{"front_gain": 1.0, "tail_gain": 1.0}],
    )

    return LaunchDescription(
        [
            joy_node,
            joy_teleop,
            cmd_vel_to_wings,
        ]
    )

