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

    joy_incrementer = Node(
        package="marid_controller",
        executable="joy_incrementer.py",
        output="screen",
    )

    thrust_controller = Node(
        package="marid_controller",
        executable="marid_thrust_controller.py",
        name="marid_thrust_controller",
        output="screen",
        parameters=[{
            "initial_thrust": 0.0,
            "min_thrust": 0.0,
            "max_thrust": 5000.0,
            "thrust_increment": 10.0,
            "world_name": "empty",
            "model_name": "marid",
            "link_name": "base_link_front",
            "update_rate": 50.0,
            "enable_keyboard": False,
            "enable_differential": False,
            "use_thruster_plugin": True,
            "use_center_thruster": True,
            "use_sim_time": True,
        }],
    )

    pose_estimator_logger = Node(
        package="marid_logging",
        executable="eskf_gt_logger",
        name="eskf_gt_logger",
        output="screen",
        parameters=[{
            'log_directory': '~/marid_ws/data_sync',
            'log_rate': 50.0,
            'samples_per_file': 10000,
            'enable_logging': True,
        }],
    )

    return LaunchDescription(
        [
            joy_node,
            joy_teleop,
            cmd_vel_to_wings,
            joy_incrementer,
            thrust_controller,
            pose_estimator_logger,
        ]
    )
