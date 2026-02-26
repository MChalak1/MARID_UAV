from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
import os
from ament_index_python.packages import get_package_share_directory
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command, LaunchConfiguration


def generate_launch_description():

    model_arg = DeclareLaunchArgument(
        name="model",
        default_value=os.path.join(get_package_share_directory("marid_description"), "urdf", "marid_new.urdf.xacro"),
        description="Absolute path to marid URDF file"
    )

    marid_description = ParameterValue(Command(["xacro ", LaunchConfiguration("model")]), value_type=str)

    marid_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": marid_description}]

    )

    joint_state_publisher_gui = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui"
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", os.path.join(get_package_share_directory("marid_description"), "rviz", "display.rviz")]
    )

    return LaunchDescription([
        model_arg,
        marid_state_publisher,
        joint_state_publisher_gui,
        rviz_node
    ])
