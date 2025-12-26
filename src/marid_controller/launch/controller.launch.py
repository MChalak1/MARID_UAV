from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():



    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager"
        ],
        parameters=[{'use_sim_time': True}]

    )


    simple_position_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "simple_position_controller",
            "--controller-manager",
            "/controller_manager"
        ],
        parameters=[{'use_sim_time': True}]
    )

    marid_odom_spawner = Node(
        package="marid_controller",
        executable="marid_odom_pub.py",      
        name="marid_odom_node",
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    return LaunchDescription([
        joint_state_broadcaster_spawner,
        simple_position_controller,



    ])