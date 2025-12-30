import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.substitutions import Command, LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    marid_description = get_package_share_directory("marid_description")

    model_arg = DeclareLaunchArgument(name="model", default_value=os.path.join(
                                        marid_description, "urdf", "marid.urdf.xacro"
                                        ),
                                      description="Absolute path to robot urdf file"
    )

    gazebo_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=[
            str(Path(marid_description).parent.resolve())
            ]
        )
    
    robot_description = ParameterValue(Command([
            "xacro ",
            LaunchConfiguration("model")
        ]),
        value_type=str
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description,
                     "use_sim_time": True}]
    )




    world_path = os.path.join(get_package_share_directory("marid_description"),
                          "worlds", "wt.sdf")

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")
        ]),
        launch_arguments=[
            ("gz_args", f"-r -v 4 {world_path}"),
            ("on_exit_shutdown", "true")
        ]
    )




    # Wait for world to load before spawning entity
    gz_spawn_entity = TimerAction(
        period=3.0,  # Wait 3 seconds for world to fully load
        actions=[
            Node(
                package="ros_gz_sim",
                executable="create",
                output="screen",
                arguments=["-topic", "robot_description",
                           "-name", "marid",
                           "-x", "0", "-y", "0", "-z", "5.0"]
            )
        ]
    )

    imu_bridge = Node(
    package="ros_gz_bridge",
    executable="parameter_bridge",
    arguments=[
        "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
        "/imu@sensor_msgs/msg/Imu[gz.msgs.IMU",
        "/baro/pressure@sensor_msgs/msg/FluidPressure[gz.msgs.FluidPressure",
        "/gps/fix@sensor_msgs/msg/NavSatFix[gz.msgs.NavSat",
        "/world/wt/state@gz.msgs.World[gz.msgs.World"
        # Thruster bridge removed - using ApplyLinkWrench plugin directly via gz topic commands
    ],
    output='screen'
    )

    # Include localization launch file
    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory("marid_localization"),
                "launch",
                "local_localization.launch.py"
            )
        ])
    )

    return LaunchDescription([
        model_arg,
        gazebo_resource_path,
        robot_state_publisher_node,
        gazebo,
        gz_spawn_entity,
        imu_bridge,
        localization_launch
    ])