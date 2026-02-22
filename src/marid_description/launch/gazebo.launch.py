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
                          "worlds", "empty.sdf")

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")
        ]),
        launch_arguments=[
            ("gz_args", f"-r -v 4 {world_path}"),
            ("on_exit_shutdown", "true")
        ]
    )




    # Timer Action to wait for world spawning
    gz_spawn_entity = TimerAction(
        period=3.0,  # 3 seconds for world to fully load
        actions=[
            Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=["-topic", "robot_description",
                   "-name", "marid",
                   "-z", "1.0"]
            )
        ]
    )

    imu_bridge = Node(
    package="ros_gz_bridge",
    executable="parameter_bridge",
    name="marid_ros_gz_bridge",
    parameters=[{"use_sim_time": True}],
    remappings=[
        ("/model/marid/odometry", "/gazebo/odom"),
    ],
    arguments=[
        "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
        "/imu@sensor_msgs/msg/Imu[gz.msgs.IMU",
        "/model/marid/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry",
        "/baro/pressure@sensor_msgs/msg/FluidPressure[gz.msgs.FluidPressure",
        "/gps/fix@sensor_msgs/msg/NavSatFix[gz.msgs.NavSat",
        "/world/wt/state@gz.msgs.World[gz.msgs.World",
        # Bridge thruster command topics (ROS2 -> Gazebo Transport)
        # Note: ] means ROS2 -> Gazebo, [ means Gazebo -> ROS2
        "/model/marid/joint/thruster_L_joint/cmd_vel@std_msgs/msg/Float64]gz.msgs.Double",
        "/model/marid/joint/thruster_R_joint/cmd_vel@std_msgs/msg/Float64]gz.msgs.Double",
        "/model/marid/joint/thruster_center_joint/cmd_vel@std_msgs/msg/Float64]gz.msgs.Double",
        # Bridge wing and tail joint command topics (ROS2 -> Gazebo Transport)
        # Using custom topic names without /0/ to be ROS2-compatible
        "/model/marid/joint/left_wing_joint/cmd_pos@std_msgs/msg/Float64]gz.msgs.Double",
        "/model/marid/joint/right_wing_joint/cmd_pos@std_msgs/msg/Float64]gz.msgs.Double",
        "/model/marid/joint/tail_left_joint/cmd_pos@std_msgs/msg/Float64]gz.msgs.Double",
        "/model/marid/joint/tail_right_joint/cmd_pos@std_msgs/msg/Float64]gz.msgs.Double",
        "/lidar/scan@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
        # Camera bridge:
        "/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image",
        "/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
    
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
        localization_launch,
    ])
