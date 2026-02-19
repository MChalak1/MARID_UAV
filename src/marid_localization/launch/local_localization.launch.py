#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration, Command
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('marid_localization')  

    config_file = os.path.join(pkg_dir, 'config', 'ekf.yaml')
    navsat_config_file = os.path.join(pkg_dir, 'config', 'navsat_transform.yaml')
    imu_filter_config_file = os.path.join(pkg_dir, 'config', 'imu_filter_madgwick.yaml')

    return LaunchDescription([


        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            arguments=["--x", "0", "--y", "0","--z", "0.103",
                    "--qx", "0", "--qy", "0", "--qz", "0", "--qw", "1",
                    "--frame-id", "base_link_front",
                    "--child-frame-id", "imu_link_ekf"],
        ),
        
        # Align FAST-LIO's camera_init frame with odom (Gazebo world) for EKF fusion.
        # camera_init origin = first LiDAR scan = robot spawn at (0, 0, 5) in Gazebo.
        # Enables robot_localization to correctly fuse /Odometry (LiDAR) with /gazebo/odom.
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            arguments=["--x", "0", "--y", "0", "--z", "5.0",
                       "--qx", "0", "--qy", "0", "--qz", "0", "--qw", "1",
                       "--frame-id", "odom",
                       "--child-frame-id", "camera_init"],
        ),
        
        # Convert GPS NavSatFix to Odometry in map frame
        Node(
            package='robot_localization',
            executable='navsat_transform_node',
            name='navsat_transform',
            output='screen',
            parameters=[navsat_config_file],  # Load from config file
            remappings=[
                ('/imu', '/imu_ekf'),
                ('/odometry/filtered', '/odometry/filtered/local'),
                ('/odometry/gps', '/gps/odometry')
            ]
        ),
        
        # Local EKF - fuses IMU + Barometer
        # Delay startup by 3 seconds to allow sensors to initialize and publish valid data
        TimerAction(
            period=3.0,
            actions=[
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node_odom',
            output='screen',
            parameters=[config_file,  {'use_sim_time': True}],
            remappings=[
                ('/odometry/filtered', '/odometry/filtered/local')
                    ]
                )
            ]
        ),
        
        # Global EKF - fuses local odometry + GPS
        # Delay startup by 4 seconds (after local EKF) to ensure local EKF is publishing
        TimerAction(
            period=4.0,
            actions=[
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node_map',
            output='screen',
            parameters=[config_file,  {'use_sim_time': True}],
            remappings=[
                ('/odometry/filtered', '/odometry/filtered/global')
                    ]
                )
            ]
        ),

        
        # Barometer altitude converter (you'll need to create this)
        Node(
            package='marid_localization',
            executable='baro_to_alt.py',
            name='barometer_altitude_converter',
            output='screen',
            parameters=[{
                'sea_level_pressure': 101325.0,  # Pa
                'temperature': 288.15,  # K (15°C)
                'use_sim_time': True,
            }]
        ),

        # Add gravity to IMU so Madgwick sees ~9.81 m/s² when level (Gazebo publishes specific force)
        Node(
            package='marid_localization',
            executable='imu_add_gravity.py',
            name='imu_add_gravity',
            output='screen',
            parameters=[{'use_sim_time': True}],
        ),
        # Madgwick filter: /imu/with_gravity -> orientation, publishes to /imu/data
        Node(
            package='imu_filter_madgwick',
            executable='imu_filter_madgwick_node',
            name='imu_filter_madgwick',
            output='screen',
            parameters=[imu_filter_config_file, {'use_sim_time': True}],
            remappings=[('/imu/data_raw', '/imu/with_gravity')],
        ),
        # Republisher: subscribes to Madgwick output (/imu/data), publishes to /imu_ekf (no changes elsewhere)
        Node(
            package='marid_localization',
            executable='imu_republisher.py',
            name='imu_republisher',
            output='screen',
            parameters=[{'use_sim_time': True}],
            remappings=[('/imu', '/imu/data')],
        ),
        
        # Convert Gazebo pose to odometry (ground truth)
        Node(
            package='marid_localization',
            executable='gazebo_pose_to_odom.py',
            name='gazebo_pose_to_odom',
            output='screen',
            parameters=[{
                'model_name': 'marid',
                'odom_frame_id': 'odom',
                'base_frame_id': 'base_link_front',
                'publish_tf': False,  # EKF handles TF
                'use_sim_time': True,
                'airspeed_topic': '/airspeed/velocity',  # Use converted airspeed topic
            }]
        ),
        
        # Airspeed converter - converts gz.msgs.AirSpeed to std_msgs/Float64
        Node(
            package='marid_localization',
            executable='airspeed_converter.py',
            name='airspeed_converter',
            output='screen',
            parameters=[{
                'gz_topic': '/airspeed',
                'output_topic': '/airspeed/velocity',  # Changed to avoid conflict with FluidPressure publisher
                'publish_rate': 50.0,
            }]
        ),
        
        # Wind estimator - uses EKF odometry + pitot airspeed to estimate 3D wind
        Node(
            package='marid_localization',
            executable='wind_estimator.py',
            name='wind_estimator',
            output='screen',
            parameters=[{
                'odom_topic': '/odometry/filtered/local',
                'airspeed_topic': '/airspeed/velocity',  # Use converted airspeed topic
                'wind_frame_id': 'odom',
            }]
        ),
        
        # Air density calculator - calculates density based on altitude
        Node(
            package='marid_localization',
            executable='air_density_calculator.py',
            name='air_density_calculator',
            output='screen',
            parameters=[{
                'update_rate': 10.0,  # Hz
                'use_barometer': True,  # Use barometer altitude if available
                'sea_level_density': 1.225,  # kg/m³
                'sea_level_pressure': 101325.0,  # Pa
                'sea_level_temperature': 288.15,  # K (15°C)
                'use_sim_time': True,
            }]
        ),
    ])