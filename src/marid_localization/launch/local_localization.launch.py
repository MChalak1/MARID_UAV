#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('marid_localization')  # Change this

    config_file = os.path.join(pkg_dir, 'config', 'ekf.yaml')
    navsat_config_file = os.path.join(pkg_dir, 'config', 'navsat_transform.yaml')


    
    return LaunchDescription([


        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            arguments=["--x", "0", "--y", "0","--z", "0.103",
                    "--qx", "0", "--qy", "0", "--qz", "0", "--qw", "1",
                    "--frame-id", "base_link_front",
                    "--child-frame-id", "imu_link_ekf"],
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
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node_odom',
            output='screen',
            parameters=[config_file,  {'use_sim_time': True}],
            remappings=[
                ('/odometry/filtered', '/odometry/filtered/local')
            ]
        ),
        
        # Global EKF - fuses local odometry + GPS
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node_map',
            output='screen',
            parameters=[config_file,  {'use_sim_time': True}],
            remappings=[
                ('/odometry/filtered', '/odometry/filtered/global')
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

        Node(
            package='marid_localization',
            executable='imu_republisher.py',
            name='imu_republisher',
            output='screen',
            parameters=[{'use_sim_time': True}]
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
            }]
        ),
    ])