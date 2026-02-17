#!/usr/bin/env python3
"""FAST-LIO LiDAR-Inertial Odometry Launch File"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    fast_lio_pkg = get_package_share_directory('fast_lio')
    config_file = os.path.join(fast_lio_pkg, 'config', 'marid.yaml')
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time'
        ),
        DeclareLaunchArgument(
            'config_path',
            default_value=config_file,
            description='Path to FAST-LIO config file'
        ),
        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package='fast_lio',
                    executable='fastlio_mapping',
                    name='fast_lio',
                    output='screen',
                    parameters=[
                        LaunchConfiguration('config_path'),
                        {'use_sim_time': LaunchConfiguration('use_sim_time')}
                    ],
                ),
            ]
        ),
    ])