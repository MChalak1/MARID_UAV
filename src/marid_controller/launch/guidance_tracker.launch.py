#!/usr/bin/env python3
"""
Launch file for Guidance Tracker (standalone)
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction

def generate_launch_description():
    return LaunchDescription([
        TimerAction(
            period=5.5,
            actions=[
                Node(
                    package="marid_controller",
                    executable="marid_guidance_tracker.py",
                    name="marid_guidance_tracker",
                    output='screen',
                    parameters=[{
                        'update_rate': 50.0,
                        'thrust_to_weight_ratio': 2.5,
                        'max_thrust': None,
                        'min_thrust': 0.0,
                        'max_yaw_differential': 0.2,
                        'speed_kp': 1.0,
                        'speed_ki': 0.05,
                        'speed_kd': 0.3,
                        'heading_rate_kp': 1.0,
                        'heading_rate_ki': 0.1,
                        'heading_rate_kd': 0.3,
                        'altitude_kp': 2.0,
                        'altitude_ki': 0.1,
                        'altitude_kd': 0.5,
                        'target_altitude': 5.0,
                        'use_sim_time': True,
                    }]
                )
            ]
        ),
    ])
