#!/usr/bin/env python3
"""
Launch file for MARID Safety Node
"""
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="marid_controller",
            executable="marid_safety_node.py",
            name="marid_safety_node",
            output='screen',
            parameters=[{
                'update_rate': 10.0,
                'enable_self_destruct': True,
                'critical_altitude_threshold': 1.0,  # Critical altitude (m)
                'altitude_drop_rate_threshold': -5.0,  # m/s
                'altitude_drop_time_window': 2.0,  # seconds
                'min_thrust_threshold': 0.5,  # Minimum expected thrust (N)
                'thrust_failure_time': 3.0,  # seconds
                'enemy_ao_enabled': True,  # Enable enemy AO monitoring
                'enemy_ao_center_x': 50.0,  # Enemy AO center X (m)
                'enemy_ao_center_y': 50.0,  # Enemy AO center Y (m)
                'enemy_ao_radius': 100.0,  # Enemy AO radius (m)
                'self_destruct_delay': 1.0,  # Delay before self-destruct (s)
                'self_destruct_thrust': 0.0,  # Thrust on self-destruct
                'use_sim_time': True,
            }]
        )
    ])

