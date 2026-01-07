#!/usr/bin/env python3
"""
Launch file for MARID Thrust Controller
Applies thrust forces to the drone using the ApplyLinkWrench plugin in Gazebo.
"""
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="marid_controller",
            executable="marid_thrust_controller.py",
            name="marid_thrust_controller",
            output='screen',
            parameters=[{
                'initial_thrust': 0.0,         # Initial thrust: 0N (AI controller will set thrust)
                'min_thrust': 0.0,             # Minimum thrust: 0N
                'max_thrust': 30.0,            # Maximum thrust: 30N
                'thrust_increment': 1.0,       # Increment per keypress: 1N
                'world_name': 'wt',            # Gazebo world name
                'model_name': 'marid',         # Model name
                'link_name': 'base_link_front', # Link to apply force to
                'update_rate': 10.0,          # Update rate for persistent wrench (Hz)
                'enable_keyboard': True,       # Enable keyboard control
                'enable_differential': False,  # Start with equal thrust
                'use_sim_time': True,
            }]
        )
    ])

