#!/usr/bin/env python3
"""
Launch file for MARID Attitude Controller
Controls roll, pitch, and yaw using control surfaces (wings and tail).
"""
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="marid_controller",
            executable="marid_attitude_controller.py",
            name="marid_attitude_controller",
            output='screen',
            parameters=[{
                'update_rate': 50.0,  # Control loop frequency (Hz)
                
                # PID gains
                'roll_kp': 1.0,
                'roll_ki': 0.0,
                'roll_kd': 0.3,
                
                'pitch_kp': 1.5,
                'pitch_ki': 0.0,
                'pitch_kd': 0.5,
                
                'yaw_kp': 1.0,
                'yaw_ki': 0.0,
                'yaw_kd': 0.3,
                
                # Control surface limits (radians)
                'wing_max_deflection': 0.5,  # ~28.6 degrees
                'tail_max_deflection': 0.5,   # ~28.6 degrees
                
                # Waypoint navigation (should match AI controller)
                'destination_latitude': 34.0522,  # Los Angeles, CA (default)
                'destination_longitude': -118.2437,
                'datum_latitude': 37.45397139527321,  # SF Bay Area
                'datum_longitude': -122.16791304213365,
                'waypoint_tolerance': 2.0,
                
                'use_sim_time': True,
            }]
        )
    ])

