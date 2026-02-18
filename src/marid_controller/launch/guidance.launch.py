#!/usr/bin/env python3
"""
Launch file for Option A Guidance Node (standalone)
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction

def generate_launch_description():
    destination_lat = 34.0522  # Los Angeles, CA
    destination_lon = -118.2437
    datum_lat = 37.4  # Match Gazebo world origin (wt.sdf)
    datum_lon = -122.1
    
    return LaunchDescription([
        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package="marid_controller",
                    executable="marid_ai_guidance.py",
                    name="marid_ai_guidance",
                    output='screen',
                    parameters=[{
                        'control_mode': 'pid',
                        'model_path': '',
                        'normalizer_path': '',
                        'update_rate': 50.0,
                        'enable_ai': True,
                        'enable_pid_fallback': True,
                        'destination_latitude': destination_lat,
                        'destination_longitude': destination_lon,
                        'datum_latitude': datum_lat,
                        'datum_longitude': datum_lon,
                        'target_altitude': 5.0,
                        'target_velocity': 10.0,
                        'altitude_min': 3.0,
                        'altitude_max': 10.0,
                        'waypoint_tolerance': 2.0,
                        'max_heading_rate': 0.5,
                        'min_speed': 10.0,
                        'max_speed': 50.0,
                        'use_sim_time': True,
                    }]
                )
            ]
        ),
    ])
