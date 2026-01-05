#!/usr/bin/env python3
"""
Launch file for MARID AI Controller with Waypoint Navigation
"""
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="marid_controller",
            executable="marid_ai_controller.py",
            name="marid_ai_controller",
            output='screen',
            parameters=[{
                'control_mode': 'pid',  # Start with PID
                'model_path': '',
                'update_rate': 50.0,
                'enable_ai': True,
                'enable_pid_fallback': True,
                # Waypoint navigation - GPS coordinates (preferred)
                # Use either GPS coordinates OR local x/y, not both
                'destination_latitude': None,  # Target latitude (degrees) - set this for GPS navigation
                'destination_longitude': None,  # Target longitude (degrees) - set this for GPS navigation
                # OR use local coordinates (backward compatibility)
                'destination_x': 100.0,  # Target X (m) - only if lat/lon not set
                'destination_y': 100.0,  # Target Y (m) - only if lat/lon not set
                # Datum (reference point) - must match navsat_transform.yaml
                'datum_latitude': 37.45397139527321,  # Reference latitude (degrees)
                'datum_longitude': -122.16791304213365,  # Reference longitude (degrees)
                # Altitude and velocity
                'altitude_min': 3.0,  # Minimum altitude (m)
                'altitude_max': 10.0,  # Maximum altitude (m)
                'target_altitude': 5.0,  # Preferred altitude (m)
                'target_velocity': 10.0,  # Average speed (m/s)
                'waypoint_tolerance': 2.0,  # Waypoint reach tolerance (m)
                'altitude_tolerance': 1.0,  # Altitude tolerance (m)
                # Control limits
                'min_thrust': 0.0,
                'max_thrust': 30.0,
                'max_yaw_differential': 0.2,
                'use_sim_time': True,
            }]
        )
    ])

