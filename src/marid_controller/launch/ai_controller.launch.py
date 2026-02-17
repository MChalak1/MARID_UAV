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
                # Waypoint navigation - GPS coordinates (default: Los Angeles, CA)
                'destination_latitude': 34.0522,  # Los Angeles, CA (degrees)
                'destination_longitude': -118.2437,  # Los Angeles, CA (degrees)
                # Local coordinates (backward compatibility - only used if GPS not set)
                # 'destination_x': 100.0,  # Target X (m) - only if lat/lon not set
                # 'destination_y': 100.0,  # Target Y (m) - only if lat/lon not set
                # Datum (reference point) - must match navsat_transform.yaml
                'datum_latitude': 37.45397139527321,  # Reference latitude (degrees) - SF Bay Area
                'datum_longitude': -122.16791304213365,  # Reference longitude (degrees) - SF Bay Area
                # Altitude and velocity
                'altitude_min': 3.0,  # Minimum altitude (m)
                'altitude_max': 10.0,  # Maximum altitude (m)
                'target_altitude': 5.0,  # Preferred altitude (m)
                'target_velocity': 10.0,  # Average speed (m/s)
                'waypoint_tolerance': 2.0,  # Waypoint reach tolerance (m)
                'altitude_tolerance': 1.0,  # Altitude tolerance (m)
                # Control limits - AUTO-CALCULATION MODE
                'min_thrust': 0.0,
                # Note: Omit max_thrust to auto-calculate from aircraft mass
                # 'max_thrust': 200.0,  # Uncomment to set fixed max thrust (N)
                'thrust_to_weight_ratio': 2.5,  # Thrust-to-weight ratio (2.5x weight for high-speed flight)
                # Alternative: Override auto-calculation with fixed value
                # 'base_thrust_override': 200.0,  # Uncomment to use fixed thrust instead
                'max_yaw_differential': 0.2,
                'use_sim_time': True,
            }]
        )
    ])

