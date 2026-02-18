#!/usr/bin/env python3
"""
Master launch file for MARID Full Control System
Launches controller manager, localization, thrust controller, AI controller, and attitude controller.
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Shared waypoint parameters
    destination_lat = 34.0522  # Los Angeles, CA
    destination_lon = -118.2437
    datum_lat = 37.4  # Match Gazebo world origin (wt.sdf)
    datum_lon = -122.1
    
    # Get launch file paths
    marid_localization_dir = get_package_share_directory('marid_localization')
    localization_launch = os.path.join(
        marid_localization_dir,
        'launch',
        'local_localization.launch.py'
    )
    
    marid_controller_dir = get_package_share_directory('marid_controller')
    controller_launch = os.path.join(
        marid_controller_dir,
        'launch',
        'controller.launch.py'
    )
    
    return LaunchDescription([
        # Include controller manager (spawns joint controllers)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([controller_launch])
        ),
        
        # Include localization stack (EKF, sensors, etc.)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([localization_launch])
        ),
        
        # Thrust Controller - applies forces to drone in Gazebo
        # Needs to start early to be ready when AI controller publishes commands
        Node(
            package="marid_controller",
            executable="marid_thrust_controller.py",
            name="marid_thrust_controller",
            output='screen',
            parameters=[{
                'initial_thrust': 0.0,         # Start at 0N (AI controller will set thrust)
                'min_thrust': 0.0,
                'max_thrust': 30.0,            # Can be None for auto-calculation
                'thrust_increment': 1.0,
                'world_name': 'wt',
                'model_name': 'marid',
                'link_name': 'base_link_front',
                'update_rate': 10.0,
                'enable_keyboard': True,
                'enable_differential': False,
                'use_sim_time': True,
            }]
        ),
        
        # AI Controller - handles thrust and waypoint navigation
        # Delay startup by 5 seconds to allow localization to initialize
        TimerAction(
            period=5.0,
            actions=[
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
                        'destination_latitude': destination_lat,
                        'destination_longitude': destination_lon,
                        # Datum (reference point) - must match navsat_transform.yaml
                        'datum_latitude': datum_lat,
                        'datum_longitude': datum_lon,
                        # Altitude and velocity
                        'altitude_min': 3.0,  # Minimum altitude (m)
                        'altitude_max': 10.0,  # Maximum altitude (m)
                        'target_altitude': 5.0,  # Preferred altitude (m)
                        'target_velocity': 10.0,  # Average speed (m/s)
                        'waypoint_tolerance': 2.0,  # Waypoint reach tolerance (m)
                        'altitude_tolerance': 1.0,  # Altitude tolerance (m)
                        # Control limits - AUTO-CALCULATION MODE
                        'min_thrust': 0.0,
                        'thrust_to_weight_ratio': 2.5,  # Thrust-to-weight ratio (2.5x weight for high-speed flight)
                        'max_yaw_differential': 0.2,
                        'use_sim_time': True,
                    }]
                )
            ]
        ),
        
        # Attitude Controller - handles control surfaces (wings and tail)
        # Delay startup by 5 seconds to allow localization to initialize
        TimerAction(
            period=5.0,
            actions=[
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
                        
                        # Waypoint navigation (matches AI controller)
                        'destination_latitude': destination_lat,
                        'destination_longitude': destination_lon,
                        'datum_latitude': datum_lat,
                        'datum_longitude': datum_lon,
                        'waypoint_tolerance': 2.0,
                        
                        'use_sim_time': True,
                    }]
                )
            ]
        ),
    ])
