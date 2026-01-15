#!/usr/bin/env python3
"""
Option A Architecture Launch File - High-Level Guidance Control
Launches: Guidance Node (AI/PID) → Guidance Tracker → Actuator Controllers

Architecture:
  1. Guidance Node: Computes guidance targets (desired_heading_rate, desired_speed)
  2. Guidance Tracker: Tracks guidance targets and outputs actuator commands (thrust, yaw_diff)
  3. Thrust Controller: Applies actuator commands to Gazebo
  4. Attitude Controller: Controls surfaces based on guidance (optional, can also use waypoint directly)
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
    datum_lat = 37.45397139527321  # SF Bay Area
    datum_lon = -122.16791304213365
    
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
        # Subscribes to /marid/thrust/total and /marid/yaw/differential from guidance tracker
        Node(
            package="marid_controller",
            executable="marid_thrust_controller.py",
            name="marid_thrust_controller",
            output='screen',
            parameters=[{
                'initial_thrust': 0.0,
                'min_thrust': 0.0,
                'thrust_to_weight_ratio': 2.5,
                'thrust_increment': 1.0,
                'world_name': 'wt',
                'model_name': 'marid',
                'link_name': 'base_link_front',
                'update_rate': 10.0,
                'enable_keyboard': False,  # Disabled in Option A (guidance controls it)
                'enable_differential': True,  # Enable differential for yaw control
                'use_sim_time': True,
            }]
        ),
        
        # AI GUIDANCE NODE (Option A) - Computes guidance targets
        # Delay startup by 5 seconds to allow localization to initialize
        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package="marid_controller",
                    executable="marid_ai_guidance.py",
                    name="marid_ai_guidance",
                    output='screen',
                    parameters=[{
                        'control_mode': 'pid',  # Start with PID guidance, switch to 'ai' when model loaded
                        'model_path': '',  # Set path to trained model when available
                        'normalizer_path': '',  # Set path to normalization parameters when available
                        'update_rate': 50.0,
                        'enable_ai': True,
                        'enable_pid_fallback': True,
                        # Waypoint navigation - GPS coordinates
                        'destination_latitude': destination_lat,
                        'destination_longitude': destination_lon,
                        'datum_latitude': datum_lat,
                        'datum_longitude': datum_lon,
                        # Guidance parameters
                        'target_altitude': 5.0,  # m
                        'target_velocity': 10.0,  # m/s
                        'altitude_min': 3.0,  # m
                        'altitude_max': 10.0,  # m
                        'waypoint_tolerance': 2.0,  # m
                        # Guidance limits
                        'max_heading_rate': 0.5,  # rad/s
                        'min_speed': 10.0,  # m/s
                        'max_speed': 50.0,  # m/s
                        'use_sim_time': True,
                    }]
                )
            ]
        ),
        
        # GUIDANCE TRACKER (Option A) - Tracks guidance targets and outputs actuator commands
        # Subscribes to /marid/guidance/desired_heading_rate and /marid/guidance/desired_speed
        # Publishes to /marid/thrust/total and /marid/yaw/differential
        TimerAction(
            period=5.5,  # Start slightly after guidance node
            actions=[
                Node(
                    package="marid_controller",
                    executable="marid_guidance_tracker.py",
                    name="marid_guidance_tracker",
                    output='screen',
                    parameters=[{
                        'update_rate': 50.0,
                        # Thrust parameters
                        'thrust_to_weight_ratio': 2.5,
                        'min_thrust': 0.0,
                        'max_yaw_differential': 0.2,
                        # Physics-based thrust
                        'use_physics_thrust': True,
                        'use_airspeed_sensor': True,
                        'drag_coefficient': 0.1,  # Tune based on your model
                        'air_density': 1.225,  # kg/m³ at sea level
                        # Wind vector (from world file: [0, 1, 0] m/s)
                        'wind_x': 0.0,
                        'wind_y': 1.0,
                        'wind_z': 0.0,
                        # PID gains for tracking guidance targets
                        'speed_kp': 1.0,
                        'speed_ki': 0.05,
                        'speed_kd': 0.3,
                        'heading_rate_kp': 1.0,
                        'heading_rate_ki': 0.1,
                        'heading_rate_kd': 0.3,
                        # Altitude control
                        'altitude_kp': 2.0,
                        'altitude_ki': 0.1,
                        'altitude_kd': 0.5,
                        'target_altitude': 5.0,  # m
                        'use_sim_time': True,
                    }]
                )
            ]
        ),
        
        # Attitude Controller - Controls surfaces (can subscribe to guidance or use waypoint directly)
        # Optionally can subscribe to guidance heading_rate for coordinated turns
        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package="marid_controller",
                    executable="marid_attitude_controller.py",
                    name="marid_attitude_controller",
                    output='screen',
                    parameters=[{
                        'update_rate': 50.0,
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
                        # Control surface limits
                        'wing_max_deflection': 0.5,
                        'tail_max_deflection': 0.5,
                        # Waypoint navigation (for attitude control)
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
