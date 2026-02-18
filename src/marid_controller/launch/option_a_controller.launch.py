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
from launch.actions import IncludeLaunchDescription, TimerAction, DeclareLaunchArgument, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    return LaunchDescription([
        # Declare launch arguments for destination coordinates
        DeclareLaunchArgument(
            'destination_latitude',
            default_value='34.0522',  # Los Angeles, CA
            description='Destination latitude in degrees'
        ),
        DeclareLaunchArgument(
            'destination_longitude',
            default_value='-118.2437',  # Los Angeles, CA
            description='Destination longitude in degrees'
        ),
        DeclareLaunchArgument(
            'destination_x',
            default_value='-1.0',  # -1.0 means not set, use GPS coordinates instead
            description='Destination X coordinate in meters (local ENU frame). Use -1.0 to use GPS coordinates.'
        ),
        DeclareLaunchArgument(
            'destination_y',
            default_value='-1.0',  # -1.0 means not set, use GPS coordinates instead
            description='Destination Y coordinate in meters (local ENU frame). Use -1.0 to use GPS coordinates.'
        ),
        # Use OpaqueFunction to access launch arguments and convert to floats
        OpaqueFunction(function=launch_setup)
    ])

def launch_setup(context):
    # Get launch argument values and convert to float
    destination_lat = float(context.launch_configurations['destination_latitude'])
    destination_lon = float(context.launch_configurations['destination_longitude'])
    destination_x = float(context.launch_configurations.get('destination_x', '-1.0'))
    destination_y = float(context.launch_configurations.get('destination_y', '-1.0'))
    datum_lat = 37.4  # Match Gazebo world origin (wt.sdf)
    datum_lon = -122.1
    
    # Determine if local coordinates are set (both must be != -1.0)
    use_local_coords = (destination_x != -1.0 and destination_y != -1.0)
    
    # Get launch file paths
    marid_controller_dir = get_package_share_directory('marid_controller')
    controller_launch = os.path.join(
        marid_controller_dir,
        'launch',
        'controller.launch.py'
    )
    
    return [
        # Include controller manager (spawns joint controllers)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([controller_launch])
        ),
        # Localization is launched by marid_description/launch/gazebo.launch.py
        # to avoid duplicate nodes (EKF, wind_estimator, airspeed_converter, etc.).
        
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
                'thrust_to_weight_ratio': 0.65,  # Reduced from 2.5: gives max thrust ~1.3x weight (reasonable for aircraft)
                'thrust_increment': 1.0,
                'world_name': 'wt',
                'model_name': 'marid',
                'link_name': 'base_link_front',
                'update_rate': 50.0,  # Increased to match thruster plugin update rate
                'enable_keyboard': False,  # Disabled in Option A (guidance controls it)
                'enable_differential': False,  # Disable differential thrust - yaw controlled aerodynamically
                'thrust_to_angvel_gain': 50.0,  # Conversion factor: omega = gain * sqrt(thrust)
                'use_thruster_plugin': True,  # Use Gazebo Thruster plugin (True) or legacy wrench (False)
                'use_center_thruster': False,  # Use single center thruster (True) or dual left/right (False) - set to True for testing
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
                        # Waypoint navigation - use local coordinates if set, otherwise GPS
                        'destination_latitude': destination_lat if not use_local_coords else -1.0,
                        'destination_longitude': destination_lon if not use_local_coords else -1.0,
                        'destination_x': destination_x,
                        'destination_y': destination_y,
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
                        'base_thrust_override': 1.0,  # Fixed thrust of 1.0N (overrides PID calculations)
                        'thrust_to_weight_ratio': 0.65,  # Not used when override is set
                        'min_thrust': 0.0,
                        'max_yaw_differential': 0.2,
                        # Physics-based thrust (disabled for fixed thrust test)
                        'use_physics_thrust': False,
                        'use_airspeed_sensor': True,
                        'drag_coefficient': 0.1,  # Tune based on your model
                        'air_density': 1.225,  # kg/m³ at sea level
                        # Wind vector (from world file: [0, 0, 0] m/s - disabled for testing)
                        'wind_x': 0.0,
                        'wind_y': 0.0,
                        'wind_z': 0.0,
                        # PID gains (disabled for fixed thrust test, but can still be used for attitude control)
                        'speed_kp': 0.0,  # Disabled - not used when override is set
                        'speed_ki': 0.0,
                        'speed_kd': 0.0,
                        'heading_rate_kp': 1.0,  # Still active for yaw control
                        'heading_rate_ki': 0.1,
                        'heading_rate_kd': 0.3,
                        # Altitude control (re-enabled for altitude maintenance)
                        'altitude_kp': 2.0,  # Re-enable for altitude maintenance
                        'altitude_ki': 0.1,
                        'altitude_kd': 0.5,
                        'target_altitude': 5.0,
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
                        # PID gains (reduced P, increased D for better damping)
                        'roll_kp': 0.5,  # Reduced from 1.0
                        'roll_ki': 0.0,
                        'roll_kd': 0.8,  # Increased from 0.3 for better damping
                        'pitch_kp': 0.8,  # Reduced from 1.5
                        'pitch_ki': 0.0,
                        'pitch_kd': 1.0,  # Increased from 0.5 for better damping
                        'yaw_kp': 0.5,  # Reduced from 1.0
                        'yaw_ki': 0.0,
                        'yaw_kd': 0.8,  # Increased from 0.3 for better damping
                        # Control surface limits
                        'wing_max_deflection': 0.5,
                        'tail_max_deflection': 0.5,
                        # Waypoint navigation (for attitude control) - use local coordinates if set, otherwise GPS
                        'destination_latitude': destination_lat if not use_local_coords else -1.0,
                        'destination_longitude': destination_lon if not use_local_coords else -1.0,
                        'destination_x': destination_x,
                        'destination_y': destination_y,
                        'datum_latitude': datum_lat,
                        'datum_longitude': datum_lon,
                        'waypoint_tolerance': 2.0,
                        'use_sim_time': True,
                    }]
                )
            ]
        ),
    ]
