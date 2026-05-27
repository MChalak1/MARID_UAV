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
import math
import random

def generate_launch_description():
    return LaunchDescription([
        # Declare launch arguments for destination coordinates
        DeclareLaunchArgument(
            'destination_latitude',
            default_value='48.8566',  # Paris, France randomization center
            description='Destination latitude in degrees, or random destination center when random_location is true'
        ),
        DeclareLaunchArgument(
            'destination_longitude',
            default_value='2.3522',  # Paris, France randomization center
            description='Destination longitude in degrees, or random destination center when random_location is true'
        ),
        DeclareLaunchArgument(
            'random_location',
            default_value='true',
            description='Generate a random GPS destination near destination_latitude/longitude. Set false to use the exact destination.'
        ),
        DeclareLaunchArgument(
            'random_location_min_radius_m',
            default_value='1000.0',
            description='Minimum random destination distance from destination_latitude/longitude in meters'
        ),
        DeclareLaunchArgument(
            'random_location_max_radius_m',
            default_value='5000.0',
            description='Maximum random destination distance from destination_latitude/longitude in meters'
        ),
        DeclareLaunchArgument(
            'random_location_seed',
            default_value='',
            description='Optional RNG seed for reproducible random destinations'
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
        DeclareLaunchArgument(
            'use_center_thruster',
            default_value='true',
            description='Use single center thruster (true) or dual left/right (false)'
        ),
        DeclareLaunchArgument(
            'initial_thrust',
            default_value='10.0',
            description='Initial thrust in Newtons'
        ),

        DeclareLaunchArgument(
            'target_altitude',
            default_value='2380.0',
            description='Target altitude for guidance in meters'
        ),
        DeclareLaunchArgument(
            'random_altitude',
            default_value='true',
            description='Generate a random target altitude. Set false to use target_altitude exactly.'
        ),
        DeclareLaunchArgument(
            'random_altitude_min_m',
            default_value='300.0',
            description='Minimum random target altitude in meters'
        ),
        DeclareLaunchArgument(
            'random_altitude_max_m',
            default_value='3000.0',
            description='Maximum random target altitude in meters'
        ),

        DeclareLaunchArgument(
            'enable_logging',
            default_value='true',
            description='Enable attitude controller logging'
        ),
        # Use OpaqueFunction to access launch arguments and convert to floats
        OpaqueFunction(function=launch_setup)
    ])

def _as_bool(value):
    return str(value).lower() in ('true', '1', 'yes', 'on')

def _random_lat_lon_near(lat_deg, lon_deg, min_radius_m, max_radius_m):
    min_radius_m = max(0.0, min_radius_m)
    max_radius_m = max(min_radius_m, max_radius_m)
    distance_m = random.uniform(min_radius_m, max_radius_m)
    bearing_rad = random.uniform(0.0, 2.0 * math.pi)

    earth_radius_m = 6371000.0
    lat1 = math.radians(lat_deg)
    lon1 = math.radians(lon_deg)
    angular_distance = distance_m / earth_radius_m

    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular_distance)
        + math.cos(lat1) * math.sin(angular_distance) * math.cos(bearing_rad)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing_rad) * math.sin(angular_distance) * math.cos(lat1),
        math.cos(angular_distance) - math.sin(lat1) * math.sin(lat2),
    )
    lon2 = (lon2 + math.pi) % (2.0 * math.pi) - math.pi
    return math.degrees(lat2), math.degrees(lon2), distance_m, math.degrees(bearing_rad)

def launch_setup(context):
    # Get launch argument values and convert to float
    destination_lat = float(context.launch_configurations['destination_latitude'])
    destination_lon = float(context.launch_configurations['destination_longitude'])
    destination_x = float(context.launch_configurations.get('destination_x', '-1.0'))
    destination_y = float(context.launch_configurations.get('destination_y', '-1.0'))
    random_location = _as_bool(context.launch_configurations.get('random_location', 'true'))
    random_min_radius_m = float(context.launch_configurations.get('random_location_min_radius_m', '1000.0'))
    random_max_radius_m = float(context.launch_configurations.get('random_location_max_radius_m', '5000.0'))
    random_seed = context.launch_configurations.get('random_location_seed', '')
    use_center_thruster = context.launch_configurations.get('use_center_thruster', 'true').lower() == 'true'
    initial_thrust = float(context.launch_configurations.get('initial_thrust', '10.0'))
    datum_lat = 37.4  # Match Gazebo world origin (wt.sdf)
    datum_lon = -122.1
    target_altitude = float(context.launch_configurations.get('target_altitude', '800.0'))
    random_altitude = _as_bool(context.launch_configurations.get('random_altitude', 'true'))
    random_altitude_min_m = float(context.launch_configurations.get('random_altitude_min_m', '300.0'))
    random_altitude_max_m = float(context.launch_configurations.get('random_altitude_max_m', '3000.0'))
    enable_logging = context.launch_configurations.get('enable_logging', 'true').lower() == 'true'
    
    # Determine if local coordinates are set (both must be != -1.0)
    use_local_coords = (destination_x != -1.0 and destination_y != -1.0)

    if random_location and not use_local_coords:
        center_lat = destination_lat
        center_lon = destination_lon
        if random_seed:
            random.seed(random_seed)
        destination_lat, destination_lon, random_distance_m, random_bearing_deg = _random_lat_lon_near(
            center_lat,
            center_lon,
            random_min_radius_m,
            random_max_radius_m,
        )
        print(
            f"[option_a_controller] random_location=true: destination="
            f"({destination_lat:.6f}, {destination_lon:.6f}), "
            f"center=({center_lat:.6f}, {center_lon:.6f}), "
            f"distance={random_distance_m:.1f} m, bearing={random_bearing_deg:.1f} deg"
        )

    if random_altitude:
        random_altitude_min_m = max(0.0, random_altitude_min_m)
        random_altitude_max_m = max(random_altitude_min_m, random_altitude_max_m)
        target_altitude = random.uniform(random_altitude_min_m, random_altitude_max_m)
        print(
            f"[option_a_controller] random_altitude=true: target_altitude="
            f"{target_altitude:.1f} m, range=({random_altitude_min_m:.1f}, "
            f"{random_altitude_max_m:.1f}) m"
        )
    
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
                'initial_thrust': initial_thrust,
                'min_thrust': 0.0,
                'thrust_to_weight_ratio': 0.45,  # Reduced from 2.5: gives max thrust ~1.3x weight (reasonable for aircraft)
                'thrust_increment': 10.0,
                'world_name': 'empty',
                'model_name': 'marid',
                'link_name': 'base_link_front',
                'update_rate': 50.0,  # Increased to match thruster plugin update rate
                'enable_keyboard': False,  # Disabled in Option A (guidance controls it)
                'enable_differential': False,  # Disable differential thrust - yaw controlled aerodynamically
                'thrust_to_angvel_gain': 50.0,  # Conversion factor: omega = gain * sqrt(thrust)
                'use_thruster_plugin': True,  # Use Gazebo Thruster plugin (True) or legacy wrench (False)
                'use_center_thruster': use_center_thruster,  # Use single center thruster (True) or dual left/right (False)
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
                        'target_altitude': target_altitude,  # m
                        'target_velocity': 75.0,  # m/s 
                        'altitude_min': target_altitude*0.9,  # m
                        'altitude_max': target_altitude*1.1,  # m
                        'waypoint_tolerance': 2.0,  # m
                        # Guidance limits
                        'max_heading_rate': 0.5,  # rad/s
                        'min_speed': 10.0,  # m/s
                        'max_speed': 100.0,  # m/s
                        # Altitude gate: hold target altitude before navigating
                        'altitude_tolerance': 15.0,        # m — band around target altitude
                        'altitude_stable_duration': 2.0,  # s — hold time before phase transition
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
                        'base_thrust_override': -1.0, # Fixed thrust (overrides PID calculations)
                        'thrust_to_weight_ratio': 0.65,  # Not used when override is set
                        'min_thrust': 0.0,
                        # Physics-based thrust (disabled for fixed thrust test)
                        'use_physics_thrust': True,
                        'use_airspeed_sensor': True,
                        'drag_coefficient': 0.25,  
                        'air_density': 1.225,  # kg/m³ at sea level
                        # Wind vector (from world file: [0, 0, 0] m/s - disabled for testing)
                        'wind_x': 0.0,
                        'wind_y': 0.0,
                        'wind_z': 0.0,
                        # PID gains (disabled for fixed thrust test, but can still be used for attitude control)
                        'speed_kp': 0.5,  # Feedforward correction (base_thrust_override is -1.0, so PID is active)
                        'speed_ki': 0.02,
                        'speed_kd': 0.1,
                        # Altitude control (re-enabled for altitude maintenance)
                        'altitude_kp': 2.0,  # Re-enable for altitude maintenance
                        'altitude_ki': 0.1,
                        'altitude_kd': 0.5,
                        'target_altitude': target_altitude,
                        'use_sim_time': True,
                    }]
                )
            ]
        ),

        # ESKF Ground-Truth Logger - records ESKF vs Gazebo pairs for ML training.
        # Lock-protected: if joystick_teleop_wings.launch is also running, whichever
        # started first holds the lock and this instance silently disables itself.
        Node(
           package="marid_logging",
           executable="eskf_gt_logger",
           name="eskf_gt_logger",
           output="screen",
           parameters=[{
               'log_directory': '~/marid_ws/data_extended',
               'log_rate': 50.0,
               'samples_per_file': 10000,
               'enable_logging': enable_logging,
           }],
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
                        'wing_max_deflection': 0.005,
                        'tail_max_deflection': 0.045,
                        # Altitude control (tail/wings pitch - works with thrust)
                        'target_altitude': target_altitude,
                        'altitude_pitch_gain': 0.03,
                        'climb_wing_incidence': 0.0,  # rad — fixed wing AoA during climb
                        'airborne_altitude_threshold': 50.0,  # m — pitch command always active (disabled for testing, set to 0.5 m for real runs)
                        'airborne_speed_threshold': 13.0,   # m/s — must also exceed this to be considered airborne
                        'pitch_slew_rate': 50.0,  # rad/s — ~1°/s max nose-up rate after liftoff
                        'roll_slew_rate': 40.0,    # rad/s — ~ limit roll rate for better stability during climb
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
