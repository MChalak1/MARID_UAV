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
            default_value='37.4',  # Gazebo world datum / randomization center
            description='Destination latitude in degrees, or random destination center when random_location is true'
        ),
        DeclareLaunchArgument(
            'destination_longitude',
            default_value='-122.1',  # Gazebo world datum / randomization center
            description='Destination longitude in degrees, or random destination center when random_location is true'
        ),
        DeclareLaunchArgument(
            'random_location',
            default_value='true',
            description='Generate a random GPS destination near destination_latitude/longitude. Set false to use the exact destination.'
        ),
        DeclareLaunchArgument(
            'random_location_min_radius_m',
            default_value='500000.0',
            description='Minimum random destination distance from destination_latitude/longitude in meters'
        ),
        DeclareLaunchArgument(
            'random_location_max_radius_m',
            default_value='1000000.0',
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
    random_min_radius_m = float(context.launch_configurations.get('random_location_min_radius_m', '500000.0'))
    random_max_radius_m = float(context.launch_configurations.get('random_location_max_radius_m', '1000000.0'))
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
               'log_directory': '~/marid_ws/data_sync',
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
                        'update_rate': 50.0,   # Hz — control loop rate

                        # ── Legacy PID gains (unused when ESKF + rate-loop active) ──────────
                        # These feed the PIDController objects but the main path uses
                        # angle→rate→deflection chains below. Ki=0 everywhere: no integrator
                        # windup risk, steady-state error handled by the outer altitude loop.
                        'roll_kp': 0.12,
                        'roll_ki': 0.0,
                        'roll_kd': 0.55,
                        'pitch_kp': 0.8,
                        'pitch_ki': 0.0,
                        'pitch_kd': 1.0,
                        'yaw_kp': 0.5,
                        'yaw_ki': 0.0,
                        'yaw_kd': 0.8,

                        # ── Control surface deflection limits ────────────────────────────────
                        # Hard clamp on every surface command. Tail drives pitch+yaw mixing;
                        # wings drive roll only in cruise.
                        'wing_max_deflection': math.radians(5.0),   # max wing deflection (rad)
                        'tail_max_deflection': math.radians(10.0),  # max tail deflection (rad)

                        # ── Altitude hold — vz cascade ───────────────────────────────────────
                        # Two-stage: altitude_error → desired_vz → desired_pitch.
                        # Stage 1: altitude_error (m) × altitude_pitch_gain → desired_vz (m/s),
                        #          clipped to ±altitude_vz_max.
                        # Stage 2: vz_error (m/s) × altitude_vz_pitch_gain → pitch angle (rad),
                        #          clipped to ±altitude_pitch_max.
                        # Settles naturally: at target altitude vz→0, then cruise_pitch_trim
                        # provides the small steady cruise attitude bias.
                        'target_altitude': target_altitude,
                        'altitude_pitch_gain': 0.1,      # altitude error (m) → desired vz (m/s)
                        'altitude_vz_max': 5.0,           # max target vertical speed (m/s)
                        'altitude_vz_pitch_gain': 0.025, # vz error (m/s) → pitch angle (rad)
                        'altitude_pitch_max': 0.20,       # max pitch from altitude loop (rad ≈ ±11.5°)

                        # ── Phase gates ──────────────────────────────────────────────────────
                        # Drone is considered airborne (pitch/roll active) only when BOTH
                        # thresholds are exceeded. Guards against spurious commands during
                        # ground roll and slow taxi.
                        'airborne_altitude_threshold': 5.0,   # m AGL — below this: ground mode
                        'airborne_speed_threshold': 13.0,     # m/s — below this: ground mode
                        'climb_wing_incidence': 0.0,          # fixed wing AoA offset during climb (rad)

                        # ── Pitch target slew ────────────────────────────────────────────────
                        # Limits how fast the desired pitch TARGET moves.
                        # Prevents the −30° nose-up command from stepping instantaneously
                        # at liftoff. Also smooths the climb→cruise transition step.
                        # In cruise the slew is linear at cruise_pitch_slew_rate.
                        # pitch_slew_slow_radius: error at which full slew rate is reached;
                        #   below this the cubic shaping reduces the rate to prevent overshoot
                        #   near the −30° climb target.
                        # pitch_slew_curve_exponent: shape of rate reduction (3 = cubic).
                        # pitch_slew_snap_threshold: error below which target is snapped
                        #   directly (avoids infinite asymptotic approach).
                        'pitch_slew_rate': math.radians(120.0),        # max slew rate (rad/s)
                        'pitch_slew_slow_radius': math.radians(25.0),  # full-rate onset (rad)
                        'pitch_slew_curve_exponent': 3.0,              # shaping exponent
                        'pitch_slew_snap_threshold': math.radians(0.05),  # snap-to threshold (rad)

                        # ── Cruise transition / surface command slew ─────────────────────────
                        # cruise_pitch_slew_rate limits how fast the pitch target moves after
                        # entering cruise. roll_slew_rate/cruise_roll_slew_rate limit roll
                        # surface command changes so phase transitions do not step directly
                        # into large wing inputs.
                        'roll_slew_rate': math.radians(8.0),  # max roll command slew (rad/s)
                        'cruise_pitch_slew_rate': math.radians(6.0),   # cruise pitch target slew (rad/s)
                        'cruise_roll_slew_rate': math.radians(4.0),    # cruise roll surface slew (rad/s)
                        'cruise_navigation_delay': 10.0,       # seconds to settle in cruise before waypoint turns
                        'cruise_pitch_trim': math.radians(-2.0),  # neutral cruise pitch target
                        'climb_pitch_up_boost': 1.15,          # extra tail authority for nose-up climb corrections
                        'cruise_pitch_up_boost': 1.30,         # extra tail authority for nose-up cruise corrections

                        # ── Roll rate loop ───────────────────────────────────────────────────
                        # roll_angle_to_rate_gain: roll error (rad) → desired roll rate (rad/s).
                        # max/min_roll_rate_command: clamp on the desired roll rate.
                        # roll_angle_deadband: ignore roll errors smaller than this (rad).
                        # roll_rate_kp: roll rate error → surface deflection gain.
                        # roll_gain_reference_speed: airspeed (m/s) at which gains are nominal.
                        #   Above this speed gains reduce as (ref/v)² — higher q needs less deflection.
                        # min_roll_speed_scale: floor on the speed-scaling factor (prevents
                        #   gains going to zero at very high speed).
                        'roll_angle_to_rate_gain': 1.5,
                        'max_roll_rate_command': math.radians(6.0),
                        'min_roll_rate_command': math.radians(0.7),
                        'roll_angle_deadband': math.radians(0.15),
                        'roll_rate_kp': 0.035,
                        'roll_gain_reference_speed': 20.0,
                        'min_roll_speed_scale': 0.35,

                        # ── Fallback attitude-rate filter ────────────────────────────────────
                        # When odometry angular rates are unavailable, roll/pitch/yaw rates
                        # are estimated from finite-differenced Euler angles. This EMA filter
                        # smooths those estimates.
                        # alpha: EMA coefficient (lower = smoother, more lag).
                        # max_fallback_attitude_rate: sanity clamp on the estimated rate.
                        'fallback_rate_filter_alpha': 0.08,
                        'max_fallback_attitude_rate': math.radians(25.0),

                        # ── Pitch rate loop ──────────────────────────────────────────────────
                        # pitch_angle_to_rate_gain: pitch error (rad) → desired pitch rate (rad/s).
                        #   At 10° error: desired rate = 2.4 × 0.175 = 0.42 rad/s = 24°/s.
                        # max_pitch_rate_command: clamp on desired pitch rate. Saturates at
                        #   pitch_error = max_rate / angle_to_rate_gain = 45/2.4 ≈ 18.75°.
                        # pitch_rate_kp: pitch rate error → surface deflection gain.
                        # pitch_gain_reference_speed / min_pitch_speed_scale: same q-scaling
                        #   as roll — gains reduce with (ref/v)², floored at min_scale.
                        'pitch_angle_to_rate_gain': 2.4,
                        'max_pitch_rate_command': math.radians(30.0),
                        'pitch_rate_kp': 0.12,
                        'pitch_gain_reference_speed': 20.0,
                        'min_pitch_speed_scale': 0.60,

                        # ── Pitch deflection shaping ─────────────────────────────────────────
                        # Scales the final pitch surface command based on pitch error magnitude.
                        # In both climb and cruise it keys off pitch error, so pitch control
                        # gets gentler near the target instead of kicking hard around 0°.
                        # pitch_deflection_slow_radius: error at which scale reaches 1.0.
                        #   Below this, scale reduces via cubic curve to min_scale.
                        # pitch_deflection_curve_exponent: shaping (3 = cubic drop-off).
                        # min_pitch_deflection_scale: floor authority at zero pitch error.
                        #   Prevents full deflection when pitch error is tiny (overshoot guard).
                        'pitch_deflection_slow_radius': math.radians(45.0),
                        'pitch_deflection_curve_exponent': 4.0,
                        'min_pitch_deflection_scale': 0.3625,

                        # ── Yaw rate loop ────────────────────────────────────────────────────
                        # Same structure as roll/pitch: yaw_error → rate → deflection.
                        # Yaw is mixed into both tail surfaces (differential).
                        # yaw_rate_deadband zeros tiny yaw-rate errors so climb/cruise logs do
                        # not show constant tail twitches from measurement noise.
                        'yaw_angle_to_rate_gain': 1.0,
                        'max_yaw_rate_command': math.radians(8.0),
                        'yaw_rate_kp': 0.05,
                        'yaw_rate_deadband': math.radians(0.5),

                        # ── Waypoint navigation ──────────────────────────────────────────────
                        # Gentler cruise bank demand prevents the climb→cruise handoff from
                        # suddenly asking for a large turn correction.
                        'heading_to_bank_gain': 0.35,          # heading error (rad) → desired bank (rad)
                        'max_bank_angle': math.radians(15.0),  # max commanded cruise bank
                        'destination_latitude': destination_lat if not use_local_coords else -1.0,
                        'destination_longitude': destination_lon if not use_local_coords else -1.0,
                        'destination_x': destination_x,
                        'destination_y': destination_y,
                        'datum_latitude': datum_lat,
                        'datum_longitude': datum_lon,
                        'waypoint_tolerance': 2.0,   # m — distance to declare waypoint reached
                        'use_sim_time': True,
                    }]
                )
            ]
        ),
    ]
