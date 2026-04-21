#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, TimerAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('marid_localization')  

    config_file = os.path.join(pkg_dir, 'config', 'ekf.yaml')
    navsat_config_file = os.path.join(pkg_dir, 'config', 'navsat_transform.yaml')
    imu_filter_config_file = os.path.join(pkg_dir, 'config', 'imu_filter_madgwick.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time'
        ),
        DeclareLaunchArgument(
            'use_fast_lio',
            default_value='true',
            description='Whether to launch FAST-LIO LiDAR-inertial odometry (publishes /Odometry for EKF)'
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="imu_link_ekf_broadcaster",
            arguments=["--x", "0", "--y", "0","--z", "0.2",
                    "--qx", "0", "--qy", "0", "--qz", "0", "--qw", "1",
                    "--frame-id", "base_link_front",
                    "--child-frame-id", "imu_link_ekf"],
        ),

        # Align FAST-LIO's camera_init world frame with odom at startup.
        # camera_init is FAST-LIO's inertial reference frame (set at first LiDAR scan).
        # It must be a child of odom (not base_link_front) so robot_localization can
        # transform /Odometry into the odom frame for EKF fusion.
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="camera_init_broadcaster",
            arguments=["--x", "0", "--y", "0", "--z", "0.2",
                       "--qx", "0", "--qy", "0", "--qz", "0", "--qw", "1",
                       "--frame-id", "odom",
                       "--child-frame-id", "camera_init"],
        ),
        # FAST-LIO LiDAR-inertial odometry (optional; publishes /Odometry for local EKF)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_dir, 'launch', 'fast_lio.launch.py')
            ),
            condition=IfCondition(LaunchConfiguration('use_fast_lio')),
            launch_arguments=[
                ('use_sim_time', LaunchConfiguration('use_sim_time')),
            ],
        ),
        # Convert GPS NavSatFix to Odometry in map frame

        Node(
            package='robot_localization',
            executable='navsat_transform_node',
            name='navsat_transform',
            output='screen',
            parameters=[navsat_config_file],  # Load from config file
            remappings=[
                ('/imu', '/imu_ekf'),
                ('/odometry/filtered', '/odometry/filtered/local'),
                ('/odometry/gps', '/gps/odometry')
                ]
        ),


        
        # Local EKF - fuses IMU + Barometer + /gazebo/odom (+ FAST-LIO, marid/odom)
        # Short delay so /gazebo/odom and other sensors are publishing (bridge uses use_sim_time)
        TimerAction(
            period=5.0,
            actions=[
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node_odom',
            output='screen',
            parameters=[config_file,  {'use_sim_time': True}],
            remappings=[
                ('/odometry/filtered', '/odometry/filtered/local')
                    ]
                )
            ]
        ),
        
        # Global EKF - fuses local odometry + GPS
        # Delay startup after local EKF (local starts at 5s) so map EKF gets /odometry/filtered/local
        TimerAction(
            period=20.0,
            actions=[
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node_map',
            output='screen',
            parameters=[config_file,  {'use_sim_time': True}],
            remappings=[
                ('/odometry/filtered', '/odometry/filtered/global')
                    ]
                )
            ]
        ),

        
        # Barometer altitude converter (you'll need to create this)
        Node(
            package='marid_localization',
            executable='baro_to_alt.py',
            name='barometer_altitude_converter',
            output='screen',
            parameters=[{
                'sea_level_pressure': 101325.0,  # Pa
                'temperature': 288.15,  # K (15°C)
                'use_sim_time': True,
            }]
        ),

        # Add gravity to IMU so Madgwick sees ~9.81 m/s² when level (Gazebo publishes specific force)
        Node(
            package='marid_localization',
            executable='imu_add_gravity.py',
            name='imu_add_gravity',
            output='screen',
            parameters=[{'use_sim_time': True}],
        ),
        # Madgwick filter: /imu/with_gravity -> orientation, publishes to /imu/data
        Node(
            package='imu_filter_madgwick',
            executable='imu_filter_madgwick_node',
            name='imu_filter_madgwick',
            output='screen',
            parameters=[imu_filter_config_file, {'use_sim_time': True}],
            remappings=[('/imu/data_raw', '/imu/with_gravity'), ('/imu/mag', '/magnetometer')],
        ),
        # Republisher: subscribes to Madgwick output (/imu/data), publishes to /imu_ekf (no changes elsewhere)
        Node(
            package='marid_localization',
            executable='imu_republisher.py',
            name='imu_republisher',
            output='screen',
            parameters=[{'use_sim_time': True}],
            remappings=[('/imu', '/imu/data')],
        ),
        # Camera HUD overlay: draws filtered IMU attitude over camera feed
        Node(
            package='marid_localization',
            executable='camera_hud_overlay.py',
            name='camera_hud_overlay',
            output='screen',
            parameters=[{
                'image_topic': '/camera/image_raw',
                'imu_topic': '/imu_ekf',
                'output_topic': '/camera/image_hud',
                'smoothing_alpha': 0.15,
            }],
        ),
        # IMU-based odometry (publishes /marid/odom for EKF when FAST-LIO/Gazebo pose unavailable)
        Node(
            package='marid_controller',
            executable='marid_odom_pub.py',
            name='marid_odom_node',
            output='screen',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'calibration_required': 20,
                'use_fastlio': False,
            }],
        ),
        # Wheel odometry — taxiing position from joint velocities + IMU heading.
        # Only integrates while sonar AGL <= 0.30 m (wheels on ground).
        # Publishes /wheel/odometry for the EKF to fuse during ground operations.
        Node(
            package='marid_localization',
            executable='wheel_odometry.py',
            name='wheel_odometry_node',
            output='screen',
            parameters=[{
                'front_wheel_radius': 0.089,
                'rear_wheel_radius':  0.080,
                'ground_threshold':   0.30,
                'publish_tf':         False,
                'use_sim_time':       True,
            }],
            remappings=[('/joint_states', '/world/empty/model/marid/joint_state')],
        ),

        # Optical flow velocity estimator - downward camera + sonar → body-frame velocity
        Node(
            package='marid_localization',
            executable='optical_flow_estimator.py',
            name='optical_flow_estimator',
            output='screen',
            parameters=[{
                'camera_topic': '/optical_flow/camera',
                'sonar_topic': '/sonar/scan',
                'output_topic': '/optical_flow/velocity',
                'sonar_range_topic': '/sonar/range',
                'image_width': 320,
                'image_height': 240,
                'horizontal_fov': 1.047,     # 60 deg
                'sign_vx': 1.0,             
                'sign_vy': 1.0,
                'min_altitude': 0.3,
                'max_altitude': 5.0,
                'velocity_variance': 0.05,
                'use_sim_time': True,
            }],
        ),

        # Forward camera velocity estimator — LK sparse flow, gyro-compensated.
        # Provides lateral (vy) and vertical (vz) body-frame velocity from the
        # forward-facing camera. Active above min_altitude when downward OF drops out.
        Node(
            package='marid_localization',
            executable='forward_flow_estimator.py',
            name='forward_flow_estimator',
            output='screen',
            parameters=[{
                'camera_topic':        '/camera/image_raw',
                'output_topic':        '/forward_camera/velocity',
                'min_altitude':        1.0,
                'depth_scale':         3.0,
                'velocity_variance':   0.1,
                'use_sim_time':        True,
            }],
        ),

        # Airspeed converter - converts gz.msgs.AirSpeed to std_msgs/Float64
        Node(
            package='marid_localization',
            executable='airspeed_converter.py',
            name='airspeed_converter',
            output='screen',
            parameters=[{
                'gz_topic': '/airspeed',
                'output_topic': '/airspeed/velocity',
            }]
        ),
        
        # Wind estimator - uses EKF odometry + pitot airspeed to estimate 3D wind
        Node(
            package='marid_localization',
            executable='wind_estimator.py',
            name='wind_estimator',
            output='screen',
            parameters=[{
                'odom_topic': '/odometry/filtered/local',
                'airspeed_topic': '/airspeed/velocity',  # Use converted airspeed topic
                'wind_frame_id': 'odom',
            }]
        ),
        
        # Air density calculator - calculates density based on altitude
        Node(
            package='marid_localization',
            executable='air_density_calculator.py',
            name='air_density_calculator',
            output='screen',
            parameters=[{
                'update_rate': 10.0,  # Hz
                'use_barometer': True,  # Use barometer altitude if available
                'sea_level_density': 1.225,  # kg/m³
                'sea_level_pressure': 101325.0,  # Pa
                'sea_level_temperature': 288.15,  # K (15°C)
                'use_sim_time': True,
            }]
        ),

        # VX error monitor — compares ground truth vs estimated forward velocity for PlotJuggler
        Node(
            package='marid_localization',
            executable='vx_error_monitor.py',
            name='vx_error_monitor',
            output='screen',
            parameters=[{
                'gt_topic':  '/gazebo/odom',
                'est_topic': '/marid/odom',
                'min_speed': 0.5,
            }]
        ),
    ])
