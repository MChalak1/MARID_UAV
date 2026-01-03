#!/usr/bin/env python3
"""
MARID Safety and Self-Destruct Node
Monitors critical safety conditions and triggers self-destruct in emergency scenarios.
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64, String
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseWithCovarianceStamped
import numpy as np
import math
import time


class MaridSafetyNode(Node):
    def __init__(self):
        super().__init__('marid_safety_node')
        
        # Parameters
        self.declare_parameter('update_rate', 10.0)  # Safety check frequency (Hz)
        self.declare_parameter('enable_self_destruct', True)
        
        # Altitude safety parameters
        self.declare_parameter('critical_altitude_threshold', 1.0)  # Critical altitude (m)
        self.declare_parameter('altitude_drop_rate_threshold', -5.0)  # m/s (negative = dropping)
        self.declare_parameter('altitude_drop_time_window', 2.0)  # Time window for drop detection (s)
        
        # Engine failure detection
        self.declare_parameter('min_thrust_threshold', 0.5)  # Minimum expected thrust (N)
        self.declare_parameter('thrust_failure_time', 3.0)  # Time without thrust before failure (s)
        
        # Enemy area of operation (AO) parameters
        self.declare_parameter('enemy_ao_enabled', False)
        self.declare_parameter('enemy_ao_center_x', 0.0)
        self.declare_parameter('enemy_ao_center_y', 0.0)
        self.declare_parameter('enemy_ao_radius', 100.0)  # Radius in meters
        
        # Self-destruct parameters
        self.declare_parameter('self_destruct_delay', 1.0)  # Delay before self-destruct (s)
        self.declare_parameter('self_destruct_thrust', 0.0)  # Set thrust to 0 on self-destruct
        
        # Get parameters
        self.update_rate_ = self.get_parameter('update_rate').value
        self.enable_self_destruct_ = self.get_parameter('enable_self_destruct').value
        self.critical_altitude_ = self.get_parameter('critical_altitude_threshold').value
        self.altitude_drop_rate_ = self.get_parameter('altitude_drop_rate_threshold').value
        self.altitude_drop_window_ = self.get_parameter('altitude_drop_time_window').value
        self.min_thrust_ = self.get_parameter('min_thrust_threshold').value
        self.thrust_failure_time_ = self.get_parameter('thrust_failure_time').value
        self.enemy_ao_enabled_ = self.get_parameter('enemy_ao_enabled').value
        self.enemy_ao_center_ = np.array([
            self.get_parameter('enemy_ao_center_x').value,
            self.get_parameter('enemy_ao_center_y').value
        ])
        self.enemy_ao_radius_ = self.get_parameter('enemy_ao_radius').value
        self.self_destruct_delay_ = self.get_parameter('self_destruct_delay').value
        self.self_destruct_thrust_ = self.get_parameter('self_destruct_thrust').value
        
        # State variables
        self.current_odom_ = None
        self.current_altitude_ = None
        self.current_thrust_ = 0.0
        self.altitude_history_ = []  # [(time, altitude), ...]
        self.thrust_history_ = []  # [(time, thrust), ...]
        self.self_destruct_triggered_ = False
        self.self_destruct_time_ = None
        
        # Subscribers
        self.odom_sub_ = self.create_subscription(
            Odometry,
            '/odometry/filtered/local',
            self.odom_callback,
            10
        )
        
        self.altitude_sub_ = self.create_subscription(
            PoseWithCovarianceStamped,
            '/barometer/altitude',
            self.altitude_callback,
            10
        )
        
        self.thrust_sub_ = self.create_subscription(
            Float64,
            '/marid/thrust/total',
            self.thrust_callback,
            10
        )
        
        # Publishers
        self.self_destruct_pub_ = self.create_publisher(
            Bool,
            '/marid/safety/self_destruct',
            10
        )
        
        self.safety_status_pub_ = self.create_publisher(
            String,
            '/marid/safety/status',
            10
        )
        
        self.emergency_thrust_pub_ = self.create_publisher(
            Float64,
            '/marid/thrust/total',
            10
        )
        
        # Safety check timer
        timer_period = 1.0 / self.update_rate_
        self.safety_timer_ = self.create_timer(timer_period, self.safety_check)
        
        self.get_logger().info('MARID Safety Node initialized')
        self.get_logger().info(f'Self-destruct enabled: {self.enable_self_destruct_}')
        self.get_logger().info(f'Critical altitude threshold: {self.critical_altitude_} m')
        self.get_logger().info(f'Enemy AO monitoring: {self.enemy_ao_enabled_}')
        if self.enemy_ao_enabled_:
            self.get_logger().info(f'Enemy AO center: ({self.enemy_ao_center_[0]:.2f}, {self.enemy_ao_center_[1]:.2f}), radius: {self.enemy_ao_radius_:.2f} m')
    
    def odom_callback(self, msg):
        """Store current odometry"""
        self.current_odom_ = msg
    
    def altitude_callback(self, msg):
        """Store current altitude"""
        self.current_altitude_ = msg.pose.pose.position.z
    
    def thrust_callback(self, msg):
        """Store current thrust command"""
        self.current_thrust_ = msg.data
    
    def is_in_enemy_ao(self):
        """Check if drone is in enemy area of operation"""
        if not self.enemy_ao_enabled_ or self.current_odom_ is None:
            return False
        
        current_pos = np.array([
            self.current_odom_.pose.pose.position.x,
            self.current_odom_.pose.pose.position.y
        ])
        
        distance = np.linalg.norm(current_pos - self.enemy_ao_center_)
        return distance <= self.enemy_ao_radius_
    
    def check_altitude_drop(self):
        """Check for sudden altitude drop"""
        if self.current_altitude_ is None:
            return False
        
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # Add current altitude to history
        self.altitude_history_.append((current_time, self.current_altitude_))
        
        # Remove old entries outside time window
        self.altitude_history_ = [
            (t, alt) for t, alt in self.altitude_history_
            if current_time - t <= self.altitude_drop_window_
        ]
        
        if len(self.altitude_history_) < 2:
            return False
        
        # Check if altitude is below critical threshold
        if self.current_altitude_ < self.critical_altitude_:
            # Check if in enemy AO
            if self.is_in_enemy_ao():
                self.get_logger().error(f'CRITICAL: Altitude below threshold ({self.current_altitude_:.2f} m) in enemy AO!')
                return True
        
        # Check altitude drop rate
        oldest_time, oldest_alt = self.altitude_history_[0]
        newest_time, newest_alt = self.altitude_history_[-1]
        
        if newest_time - oldest_time > 0:
            drop_rate = (newest_alt - oldest_alt) / (newest_time - oldest_time)
            
            if drop_rate < self.altitude_drop_rate_ and self.current_altitude_ < self.critical_altitude_:
                if self.is_in_enemy_ao():
                    self.get_logger().error(f'CRITICAL: Rapid altitude drop ({drop_rate:.2f} m/s) in enemy AO!')
                    return True
        
        return False
    
    def check_engine_failure(self):
        """Check for engine failure (no thrust for extended period)"""
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # Add current thrust to history
        self.thrust_history_.append((current_time, self.current_thrust_))
        
        # Remove old entries
        self.thrust_history_ = [
            (t, thrust) for t, thrust in self.thrust_history_
            if current_time - t <= self.thrust_failure_time_ + 1.0
        ]
        
        # Check if thrust has been below threshold for failure time
        if len(self.thrust_history_) < 2:
            return False
        
        # Find continuous period of low thrust
        low_thrust_start = None
        for t, thrust in self.thrust_history_:
            if thrust < self.min_thrust_:
                if low_thrust_start is None:
                    low_thrust_start = t
            else:
                low_thrust_start = None
        
        if low_thrust_start is not None:
            duration = current_time - low_thrust_start
            if duration >= self.thrust_failure_time_:
                self.get_logger().error(f'CRITICAL: Engine failure detected (no thrust for {duration:.2f} s)!')
                return True
        
        return False
    
    def trigger_self_destruct(self, reason):
        """Trigger self-destruct sequence"""
        if self.self_destruct_triggered_:
            return  # Already triggered
        
        self.self_destruct_triggered_ = True
        self.self_destruct_time_ = self.get_clock().now().nanoseconds / 1e9
        
        self.get_logger().error('=' * 60)
        self.get_logger().error(f'SELF-DESTRUCT TRIGGERED: {reason}')
        self.get_logger().error(f'Delay: {self.self_destruct_delay_} seconds')
        self.get_logger().error('=' * 60)
        
        # Publish self-destruct signal
        destruct_msg = Bool()
        destruct_msg.data = True
        self.self_destruct_pub_.publish(destruct_msg)
        
        # Publish safety status
        status_msg = String()
        status_msg.data = f'SELF_DESTRUCT: {reason}'
        self.safety_status_pub_.publish(status_msg)
    
    def safety_check(self):
        """Main safety check loop"""
        if not self.enable_self_destruct_:
            return
        
        if self.self_destruct_triggered_:
            # Execute self-destruct sequence
            current_time = self.get_clock().now().nanoseconds / 1e9
            elapsed = current_time - self.self_destruct_time_
            
            if elapsed >= self.self_destruct_delay_:
                # Set thrust to zero (or self-destruct value)
                thrust_msg = Float64()
                thrust_msg.data = self.self_destruct_thrust_
                self.emergency_thrust_pub_.publish(thrust_msg)
                
                self.get_logger().error('SELF-DESTRUCT EXECUTED: Thrust set to zero')
                
                # Publish status
                status_msg = String()
                status_msg.data = 'SELF_DESTRUCT_EXECUTED'
                self.safety_status_pub_.publish(status_msg)
            
            return
        
        # Check safety conditions
        if self.check_altitude_drop():
            self.trigger_self_destruct('Sudden altitude drop in enemy AO')
            return
        
        if self.check_engine_failure():
            self.trigger_self_destruct('Engine failure detected')
            return
        
        # Publish normal status
        status_msg = String()
        status_msg.data = 'NORMAL'
        self.safety_status_pub_.publish(status_msg)


def main():
    rclpy.init()
    node = MaridSafetyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

