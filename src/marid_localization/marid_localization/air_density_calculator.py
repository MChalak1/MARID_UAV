#!/usr/bin/env python3
"""
Air Density Calculator for MARID UAV
Calculates air density based on altitude using standard atmosphere model.
Publishes density for monitoring. Note: Dynamic plugin updates require
Gazebo service calls or plugin modification.
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
import math

class AirDensityCalculator(Node):
    def __init__(self):
        super().__init__('air_density_calculator')
        
        # Parameters
        self.declare_parameter('update_rate', 10.0)  # Hz
        self.declare_parameter('use_barometer', True)  # Use barometer altitude if available
        
        # Standard atmosphere constants (ISA - International Standard Atmosphere)
        self.declare_parameter('sea_level_density', 1.225)  # kg/m³ at sea level
        self.declare_parameter('sea_level_pressure', 101325.0)  # Pa
        self.declare_parameter('sea_level_temperature', 288.15)  # K (15°C)
        self.declare_parameter('gas_constant', 287.05)  # J/kg·K for dry air
        self.declare_parameter('lapse_rate', 0.0065)  # K/m (temperature decrease with altitude)
        self.declare_parameter('gravity', 9.80665)  # m/s²
        
        self.update_rate_ = self.get_parameter('update_rate').value
        self.use_barometer_ = self.get_parameter('use_barometer').value
        
        # Atmosphere constants
        self.rho0_ = self.get_parameter('sea_level_density').value
        self.p0_ = self.get_parameter('sea_level_pressure').value
        self.T0_ = self.get_parameter('sea_level_temperature').value
        self.R_ = self.get_parameter('gas_constant').value
        self.L_ = self.get_parameter('lapse_rate').value
        self.g_ = self.get_parameter('gravity').value
        
        # Current state
        self.current_altitude_ = 0.0
        self.current_density_ = self.rho0_
        self.last_log_time_ = 0.0
        
        # Subscribers - try multiple sources for altitude
        if self.use_barometer_:
            self.altitude_sub_ = self.create_subscription(
                PoseWithCovarianceStamped,
                '/barometer/altitude',
                self.altitude_callback,
                10
            )
        
        self.odom_sub_ = self.create_subscription(
            Odometry,
            '/odometry/filtered/local',
            self.odom_callback,
            10
        )
        
        # Publisher for air density
        self.density_pub_ = self.create_publisher(
            Float64,
            '/marid/air_density',
            10
        )
        
        # Update timer
        timer_period = 1.0 / self.update_rate_
        self.update_timer_ = self.create_timer(timer_period, self.update_density)
        
        self.get_logger().info('Air Density Calculator initialized')
        self.get_logger().info(f'Sea level density: {self.rho0_:.4f} kg/m³')
        self.get_logger().info(f'Update rate: {self.update_rate_} Hz')
    
    def altitude_callback(self, msg):
        """Update altitude from barometer (preferred source)"""
        altitude = msg.pose.pose.position.z
        
        # Validate altitude
        if not (math.isnan(altitude) or math.isinf(altitude)):
            self.current_altitude_ = altitude
    
    def odom_callback(self, msg):
        """Update altitude from odometry (fallback source)"""
        altitude = msg.pose.pose.position.z
        
        # Use odometry if barometer not available or significantly different
        if not (math.isnan(altitude) or math.isinf(altitude)):
            if not self.use_barometer_ or abs(altitude - self.current_altitude_) > 5.0:
                self.current_altitude_ = altitude
    
    def calculate_air_density(self, altitude):
        """
        Calculate air density at given altitude using standard atmosphere model.
        
        Uses the barometric formula for the troposphere (0-11 km):
        - Temperature: T = T₀ - L·h
        - Pressure: p = p₀·(T/T₀)^(g/(R·L))
        - Density: ρ = p/(R·T) (ideal gas law)
        
        Args:
            altitude: Altitude above sea level in meters
            
        Returns:
            Air density in kg/m³
        """
        # Clamp altitude to reasonable range (0-20 km)
        altitude = max(0.0, min(altitude, 20000.0))
        
        # Temperature at altitude (linear decrease in troposphere)
        T = self.T0_ - self.L_ * altitude
        
        # Handle edge cases
        if T <= 0:
            # Above tropopause, use exponential model
            # For altitudes > 11 km, temperature is constant
            T = 216.65  # K (temperature at tropopause)
            h_tropopause = 11000.0  # m
            p_tropopause = self.p0_ * (T / self.T0_) ** (self.g_ / (self.R_ * self.L_))
            
            if altitude > h_tropopause:
                # Exponential decrease above tropopause
                p = p_tropopause * math.exp(-self.g_ * (altitude - h_tropopause) / (self.R_ * T))
            else:
                # Still in troposphere
                p = self.p0_ * (T / self.T0_) ** (self.g_ / (self.R_ * self.L_))
        else:
            # Standard troposphere calculation
            p = self.p0_ * (T / self.T0_) ** (self.g_ / (self.R_ * self.L_))
        
        # Density from ideal gas law: ρ = p / (R · T)
        if T > 0:
            rho = p / (self.R_ * T)
        else:
            rho = 0.0
        
        # Clamp to reasonable values (0 to 2 kg/m³)
        rho = max(0.0, min(rho, 2.0))
        
        return rho
    
    def update_density(self):
        """Calculate and publish current air density"""
        # Calculate density at current altitude
        rho = self.calculate_air_density(self.current_altitude_)
        self.current_density_ = rho
        
        # Publish density
        density_msg = Float64()
        density_msg.data = rho
        self.density_pub_.publish(density_msg)
        
        # Log periodically (every 5 seconds)
        current_time = self.get_clock().now().nanoseconds / 1e9
        if current_time - self.last_log_time_ > 5.0:
            density_ratio = (rho / self.rho0_) * 100.0
            self.get_logger().info(
                f'Altitude: {self.current_altitude_:.1f} m | '
                f'Air density: {rho:.4f} kg/m³ ({density_ratio:.1f}% of sea level) | '
                f'Sea level: {self.rho0_:.4f} kg/m³'
            )
            self.last_log_time_ = current_time


def main():
    rclpy.init()
    node = AirDensityCalculator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


