#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import FluidPressure
from geometry_msgs.msg import PoseWithCovarianceStamped
import math

class BarometerConverter(Node):
    def __init__(self):
        super().__init__('barometer_altitude_converter')
        
        # Parameters
        self.declare_parameter('sea_level_pressure', 101325.0)  # Pa
        self.declare_parameter('temperature', 288.15)  # K (15°C)
        self.declare_parameter('altitude_variance', 0.25)  # m^2
        
        self.p0_ = self.get_parameter('sea_level_pressure').value
        self.T_ = self.get_parameter('temperature').value
        self.alt_var_ = self.get_parameter('altitude_variance').value
        
        # Constants
        self.R = 287.05  # Gas constant for dry air (J/kg·K)
        self.g = 9.80665  # Gravity (m/s^2)
        self.L = 0.0065   # Temperature lapse rate (K/m)
        
        # Subscribers and publishers
        self.pressure_sub_ = self.create_subscription(
            FluidPressure,
            '/baro/pressure',  # Change to your barometer topic
            self.pressure_callback,
            10
        )
        
        self.altitude_pub_ = self.create_publisher(
            PoseWithCovarianceStamped,
            '/barometer/altitude',
            10
        )
        
        self.get_logger().info('Barometer altitude converter initialized')
    
    def pressure_to_altitude(self, pressure):
        """
        Convert atmospheric pressure to altitude using barometric formula.
        Using the standard atmosphere model.
        """
        # Validate input pressure
        if math.isnan(pressure) or math.isinf(pressure) or pressure <= 0:
            return float('nan')
        
        # Barometric formula for altitude
        try:
        altitude = (self.T_ / self.L) * (1.0 - math.pow(pressure / self.p0_, 
                                         (self.R * self.L) / self.g))
            # Validate output
            if math.isnan(altitude) or math.isinf(altitude):
                return float('nan')
        return altitude
        except (ValueError, OverflowError) as e:
            self.get_logger().warn(f'Error calculating altitude from pressure {pressure}: {e}')
            return float('nan')
    
    

    def pressure_callback(self, msg: FluidPressure):
        """Convert pressure measurement to altitude pose"""
        pressure = msg.fluid_pressure
        
        # Validate pressure is valid
        if math.isnan(pressure) or math.isinf(pressure) or pressure <= 0 or pressure > 200000:
            if not hasattr(self, '_invalid_pressure_count'):
                self._invalid_pressure_count = 0
            self._invalid_pressure_count += 1
            if self._invalid_pressure_count % 50 == 0:  # Log periodically
                self.get_logger().warn(f'Invalid pressure reading: {pressure} Pa, skipping')
            return
        
        altitude = self.pressure_to_altitude(pressure)
        
        # Validate altitude is valid
        if math.isnan(altitude) or math.isinf(altitude):
            if not hasattr(self, '_invalid_altitude_count'):
                self._invalid_altitude_count = 0
            self._invalid_altitude_count += 1
            if self._invalid_altitude_count % 50 == 0:  # Log periodically
                self.get_logger().warn(f'Calculated invalid altitude: {altitude} m from pressure {pressure} Pa, skipping')
            return
        
        # Create pose message with only Z position
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header = msg.header
        pose_msg.header.frame_id = 'odom'
        
        # Set altitude as Z position
        pose_msg.pose.pose.position.z = altitude
        
        # Set orientation to identity (not used)
        pose_msg.pose.pose.orientation.w = 1.0
        
        # Set covariance - only Z position has meaningful variance
        pose_msg.pose.covariance = [0.0] * 36
        pose_msg.pose.covariance[14] = self.alt_var_
        
        self.altitude_pub_.publish(pose_msg)

def main():
    rclpy.init()
    node = BarometerConverter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()