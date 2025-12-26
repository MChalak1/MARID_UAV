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
        # Barometric formula for altitude
        altitude = (self.T_ / self.L) * (1.0 - math.pow(pressure / self.p0_, 
                                         (self.R * self.L) / self.g))
        return altitude
    
    

    def pressure_callback(self, msg: FluidPressure):
        """Convert pressure measurement to altitude pose"""
        pressure = msg.fluid_pressure
        
        # Debug logging
        # self.get_logger().info(f'Received pressure: {pressure} Pa')
        
        altitude = self.pressure_to_altitude(pressure)
        
        # self.get_logger().info(f'Calculated altitude: {altitude:.2f} m')
        
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