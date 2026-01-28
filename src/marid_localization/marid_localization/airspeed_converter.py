#!/usr/bin/env python3
"""
Airspeed Converter Node
Reads gz.msgs.AirSpeed from Gazebo and publishes std_msgs/Float64
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import subprocess
import re
import math

class AirspeedConverter(Node):
    def __init__(self):
        super().__init__('airspeed_converter')
        
        self.declare_parameter('gz_topic', '/airspeed')
        self.declare_parameter('output_topic', '/airspeed')
        self.declare_parameter('publish_rate', 50.0)
        
        self.gz_topic_ = self.get_parameter('gz_topic').value
        self.output_topic_ = self.get_parameter('output_topic').value
        self.publish_rate_ = self.get_parameter('publish_rate').value
        
        self.airspeed_pub_ = self.create_publisher(Float64, self.output_topic_, 10)
        
        timer_period = 1.0 / self.publish_rate_
        self.timer_ = self.create_timer(timer_period, self.query_and_publish_airspeed)
        
        self.get_logger().info(f'Airspeed converter initialized')
        self.get_logger().info(f'Reading from Gazebo topic: {self.gz_topic_}')
        self.get_logger().info(f'Publishing to ROS2 topic: {self.output_topic_}')
    
    def query_airspeed_from_gz(self):
        """Query airspeed using gz topic command"""
        try:
            cmd = ['gz', 'topic', '-e', '-t', self.gz_topic_, '-n', '1']
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1.0  # Increased from 0.2 to 1.0 seconds
            )
            
            if result.returncode == 0:
                if result.stdout:
                    parsed_velocity = self.parse_airspeed_output(result.stdout)
                    if parsed_velocity is not None:
                        return parsed_velocity
                    else:
                        # Log parsing failure (first time only)
                        if not hasattr(self, '_parse_failure_logged'):
                            self.get_logger().warn(f'Failed to parse airspeed from gz topic output. Return code: {result.returncode}')
                            if result.stderr:
                                self.get_logger().warn(f'stderr: {result.stderr[:200]}')
                            self._parse_failure_logged = True
                else:
                    # No output (first time only)
                    if not hasattr(self, '_no_output_logged'):
                        self.get_logger().warn(f'gz topic command returned no output for topic {self.gz_topic_}')
                        self._no_output_logged = True
            else:
                # Command failed (log first few times)
                if not hasattr(self, '_cmd_error_count'):
                    self._cmd_error_count = 0
                self._cmd_error_count += 1
                if self._cmd_error_count <= 3:
                    self.get_logger().warn(f'gz topic command failed with return code {result.returncode}')
                    if result.stderr:
                        self.get_logger().warn(f'stderr: {result.stderr[:200]}')
                        
        except subprocess.TimeoutExpired:
            if not hasattr(self, '_timeout_logged'):
                self.get_logger().warn(f'gz topic command timed out for topic {self.gz_topic_}')
                self._timeout_logged = True
        except Exception as e:
            if not hasattr(self, '_error_count'):
                self._error_count = 0
            self._error_count += 1
            if self._error_count <= 3:
                self.get_logger().error(f'Error querying airspeed: {e}')
        
        return None
    
    def parse_airspeed_output(self, output):
        """Parse diff_pressure and temperature from gz.msgs.AirSpeed protobuf output
        and calculate airspeed using pitot tube formula: v = sqrt(2 * ΔP / ρ)
        """
        try:
            # Extract diff_pressure and temperature from protobuf output
            # Format: diff_pressure: VALUE
            #         temperature: VALUE
            diff_pressure_match = re.search(r'diff_pressure:\s*([-\d.eE]+)', output, re.IGNORECASE)
            temperature_match = re.search(r'temperature:\s*([-\d.eE]+)', output, re.IGNORECASE)
            
            if diff_pressure_match and temperature_match:
                diff_pressure = float(diff_pressure_match.group(1))  # Pa (Pascal)
                temperature = float(temperature_match.group(1))  # K (Kelvin)
                
                # Validate values
                if math.isnan(diff_pressure) or math.isinf(diff_pressure):
                    return None
                if math.isnan(temperature) or math.isinf(temperature) or temperature < 0:
                    return None
                
                # Calculate air density from temperature using ideal gas law approximation
                # ρ = P / (R * T), where P is static pressure, R is gas constant
                # Using sea-level pressure (101325 Pa) and R = 287.05 J/(kg·K)
                R_air = 287.05  # J/(kg·K) - specific gas constant for dry air
                P_static = 101325.0  # Pa - sea level static pressure
                rho = P_static / (R_air * temperature)  # kg/m³
                
                # Calculate airspeed from differential pressure using pitot tube formula
                # v = sqrt(2 * ΔP / ρ) for incompressible flow
                if rho > 0:
                    abs_diff_pressure = abs(diff_pressure)
                    velocity = math.sqrt(2.0 * abs_diff_pressure / rho)  # m/s
                    
                    # Validate velocity is reasonable (0-1000 m/s)
                    if 0 <= velocity <= 1000:
                        return velocity
                    else:
                        if not hasattr(self, '_invalid_velocity_logged'):
                            self.get_logger().warn(f'Calculated velocity out of range: {velocity:.2f} m/s (diff_pressure={diff_pressure:.2f} Pa, temp={temperature:.2f} K)')
                            self._invalid_velocity_logged = True
                else:
                    if not hasattr(self, '_invalid_density_logged'):
                        self.get_logger().warn(f'Invalid air density: {rho:.4f} kg/m³')
                        self._invalid_density_logged = True
            else:
                # Log for debugging (first time only)
                if not hasattr(self, '_parse_debug_logged'):
                    self.get_logger().warn(f'Could not parse diff_pressure or temperature from output. First 500 chars: {output[:500]}')
                    self._parse_debug_logged = True
                
        except Exception as e:
            if not hasattr(self, '_parse_error_count'):
                self._parse_error_count = 0
            self._parse_error_count += 1
            if self._parse_error_count == 1:
                self.get_logger().error(f'Error parsing airspeed output: {e}')
        
        return None
    
    def query_and_publish_airspeed(self):
        """Query airspeed and publish"""
        airspeed = self.query_airspeed_from_gz()
        
        if airspeed is not None:
            msg = Float64()
            msg.data = airspeed
            self.airspeed_pub_.publish(msg)
            
            if not hasattr(self, '_success_logged'):
                self.get_logger().info(f'✓ Successfully reading airspeed from Gazebo: {airspeed:.2f} m/s')
                self._success_logged = True

def main():
    rclpy.init()
    node = AirspeedConverter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
