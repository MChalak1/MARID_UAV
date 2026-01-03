#!/usr/bin/env python3
"""
Gazebo Pose to Odometry Converter
Uses ros_gz_bridge to get model pose from Gazebo world state
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import subprocess
import os
import re
import math
import time

class GazeboPoseToOdom(Node):
    def __init__(self):
        super().__init__('gazebo_pose_to_odom')
        
        # Parameters
        self.declare_parameter('model_name', 'marid')
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('base_frame_id', 'base_link_front')
        self.declare_parameter('publish_rate', 50.0)
        self.declare_parameter('use_gz_model_command', True)  # Use gz model command as fallback
        
        self.model_name_ = self.get_parameter('model_name').value
        self.odom_frame_id_ = self.get_parameter('odom_frame_id').value
        self.base_frame_id_ = self.get_parameter('base_frame_id').value
        self.publish_rate_ = self.get_parameter('publish_rate').value
        self.use_gz_model_ = self.get_parameter('use_gz_model_command').value
        
        # State
        self.last_position_ = None
        self.last_time_ = None
        self.pose_received_ = False
        
        # Publisher
        self.odom_pub_ = self.create_publisher(
            Odometry,
            '/gazebo/odom',
            10
        )
        
        # Timer to query pose
        timer_period = 1.0 / self.publish_rate_
        self.timer_ = self.create_timer(timer_period, self.query_and_publish_pose)
        
        self.get_logger().info(f'Gazebo Pose to Odometry converter initialized for model: {self.model_name_}')
        self.get_logger().info('Using gz model command to query pose...')
    
    def query_pose_using_gz_model(self):
        """Query model pose using gz model command"""
        max_retries = 2
        timeout = 5.0  # Increased from 3.0 to 5.0 seconds
        
        for attempt in range(max_retries):
            try:
                # Use gz model -p command to get pose
                cmd_str = f'gz model -m {self.model_name_} -p'
                result = subprocess.run(
                    cmd_str,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    executable='/bin/bash'
                )
                
                if result.returncode == 0 and result.stdout:
                    # Parse output format:
                    parsed = self.parse_gz_model_output(result.stdout)
                    if parsed is not None:
                        return parsed
                    elif result.stdout:
                        # Log the actual output for debugging (only on last attempt)
                        if attempt == max_retries - 1:
                            self.get_logger().warn(f'Failed to parse output. First 200 chars: {result.stdout[:200]}')
                else:
                    if result.returncode != 0:
                        if attempt == max_retries - 1:  # Last attempt
                            self.get_logger().debug(f'gz model command returned code {result.returncode}, stderr: {result.stderr[:100]}')
                        # Check if model doesn't exist
                        if 'No model named' in result.stderr or 'No model named' in result.stdout:
                            if attempt == max_retries - 1:
                                self.get_logger().warn(f'Model "{self.model_name_}" not found in Gazebo. Is it spawned?')
                            return None
                    # Retry on failure
                    if attempt < max_retries - 1:
                        time.sleep(0.5)
                        continue
                        
            except subprocess.TimeoutExpired:
                if attempt == max_retries - 1:  # Last attempt
                    self.get_logger().warn(f'gz model command timed out after {max_retries} attempts (timeout: {timeout}s)')
                    self.get_logger().warn('Is Gazebo running? Is the model spawned?')
                else:
                    timeout += 2.0  # Increase timeout for next attempt
                    time.sleep(0.5)
                    continue
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    self.get_logger().error(f'gz model command failed: {e}')
                else:
                    time.sleep(0.5)
                    continue
        
        return None
    
    def parse_gz_model_output(self, output):
        """Parse pose from gz model -p output"""
        try:
            from geometry_msgs.msg import PoseWithCovarianceStamped
            from tf_transformations import quaternion_from_euler
            import re
            
            # Parse format:
            # Pose [ XYZ (m) ] [ RPY (rad) ]:
            # [0.000000 2.000000 3.000000]  <- XYZ line
            # [0.000000 -0.000000 0.000000]  <- RPY line
            
            # Pattern to match [number number number]
            pattern = r'\[\s*([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)\s*\]'
            
            # Find the "Pose" section to avoid matching "Model: [10]"
            pose_section_start = output.find('Pose')
            if pose_section_start == -1:
                self.get_logger().warn('Could not find "Pose" section in output')
                return None
            
            # Get everything after "Pose"
            after_pose = output[pose_section_start:]
            
            # Find all matches of [number number number] after "Pose"
            matches = re.findall(pattern, after_pose)
            
            if len(matches) >= 2:
                # First match should be XYZ, second should be RPY
                xyz_values = matches[0]
                rpy_values = matches[1]
                
                pose_msg = PoseWithCovarianceStamped()
                pose_msg.header.frame_id = self.odom_frame_id_
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                
                # Extract position
                x = float(xyz_values[0])
                y = float(xyz_values[1])
                z = float(xyz_values[2])
                roll = float(rpy_values[0])
                pitch = float(rpy_values[1])
                yaw = float(rpy_values[2])
                
                # Validate values are not NaN or infinite
                if (math.isnan(x) or math.isnan(y) or math.isnan(z) or
                    math.isnan(roll) or math.isnan(pitch) or math.isnan(yaw) or
                    math.isinf(x) or math.isinf(y) or math.isinf(z) or
                    math.isinf(roll) or math.isinf(pitch) or math.isinf(yaw)):
                    self.get_logger().warn(f'Parsed NaN or infinite values from Gazebo pose: x={x}, y={y}, z={z}, roll={roll}, pitch={pitch}, yaw={yaw}')
                    return None
                
                pose_msg.pose.pose.position.x = x
                pose_msg.pose.pose.position.y = y
                pose_msg.pose.pose.position.z = z
                
                # Convert Euler to quaternion
                qx, qy, qz, qw = quaternion_from_euler(roll, pitch, yaw)
                
                # Validate quaternion is not NaN
                if (math.isnan(qx) or math.isnan(qy) or math.isnan(qz) or math.isnan(qw) or
                    math.isinf(qx) or math.isinf(qy) or math.isinf(qz) or math.isinf(qw)):
                    self.get_logger().warn(f'Quaternion conversion produced NaN or infinite values: qx={qx}, qy={qy}, qz={qz}, qw={qw}')
                    return None
                
                pose_msg.pose.pose.orientation.x = qx
                pose_msg.pose.pose.orientation.y = qy
                pose_msg.pose.pose.orientation.z = qz
                pose_msg.pose.pose.orientation.w = qw
                
                # Set covariance (low uncertainty for ground truth)
                pose_msg.pose.covariance = [0.01] * 36
                
                return pose_msg
            else:
                self.get_logger().warn(f'Found {len(matches)} matches, expected at least 2')
                
        except Exception as e:
            self.get_logger().error(f'Error parsing gz model output: {e}')
            self.get_logger().error(f'Output was: {output[:500]}')  # First 500 chars
        
        return None
    
    def query_and_publish_pose(self):
        """Query pose and publish as odometry"""
        if not self.use_gz_model_:
            return
        
        pose_msg = self.query_pose_using_gz_model()
        
        if pose_msg is None:
            # Log periodically if we're not getting pose (every 5 seconds at 50Hz = 250 calls)
            if not hasattr(self, '_warn_count'):
                self._warn_count = 0
            self._warn_count += 1
            if self._warn_count % 250 == 0:  # Every 5 seconds
                self.get_logger().warn('Still unable to query pose from Gazebo. Is Gazebo running?')
            return
        
        # Validate pose values are not NaN before publishing
        pos = pose_msg.pose.pose.position
        ori = pose_msg.pose.pose.orientation
        
        if (math.isnan(pos.x) or math.isnan(pos.y) or math.isnan(pos.z) or
            math.isnan(ori.x) or math.isnan(ori.y) or math.isnan(ori.z) or math.isnan(ori.w) or
            math.isinf(pos.x) or math.isinf(pos.y) or math.isinf(pos.z) or
            math.isinf(ori.x) or math.isinf(ori.y) or math.isinf(ori.z) or math.isinf(ori.w)):
            if not hasattr(self, '_nan_warn_count'):
                self._nan_warn_count = 0
            self._nan_warn_count += 1
            if self._nan_warn_count % 250 == 0:  # Every 5 seconds at 50Hz
                self.get_logger().warn('Received NaN or infinite values from Gazebo pose query, skipping publication')
            return
        
        if not self.pose_received_:
            self.get_logger().info('✓ Successfully querying pose from Gazebo using gz model!')
            self.get_logger().info(f'  Position: x={pos.x:.3f}, y={pos.y:.3f}, z={pos.z:.3f}')
            self.pose_received_ = True
        
        try:
            # Convert to odometry
            odom_msg = Odometry()
            odom_msg.header = pose_msg.header
            odom_msg.header.frame_id = self.odom_frame_id_
            odom_msg.child_frame_id = self.base_frame_id_
            
            # Copy pose
            odom_msg.pose.pose = pose_msg.pose.pose
            odom_msg.pose.covariance = pose_msg.pose.covariance
            
            # Calculate velocity
            now = self.get_clock().now()
            if self.last_position_ is not None and self.last_time_ is not None:
                dt = (now.nanoseconds - self.last_time_) / 1e9
                if dt > 0 and dt < 1.0:
                    dx = pose_msg.pose.pose.position.x - self.last_position_[0]
                    dy = pose_msg.pose.pose.position.y - self.last_position_[1]
                    dz = pose_msg.pose.pose.position.z - self.last_position_[2]
                    
                    odom_msg.twist.twist.linear.x = dx / dt
                    odom_msg.twist.twist.linear.y = dy / dt
                    odom_msg.twist.twist.linear.z = dz / dt
                else:
                    odom_msg.twist.twist.linear.x = 0.0
                    odom_msg.twist.twist.linear.y = 0.0
                    odom_msg.twist.twist.linear.z = 0.0
            else:
                odom_msg.twist.twist.linear.x = 0.0
                odom_msg.twist.twist.linear.y = 0.0
                odom_msg.twist.twist.linear.z = 0.0
            
            odom_msg.twist.twist.angular.x = 0.0
            odom_msg.twist.twist.angular.y = 0.0
            odom_msg.twist.twist.angular.z = 0.0
            odom_msg.twist.covariance = [0.1] * 36
            
            odom_msg.header.stamp = now.to_msg()
            self.odom_pub_.publish(odom_msg)
            
            # Update state
            self.last_position_ = [
                pose_msg.pose.pose.position.x,
                pose_msg.pose.pose.position.y,
                pose_msg.pose.pose.position.z
            ]
            self.last_time_ = now.nanoseconds
            
        except Exception as e:
            self.get_logger().error(f'Error publishing odometry: {e}')

def main():
    rclpy.init()
    node = GazeboPoseToOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
