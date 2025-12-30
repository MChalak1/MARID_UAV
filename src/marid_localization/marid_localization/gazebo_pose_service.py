#!/usr/bin/env python3
"""
Gazebo Pose Service Query
Queries Gazebo for model pose using services and publishes as odometry
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
import subprocess
import json
import math

class GazeboPoseService(Node):
    def __init__(self):
        super().__init__('gazebo_pose_service')
        
        # Parameters
        self.declare_parameter('model_name', 'marid')
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('base_frame_id', 'base_link_front')
        self.declare_parameter('publish_rate', 50.0)
        
        self.model_name_ = self.get_parameter('model_name').value
        self.odom_frame_id_ = self.get_parameter('odom_frame_id').value
        self.base_frame_id_ = self.get_parameter('base_frame_id').value
        self.publish_rate_ = self.get_parameter('publish_rate').value
        
        # Publisher
        self.odom_pub_ = self.create_publisher(
            Odometry,
            '/gazebo/odom',
            10
        )
        
        # State
        self.last_position_ = None
        self.last_time_ = None
        
        # Timer to query and publish pose
        timer_period = 1.0 / self.publish_rate_
        self.timer_ = self.create_timer(timer_period, self.query_and_publish)
        
        self.get_logger().info(f'Gazebo Pose Service node initialized for model: {self.model_name_}')
        self.get_logger().warn('This node queries Gazebo for pose - make sure Gazebo is running')
    
    def query_gazebo_pose(self):
        """Query Gazebo for model pose using gz service"""
        try:
            # Use gz service to get model pose
            # Format: gz service -s /world/wt/model/marid/pose
            cmd = ['gz', 'service', '-s', f'/world/wt/model/{self.model_name_}/pose', '-r', 'gz.msgs.Pose']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=0.1)
            
            if result.returncode == 0:
                # Parse the pose from output (format may vary)
                # This is a simplified parser - may need adjustment
                return self.parse_pose_output(result.stdout)
        except Exception as e:
            # Silently fail - service might not be available
            pass
        
        return None
    
    def parse_pose_output(self, output):
        """Parse pose from gz service output"""
        # This is a placeholder - actual parsing depends on output format
        # For now, return None and we'll use topic subscription instead
        return None
    
    def query_and_publish(self):
        """Query pose and publish odometry"""
        # For now, this is a placeholder
        # The actual implementation should use gz service or topic subscription
        # Since service approach is complex, we'll rely on topic subscription
        # which is handled by gazebo_pose_to_odom.py
        pass

def main():
    rclpy.init()
    node = GazeboPoseService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

