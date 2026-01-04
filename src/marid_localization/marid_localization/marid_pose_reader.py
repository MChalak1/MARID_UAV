#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion

class MaridPoseReader(Node):
    def __init__(self):
        super().__init__("marid_pose_reader")
        
        # Subscribe to your global EKF output
        self.odom_sub = self.create_subscription(
            Odometry, 
            "/odometry/global",  
            self.odom_callback, 
            10
        )
        
        self.current_pose = {
            'x': 0.0, 'y': 0.0, 'z': 0.0,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
        }
    
    def odom_callback(self, msg):
        # Extract position
        self.current_pose['x'] = msg.pose.pose.position.x
        self.current_pose['y'] = msg.pose.pose.position.y
        self.current_pose['z'] = msg.pose.pose.position.z
        
        # Extract orientation
        q = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ]
        roll, pitch, yaw = euler_from_quaternion(q)
        
        self.current_pose['roll'] = roll
        self.current_pose['pitch'] = pitch
        self.current_pose['yaw'] = yaw
        
        # Log it
        self.get_logger().info(
            f"Position: x={self.current_pose['x']:.2f}, "
            f"y={self.current_pose['y']:.2f}, "
            f"z={self.current_pose['z']:.2f} | "
            f"Orientation: roll={roll:.2f}, pitch={pitch:.2f}, yaw={yaw:.2f}"
        )
    
    def get_pose(self):
        """Call this method to get the current pose"""
        return self.current_pose

def main():
    rclpy.init()
    node = MaridPoseReader()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()