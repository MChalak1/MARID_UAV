#!/usr/bin/env python3
import rclpy
import time
import math
from rclpy.node import Node
from sensor_msgs.msg import Imu

imu_pub = None

def imuCallback(imu):
    global imu_pub
    
    # Validate IMU data is not NaN or infinite
    if (math.isnan(imu.orientation.x) or math.isnan(imu.orientation.y) or 
        math.isnan(imu.orientation.z) or math.isnan(imu.orientation.w) or
        math.isnan(imu.angular_velocity.x) or math.isnan(imu.angular_velocity.y) or
        math.isnan(imu.angular_velocity.z) or
        math.isnan(imu.linear_acceleration.x) or math.isnan(imu.linear_acceleration.y) or
        math.isnan(imu.linear_acceleration.z) or
        math.isinf(imu.orientation.x) or math.isinf(imu.orientation.y) or 
        math.isinf(imu.orientation.z) or math.isinf(imu.orientation.w) or
        math.isinf(imu.angular_velocity.x) or math.isinf(imu.angular_velocity.y) or
        math.isinf(imu.angular_velocity.z) or
        math.isinf(imu.linear_acceleration.x) or math.isinf(imu.linear_acceleration.y) or
        math.isinf(imu.linear_acceleration.z)):
        # Not to publish invalid IMU data - log occasionally
        if not hasattr(imuCallback, '_nan_warn_count'):
            imuCallback._nan_warn_count = 0
        imuCallback._nan_warn_count += 1
        if imuCallback._nan_warn_count % 100 == 0:  # Log periodically
            if hasattr(imuCallback, '_node') and imuCallback._node is not None:
                imuCallback._node.get_logger().warning('Received NaN or infinite values in IMU data, skipping publication')
        return

    imu.header.frame_id = "imu_link_ekf"
    imu_pub.publish(imu)


def main(args=None):
    global imu_pub
    rclpy.init(args=args)
    node = Node('imu_republisher_node')
    time.sleep(1)
    imu_pub = node.create_publisher(Imu, "/imu_ekf", 10)
    imu_sub = node.create_subscription(Imu, "/imu", imuCallback, 10)
    
    # Store node reference for logging in callback
    imuCallback._node = node
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()