#!/usr/bin/env python3
"""
IMU Logger Node (MARID)
Subscribes to sensor_msgs/Imu and writes timestamp, linear_acceleration, angular_velocity, orientation to CSV.
"""
import csv
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu


class ImuLogger(Node):
    def __init__(self):
        super().__init__('imu_logger')

        self.declare_parameter('imu_topic', '/imu')
        self.declare_parameter('output_file', 'imu_log.csv')
        self.declare_parameter('output_dir', '')

        self.imu_topic_ = self.get_parameter('imu_topic').value
        self.output_file_ = self.get_parameter('output_file').value
        self.output_dir_ = self.get_parameter('output_dir').value

        path = self.output_file_
        if self.output_dir_:
            path = os.path.join(self.output_dir_, self.output_file_)
        self.file_path_ = path
        self.file_ = None
        self.writer_ = None
        self.header_written_ = False

        self.sub_ = self.create_subscription(
            Imu,
            self.imu_topic_,
            self.imu_callback,
            10
        )

        self.get_logger().info(f'IMU logger subscribed to {self.imu_topic_}, writing to {self.file_path_}')

    def open_file(self):
        if self.file_ is not None:
            return
        dirname = os.path.dirname(self.file_path_)
        if dirname and not os.path.isdir(dirname):
            os.makedirs(dirname, exist_ok=True)
        self.file_ = open(self.file_path_, 'w', newline='')
        self.writer_ = csv.writer(self.file_)

    def write_header(self):
        if self.header_written_:
            return
        self.writer_.writerow([
            'timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz',
            'qx', 'qy', 'qz', 'qw'
        ])
        self.header_written_ = True

    def imu_callback(self, msg):
        self.open_file()
        self.write_header()
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        acc = msg.linear_acceleration
        gyro = msg.angular_velocity
        ori = msg.orientation
        self.writer_.writerow([
            t, acc.x, acc.y, acc.z, gyro.x, gyro.y, gyro.z,
            ori.x, ori.y, ori.z, ori.w
        ])
        self.file_.flush()

    def close_file(self):
        if self.file_ is not None:
            self.file_.close()
            self.file_ = None
            self.get_logger().info(f'Closed {self.file_path_}')


def main():
    rclpy.init()
    node = ImuLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close_file()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
