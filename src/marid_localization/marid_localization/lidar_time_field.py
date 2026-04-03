#!/usr/bin/env python3
"""
LiDAR Time Field Injector
Gazebo's gpu_lidar does not publish a per-point 'time' field in PointCloud2.
FAST-LIO (Velodyne type) expects this field and PCL warns on every scan when absent.
This node republishes the cloud with a zero-valued 'time' field appended so FAST-LIO
can map it without warnings. Motion compensation is not affected — the sim scan is
instantaneous so all per-point times are correctly zero.
"""
import struct
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField


class LidarTimeField(Node):
    def __init__(self):
        super().__init__('lidar_time_field')
        self.sub_ = self.create_subscription(
            PointCloud2, 'input', self.callback, 10)
        self.pub_ = self.create_publisher(PointCloud2, 'output', 10)

    def callback(self, msg: PointCloud2):
        # Check if 'time' field already present
        if any(f.name == 'time' for f in msg.fields):
            self.pub_.publish(msg)
            return

        # Compute current point step and stride to next field offset
        old_point_step = msg.point_step
        time_offset = old_point_step  # append after existing fields
        new_point_step = old_point_step + 4  # float32 = 4 bytes

        # Rebuild data buffer with 4 zero bytes appended to each point
        old_data = msg.data
        n_points = msg.width * msg.height
        new_data = bytearray(n_points * new_point_step)
        zero = struct.pack('f', 0.0)

        for i in range(n_points):
            src = i * old_point_step
            dst = i * new_point_step
            new_data[dst:dst + old_point_step] = old_data[src:src + old_point_step]
            new_data[dst + time_offset:dst + time_offset + 4] = zero

        out = PointCloud2()
        out.header = msg.header
        out.height = msg.height
        out.width = msg.width
        out.fields = list(msg.fields) + [
            PointField(name='time', offset=time_offset,
                       datatype=PointField.FLOAT32, count=1)
        ]
        out.is_bigendian = msg.is_bigendian
        out.point_step = new_point_step
        out.row_step = new_point_step * msg.width
        out.data = bytes(new_data)
        out.is_dense = msg.is_dense
        self.pub_.publish(out)


def main():
    rclpy.init()
    node = LidarTimeField()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
