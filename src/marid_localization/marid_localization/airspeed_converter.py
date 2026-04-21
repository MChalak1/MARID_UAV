#!/usr/bin/env python3
"""
Airspeed Converter Node

Subscribes directly to the Gazebo /airspeed topic via gz.transport (bypassing
the ROS bridge, which lacks AirSpeed support in Jazzy) and converts differential
pressure + temperature to airspeed (m/s) using the pitot formula:
    v = sqrt(2 * |ΔP| / ρ)    where ρ = P_static / (R_air * T)

Publishes std_msgs/Float64 to /airspeed/velocity.
"""
import math
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

from gz.transport13 import Node as GzNode
from gz.msgs10.air_speed_pb2 import AirSpeed


R_AIR = 287.05      # J/(kg·K) — specific gas constant for dry air
P_STATIC = 101325.0  # Pa — sea-level static pressure


class AirspeedConverter(Node):
    def __init__(self):
        super().__init__('airspeed_converter')

        self.declare_parameter('gz_topic', '/airspeed')
        self.declare_parameter('output_topic', '/airspeed/velocity')
        # Low-pass filter coefficient (0=no filtering, 1=freeze).
        # Pitot pressure noise is amplified by the sqrt formula at low speeds,
        # creating a ~0.4 m/s noise floor when stationary. alpha=0.1 cuts this
        # to well below min_airspeed_for_fusion without lagging at cruise speed.
        self.declare_parameter('filter_alpha', 0.1)

        gz_topic = self.get_parameter('gz_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.alpha_ = float(self.get_parameter('filter_alpha').value)

        self.filtered_velocity_ = 0.0

        self.pub_ = self.create_publisher(Float64, output_topic, 10)

        self.gz_node_ = GzNode()
        ok = self.gz_node_.subscribe(AirSpeed, gz_topic, self._gz_callback)
        if not ok:
            self.get_logger().error(f'Failed to subscribe to gz topic: {gz_topic}')
        else:
            self.get_logger().info(f'Airspeed converter: gz:{gz_topic} -> {output_topic}')

    def _gz_callback(self, msg: AirSpeed):
        diff_pressure = float(msg.diff_pressure)
        temperature = float(msg.temperature)

        if not (math.isfinite(diff_pressure) and math.isfinite(temperature)):
            return
        if temperature <= 0.0:
            return

        rho = P_STATIC / (R_AIR * temperature)
        if rho <= 0.0:
            return

        velocity = math.sqrt(2.0 * abs(diff_pressure) / rho)
        if not math.isfinite(velocity):
            return

        self.filtered_velocity_ = (
            self.alpha_ * velocity + (1.0 - self.alpha_) * self.filtered_velocity_
        )

        out = Float64()
        out.data = self.filtered_velocity_
        self.pub_.publish(out)


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
