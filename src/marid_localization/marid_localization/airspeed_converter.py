#!/usr/bin/env python3
"""
Airspeed Converter Node

Subscribes to the bridged ros_gz_interfaces/msg/AirSpeed topic and converts
differential pressure + temperature to airspeed (m/s) using the pitot formula:
    v = sqrt(2 * |ΔP| / ρ)    where ρ = P_static / (R_air * T)

Publishes std_msgs/Float64 to /airspeed/velocity.
"""
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from ros_gz_interfaces.msg import AirSpeed


R_AIR = 287.05     # J/(kg·K) — specific gas constant for dry air
P_STATIC = 101325.0  # Pa — sea-level static pressure


class AirspeedConverter(Node):
    def __init__(self):
        super().__init__('airspeed_converter')

        self.declare_parameter('input_topic', '/airspeed')
        self.declare_parameter('output_topic', '/airspeed/velocity')

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value

        self.pub_ = self.create_publisher(Float64, output_topic, 10)
        self.sub_ = self.create_subscription(
            AirSpeed, input_topic, self.airspeed_callback, 10
        )

        self.get_logger().info(f'Airspeed converter: {input_topic} -> {output_topic}')

    def airspeed_callback(self, msg: AirSpeed):
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

        out = Float64()
        out.data = velocity
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
