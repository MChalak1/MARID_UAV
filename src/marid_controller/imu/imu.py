#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray
from tf_transformations import euler_from_quaternion

# --- helpers ---
def clamp(x, lo, hi): 
    return max(lo, min(hi, x))

def clamp_abs(x, limit):
    return clamp(x, -limit, limit)

def angdiff(a, b):
    """Shortest signed angle from a to b (b - a) in [-pi, pi]."""
    return (b - a + math.pi) % (2*math.pi) - math.pi

class MaridImuController(Node):
    def __init__(self):
        super().__init__('marid_imu_control_node')
        self.get_logger().info('MARID IMU controller (4-surface mixer) started.')

        # ------------------ INTERFACES ------------------
        self.cmd_topic = '/simple_position_controller/commands'
        imu_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=50
        )
        # Use /imu/filtered if you run the KF; otherwise /imu/out
        self.subscription = self.create_subscription(Imu, '/imu/out', self.imu_callback, imu_qos)
        self.pub = self.create_publisher(Float64MultiArray, self.cmd_topic, 10)

        # ------------------ TARGETS (rad) ------------------
        self.target_roll  = 0.0
        self.target_pitch = 0.0
        # REP-103: +X forward. To face +Y, set +pi/2
        self.target_yaw   = math.pi/2

        # ------------------ PID GAINS ------------------
        self.kp_roll,  self.ki_roll,  self.kd_roll  = 2.0, 0.0, 0.30
        self.kp_pitch, self.ki_pitch, self.kd_pitch = 1.2, 0.0, 0.25
        self.kp_yaw,   self.ki_yaw,   self.kd_yaw   = 1.0, 0.0, 0.15
        self.int_limit = 0.5
        self.yaw_int_limit = 0.5

        # ------------------ MIXER WEIGHTS ------------------
        # Main wings: roll & pitch; optional small yaw assist
        self.w_roll_main   = 1.0    # roll authority on main wings
        self.w_pitch_main  = 0.20   # pitch share on main wings
        self.w_yaw_main    = 0.00   # usually 0; set 0.05–0.15 if needed

        # Tail pair: pitch symmetric, yaw differential
        self.w_pitch_tail  = 0.30   # pitch share on tail (increase for more elevator)
        self.w_yaw_tail    = 0.30   # yaw authority on tail differential

        # ------------------ LIMITS / RATE ------------------
        self.joint_min, self.joint_max = -0.5, 0.5   # rad
        self.max_slew = 1.0  # rad/s per joint

        # ------------------ STATE ------------------
        self.prev_roll_err = 0.0
        self.prev_pitch_err = 0.0
        self.prev_yaw_err = 0.0
        self.roll_int = 0.0
        self.pitch_int = 0.0
        self.yaw_int = 0.0
        self.last_imu_time_ros = None

        # last_u order MUST match simple_position_controller.joints
        self.last_u = [0.0, 0.0, 0.0, 0.0]  # [L, R, TL, TR]
        self._curr_cmd = [0.0, 0.0, 0.0, 0.0]

        # ------------------ TIMERS ------------------
        self.control_rate_hz = 100.0
        self.dt_ctrl = 1.0 / self.control_rate_hz
        self.create_timer(self.dt_ctrl, self.control_loop)
        self.create_timer(1.0, self.diag)

    def now_ros(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def imu_callback(self, msg: Imu):
        # Orientation
        q = msg.orientation
        roll, pitch, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        # dt from wall-clock between IMU callbacks (ok in sim)
        now = self.now_ros()
        dt = 0.0 if self.last_imu_time_ros is None else max(1e-4, now - self.last_imu_time_ros)
        self.last_imu_time_ros = now

        # ----------------- ROLL PID -----------------
        er = self.target_roll - roll
        dr = (er - self.prev_roll_err) / dt if dt > 0 else 0.0
        self.roll_int = clamp_abs(self.roll_int + er * dt, self.int_limit)
        u_roll = self.kp_roll*er + self.ki_roll*self.roll_int + self.kd_roll*dr
        self.prev_roll_err = er

        # ----------------- PITCH PID ----------------
        ep = self.target_pitch - pitch
        dp = (ep - self.prev_pitch_err) / dt if dt > 0 else 0.0
        self.pitch_int = clamp_abs(self.pitch_int + ep * dt, self.int_limit)
        u_pitch = self.kp_pitch*ep + self.ki_pitch*self.pitch_int + self.kd_pitch*dp
        self.prev_pitch_err = ep

        # ----------------- YAW PID ------------------
        ey = angdiff(yaw, self.target_yaw)
        dy = (ey - self.prev_yaw_err) / dt if dt > 0 else 0.0
        self.yaw_int = clamp_abs(self.yaw_int + ey * dt, self.yaw_int_limit)
        u_yaw = self.kp_yaw*ey + self.ki_yaw*self.yaw_int + self.kd_yaw*dy
        self.prev_yaw_err = ey

        # ----------------- MIXER (4 surfaces) -----------------
        # Main wings: roll differential, pitch symmetric, (optional yaw assist)
        left_main  = (+self.w_roll_main)*u_roll + (+self.w_pitch_main)*u_pitch + (+self.w_yaw_main)*u_yaw
        right_main = (-self.w_roll_main)*u_roll + (+self.w_pitch_main)*u_pitch + (-self.w_yaw_main)*u_yaw

        # Tails: pitch symmetric, yaw differential
        tail_left  = (+self.w_pitch_tail)*u_pitch + (+self.w_yaw_tail)*u_yaw
        tail_right = (+self.w_pitch_tail)*u_pitch + (-self.w_yaw_tail)*u_yaw

        # Clamp
        left_main  = clamp(left_main,  self.joint_min, self.joint_max)
        right_main = clamp(right_main, self.joint_min, self.joint_max)
        tail_left  = clamp(tail_left,  self.joint_min, self.joint_max)
        tail_right = clamp(tail_right, self.joint_min, self.joint_max)

        # Store in controller's expected order: [L, R, TL, TR]
        self.last_u = [left_main, right_main, tail_left, tail_right]

    def control_loop(self):
        # Fail-safe: zero if IMU stale > 0.5s
        if self.last_imu_time_ros is None or (self.now_ros() - self.last_imu_time_ros) > 0.5:
            target = [0.0, 0.0, 0.0, 0.0]
        else:
            target = self.last_u

        # Slew-rate limit per joint
        next_cmd = []
        for c, t in zip(self._curr_cmd, target):
            delta = clamp(t - c, -self.max_slew*self.dt_ctrl, self.max_slew*self.dt_ctrl)
            next_cmd.append(c + delta)
        self._curr_cmd = next_cmd

        msg = Float64MultiArray()
        msg.data = [float(x) for x in self._curr_cmd]  # [L, R, TL, TR]
        self.pub.publish(msg)

    def diag(self):
        age = float('inf') if self.last_imu_time_ros is None else (self.now_ros() - self.last_imu_time_ros)
        L, R, TL, TR = self.last_u
        self.get_logger().info(
            f"cmd {self.cmd_topic} = [L:{L:+.3f}, R:{R:+.3f}, TL:{TL:+.3f}, TR:{TR:+.3f}] | IMU age: {age:.2f}s"
        )

def main(args=None):
    rclpy.init(args=args)
    node = MaridImuController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
