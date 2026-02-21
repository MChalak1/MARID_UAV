#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from nav_msgs.msg import Odometry
import numpy as np
from sensor_msgs.msg import Imu
from tf_transformations import quaternion_matrix
from std_srvs.srv import Empty
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
import math

class MaridOdomPublisher(Node):
    def __init__(self):
        super().__init__("marid_odom_node")
        
        # Parameters
        self.declare_parameter("imu_outputs_specific_force", True)
        self.declare_parameter("gravity", 9.81)
        self.declare_parameter("publish_rate", 50.0)
        self.declare_parameter("max_velocity", 10.0)  # m/s safety limit
        self.declare_parameter("max_acceleration", 50.0)  # m/s^2 safety limit
        self.declare_parameter("velocity_decay_rate", 0.95)  # Damping factor
        self.declare_parameter("base_position_variance", 0.01)
        self.declare_parameter("base_velocity_variance", 0.1)
        self.declare_parameter("variance_growth_rate", 0.001)  # per second
        self.declare_parameter("child_frame_id", "base_link_front")
        
        # Get parameters
        self.accel_is_sf_ = self.get_parameter("imu_outputs_specific_force").value
        self.g_ = self.get_parameter("gravity").value
        self.publish_rate_ = self.get_parameter("publish_rate").value
        self.max_velocity_ = self.get_parameter("max_velocity").value
        self.max_acceleration_ = self.get_parameter("max_acceleration").value
        self.velocity_decay_ = self.get_parameter("velocity_decay_rate").value
        self.base_pos_var_ = self.get_parameter("base_position_variance").value
        self.base_vel_var_ = self.get_parameter("base_velocity_variance").value
        self.var_growth_rate_ = self.get_parameter("variance_growth_rate").value
        self.child_frame_id_ = self.get_parameter("child_frame_id").value
        self.publish_tf_ = self.get_parameter("publish_tf").value
        
        # State variables
        self.x_ = 0.0
        self.y_ = 0.0
        self.z_ = 0.0
        self.vx_ = 0.0
        self.vy_ = 0.0
        self.vz_ = 0.0
        self.wz_ = 0.0
        self.qx_, self.qy_, self.qz_, self.qw_ = 0.0, 0.0, 0.0, 1.0
        
        # Bias estimation (simple moving average)
        self.accel_bias_ = np.zeros(3)
        self.bias_samples_ = []
        self.max_bias_samples_ = 100
        self.calibrating_ = True
        self.calibration_samples_ = 0
        self.calibration_required_ = 10
        
        # Statistics
        self.integration_count_ = 0
        self.dropped_samples_ = 0
        self.warning_count_ = 0
        
        # Subscriptions and publishers
        self.pose_sub_ = self.create_subscription(
            Imu, "/imu_ekf", self.imuCallback, 10
        )
        self.odom_pub_ = self.create_publisher(Odometry, "/marid/odom", 10)
        self.diag_pub_ = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
        
        # Timer
        timer_period = 1.0 / self.publish_rate_
        self.timer_ = self.create_timer(timer_period, self.publish_odom)
        
        # Services
        self.reset_srv_ = self.create_service(
            Empty, "~/reset_odometry", self.reset_odometry_callback
        )
        
        # Odometry message
        self.odom_msg_ = Odometry()
        self.odom_msg_.header.frame_id = "odom"
        self.odom_msg_.child_frame_id = self.child_frame_id_
        
        # Time tracking
        self.start_time_ = self.get_clock().now()
        self.prev_time_ = self.start_time_
        self.prev_imu_time_ = None
        
        self.get_logger().info(f"IMU Odometry Node initialized")
        self.get_logger().info(f"Calibrating accelerometer bias ({self.calibration_required_} samples)...")

    def reset_odometry_callback(self, request, response):
        """Service to reset odometry to zero"""
        self.x_ = 0.0
        self.y_ = 0.0
        self.z_ = 0.0
        self.vx_ = 0.0
        self.vy_ = 0.0
        self.vz_ = 0.0
        self.start_time_ = self.get_clock().now()
        self.get_logger().info("Odometry reset to zero")
        return response

    def normalize_quaternion(self, qx, qy, qz, qw):
        """Ensure quaternion is normalized; return identity if invalid (NaN/inf/zero norm)."""
        if (math.isnan(qx) or math.isnan(qy) or math.isnan(qz) or math.isnan(qw) or
            math.isinf(qx) or math.isinf(qy) or math.isinf(qz) or math.isinf(qw)):
            return 0.0, 0.0, 0.0, 1.0
        norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        if norm < 1e-6 or math.isnan(norm) or math.isinf(norm):
            return 0.0, 0.0, 0.0, 1.0
        return qx/norm, qy/norm, qz/norm, qw/norm

    def imuCallback(self, msg: Imu):
        try:
            t = Time.from_msg(msg.header.stamp)
            
            # First message initialization
            if self.prev_imu_time_ is None:
                self.prev_imu_time_ = t
                self.qx_, self.qy_, self.qz_, self.qw_ = self.normalize_quaternion(
                    msg.orientation.x, msg.orientation.y, 
                    msg.orientation.z, msg.orientation.w
                )
                return
            
            # Calculate time delta
            dt = (t - self.prev_imu_time_).nanoseconds / 1e9
            
            # Validate timestamp
            if dt <= 0.0:
                self.get_logger().warn(f"Non-positive dt: {dt}s, skipping sample")
                self.dropped_samples_ += 1
                return
            
            if dt > 1.0:
                self.get_logger().warn(f"Large dt detected: {dt}s, limiting to 1.0s")
                dt = 1.0
                self.dropped_samples_ += 1
            
            self.prev_imu_time_ = t
            
            # Update orientation with normalization
            self.qx_, self.qy_, self.qz_, self.qw_ = self.normalize_quaternion(
                msg.orientation.x, msg.orientation.y,
                msg.orientation.z, msg.orientation.w
            )
            
            # Update angular velocity
            self.wz_ = msg.angular_velocity.z
            
            # Body-frame acceleration
            a_b = np.array([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ])
            
            # Check for invalid accelerations
            if np.any(np.isnan(a_b)) or np.any(np.isinf(a_b)):
                self.get_logger().warn("NaN/Inf in acceleration data, skipping")
                self.dropped_samples_ += 1
                return
            
            # Acceleration magnitude check
            accel_mag = np.linalg.norm(a_b)
            if accel_mag > self.max_acceleration_:
                self.get_logger().warn(
                    f"Acceleration {accel_mag:.2f} exceeds limit {self.max_acceleration_}, clamping"
                )
                a_b = a_b * (self.max_acceleration_ / accel_mag)
                self.warning_count_ += 1
            
            # Bias calibration phase
            if self.calibrating_:
                self.bias_samples_.append(a_b.copy())
                self.calibration_samples_ += 1
                
                if self.calibration_samples_ >= self.calibration_required_:
                    # Assume robot is stationary, so acceleration should be [0, 0, -g] in body frame
                    # or [0, 0, g] if specific force
                    self.accel_bias_ = np.mean(self.bias_samples_, axis=0)
                    if self.accel_is_sf_:
                        # For specific force, bias is what deviates from expected [0, 0, 0] in horizontal
                        self.accel_bias_[2] = 0.0  # Don't bias out gravity component
                    self.calibrating_ = False
                    self.get_logger().info(
                        f"Calibration complete. Bias: [{self.accel_bias_[0]:.4f}, "
                        f"{self.accel_bias_[1]:.4f}, {self.accel_bias_[2]:.4f}]"
                    )
                else:
                    return
            
            # Apply bias correction
            a_b_corrected = a_b - self.accel_bias_
            
            # Rotation matrix body->world
            R_bw = quaternion_matrix([self.qx_, self.qy_, self.qz_, self.qw_])[:3, :3]
            
            # Convert acceleration to world frame
            if self.accel_is_sf_:
                # Specific force to proper acceleration
                f_w = R_bw @ a_b_corrected
                a_w = f_w + np.array([0.0, 0.0, self.g_])
            else:
                # Already true acceleration
                a_w = R_bw @ a_b_corrected
            
            # Integration with velocity damping
            self.vx_ = self.vx_ * self.velocity_decay_ + a_w[0] * dt
            self.vy_ = self.vy_ * self.velocity_decay_ + a_w[1] * dt
            self.vz_ = self.vz_ * self.velocity_decay_ + a_w[2] * dt
            
            # Velocity limiting
            v_mag = np.sqrt(self.vx_**2 + self.vy_**2 + self.vz_**2)
            if v_mag > self.max_velocity_:
                scale = self.max_velocity_ / v_mag
                self.vx_ *= scale
                self.vy_ *= scale
                self.vz_ *= scale
                self.warning_count_ += 1
            
            # Position integration
            self.x_ += self.vx_ * dt
            self.y_ += self.vy_ * dt
            self.z_ += self.vz_ * dt
            
            self.integration_count_ += 1
            
        except Exception as e:
            self.get_logger().error(f"Error in IMU callback: {str(e)}")
            self.dropped_samples_ += 1

    def publish_odom(self):
        
        now = self.get_clock().now()
        
        # Validate state variables are not NaN or infinite before publishing
        if (math.isnan(self.x_) or math.isnan(self.y_) or math.isnan(self.z_) or
            math.isnan(self.vx_) or math.isnan(self.vy_) or math.isnan(self.vz_) or
            math.isnan(self.qx_) or math.isnan(self.qy_) or math.isnan(self.qz_) or math.isnan(self.qw_) or
            math.isinf(self.x_) or math.isinf(self.y_) or math.isinf(self.z_) or
            math.isinf(self.vx_) or math.isinf(self.vy_) or math.isinf(self.vz_) or
            math.isinf(self.qx_) or math.isinf(self.qy_) or math.isinf(self.qz_) or math.isinf(self.qw_)):
            if not hasattr(self, '_nan_warn_count'):
                self._nan_warn_count = 0
            self._nan_warn_count += 1
            if self._nan_warn_count % 50 == 0:  # Log periodically
                self.get_logger().warn('State variables contain NaN or infinite values, skipping odometry publication')
                self.get_logger().warn(f'  Position: x={self.x_}, y={self.y_}, z={self.z_}')
                self.get_logger().warn(f'  Velocity: vx={self.vx_}, vy={self.vy_}, vz={self.vz_}')
                self.get_logger().warn(f'  Orientation: qx={self.qx_}, qy={self.qy_}, qz={self.qz_}, qw={self.qw_}')
            return  # Don't publish invalid data
        
        # Re-normalize quaternion before publish (avoid denormalized/NaN reaching EKF)
        self.qx_, self.qy_, self.qz_, self.qw_ = self.normalize_quaternion(
            self.qx_, self.qy_, self.qz_, self.qw_
        )
        if (math.isnan(self.qx_) or math.isnan(self.qy_) or math.isnan(self.qz_) or math.isnan(self.qw_)):
            return
        
        # Calculate growing covariance based on time since start
        elapsed_time = (now - self.start_time_).nanoseconds / 1e9
        pos_variance = self.base_pos_var_ + self.var_growth_rate_ * elapsed_time
        vel_variance = self.base_vel_var_
        
        # Populate odometry message
        self.odom_msg_.header.stamp = now.to_msg()
        
        # Position
        self.odom_msg_.pose.pose.position.x = self.x_
        self.odom_msg_.pose.pose.position.y = self.y_
        self.odom_msg_.pose.pose.position.z = self.z_
        
        # Orientation
        self.odom_msg_.pose.pose.orientation.x = self.qx_
        self.odom_msg_.pose.pose.orientation.y = self.qy_
        self.odom_msg_.pose.pose.orientation.z = self.qz_
        self.odom_msg_.pose.pose.orientation.w = self.qw_
        
        # Velocity
        self.odom_msg_.twist.twist.linear.x = self.vx_
        self.odom_msg_.twist.twist.linear.y = self.vy_
        self.odom_msg_.twist.twist.linear.z = self.vz_
        self.odom_msg_.twist.twist.angular.z = self.wz_
        
        # Pose covariance (6x6: x, y, z, roll, pitch, yaw)
        self.odom_msg_.pose.covariance = [
            pos_variance, 0, 0, 0, 0, 0,
            0, pos_variance, 0, 0, 0, 0,
            0, 0, pos_variance, 0, 0, 0,
            0, 0, 0, 0.01, 0, 0,  # Orientation from IMU EKF (low uncertainty)
            0, 0, 0, 0, 0.01, 0,
            0, 0, 0, 0, 0, 0.01
        ]
        
        # Twist covariance (6x6: vx, vy, vz, wx, wy, wz)
        self.odom_msg_.twist.covariance = [
            vel_variance, 0, 0, 0, 0, 0,
            0, vel_variance, 0, 0, 0, 0,
            0, 0, vel_variance, 0, 0, 0,
            0, 0, 0, 0.1, 0, 0,
            0, 0, 0, 0, 0.1, 0,
            0, 0, 0, 0, 0, 0.1
        ]
        
        self.odom_pub_.publish(self.odom_msg_)
        
        # Publish diagnostics periodically (every 2 seconds)
        if self.integration_count_ % (int(self.publish_rate_) * 2) == 0:
            self.publish_diagnostics(now)

    def publish_diagnostics(self, now):
        """Publish diagnostic information"""
        diag_array = DiagnosticArray()
        diag_array.header.stamp = now.to_msg()
        
        status = DiagnosticStatus()
        status.name = "IMU Odometry"
        status.hardware_id = "marid_odom_node"
        
        # Determine status level
        if self.dropped_samples_ > 100 or self.warning_count_ > 50:
            status.level = DiagnosticStatus.WARN
            status.message = "High error rate detected"
        else:
            status.level = DiagnosticStatus.OK
            status.message = "Operating normally"
        
        # Add key-value pairs
        status.values.append(KeyValue(key="Integration Count", value=str(self.integration_count_)))
        status.values.append(KeyValue(key="Dropped Samples", value=str(self.dropped_samples_)))
        status.values.append(KeyValue(key="Warnings", value=str(self.warning_count_)))
        status.values.append(KeyValue(key="Position X", value=f"{self.x_:.3f}"))
        status.values.append(KeyValue(key="Position Y", value=f"{self.y_:.3f}"))
        status.values.append(KeyValue(key="Position Z", value=f"{self.z_:.3f}"))
        status.values.append(KeyValue(key="Velocity Magnitude", 
                                     value=f"{np.sqrt(self.vx_**2 + self.vy_**2 + self.vz_**2):.3f}"))
        status.values.append(KeyValue(key="Accel Bias X", value=f"{self.accel_bias_[0]:.4f}"))
        status.values.append(KeyValue(key="Accel Bias Y", value=f"{self.accel_bias_[1]:.4f}"))
        status.values.append(KeyValue(key="Accel Bias Z", value=f"{self.accel_bias_[2]:.4f}"))
        
        diag_array.status.append(status)
        self.diag_pub_.publish(diag_array)

def main():
    rclpy.init()
    node = MaridOdomPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()