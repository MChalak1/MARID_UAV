#!/usr/bin/env python3
"""
Simulated sun sensor for MARID GPS-denied navigation.

Computes the solar position (azimuth + elevation) from sim time and known
lat/lon, then rotates the sun ENU unit vector into body frame using the
Gazebo ground-truth pose from /gazebo/odom.  Publishes what a real sun
sensor would physically measure, independently of the estimation pipeline.

Published topics:
  /sun_sensor/sun_vector_body  (geometry_msgs/Vector3Stamped)
      Body-frame unit vector pointing toward the sun.
  /sun_sensor/sun_azimuth_enu_rad  (std_msgs/Float64)
      Sun azimuth in ENU frame (from East, CCW positive, radians).
  /sun_sensor/sun_elevation_deg  (std_msgs/Float64)
      Sun elevation above horizon in degrees.

Parameters:
  lat_deg              Geodetic latitude  (default 37.4 — SF Bay Area)
  lon_deg              Geodetic longitude (default -122.1)
  reference_utc        ISO-8601 epoch string that sim time=0 corresponds to
                       (default "2026-04-26T12:00:00")
  sun_elevation_min_deg  Minimum elevation for a valid reading (default 10.0)
  output_rate_hz       Publish rate in Hz (default 20.0)
  use_sim_time         Standard ROS2 flag (default True)
"""

import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Float64
from datetime import datetime, timezone, timedelta


# ---------------------------------------------------------------------------
# Solar position (simplified NREL SPA, ~0.01° accuracy, stdlib only)
# ---------------------------------------------------------------------------

def _julian_day(dt_utc: datetime) -> float:
    """Convert UTC datetime to Julian Day Number."""
    year  = dt_utc.year
    month = dt_utc.month
    day   = dt_utc.day
    frac  = (dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0) / 24.0
    if month <= 2:
        year  -= 1
        month += 12
    A = int(year / 100)
    B = 2 - A + int(A / 4)
    return int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + frac + B - 1524.5


def _sun_enu(jd: float, lat_deg: float, lon_deg: float):
    """
    Return (sun_ENU, elevation_deg, az_enu_rad) or (None, elevation_deg, 0)
    if sun is below horizon or near zenith.

    sun_ENU : np.ndarray-like list [E, N, U] unit vector (ENU convention).
    az_enu_rad : azimuth from East, CCW positive (radians).
    """
    n = jd - 2451545.0                          # days since J2000.0

    L = (280.460 + 0.9856474 * n) % 360.0      # mean longitude (deg)
    g = math.radians((357.528 + 0.9856003 * n) % 360.0)  # mean anomaly

    lam_deg = L + 1.915 * math.sin(g) + 0.020 * math.sin(2.0 * g)
    eps_deg = 23.439 - 4.0e-7 * n              # obliquity (deg)

    lam = math.radians(lam_deg)
    eps = math.radians(eps_deg)

    # Equatorial coordinates
    ra  = math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))
    dec = math.asin(math.sin(eps) * math.sin(lam))

    # Greenwich Mean Sidereal Time (deg)
    T = n / 36525.0
    gmst_deg = (280.46061837 + 360.98564736629 * n
                + 0.000387933 * T * T - T * T * T / 38710000.0) % 360.0

    lha = math.radians(gmst_deg + lon_deg) - ra  # local hour angle (rad)

    lat = math.radians(lat_deg)
    sin_el = (math.sin(lat) * math.sin(dec)
              + math.cos(lat) * math.cos(dec) * math.cos(lha))
    sin_el  = max(-1.0, min(1.0, sin_el))
    elevation = math.asin(sin_el)
    el_deg = math.degrees(elevation)

    cos_el = math.cos(elevation)
    if cos_el < 1e-6:
        return None, el_deg, 0.0            # near zenith — azimuth undefined

    # Azimuth from North, clockwise (standard meteorological)
    cos_az_N = max(-1.0, min(1.0,
        (math.sin(dec) - sin_el * math.sin(lat)) / (cos_el * math.cos(lat))
    ))
    az_from_N = math.acos(cos_az_N)
    if math.sin(lha) > 0.0:
        az_from_N = 2.0 * math.pi - az_from_N  # afternoon — West side

    # ENU unit vector (X=East, Y=North, Z=Up)
    sun_ENU = [
        cos_el * math.sin(az_from_N),   # East
        cos_el * math.cos(az_from_N),   # North
        sin_el,                          # Up
    ]

    # Azimuth from East, CCW positive (ENU convention)
    az_enu_rad = math.atan2(sun_ENU[1], sun_ENU[0])

    return sun_ENU, el_deg, az_enu_rad


# ---------------------------------------------------------------------------
# ROS2 node
# ---------------------------------------------------------------------------

class SunSensorSim(Node):
    def __init__(self):
        super().__init__("sun_sensor_sim")

        self.declare_parameter("lat_deg",               37.4)
        self.declare_parameter("lon_deg",              -122.1)
        self.declare_parameter("reference_utc",  "2026-04-26T19:00:00")
        self.declare_parameter("sun_elevation_min_deg", 10.0)
        self.declare_parameter("output_rate_hz",        20.0)

        self.lat_deg_  = float(self.get_parameter("lat_deg").value)
        self.lon_deg_  = float(self.get_parameter("lon_deg").value)
        ref_str        = str(self.get_parameter("reference_utc").value)
        self.sun_el_min_ = float(self.get_parameter("sun_elevation_min_deg").value)
        rate_hz        = float(self.get_parameter("output_rate_hz").value)

        # Parse reference UTC epoch (sim time=0 corresponds to this moment)
        try:
            self.ref_epoch_ = datetime.fromisoformat(ref_str).replace(tzinfo=timezone.utc)
        except ValueError:
            self.get_logger().error(
                f"Cannot parse reference_utc '{ref_str}'. Using 2026-04-26T12:00:00.")
            self.ref_epoch_ = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)

        # Ground-truth orientation from Gazebo
        self.gt_qx_ = 0.0
        self.gt_qy_ = 0.0
        self.gt_qz_ = 0.0
        self.gt_qw_ = 1.0
        self.gt_ready_ = False

        self.gt_sub_ = self.create_subscription(
            Odometry, "/gazebo/odom", self._gt_callback, 10
        )

        self.vec_pub_ = self.create_publisher(
            Vector3Stamped, "/sun_sensor/sun_vector_body", 10
        )
        self.az_pub_  = self.create_publisher(
            Float64, "/sun_sensor/sun_azimuth_enu_rad", 10
        )
        self.el_pub_  = self.create_publisher(
            Float64, "/sun_sensor/sun_elevation_deg", 10
        )

        period = 1.0 / max(1.0, rate_hz)
        self.timer_ = self.create_timer(period, self._publish)
        self.get_logger().info(
            f"Sun sensor sim started: lat={self.lat_deg_}° lon={self.lon_deg_}° "
            f"ref_epoch={self.ref_epoch_.isoformat()}"
        )

    def _gt_callback(self, msg: Odometry):
        o = msg.pose.pose.orientation
        self.gt_qx_ = o.x
        self.gt_qy_ = o.y
        self.gt_qz_ = o.z
        self.gt_qw_ = o.w
        self.gt_ready_ = True

    def _publish(self):
        now = self.get_clock().now()
        sim_sec = now.nanoseconds * 1e-9
        utc_now = self.ref_epoch_ + timedelta(seconds=sim_sec)
        jd = _julian_day(utc_now)

        sun_ENU, el_deg, az_enu_rad = _sun_enu(jd, self.lat_deg_, self.lon_deg_)

        # Always publish elevation (used for validity gating by the fusion node)
        el_msg = Float64()
        el_msg.data = el_deg
        self.el_pub_.publish(el_msg)

        if sun_ENU is None or el_deg < self.sun_el_min_:
            return                          # sun below threshold — no vector output

        # Publish azimuth (ENU, from East CCW) so fusion node can extract heading
        az_msg = Float64()
        az_msg.data = az_enu_rad
        self.az_pub_.publish(az_msg)

        if not self.gt_ready_:
            return                          # no ground truth yet

        # Rotate sun ENU into body frame using Gazebo ground-truth quaternion.
        # R_body_to_world (body→world) applied as R.T to get world→body.
        # DCM from [x,y,z,w] quaternion: standard body-to-world formula.
        qx, qy, qz, qw = self.gt_qx_, self.gt_qy_, self.gt_qz_, self.gt_qw_
        norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if norm < 1e-9:
            return
        qx /= norm; qy /= norm; qz /= norm; qw /= norm

        # Body-to-world DCM columns (R columns = body axes in world)
        # R.T row i = world axis i expressed in body frame
        # sun_body = R.T @ sun_ENU
        se = sun_ENU
        sun_bx = ((1-2*(qy*qy+qz*qz))*se[0] + 2*(qx*qy+qw*qz)*se[1] + 2*(qx*qz-qw*qy)*se[2])
        sun_by = (2*(qx*qy-qw*qz)*se[0] + (1-2*(qx*qx+qz*qz))*se[1] + 2*(qy*qz+qw*qx)*se[2])
        sun_bz = (2*(qx*qz+qw*qy)*se[0] + 2*(qy*qz-qw*qx)*se[1] + (1-2*(qx*qx+qy*qy))*se[2])

        vec_msg = Vector3Stamped()
        vec_msg.header.stamp = now.to_msg()
        vec_msg.header.frame_id = "base_link_front"
        vec_msg.vector.x = sun_bx
        vec_msg.vector.y = sun_by
        vec_msg.vector.z = sun_bz
        self.vec_pub_.publish(vec_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SunSensorSim()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
