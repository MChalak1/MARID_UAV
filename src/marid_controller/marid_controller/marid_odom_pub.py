#!/usr/bin/env python3
"""MARID GPS-denied odometry publisher.

Navigation map:
  MARIDESKF
    - owns the error-state filter math only: nominal state, covariance,
      process prediction, and measurement updates.

  MaridOdomPublisher
    - owns ROS I/O, sensor freshness, source priority, launch parameters,
      ESKF seeding, and final /marid/odom publication.

State ownership in the running node:
  pose.x/y       -> ESKF position when seeded; FAST-LIO/wheel/DR otherwise.
  pose.z         -> z_fused_ from sonar/barometer/IMU, not ESKF vertical drift.
  orientation    -> raw /imu roll-pitch + Madgwick/mag/sun yaw, or ESKF when sane.
  vx/vy          -> ESKF velocity when seeded; helper velocity stack otherwise.
  vz             -> IMU vertical velocity, with barometer fallback above LiDAR range.
  acceleration   -> never directly published; used as gravity/tilt evidence only.

ESKF trust/authority legend:
  Rule of thumb:
    - Measurement R params: smaller = more trust, larger = less trust.
    - Process Q params: larger = state allowed to move more freely.
    - Blend weights: larger = output pulled more strongly toward that source.

  Parameter                         Physical meaning             Consumed at
  eskf_r_pos                        FAST-LIO / wheel XY          fastlio_callback(), wheel_odom_callback()
  eskf_r_vel                        OF / fwd cam / wheel vel     of_callback(), fwd_cam_callback(), wheel_odom_callback()
  eskf_r_vel_airspeed               forward pitot Va_x           airspeed_callback()
  eskf_r_alt_sonar                  sonar z                      sonar_callback()
  eskf_r_alt_baro                   barometer z                  baro_callback()
  eskf_r_yaw                        FAST-LIO yaw                 fastlio_callback()
  eskf_r_yaw_sun                    sun-sensor yaw               sun_vector_callback()
  eskf_r_yaw_madgwick_fallback      Madgwick yaw stabilizer      imu_callback()
  eskf_r_raw_tilt                   raw /imu roll and pitch      imu_callback()
  eskf_r_acc                        accel-as-gravity tilt        eskf_raw_imu_callback()
  eskf_r_coordinated_turn           bank authority on b_q/b_r    eskf_raw_imu_callback(), update_coordinated_turn()
  coordinated_turn_yaw_rate_weight  bank authority on yaw rate   _apply_coordinated_turn_yaw_rate_input()
  coordinated_turn_min_bank_deg     min bank for yaw coupling    _apply_coordinated_turn_yaw_rate_input(), eskf_raw_imu_callback()
  coordinated_turn_max_bank_deg     max bank for yaw coupling    _apply_coordinated_turn_yaw_rate_input(), eskf_raw_imu_callback()
  coordinated_turn_min_speed_mps    min speed for yaw coupling   _apply_coordinated_turn_yaw_rate_input(), eskf_raw_imu_callback()
  coordinated_turn_min_agl_m        min AGL for yaw coupling     _apply_coordinated_turn_yaw_rate_input(), eskf_raw_imu_callback()
  induced_drag_frac                 bank/pitch drag authority    predict_physics(), publish_odom()

  Non-ESKF/helper authority:
  fastlio_position_weight           helper x/y position blend    fastlio_callback()
  fastlio_yaw_weight                helper yaw blend             fastlio_callback()
  airspeed_weight                   helper vx blend              publish_odom()
  optical_flow_weight               helper vx/vy blend           publish_odom()
  forward_camera_weight             helper vy blend              publish_odom()
  sonar_weight, baro_weight         z_fused_ blend               _update_z_fused()

  Note: with use_eskf=True and once seeded, publish_odom() overwrites vx_pub/vy_pub
  with self.eskf_.velocity. Helper weights still matter before seeding or when ESKF
  is disabled, but ESKF R/Q parameters govern the final published vx/vy during
  normal ESKF operation.
"""

import math
import numpy as np

import rclpy
import rclpy.duration
from rclpy.node import Node
from rclpy.time import Time

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, Range, JointState
from geometry_msgs.msg import TransformStamped, PoseWithCovarianceStamped, TwistStamped, TwistWithCovarianceStamped, Vector3Stamped
from std_msgs.msg import Float64
from tf2_ros import TransformBroadcaster, Buffer, TransformListener

from std_srvs.srv import Empty
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

from tf_transformations import quaternion_matrix


class MARIDESKF:
    """
    Error-State Kalman Filter for GPS-denied pose estimation on MARID.

    Two propagation modes:
      "imu"     — IMU-driven attitude/position propagation, δx ∈ R15
                  Error state: δp δv δtheta δba δbg
                  Nominal: p(3) v(3) q(4) ba(3) bg(3)
      "physics" — Aerodynamic-model horizontal prediction, δx ∈ R17
                  Error state: δp δv δtheta δba δkd δmh2 δbg
                  Nominal: p(3) v(3) q(4) ba(3) kd(1) mh2(1) bg(3)

    Quaternion convention: [x, y, z, w] matching tf_transformations.
    Frame: ENU (z-up). IMU input must be specific force (a_sf = a_true + g_body).

    Velocity ownership inside the filter:
      vx/vy: predicted by aerodynamic physics in physics mode, corrected by
             body-frame velocity sensors and position measurements.
      vz:    kept sensor-driven by the publisher; physics intentionally does not
             integrate vertical acceleration/lift into vz.
    """

    _CHI2_95 = {1: 3.841, 2: 5.991, 3: 7.815, 6: 12.592}

    def __init__(
        self,
        mode: str = "physics",
        g: float = 9.81,
        p_pos: float = 1.0,
        p_vel: float = 1.0,
        p_att: float = 0.1,
        p_ba:  float = 0.01,
        p_bg:  float = 1e-4,
        p_kd:  float = 1.0,
        p_mh2: float = 1.0,
        q_vel: float = 0.1,
        q_att: float = 0.01,
        q_ba:  float = 1e-4,
        q_bg:  float = 1e-6,
        q_kd:  float = 1e-4,
        q_mh2: float = 1e-4,
        kd_init:     float = 0.5,
        mh2_init:    float = 30.0,
        mass_empty:  float = 160.0,
        air_density: float = 1.225,
        sfc:         float = 5.0e-6,  # kg/(N·s) — thrust-specific fuel consumption
        # Aerodynamic model — values from marid_new_gazebo.xacro AdvancedLiftDrag plugin
        CL0:           float = 0.15188,
        CLa:           float = 5.015,
        alpha_stall:   float = 0.3391428111,
        CLa_stall:     float = -3.85,
        wing_area:     float = 2.0,
        AR:            float = 6.5,
        eff:           float = 0.97,
        CL_ctrl_wing:  float = 2.0,   # per wing surface, direction=-1 in xacro
        CL_ctrl_tail:  float = 0.2,   # per tail surface, direction=+1 in xacro
        min_aero_speed: float = 5.0,  # m/s below which velocity integration is skipped
        induced_drag_frac: float = 0.25,  # fraction of total drag that is induced (bank-corrected)
    ):
        assert mode in ("imu", "physics"), f"Unknown ESKF mode: {mode}"
        self.mode       = mode
        self.g_world    = np.array([0.0, 0.0, -g])
        self.rho        = air_density
        self.mass_empty = mass_empty
        # bg_idx: where gyro-bias block starts in error state.
        # IMU  (n=15): [δp δv δθ δba δbg]          → bg at 12
        # Phys (n=17): [δp δv δθ δba δkd δmh2 δbg] → bg at 14 (kd/mh2 keep indices 12-13)
        self.bg_idx     = 12 if mode == "imu" else 14
        self.n          = 15 if mode == "imu" else 17
        # Aerodynamic constants
        self.CL0          = CL0
        self.CLa          = CLa
        self.alpha_stall  = alpha_stall
        self.CLa_stall    = CLa_stall
        self.wing_area    = wing_area
        self.AR           = AR
        self.eff          = eff
        self.CL_ctrl_wing = CL_ctrl_wing
        self.CL_ctrl_tail = CL_ctrl_tail
        self.min_aero_speed = min_aero_speed
        self.induced_drag_frac = induced_drag_frac
        self.sfc = sfc

        self.p  = np.zeros(3)
        self.v  = np.zeros(3)
        self.q  = np.array([0.0, 0.0, 0.0, 1.0])  # [x,y,z,w]
        self.ba = np.zeros(3)  # accelerometer bias
        self.bg = np.zeros(3)  # gyro bias [b_p, b_q, b_r] rad/s
        self.kd  = kd_init  if mode == "physics" else 0.0
        self.mh2 = mh2_init if mode == "physics" else 0.0

        p_diag = [p_pos]*3 + [p_vel]*3 + [p_att]*3 + [p_ba]*3
        q_diag = [0.0]*3   + [q_vel]*3 + [q_att]*3 + [q_ba]*3
        if mode == "physics":
            p_diag += [p_kd, p_mh2]
            q_diag += [q_kd, q_mh2]
        p_diag += [p_bg] * 3   # gyro bias — always last (after kd/mh2 in physics mode)
        q_diag += [q_bg] * 3
        self.P  = np.diag(p_diag).astype(float)
        self.Qc = np.diag(q_diag).astype(float)
        self.P0 = self.P.copy()
        self.kd0 = self.kd
        self.mh20 = self.mh2

    def reset(self, p=None, v=None, q=None):
        """Reset nominal state and covariance while preserving tuning."""
        self.p = np.array(p, dtype=float) if p is not None else np.zeros(3)
        self.v = np.array(v, dtype=float) if v is not None else np.zeros(3)
        self.q = self._qnorm(np.array(q, dtype=float)) if q is not None else np.array([0.0, 0.0, 0.0, 1.0])
        self.ba = np.zeros(3)
        self.bg = np.zeros(3)
        self.kd = self.kd0 if self.mode == "physics" else 0.0
        self.mh2 = self.mh20 if self.mode == "physics" else 0.0
        self.P = self.P0.copy()

    # ------------------------------------------------------------------
    # ESKF math helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dcm(q):
        """[x,y,z,w] → 3×3 body-to-world DCM."""
        x, y, z, w = q
        return np.array([
            [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
            [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
            [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
        ])

    @staticmethod
    def _skew(v):
        return np.array([
            [ 0.0, -v[2],  v[1]],
            [ v[2],  0.0, -v[0]],
            [-v[1],  v[0],  0.0],
        ])

    @staticmethod
    def _qnorm(q):
        n = np.linalg.norm(q)
        return q / n if n > 1e-9 else np.array([0.0, 0.0, 0.0, 1.0])

    @staticmethod
    def _boxplus(q, dtheta):
        """q ⊗ exp(dtheta/2): inject attitude error δθ into quaternion."""
        angle = np.linalg.norm(dtheta)
        if angle < 1e-10:
            dq = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            s  = math.sin(angle * 0.5) / angle
            dq = np.array([dtheta[0]*s, dtheta[1]*s, dtheta[2]*s,
                           math.cos(angle * 0.5)])
        qx, qy, qz, qw = q
        dx, dy, dz, dw = dq
        return np.array([
            qw*dx + qx*dw + qy*dz - qz*dy,
            qw*dy - qx*dz + qy*dw + qz*dx,
            qw*dz + qx*dy - qy*dx + qz*dw,
            qw*dw - qx*dx - qy*dy - qz*dz,
        ])

    # ------------------------------------------------------------------
    # ESKF prediction models
    # ------------------------------------------------------------------

    def predict_imu(self, a_sf: np.ndarray, omega: np.ndarray, dt: float):
        """Propagate attitude from gyro; dead-reckon position from current velocity.

        Acceleration is intentionally not integrated into velocity. The specific-force
        convention across the Gazebo→imu_add_gravity→Madgwick pipeline is uncertain
        (gravity may be double-counted depending on sim config). Velocity is owned
        entirely by measurement updates (airspeed, OF, FAST-LIO, wheel odom), which
        bound drift to a single sensor update interval.
        """
        if not (0.0 < dt < 0.5):
            return
        omega_corr = omega - self.bg
        self.p = self.p + self.v * dt
        self.q = self._qnorm(self._boxplus(self.q, omega_corr * dt))

        n = self.n
        F = np.zeros((n, n))
        F[0:3, 3:6] = np.eye(3)
        F[6:9, 6:9] = -self._skew(omega_corr)
        F[6:9, self.bg_idx:self.bg_idx+3] = -np.eye(3)  # δθ̇ = −δbg

        Phi = np.eye(n) + F * dt
        self.P = Phi @ self.P @ Phi.T + self.Qc * dt

    def predict_physics(
        self, thrust_N: float, omega: np.ndarray,
        delta_lw: float, delta_rw: float, delta_tl: float, delta_tr: float,
        dt: float,
    ):
        """Propagate using full aerodynamic model (thrust + lift + drag + gravity).

        Lift model matches marid_new_gazebo.xacro AdvancedLiftDrag plugin.
        Velocity integration is enabled only above min_aero_speed; below that
        (ground, slow flight) position dead-reckons from current velocity estimate.
        """
        if self.mode != "physics" or not (0.0 < dt < 0.5):
            return

        C  = self._dcm(self.q)
        c1 = C[:, 0]   # body-forward in world frame
        c3 = C[:, 2]   # body-up in world frame
        m  = max(self.mass_empty + self.mh2, 1.0)
        speed = max(np.linalg.norm(self.v), 1e-6)
        q_dyn = 0.5 * self.rho * speed * speed

        # Angle of attack in body xz-plane (body forward=x, up=z per xacro convention)
        v_body  = C.T @ self.v
        alpha   = math.atan2(-v_body[2], max(abs(v_body[0]), 1e-6)) * math.copysign(1.0, v_body[0] + 1e-9)
        alpha   = max(-0.5, min(0.5, alpha))

        # CL: pre-stall / post-stall (matching AdvancedLiftDrag logic)
        if abs(alpha) < self.alpha_stall:
            CL_alpha = self.CL0 + self.CLa * alpha
        else:
            s = math.copysign(1.0, alpha)
            CL_alpha = (self.CL0 + self.CLa * self.alpha_stall * s
                        + self.CLa_stall * (abs(alpha) - self.alpha_stall) * s)

        # Control surface contributions (xacro direction: -1 for wings, +1 for tails)
        CL_surf = (-self.CL_ctrl_wing * delta_lw
                   - self.CL_ctrl_wing * delta_rw
                   + self.CL_ctrl_tail * delta_tl
                   + self.CL_ctrl_tail * delta_tr)
        CL_total = CL_alpha + CL_surf

        # Lift direction: c3 projected perpendicular to velocity (standard aero convention)
        v_hat    = self.v / speed
        lift_dir = c3 - np.dot(c3, v_hat) * v_hat
        ld_norm  = np.linalg.norm(lift_dir)
        lift_dir = lift_dir / ld_norm if ld_norm > 1e-6 else c3.copy()

        L = q_dyn * self.wing_area * CL_total
        F_lift = L * lift_dir

        # Drag: k_d augmented state (quadratic, opposes velocity).
        # Use horizontal speed to avoid inflating drag from noisy baro vz estimates.
        # Bank-angle correction (1/cos²φ) applies only to the induced-drag fraction —
        # parasitic drag is unaffected by bank angle.
        speed_h = max(math.sqrt(self.v[0]**2 + self.v[1]**2), 1e-6)
        roll = np.arctan2(C[2, 1], C[2, 2])
        cos2_phi = max(math.cos(roll) ** 2, math.cos(math.pi / 6) ** 2)
        bank_factor = 1.0 + (1.0 / cos2_phi - 1.0) * self.induced_drag_frac
        F_drag = -self.kd * self.rho * speed_h * self.v * bank_factor

        a_world = (thrust_N / m) * c1 + F_lift / m + F_drag / m + self.g_world

        # Propagate attitude always.
        # Horizontal (vx, vy) integrated from physics above min_aero_speed.
        # Vertical (vz) always dead-reckons: altitude is the best-observed state
        # (sonar + baro) and lift-model inaccuracies during climb cause z drift.
        omega_corr = omega - self.bg
        self.q = self._qnorm(self._boxplus(self.q, omega_corr * dt))
        if speed >= self.min_aero_speed:
            self.p[:2] += self.v[:2] * dt + 0.5 * a_world[:2] * (dt * dt)
            self.v[:2] += a_world[:2] * dt
        else:
            self.p[:2] += self.v[:2] * dt   # dead-reckon horizontal at low speed
        self.p[2] += self.v[2] * dt         # z always dead-reckons from sensor-corrected vz

        # Burn hydrogen: m_h2 ← m_h2 − SFC·T·Δt  (doc §3.2)
        self.mh2 = max(0.0, self.mh2 - self.sfc * thrust_N * dt)

        # F matrix — aerodynamic cross-terms enable k_d/m_h2 observability.
        # Row 5 (vz dot) is zeroed: vz is not integrated from physics (always
        # sensor-driven), so coupling its covariance to physics would be wrong.
        F = np.zeros((self.n, self.n))
        F[0:3, 3:6]   = np.eye(3)
        # Physics-driven velocity linearisation applies only to vx, vy (rows 3,4)
        v_h = np.array([self.v[0], self.v[1], 0.0])
        _drag_jac = -(self.kd * self.rho * bank_factor / m) * (
            speed_h * np.eye(3) + np.outer(self.v, v_h) / speed_h)
        _att_jac  = (thrust_N / m) * self._skew(c1) + (L / m) * self._skew(lift_dir)
        _kd_jac   = (-(self.rho * speed_h * self.v * bank_factor) / m).reshape(3, 1)
        _mh2_jac  = (-a_world / m).reshape(3, 1)
        F[3:5, 3:6]   = _drag_jac[0:2, :]   # vx, vy drag coupling
        F[3:5, 6:9]   = _att_jac[0:2, :]    # vx, vy attitude coupling
        F[3:5, 12:13] = _kd_jac[0:2]        # vx, vy vs kd
        F[3:5, 13:14] = _mh2_jac[0:2]       # vx, vy vs mh2
        # row 5 (vz) left zero — vz is sensor-driven, not physics-integrated
        F[6:9, 6:9]   = -self._skew(omega_corr)
        F[6:9, self.bg_idx:self.bg_idx+3] = -np.eye(3)  # δθ̇ = −δbg

        Phi = np.eye(self.n) + F * dt
        self.P = Phi @ self.P @ Phi.T + self.Qc * dt

    # ------------------------------------------------------------------
    # ESKF generic measurement update
    # ------------------------------------------------------------------

    def _update(self, z: np.ndarray, H: np.ndarray,
                R_noise: np.ndarray, gate_chi2=None,
                freeze_bg: bool = False) -> bool:
        """EKF update with optional Mahalanobis gate. Returns True if applied.

        freeze_bg: when True, skip the bg correction.  Use for kinematic
        measurements (position, velocity) whose innovations have no direct
        bearing on gyro bias — the cross-covariance path from P would otherwise
        corrupt bg when those measurements carry systematic errors.
        """
        S = H @ self.P @ H.T + R_noise
        if gate_chi2 is not None:
            try:
                d2 = float(z @ np.linalg.solve(S, z))
                if d2 > gate_chi2:
                    return False
            except np.linalg.LinAlgError:
                return False
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False
        if freeze_bg:
            K[self.bg_idx:self.bg_idx+3, :] = 0.0

        dx = K @ z
        self.p  = self.p  + dx[0:3]
        self.v  = self.v  + dx[3:6]
        self.q  = self._qnorm(self._boxplus(self.q, dx[6:9]))
        self.ba = self.ba + dx[9:12]
        if self.mode == "physics":
            self.kd  = max(0.0, self.kd  + float(dx[12]))
            self.mh2 = max(0.0, self.mh2 + float(dx[13]))
        self.bg = self.bg + dx[self.bg_idx:self.bg_idx+3]

        I_KH = np.eye(self.n) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_noise @ K.T
        return True

    # ------------------------------------------------------------------
    # ESKF measurement models
    # ------------------------------------------------------------------
    # H is constructed from geometry, not hardcoded frame assumptions.

    def update_position(self, p_meas: np.ndarray, r_pos: float = 0.1, gate: bool = True):
        H = np.zeros((3, self.n)); H[0:3, 0:3] = np.eye(3)
        self._update(p_meas - self.p, H, np.eye(3) * r_pos,
                     self._CHI2_95[3] if gate else None, freeze_bg=True)

    def update_position_xy(self, xy_meas: np.ndarray, r_pos: float = 0.1, gate: bool = True):
        H = np.zeros((2, self.n)); H[0:2, 0:2] = np.eye(2)
        self._update(xy_meas - self.p[0:2], H, np.eye(2) * r_pos,
                     self._CHI2_95[2] if gate else None, freeze_bg=True)

    def update_altitude(self, z_meas: float, r_alt: float = 0.05, gate: bool = True):
        H = np.zeros((1, self.n)); H[0, 2] = 1.0
        self._update(np.array([z_meas - self.p[2]]), H, np.array([[r_alt]]),
                     self._CHI2_95[1] if gate else None, freeze_bg=True)

    def update_body_velocity_1d(self, v_body_meas: float, axis: int,
                                    r_vel: float = 0.5, gate: bool = True):
        """Update from scalar body-frame velocity.

        axis=0: body X / forward probe
        axis=1: body Y / lateral probe
        axis=2: body Z / vertical probe

        Keeps attitude coupling, but limits its authority so noisy Y/Z
        airspeed does not corrupt roll.
        """

        C   = self._dcm(self.q)
        ci  = C[:, axis]
        v_b = C.T @ self.v

        ei_cross_vb = np.array([
            [0.0,     -v_b[2],  v_b[1]],
            [v_b[2],   0.0,    -v_b[0]],
            [-v_b[1],  v_b[0],  0.0],
        ])[axis]

        H = np.zeros((1, self.n))
        H[0, 3:6] = ci

        # ---- attitude-coupling limiter ----
        if axis == 0:
            att_scale = 0.05      # forward airspeed: allow weak attitude correction
            deadband = 0.5
        elif axis == 1:
            att_scale = 0.01      # lateral airspeed: much weaker, protects roll
            deadband = 1.0
        else:
            att_scale = 0.01      # vertical airspeed: much weaker
            deadband = 1.0

        # Near zero, signed pitot velocity is noisy/ambiguous.
        # Still update velocity, but do not let it correct attitude.
        if abs(v_body_meas) > deadband:
            H[0, 6:9] = att_scale * ei_cross_vb
        else:
            H[0, 6:9] = 0.0

        innov = np.array([v_body_meas - float(ci @ self.v)])

        self._update(
            innov,
            H,
            np.array([[r_vel]]),
            self._CHI2_95[1] if gate else None,
            freeze_bg=True,
        )

    def update_body_velocity_2d(self, vx_body: float, vy_body: float,
                                r_vel: float = 0.5, gate: bool = True):
        """Update from 2D body-frame velocity.
        Keeps velocity correction strong, but prevents optical flow velocity
        innovations from over-correcting attitude.
        """
        C   = self._dcm(self.q)
        c1  = C[:, 0]
        c2  = C[:, 1]
        v_b = C.T @ self.v

        H = np.zeros((2, self.n))
        H[0, 3:6] = c1
        H[1, 3:6] = c2

        # Original full attitude sensitivities
        h_att_x = np.array([0.0,     -v_b[2],  v_b[1]])
        h_att_y = np.array([v_b[2],   0.0,    -v_b[0]])

        # Optical flow is useful, but can be noisy/geometry-dependent.
        # Use weak attitude coupling.
        att_scale_x = 0.03
        att_scale_y = 0.01

        if abs(vx_body) > 0.5:
            H[0, 6:9] = att_scale_x * h_att_x
        else:
            H[0, 6:9] = 0.0

        if abs(vy_body) > 1.0:
            H[1, 6:9] = att_scale_y * h_att_y
        else:
            H[1, 6:9] = 0.0

        innov = np.array([
            vx_body - v_b[0],
            vy_body - v_b[1],
        ])

        self._update(
            innov,
            H,
            np.eye(2) * r_vel,
            self._CHI2_95[2] if gate else None,
            freeze_bg=True,
        )

    def update_airspeed_world_velocity(self, Va: float,
                                       r_fwd: float = 0.05,
                                       r_lat: float = 0.25):
        """Airspeed + zero-sideslip constraint — fully constrains 2D world velocity.

        Two simultaneous body-frame measurements:
            body-forward: c1·v = Va   (pitot, tight)
            body-lateral: c2·v = 0    (coordinated flight, loose)

        Together they make both vx and vy observable, eliminating the lateral
        unobservability that causes velocity divergence during sustained turns
        when only a 1-D forward measurement is used.
        No chi-squared gate — must recover from large accumulated velocity errors.
        """
        C  = self._dcm(self.q)
        c1 = C[:, 0]
        c2 = C[:, 1]

        H = np.zeros((2, self.n))
        H[0, 3:6] = c1
        H[1, 3:6] = c2

        innov = np.array([
            Va - float(c1 @ self.v),
            -float(c2 @ self.v),
        ])

        self._update(innov, H, np.diag([r_fwd, r_lat]),
                     gate_chi2=None, freeze_bg=True)

    def update_nhc(self, r_nhc: float = 0.01):
        """Non-Holonomic Constraint: lateral body velocity is zero on the ground.

        Wheels prevent sideways slip regardless of heading, so vy_body = 0
        even while taxiing (vx_body unconstrained).  H uses the full DCM so
        the constraint is correct at any spawn yaw.
        """
        C  = self._dcm(self.q)
        c2 = C[:, 1]   # body-Y axis expressed in world frame = [-sinψ, cosψ, ~0]
        H  = np.zeros((1, self.n))
        H[0, 3:6] = c2
        innov = np.array([-float(c2 @ self.v)])   # 0 - vy_body_est
        self._update(innov, H, np.array([[r_nhc]]),
                     gate_chi2=None, freeze_bg=True)

    def update_heading(self, yaw_meas: float, r_yaw: float = 0.05, gate: bool = True):
        """Update from yaw measurement. H = [0,0,1] in δθ block (near-level approx)."""
        x, y, z, w = self.q
        yaw_est = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        dy = yaw_meas - yaw_est
        dy = (dy + math.pi) % (2 * math.pi) - math.pi
        H  = np.zeros((1, self.n)); H[0, 8] = 1.0
        self._update(np.array([dy]), H, np.array([[r_yaw]]),
                     self._CHI2_95[1] if gate else None)

    def update_roll_pitch(self, roll_meas: float, pitch_meas: float,
                          r_tilt: float = 0.005, gate: bool = True):
        """Update roll/pitch only from a tilt measurement.

        Used in simulation where raw /imu orientation has high-quality roll/pitch,
        while Madgwick remains useful mainly for yaw observability.  The small-angle
        H maps local attitude error [δroll, δpitch] to the first two δθ states,
        matching update_heading's yaw-only approximation below.
        """
        x, y, z, w = self.q
        roll_est = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        sinp = 2*(w*y - z*x)
        pitch_est = math.asin(max(-1.0, min(1.0, sinp)))
        dr = (roll_meas - roll_est + math.pi) % (2 * math.pi) - math.pi
        dp = (pitch_meas - pitch_est + math.pi) % (2 * math.pi) - math.pi
        H = np.zeros((2, self.n))
        H[0, 6] = 1.0
        H[1, 7] = 1.0
        self._update(
            np.array([dr, dp]),
            H,
            np.eye(2) * r_tilt,
            self._CHI2_95[2] if gate else None,
        )

    def update_orientation(self, q_meas: np.ndarray, r_ori: float = 0.01,
                           gate: bool = True, gate_roll: bool = False):
        """Update from a full quaternion (Madgwick output).  H = I_3 in δθ block (doc §6.11).

        Innovation: ν = 2 · vec(q̂⁻¹ ⊗ q_meas)
        Corrects all three δθ components simultaneously, not just yaw.
        q convention: [x, y, z, w] throughout.

        gate_roll: if True, zero the δroll row of H.  Madgwick uses the accelerometer
        for tilt correction and therefore suffers the same centripetal contamination as
        update_gravity — it underestimates roll during banked turns.  Suppressing its
        roll influence lets the gyro integration (bias-corrected by b_g) own roll during banks.
        """
        qx, qy, qz, qw = self.q
        mx, my, mz, mw = q_meas
        if qx*mx + qy*my + qz*mz + qw*mw < 0.0:
            mx, my, mz, mw = -mx, -my, -mz, -mw
        # q̂⁻¹ = [−qx, −qy, −qz, qw] for a unit quaternion.
        # δq = q̂⁻¹ ⊗ q_meas  (Hamilton product, [x,y,z,w] convention):
        dx = qw*mx - qx*mw - qy*mz + qz*my
        dy = qw*my + qx*mz - qy*mw - qz*mx
        dz = qw*mz - qx*my + qy*mx - qz*mw
        nu = np.array([2.0*dx, 2.0*dy, 2.0*dz])
        H  = np.zeros((3, self.n))
        H[0:3, 6:9] = np.eye(3)
        if gate_roll:
            H[0, 6] = 0.0  # δroll row — Madgwick roll biased by centripetal acceleration
        self._update(nu, H, np.eye(3) * r_ori,
                     self._CHI2_95[3] if gate else None)

    def update_gravity(self, a_sf: np.ndarray, g: float = 9.81, r_acc: float = 0.5,
                       gate_roll: bool = False):
        """Soft gravity-alignment update using the accelerometer as a tilt sensor.

        Observable states: roll and pitch.  Yaw is unobservable from gravity alone.
        Jacobian: H[:,6:9] = [g_body]×  (right-perturbation ESKF convention).

        No hard gate.  Instead, noise is scaled by (|a|/g)² so coordinated
        high-g turns still correct tilt (direction is valid) but with reduced
        authority.  Free-fall (|a| < 0.1 g) is still skipped — no signal.

        gate_roll: if True, zero out the δroll column of H.  During banked turns
        centripetal acceleration contaminates the gravity reference in the body-y
        axis, making the ESKF think it is less banked than it actually is.
        Pitch remains observable (lift vector still has a genuine pitch component).
        """
        a_norm = np.linalg.norm(a_sf)
        if a_norm < 0.1 * g:
            return  # free-fall — no tilt information
        # Scale noise by load factor squared: high-g (banked turns) → less trust.
        # Clamp lf to >= 1 so sub-1g (dives, forward acceleration) does NOT give
        # extra authority — accelerometer is unreliable in both directions from 1g.
        load_factor = max(a_norm / g, 1.0)
        r_scaled = r_acc * (load_factor * load_factor)
        # Predicted specific force in body frame: C^T·[0,0,g]
        g_body = self._dcm(self.q).T @ np.array([0.0, 0.0, g])
        # Bias-corrected specific force: subtract estimated accelerometer bias so the
        # gravity measurement observes tilt, not a mixture of tilt and ba offset.
        a_corr = a_sf - self.ba
        H = np.zeros((3, self.n))
        H[:, 6:9]  = self._skew(g_body)
        H[:, 9:12] = -np.eye(3)   # ∂/∂δba — gravity measurement directly observes accel bias
        if gate_roll:
            H[:, 6] = 0.0  # δroll not observable — centripetal acceleration contaminates gravity
        self._update(a_corr - g_body, H,
                     np.eye(3) * (r_scaled * r_scaled),
                     self._CHI2_95[3])

    def update_coordinated_turn(
        self,
        phi: float, theta: float,
        q_raw: float, r_raw: float,
        V: float,
        g: float = 9.81,
        r_coor: float = 0.02,
    ) -> bool:
        """Yaw gyro-bias update from the coordinated-turn constraint.

        In steady coordinated banked flight the aerodynamic turn rate equals
        the kinematic yaw rate from the gyro:
          ψ̇_aero = (g/V)·tan(φ)
          ψ̇_imu  = (q·sinφ + r·cosφ) / cosθ   [Euler ZYX kinematics, bias-corrected]
        where q = body pitch rate (ω_y), r = body yaw rate (ω_z).
        Innovation: δz = ψ̇_aero − ψ̇_imu  (≈0 in coordinated flight)

        Directly observable: gyro bias b_q and b_r.  Yaw angle is NOT observable
        through this path (∂/∂δψ = 0), but yaw DRIFT is reduced because the
        accumulated bias b_r is corrected at every banked segment.

        Jacobian H (1 × n) — roll and pitch columns intentionally zeroed:
          H[δroll]  = 0  (exact term nonzero but causes roll phase lag; roll owned by gravity/Madgwick)
          H[δpitch] = 0  (exact term nonzero but biases pitch; pitch owned by gravity/Madgwick)
          H[δyaw]   = 0
          H[δbg_q]  = sinφ / cosθ   (pitch-axis gyro bias — enters ψ̇ through q·sinφ)
          H[δbg_r]  = cosφ / cosθ   (yaw-axis gyro bias — primary yaw-drift source)
        where q_c = q_raw − bg[1], r_c = r_raw − bg[2].

        Gate: χ²₁,₀.₉₅ = 3.841 (1 DOF scalar measurement).
        """
        cos_phi   = math.cos(phi)
        sin_phi   = math.sin(phi)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        if abs(cos_theta) < 1e-3 or abs(V) < 1e-3:
            return False

        q_c = q_raw - self.bg[1]
        r_c = r_raw - self.bg[2]

        psi_dot_aero = -(g / V) * math.tan(phi)
        psi_dot_imu  = (q_c * sin_phi + r_c * cos_phi) / cos_theta
        innov = np.array([psi_dot_aero - psi_dot_imu])

        H = np.zeros((1, self.n))
        # H[0, 6] = 0 — roll NOT observed here.  The exact Jacobian is nonzero but
        # applies a roll correction at every banked segment, opposing the gyro's roll
        # rate integration during bank entry/exit.  This causes phase lag.
        # Roll is well-observed by gravity and raw-IMU update_roll_pitch.
        # H[0, 7] = 0 — pitch NOT observed here.  Fires at every banked-while-climbing
        # segment, systematically biasing ESKF pitch.
        # H[0, 8] = 0  — yaw angle unobservable from this measurement
        H[0, self.bg_idx + 1] = sin_phi / cos_theta   # ∂/∂b_q (pitch-axis bias)
        H[0, self.bg_idx + 2] = cos_phi / cos_theta   # ∂/∂b_r (yaw-axis bias)

        return self._update(innov, H, np.array([[r_coor]]), self._CHI2_95[1])

    # ------------------------------------------------------------------
    # ESKF read-only accessors
    # ------------------------------------------------------------------

    @property
    def position(self):   return self.p.copy()
    @property
    def velocity(self):   return self.v.copy()
    @property
    def quaternion(self): return self.q.copy()


class MaridOdomPublisher(Node):
    """ROS-facing odometry node.

    This class intentionally keeps the architecture flat: callbacks cache sensor
    state and apply ESKF measurement updates, while publish_odom resolves source
    priority and writes /marid/odom.  The comments below group the long state list
    by physical quantity so the owner of each piece is visible at the point it is
    declared.
    """

    def __init__(self):
        super().__init__("marid_odom_node")

        # ------------------------------------------------------------------
        # Parameters: core publication, limits, and frame IDs
        # ------------------------------------------------------------------
        self.declare_parameter("imu_outputs_specific_force", True)
        self.declare_parameter("gravity", 9.81)
        self.declare_parameter("publish_rate", 50.0)

        self.declare_parameter("max_velocity", 200.0)          # m/s safety limit
        self.declare_parameter("max_acceleration", 350.0)      # m/s^2 safety limit

        # Damping interpreted as "fraction kept per second" (0.95 => keep 95% after 1 second)
        self.declare_parameter("velocity_decay_rate", 0.95)

        self.declare_parameter("base_position_variance", 0.01)
        self.declare_parameter("base_velocity_variance", 0.1)
        self.declare_parameter("variance_growth_rate", 0.001)  # per second

        self.declare_parameter("child_frame_id", "base_link_front_imu")
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("initial_ground_yaw", 0.0)       # rad — launch/spawn heading prior, used only to seed ground yaw

        # ------------------------------------------------------------------
        # Parameters: IMU bias calibration
        # ------------------------------------------------------------------
        self.declare_parameter("calibration_required", 100)    # samples
        self.declare_parameter("calibration_max_dt", 0.05)     # ignore large dt during calibration

        # ------------------------------------------------------------------
        # Parameters: altitude sensors
        # ------------------------------------------------------------------
        self.declare_parameter("use_barometer", True)
        self.declare_parameter("baro_weight", 0.98)            # fraction of baro in fused z (0..1)

        # Sonar altitude anchor
        self.declare_parameter("use_sonar", True)
        self.declare_parameter("sonar_range_topic", "/sonar/range")
        # Weight to anchor z_imu_ toward sonar AGL when sonar is valid
        self.declare_parameter("sonar_weight", 0.95)
        self.declare_parameter("max_sonar_altitude", 5.0)

        # ------------------------------------------------------------------
        # Parameters: FAST-LIO position/yaw/velocity reference
        # ------------------------------------------------------------------
        self.declare_parameter("use_fastlio", True)
        self.declare_parameter("fastlio_topic", "/Odometry")
        # How much FAST-LIO nudges position each update (0=ignore, 1=hard reset).
        # 0.7 works well in feature-rich environments (poles/walls); lower in open fields.
        self.declare_parameter("fastlio_position_weight", 0.7)
        # Blend weight for FAST-LIO yaw correction (0=ignore, 1=hard override).
        # Uses FAST-LIO orientation yaw to correct heading drift when scan-matching is reliable.
        self.declare_parameter("fastlio_yaw_weight", 0.3)
        # Altitude above which scan matching degrades (open sky, no nearby features).
        # FAST-LIO is treated as stale above this and thrust DR takes over.
        self.declare_parameter("max_fastlio_altitude", 30.0)  # m AGL
        # EMA coefficient for FAST-LIO finite-diff velocity (α ∈ (0,1]).
        # Raw finite-diff at ~10 Hz introduces ~0.5 m/s noise; α=0.3 gives τ ≈ 0.23 s
        # smoothing without meaningful lag at cruise.  α=1.0 disables smoothing.
        self.declare_parameter("fastlio_velocity_alpha", 0.3)

        # ------------------------------------------------------------------
        # Parameters: velocity sources and guards
        # ------------------------------------------------------------------
        # IMU X/Y integration toggle.
        # When False, IMU accelerations are NOT integrated into vx/vy.
        # Velocity then comes only from airspeed, optical flow, or FAST-LIO seeding,
        # and decays toward zero between updates via velocity_decay_rate.
        # Recommended when FAST-LIO is the primary position source: disabling IMU X/Y
        # integration eliminates double-integration drift that fights the FAST-LIO correction
        # and causes the characteristic sawtooth drift pattern.
        self.declare_parameter("use_imu_xy", True)

        # Airspeed complementary filter
        self.declare_parameter("use_airspeed", True)
        self.declare_parameter("airspeed_weight", 0.8)
        self.declare_parameter("min_airspeed_for_fusion", 0.5)   # m/s — only skip when truly static
        self.declare_parameter("wind_topic", "/wind/velocity")
        # Disable wind correction by default: in simulation there is no actual wind, and the
        # wind estimator uses EKF/Gazebo ground truth which leaks into GPS-denied estimation.
        # Enable only for real flight where wind is genuine and estimated independently.
        self.declare_parameter("use_wind_correction", False)
        # Airspeed sensor staleness timeout: if no message arrives within this window the
        # last reading is discarded and the next priority source is used instead.
        self.declare_parameter("airspeed_timeout", 1.0)    # seconds

        # Optical flow fusion
        self.declare_parameter("use_optical_flow", True)
        self.declare_parameter("optical_flow_topic", "/optical_flow/velocity")
        # Weight for OF velocity vs IMU velocity (0=ignore OF, 1=hard OF override)
        self.declare_parameter("optical_flow_weight", 0.7)
        # OF is suppressed above this altitude (sonar valid range)
        self.declare_parameter("max_of_altitude", 4.5)
        # OF is suppressed below this altitude (too close to ground for valid flow)
        self.declare_parameter("min_of_altitude", 0.3)
        # Maximum body pitch angle (degrees) at which OF vx is still trusted.
        # Above this threshold the downward camera tilts enough to mix pitch
        # rotation into the flow reading, corrupting the forward velocity estimate.
        self.declare_parameter("max_of_pitch_deg", 5.0)
        self.declare_parameter("of_timeout", 0.5)          # seconds

        # Forward camera optical flow fusion
        # Provides lateral (vy) and vertical (vz) body-frame velocity in flight.
        # Complements downward OF (which covers vx+vy below max_of_altitude).
        self.declare_parameter("use_forward_camera", True)
        self.declare_parameter("forward_camera_topic", "/forward_camera/velocity")
        # Blend weight for forward camera vy (0=ignore, 1=hard override)
        self.declare_parameter("forward_camera_weight", 0.5)
        # Only fuse above this altitude — below it the downward camera is preferred
        self.declare_parameter("min_forward_camera_altitude", 1.0)
        self.declare_parameter("fwd_cam_timeout", 0.5)     # seconds
        self.declare_parameter("max_body_velocity_pitch_deg", 32.5)
        self.declare_parameter("body_velocity_yaw_jump_deg", 15.0)
        self.declare_parameter("body_velocity_yaw_guard_s", 2.0)

        # Wheel odometry fusion (ground taxiing)
        self.declare_parameter("use_wheel_odom", True)
        self.declare_parameter("wheel_odom_topic", "/wheel/odometry")
        # Sonar AGL below which wheel odometry owns XY position and velocity.
        # Must match (or be slightly larger than) ground_threshold in wheel_odometry.py.
        self.declare_parameter("ground_threshold", 0.30)

        # ------------------------------------------------------------------
        # Parameters: thrust, drag, aero dead reckoning, and fuel mass
        # ------------------------------------------------------------------
        # Thrust dead reckoning + online drag calibration via FAST-LIO
        self.declare_parameter("use_thrust_dr", True)
        self.declare_parameter("thrust_topic", "/model/marid/joint/thruster_center_joint/cmd_thrust")
        self.declare_parameter("drone_mass", 190.0)           # kg
        self.declare_parameter("drag_coeff_init", 0.5)        # k_d initial guess
        self.declare_parameter("induced_drag_frac", 0.25)     # fraction of drag that is induced (bank-corrected)
        self.declare_parameter("thrust_dr_weight", 0.7)       # blend weight for vx when FAST-LIO stale
        self.declare_parameter("rls_forgetting_factor", 0.99)
        self.declare_parameter("min_speed_for_drag_id", 2.0)  # m/s — skip RLS at near-zero speed
        self.declare_parameter("air_density", 1.225)          # kg/m³ ISA sea-level default; updated dynamically from baro
        # If no thrust message arrives within this window, assume glide (thrust = 0).
        # Physics integration keeps running — drag still decelerates the estimate.
        self.declare_parameter("thrust_timeout", 0.5)         # seconds

        # H2 mass tracking — updates drone_mass_ as fuel is consumed.
        # Primary estimator: thrust-based SFC integration (always available).
        # Optional correction: ideal gas law from tank pressure + temperature sensors.
        self.declare_parameter("track_h2_mass", True)
        self.declare_parameter("drone_empty_mass", 160.0)       # kg — structural mass without fuel
        self.declare_parameter("h2_initial_mass", 30.0)         # kg — full tank at takeoff
        # Thrust-specific fuel consumption: kg of H₂ per Newton per second.
        # Approximate for a hydrogen fuel cell + electric motor at ~55% system efficiency.
        self.declare_parameter("specific_fuel_consumption", 5.0e-6)  # kg/(N·s)
        self.declare_parameter("tank_pressure_topic", "/tank/pressure")      # std_msgs/Float64, Pa
        self.declare_parameter("tank_temperature_topic", "/tank/temperature") # std_msgs/Float64, K
        self.declare_parameter("tank_volume", 0.5)              # m³ — usable gas-phase volume
        # How strongly tank sensor reading corrects the integrated estimate (0=ignore, 1=hard reset).
        self.declare_parameter("tank_sensor_weight", 0.01)

        # ------------------------------------------------------------------
        # Parameters: ESKF covariance, attitude, and coordinated-turn yaw logic
        # ------------------------------------------------------------------
        self.declare_parameter("use_eskf", True)
        self.declare_parameter("eskf_mode", "physics")     # "imu" or "physics"
        self.declare_parameter("eskf_r_pos",      0.05)    # FAST-LIO position noise (m²) — high-quality sim lidar
        self.declare_parameter("eskf_r_vel",          0.2)   # velocity sensor noise (m²/s²) — OF, fwd camera, wheel odom
        self.declare_parameter("eskf_r_vel_airspeed", 0.05)  # forward airspeed noise (m²/s²) — clean sim pitot
        self.declare_parameter("eskf_r_vel_airspeed_lat", 0.25)  # zero-sideslip noise (m²/s²) — loose lateral constraint
        self.declare_parameter("eskf_vel_yaw_jump_inflate", 50.0)  # P_v inflation (m²/s²) added on large yaw correction
        self.declare_parameter("eskf_r_alt_sonar", 0.005)  # sonar altitude noise (m²) — clean sim range
        self.declare_parameter("eskf_r_alt_baro",  0.05)   # baro altitude noise (m²) — clean sim pressure
        self.declare_parameter("eskf_r_yaw",      0.0475)  # heading noise (rad²) — keep conservative: scan-match yaw can jump
        self.declare_parameter("eskf_r_acc",      2.0)     # gravity-alignment noise (m/s²) — keep soft: acceleration is not pure gravity in flight
        self.declare_parameter("eskf_q_vel",      2.0)     # process noise — velocity
        self.declare_parameter("eskf_q_att",      0.01)    # process noise — attitude
        self.declare_parameter("eskf_q_ba",       1e-4)    # process noise — accel bias
        self.declare_parameter("eskf_p_bg",       1e-6)    # initial gyro-bias variance (rad²/s²) — optical-grade gyro
        self.declare_parameter("eskf_q_bg",       1e-8)    # process noise — gyro bias (rad²/s³); optical-grade bias stability
        self.declare_parameter("eskf_r_ori",      0.03)    # Madgwick orientation noise (rad²) — cautious: source can be dynamically biased
        self.declare_parameter("eskf_r_raw_tilt", 0.002)   # raw /imu roll/pitch noise (rad²) — high-quality sim attitude
        self.declare_parameter("eskf_r_yaw_madgwick_fallback", 0.05) # rad² — continuous Madgwick yaw stabilizer

        # Coordinated-turn yaw gyro-bias estimator
        self.declare_parameter("use_coordinated_turn",           True)
        self.declare_parameter("eskf_r_coordinated_turn",        0.05)   # rad²/s² — measurement noise (raised from 0.02: roll errors bias tan(φ) → wrong bg_r when trust is too high)
        self.declare_parameter("coordinated_turn_min_speed_mps", 5.0)    # m/s — below stall: no aero turn
        self.declare_parameter("coordinated_turn_min_bank_deg",  6.5)    # degrees — yaw bias is observable even in shallow banks with accurate roll
        self.declare_parameter("gravity_gate_roll_min_bank_deg", 2.0)    # degrees — gate accel-as-gravity roll correction during even shallow banks
        self.declare_parameter("gravity_gate_ax_mps2",           6.5)    # m/s² — gate roll when |a_x| exceeds this (thrust/dive contaminates gravity)
        self.declare_parameter("coordinated_turn_max_bank_deg",  50.0)   # degrees — extreme bank: likely uncoordinated
        self.declare_parameter("coordinated_turn_min_agl_m",     3.0)    # metres — not active on ground
        # Optional process-model assist: blend the gyro body-z rate toward the
        # body-rate implied by ψ̇=(g/V)tan(φ).  This is stronger than the bias-only
        # coordinated-turn update, but still gated by the same bank/speed/AGL checks.
        self.declare_parameter("use_coordinated_turn_yaw_rate_input", True)
        self.declare_parameter("coordinated_turn_yaw_rate_weight",    0.185)       # higher = more bank/coordinated-turn authority on yaw-rate propagation

        # ------------------------------------------------------------------
        # Parameters: absolute heading sensors
        # ------------------------------------------------------------------
        # Sun sensor fusion
        self.declare_parameter("use_sun_sensor",            True)
        self.declare_parameter("sun_sensor_topic",          "/sun_sensor/sun_vector_body")
        self.declare_parameter("sun_azimuth_topic",         "/sun_sensor/sun_azimuth_enu_rad")
        self.declare_parameter("sun_elevation_topic",       "/sun_sensor/sun_elevation_deg")
        self.declare_parameter("eskf_r_yaw_sun",            0.0025)   # rad² — strong absolute yaw, but avoid sun-only yaw yanks
        self.declare_parameter("sun_elevation_min_deg",     10.0)   # gate — below → invalid
        self.declare_parameter("sun_sensor_timeout",        1.0)    # seconds
        self.declare_parameter("sun_yaw_weight",            0.3)    # non-ESKF blend weight

        # ------------------------------------------------------------------
        # Read parameters: core publication, limits, and frame IDs
        # ------------------------------------------------------------------
        self.accel_is_sf_ = bool(self.get_parameter("imu_outputs_specific_force").value)
        self.g_ = float(self.get_parameter("gravity").value)
        self.publish_rate_ = float(self.get_parameter("publish_rate").value)

        self.max_velocity_ = float(self.get_parameter("max_velocity").value)
        self.max_acceleration_ = float(self.get_parameter("max_acceleration").value)

        self.velocity_decay_per_sec_ = float(self.get_parameter("velocity_decay_rate").value)

        self.base_pos_var_ = float(self.get_parameter("base_position_variance").value)
        self.base_vel_var_ = float(self.get_parameter("base_velocity_variance").value)
        self.var_growth_rate_ = float(self.get_parameter("variance_growth_rate").value)

        self.child_frame_id_ = str(self.get_parameter("child_frame_id").value)
        self.publish_tf_ = bool(self.get_parameter("publish_tf").value)
        self.initial_ground_yaw_ = float(self.get_parameter("initial_ground_yaw").value)

        self.calibration_required_ = int(self.get_parameter("calibration_required").value)
        self.calibration_max_dt_ = float(self.get_parameter("calibration_max_dt").value)

        # ------------------------------------------------------------------
        # Read parameters: altitude sensors
        # ------------------------------------------------------------------
        self.use_barometer_ = bool(self.get_parameter("use_barometer").value)
        self.baro_weight_ = float(self.get_parameter("baro_weight").value)
        self.baro_weight_ = max(0.0, min(1.0, self.baro_weight_))
        self.imu_weight_ = 1.0 - self.baro_weight_

        self.use_fastlio_ = bool(self.get_parameter("use_fastlio").value)
        self.fastlio_topic_ = str(self.get_parameter("fastlio_topic").value)
        self.fastlio_pos_weight_ = max(0.0, min(1.0, float(
            self.get_parameter("fastlio_position_weight").value)))
        self.fastlio_yaw_weight_ = max(0.0, min(1.0, float(
            self.get_parameter("fastlio_yaw_weight").value)))
        self.max_fastlio_alt_    = float(self.get_parameter("max_fastlio_altitude").value)
        self.fastlio_vel_alpha_  = max(0.0, min(1.0, float(
            self.get_parameter("fastlio_velocity_alpha").value)))

        self.use_imu_xy_ = bool(self.get_parameter("use_imu_xy").value)

        # ------------------------------------------------------------------
        # Read parameters: velocity sources and guards
        # ------------------------------------------------------------------
        self.use_airspeed_ = bool(self.get_parameter("use_airspeed").value)
        self.airspeed_weight_ = max(0.0, min(1.0, float(self.get_parameter("airspeed_weight").value)))
        self.min_airspeed_ = float(self.get_parameter("min_airspeed_for_fusion").value)
        self.wind_topic_ = str(self.get_parameter("wind_topic").value)
        self.use_wind_correction_ = bool(self.get_parameter("use_wind_correction").value)
        self.airspeed_timeout_   = float(self.get_parameter("airspeed_timeout").value)

        self.use_optical_flow_  = bool(self.get_parameter("use_optical_flow").value)
        self.of_topic_          = str(self.get_parameter("optical_flow_topic").value)
        self.of_weight_         = max(0.0, min(1.0, float(self.get_parameter("optical_flow_weight").value)))
        self.max_of_altitude_   = float(self.get_parameter("max_of_altitude").value)
        self.min_of_altitude_   = float(self.get_parameter("min_of_altitude").value)
        self.max_of_pitch_rad_  = math.radians(float(self.get_parameter("max_of_pitch_deg").value))
        self.of_timeout_        = float(self.get_parameter("of_timeout").value)

        self.use_sonar_         = bool(self.get_parameter("use_sonar").value)
        self.sonar_range_topic_ = str(self.get_parameter("sonar_range_topic").value)
        self.sonar_weight_      = max(0.0, min(1.0, float(self.get_parameter("sonar_weight").value)))
        self.max_sonar_alt_     = float(self.get_parameter("max_sonar_altitude").value)

        self.use_forward_camera_      = bool(self.get_parameter("use_forward_camera").value)
        self.fwd_cam_topic_           = str(self.get_parameter("forward_camera_topic").value)
        self.fwd_cam_weight_          = max(0.0, min(1.0, float(
            self.get_parameter("forward_camera_weight").value)))
        self.min_fwd_cam_altitude_    = float(self.get_parameter("min_forward_camera_altitude").value)
        self.fwd_cam_timeout_         = float(self.get_parameter("fwd_cam_timeout").value)
        self.max_body_velocity_pitch_rad_ = math.radians(float(
            self.get_parameter("max_body_velocity_pitch_deg").value))
        self.body_velocity_yaw_jump_rad_ = math.radians(float(
            self.get_parameter("body_velocity_yaw_jump_deg").value))
        self.body_velocity_yaw_guard_s_ = float(
            self.get_parameter("body_velocity_yaw_guard_s").value)

        self.use_wheel_odom_    = bool(self.get_parameter("use_wheel_odom").value)
        self.wheel_odom_topic_  = str(self.get_parameter("wheel_odom_topic").value)
        self.ground_threshold_  = float(self.get_parameter("ground_threshold").value)

        # ------------------------------------------------------------------
        # Read parameters: thrust, drag, aero dead reckoning, and fuel mass
        # ------------------------------------------------------------------
        self.use_thrust_dr_     = bool(self.get_parameter("use_thrust_dr").value)
        self.thrust_topic_      = str(self.get_parameter("thrust_topic").value)
        self.drone_mass_        = float(self.get_parameter("drone_mass").value)
        self.drag_coeff_        = float(self.get_parameter("drag_coeff_init").value)
        self.induced_drag_frac_ = float(self.get_parameter("induced_drag_frac").value)
        self.thrust_dr_weight_  = max(0.0, min(1.0, float(self.get_parameter("thrust_dr_weight").value)))
        self.rls_lambda_        = float(self.get_parameter("rls_forgetting_factor").value)
        self.min_speed_drag_id_ = float(self.get_parameter("min_speed_for_drag_id").value)
        self.air_density_       = float(self.get_parameter("air_density").value)
        self.thrust_timeout_    = float(self.get_parameter("thrust_timeout").value)

        self.track_h2_mass_    = bool(self.get_parameter("track_h2_mass").value)
        self.drone_empty_mass_ = float(self.get_parameter("drone_empty_mass").value)
        self.h2_initial_mass_  = float(self.get_parameter("h2_initial_mass").value)
        self.sfc_              = float(self.get_parameter("specific_fuel_consumption").value)
        self.tank_volume_      = float(self.get_parameter("tank_volume").value)
        self.tank_sensor_w_    = float(self.get_parameter("tank_sensor_weight").value)
        self.R_H2_             = 8314.0 / 2.016   # J/(kg·K) specific gas constant for H₂

        # ------------------------------------------------------------------
        # Read parameters: ESKF covariance, attitude, and coordinated-turn yaw logic
        # ------------------------------------------------------------------
        self.use_eskf_          = bool(self.get_parameter("use_eskf").value)
        self.eskf_mode_         = str(self.get_parameter("eskf_mode").value)
        self.eskf_r_pos_        = float(self.get_parameter("eskf_r_pos").value)
        self.eskf_r_vel_          = float(self.get_parameter("eskf_r_vel").value)
        self.eskf_r_vel_airspeed_     = float(self.get_parameter("eskf_r_vel_airspeed").value)
        self.eskf_r_vel_airspeed_lat_ = float(self.get_parameter("eskf_r_vel_airspeed_lat").value)
        self.eskf_vel_jump_inflate_   = float(self.get_parameter("eskf_vel_yaw_jump_inflate").value)
        self.eskf_r_alt_sonar_  = float(self.get_parameter("eskf_r_alt_sonar").value)
        self.eskf_r_alt_baro_   = float(self.get_parameter("eskf_r_alt_baro").value)
        self.eskf_r_yaw_        = float(self.get_parameter("eskf_r_yaw").value)
        self.eskf_r_ori_        = float(self.get_parameter("eskf_r_ori").value)
        self.eskf_r_raw_tilt_   = float(self.get_parameter("eskf_r_raw_tilt").value)
        self.eskf_r_yaw_madgwick_fallback_ = float(self.get_parameter("eskf_r_yaw_madgwick_fallback").value)
        self.eskf_r_acc_        = float(self.get_parameter("eskf_r_acc").value)

        self.use_coordinated_turn_    = bool(self.get_parameter("use_coordinated_turn").value)
        self.eskf_r_coor_turn_        = float(self.get_parameter("eskf_r_coordinated_turn").value)
        self.coor_turn_min_spd_       = float(self.get_parameter("coordinated_turn_min_speed_mps").value)
        self.coor_turn_min_bank_rad_  = math.radians(float(self.get_parameter("coordinated_turn_min_bank_deg").value))
        self.gravity_gate_roll_min_bank_rad_ = math.radians(float(self.get_parameter("gravity_gate_roll_min_bank_deg").value))
        self.gravity_gate_ax_mps2_    = float(self.get_parameter("gravity_gate_ax_mps2").value)
        self.coor_turn_max_bank_rad_  = math.radians(float(self.get_parameter("coordinated_turn_max_bank_deg").value))
        self.coor_turn_min_agl_       = float(self.get_parameter("coordinated_turn_min_agl_m").value)
        self.use_coor_turn_yaw_rate_input_ = bool(
            self.get_parameter("use_coordinated_turn_yaw_rate_input").value)
        self.coor_turn_yaw_rate_weight_ = max(0.0, min(1.0, float(
            self.get_parameter("coordinated_turn_yaw_rate_weight").value)))

        # ------------------------------------------------------------------
        # Read parameters: absolute heading sensors
        # ------------------------------------------------------------------
        # Sun sensor
        self.use_sun_sensor_       = bool(self.get_parameter("use_sun_sensor").value)
        self.sun_sensor_topic_     = str(self.get_parameter("sun_sensor_topic").value)
        self.sun_azimuth_topic_    = str(self.get_parameter("sun_azimuth_topic").value)
        self.sun_elevation_topic_  = str(self.get_parameter("sun_elevation_topic").value)
        self.eskf_r_yaw_sun_       = float(self.get_parameter("eskf_r_yaw_sun").value)
        self.sun_el_min_deg_       = float(self.get_parameter("sun_elevation_min_deg").value)
        self.sun_sensor_timeout_   = float(self.get_parameter("sun_sensor_timeout").value)
        self.sun_yaw_weight_       = float(self.get_parameter("sun_yaw_weight").value)

        # ------------------------------------------------------------------
        # Runtime objects: TF broadcaster/listener
        # ------------------------------------------------------------------
        self.tf_broadcaster_ = TransformBroadcaster(self)
        self.tf_buffer_ = Buffer()
        self.tf_listener_ = TransformListener(self.tf_buffer_, self)

        # ------------------------------------------------------------------
        # Runtime objects: ESKF and control-surface state
        # ------------------------------------------------------------------
        self.eskf_seeded_    = False
        self.eskf_imu_stamp_ = None
        # Surface angles for physics-mode predict (radians, from joint states)
        self.delta_lw_ = 0.0
        self.delta_rw_ = 0.0
        self.delta_tl_ = 0.0
        self.delta_tr_ = 0.0
        self.eskf_ = (
            MARIDESKF(
                mode=self.eskf_mode_,
                g=self.g_,
                p_att=1.0,   # large initial attitude uncertainty → chi² gate passes up to ~180° at spawn
                p_bg =float(self.get_parameter("eskf_p_bg").value),
                q_vel=float(self.get_parameter("eskf_q_vel").value),
                q_att=float(self.get_parameter("eskf_q_att").value),
                q_ba =float(self.get_parameter("eskf_q_ba").value),
                q_bg =float(self.get_parameter("eskf_q_bg").value),
                kd_init           =self.drag_coeff_,
                mh2_init          =self.h2_initial_mass_,
                mass_empty        =self.drone_empty_mass_,
                air_density       =self.air_density_,
                sfc               =self.sfc_,
                induced_drag_frac =self.induced_drag_frac_,
            ) if self.use_eskf_ else None
        )

        # ------------------------------------------------------------------
        # State ownership map
        # ------------------------------------------------------------------
        # Pose XY:
        #   ESKF owns x/y after seeding. Before that, FAST-LIO or wheel odom can
        #   anchor x_/y_; thrust DR only dead-reckons between corrections.
        self.x_ = 0.0
        self.y_ = 0.0

        # Pose Z and vertical velocity:
        #   z_imu_ integrates vertical motion; z_fused_ is the published altitude
        #   owner after sonar/baro priority.  ESKF vertical state is aligned back
        #   to these values at publish time so lift-model errors cannot own z.
        self.z_imu_ = 0.0
        self.vz_imu_ = 0.0
        self.vz_baro_ = 0.0        # baro-derived vz fallback (active above FAST-LIO gate)
        self.last_baro_z_for_vz_ = None
        self.last_baro_stamp_for_vz_ = None

        # Horizontal helper velocities:
        #   vx_/vy_ are finite-diff FAST-LIO/complementary helper velocities.
        #   When ESKF is seeded, publish_odom publishes ESKF velocity instead.
        self.vx_ = 0.0
        self.vy_ = 0.0

        # Published fused z (sonar > baro > IMU)
        self.z_fused_ = 0.0

        # Angular velocity:
        #   wx_/wy_/wz_ are pass-through diagnostics from /imu_ekf.
        #   yaw_rate_assist_ is the coordinated-turn process input delta applied
        #   to raw gyro z before ESKF prediction.
        self.wx_ = 0.0
        self.wy_ = 0.0
        self.wz_ = 0.0
        self.yaw_rate_assist_ = 0.0

        # Orientation:
        #   raw /imu owns roll/pitch in simulation; /imu_ekf/Madgwick owns yaw
        #   continuity; ESKF orientation is published only when it agrees with
        #   that hybrid attitude reference.
        self.qx_, self.qy_, self.qz_, self.qw_ = 0.0, 0.0, 0.0, 1.0
        self.madg_qx_, self.madg_qy_, self.madg_qz_, self.madg_qw_ = 0.0, 0.0, 0.0, 1.0
        # Raw /imu orientation is the high-quality simulated IMU attitude.  We use
        # its roll/pitch as tilt measurements while keeping /imu_ekf for yaw.
        self.raw_qx_, self.raw_qy_, self.raw_qz_, self.raw_qw_ = 0.0, 0.0, 0.0, 1.0
        self.raw_roll_ = 0.0
        self.raw_pitch_ = 0.0
        self.raw_yaw_ = 0.0
        self.raw_imu_orientation_ready_ = False

        # Startup bias calibration:
        #   average of measured acceleration minus expected stationary specific force.
        self.accel_bias_ = np.zeros(3, dtype=float)
        self.bias_samples_ = []
        self.calibrating_ = True
        self.calibration_samples_ = 0

        # Runtime counters and diagnostics
        self.integration_count_ = 0
        self.dropped_samples_ = 0
        self.warning_count_ = 0

        # Altitude source state: barometer
        self.baro_z_ = 0.0
        self.baro_ready_ = False

        # Pose/velocity source state: FAST-LIO
        self.fastlio_ready_ = False
        self.fastlio_vel_initialized_ = False  # True after first valid finite-diff velocity
        self.last_fastlio_stamp_ = None      # rclpy.Time of last FAST-LIO message
        self.last_fastlio_x_ = 0.0          # previous FAST-LIO position for finite-diff velocity
        self.last_fastlio_y_ = 0.0
        self.last_fastlio_z_ = 0.0
        self.fastlio_timeout_ = 0.5         # seconds — declare FAST-LIO stale after this
        # Rotation matrix from camera_init to odom.
        # camera_init is FAST-LIO's inertial frame, aligned with the drone's body frame
        # at the moment of the first LiDAR scan. Its orientation relative to odom equals
        # the drone's initial IMU attitude — captured once and held fixed.
        # None until the first FAST-LIO message sets it.
        self.camera_init_to_odom_R_ = None

        self.last_a_norm_   = 9.81   # cached |a_sf| from IMU callback for Madgwick R scaling
        self.last_phi_      = 0.0    # cached ESKF bank angle (rad) — used for coordinated-turn gate
        self.last_phi_tilt_ref_ = 0.0  # cached raw-/imu tilt bank angle for gravity roll gate
        self.last_phi_madg_ = 0.0      # cached Madgwick bank angle — diagnostic/fallback reference

        # Velocity source state: pitot / 5-hole probe.
        # Va_x corrects body-forward velocity; Va_y/Va_z correct body lateral/up.
        self.last_airspeed_ = 0.0
        self.airspeed_ready_ = False
        self.last_airspeed_stamp_ = None
        # 5-hole probe lateral (Va_y) and vertical (Va_z)
        self.last_airspeed_y_ = 0.0
        self.airspeed_y_ready_ = False
        self.last_airspeed_y_stamp_ = None
        self.last_airspeed_z_ = 0.0
        self.airspeed_z_ready_ = False
        self.last_airspeed_z_stamp_ = None
        self.wind_x_ = 0.0
        self.wind_y_ = 0.0
        self.wind_z_ = 0.0

        # Velocity source state: downward optical flow, body-frame vx/vy.
        self.of_vx_body_ = 0.0
        self.of_vy_body_ = 0.0
        self.of_ready_   = False
        self.of_seeded_  = False   # True after first OF activation seeds vx_/vy_
        self.last_of_stamp_   = None

        # Velocity source state: forward camera flow, body-frame vy/vz.
        self.fwd_cam_vy_body_ = 0.0
        self.fwd_cam_vz_body_ = 0.0
        self.fwd_cam_ready_   = False
        self.last_fwd_cam_stamp_ = None

        # Altitude/ground-contact source state: sonar.
        self.sonar_range_ = None
        self.sonar_ready_ = False
        self.last_sonar_stamp_ = None
        self.sonar_timeout_    = 1.0   # seconds — sonar declared stale after this

        # Debounce counter for on-ground detection used in vx source switching.
        # A single noisy sonar reading below ground_threshold_ must not snap vx to 0.
        # Physical location and wheel-odom position use raw _on_ground() for fast response;
        # only the vx source decision is debounced.
        self.on_ground_ticks_ = 0
        self.on_ground_debounce_ticks_ = 3   # ~60 ms at 50 Hz
        # Debounced ground flag: True only after 3 consecutive ground readings.
        # Starts False so wheel_odom cannot anchor ESKF until sonar confirms ground,
        # preventing Gazebo physics-settling from writing a bad initial position.
        self.on_ground_debounced_ = False
        # Airborne debounce: sonar noise can spike above ground_threshold_ for a
        # single tick while the drone is still on the runway (vibration, prop wash).
        # A single spike must not release the ground roll clamp or enable airspeed/
        # heading sensors — those changes persist in ESKF state permanently.
        # Require 5 consecutive not-on-ground sonar readings (~100 ms) before
        # enabling flight-mode sensors via is_airborne_.
        self.airborne_ticks_ = 0
        self.airborne_debounce_ticks_ = 5   # ~100 ms at 50 Hz
        self.is_airborne_ = False

        # Ground source state: wheel odometry owns taxi XY and body-forward speed.
        self.wheel_odom_ready_ = False
        self.wheel_x_    = 0.0
        self.wheel_y_    = 0.0
        self.wheel_vfwd_ = 0.0   # body-frame forward speed from wheel_odometry node

        # Physics source state: thrust dead reckoning and drag calibration.
        # thrust_dr_ready_ starts True: with thrust_N_=0 the model runs as a pure glide
        # from the first publish tick, applying drag and gravity even before the first
        # thrust command arrives.
        self.thrust_N_          = 0.0
        self.thrust_dr_vx_      = 0.0   # body-frame forward velocity estimate
        self.thrust_dr_ready_   = True
        self.last_thrust_stamp_ = None
        self.prev_on_ground_    = True  # tracks ground→air transition for liftoff seed
        self.liftoff_seeded_    = False  # one-shot: prevents re-seeding on sonar flicker
        self.ground_yaw_        = self.initial_ground_yaw_   # held heading for ground yaw-only attitude

        # Heading source state: sun sensor.
        self.sun_vector_body_   = None   # [sx, sy, sz] unit vector body frame
        self.sun_azimuth_enu_   = 0.0   # sun azimuth from East CCW (rad)
        self.sun_elevation_deg_ = -90.0
        self.sun_ready_         = False
        self.last_sun_stamp_    = None
        self.sun_valid_         = False  # True when elevation above threshold
        self.sun_yaw_           = 0.0   # last computed heading from sun (rad)

        # Publish fallback state:
        #   Used only when ESKF is unavailable or unseeded; ESKF velocity overwrites
        #   these at the end of publish_odom when active.
        # Last published velocities — used as hold/decay fallback so a single bad sensor
        # tick doesn't snap vx_pub to 0 (self.vx_ is always 0 when FAST-LIO is off).
        self.last_vx_pub_ = 0.0
        self.last_vy_pub_ = 0.0
        self.rls_P_            = 1.0   # RLS scalar covariance for drag_coeff_
        self.rls_vx_prev_      = None  # previous FAST-LIO body-frame vx for finite-diff
        self.rls_time_prev_    = None

        # H₂ mass tracking state
        self.h2_remaining_     = self.h2_initial_mass_
        self.tank_pressure_    = None  # Pa, from sensor (optional)
        self.tank_temperature_ = None  # K, from sensor (optional)

        # Time tracking
        self.start_time_ = self.get_clock().now()
        self.prev_imu_time_ = None
        self.prev_publish_time_ = None
        self.body_velocity_yaw_guard_until_ = None

        # ------------------------------------------------------------------
        # ROS interfaces: subscriptions, publishers, timer, services
        # ------------------------------------------------------------------
        self.pose_sub_ = self.create_subscription(
            Imu, "/imu_ekf", self.imu_callback, 10
        )

        if self.eskf_ is not None:
            # gz-sim IMU already includes gravity reaction force — subscribe to raw /imu
            # which is the unmodified sensor output at correct magnitude (~9.81 when level).
            self.eskf_imu_sub_ = self.create_subscription(
                Imu, "/imu", self.eskf_raw_imu_callback, 10
            )
            if self.eskf_mode_ == "physics":
                self.joint_state_sub_ = self.create_subscription(
                    JointState,
                    "/world/empty/model/marid/joint_state",
                    self.joint_state_callback,
                    10,
                )

        if self.use_barometer_:
            self.baro_sub_ = self.create_subscription(
                PoseWithCovarianceStamped,
                "/barometer/altitude",
                self.baro_callback,
                10,
            )

        if self.use_fastlio_:
            self.fastlio_sub_ = self.create_subscription(
                Odometry, self.fastlio_topic_, self.fastlio_callback, 10
            )

        if self.use_airspeed_:
            self.airspeed_sub_ = self.create_subscription(
                Float64, "/airspeed/velocity", self.airspeed_callback, 10
            )
            self.airspeed_y_sub_ = self.create_subscription(
                Float64, "/airspeed_y/velocity", self.airspeed_y_callback, 10
            )
            self.airspeed_z_sub_ = self.create_subscription(
                Float64, "/airspeed_z/velocity", self.airspeed_z_callback, 10
            )
            self.wind_sub_ = self.create_subscription(
                TwistStamped, self.wind_topic_, self.wind_callback, 10
            )

        if self.use_optical_flow_:
            self.of_sub_ = self.create_subscription(
                TwistWithCovarianceStamped, self.of_topic_, self.of_callback, 10
            )

        if self.use_sonar_:
            self.sonar_sub_ = self.create_subscription(
                Range, self.sonar_range_topic_, self.sonar_callback, 10
            )

        if self.use_forward_camera_:
            self.fwd_cam_sub_ = self.create_subscription(
                TwistWithCovarianceStamped, self.fwd_cam_topic_, self.fwd_cam_callback, 10
            )

        if self.use_wheel_odom_:
            self.wheel_odom_sub_ = self.create_subscription(
                Odometry, self.wheel_odom_topic_, self.wheel_odom_callback, 10
            )

        if self.use_thrust_dr_:
            self.thrust_sub_ = self.create_subscription(
                Float64, self.thrust_topic_, self.thrust_callback, 10
            )

        if self.track_h2_mass_:
            tank_p_topic = self.get_parameter("tank_pressure_topic").value
            tank_t_topic = self.get_parameter("tank_temperature_topic").value
            self.tank_p_sub_ = self.create_subscription(
                Float64, tank_p_topic, self.tank_pressure_callback, 10
            )
            self.tank_t_sub_ = self.create_subscription(
                Float64, tank_t_topic, self.tank_temperature_callback, 10
            )

        if self.use_sun_sensor_:
            self.sun_vec_sub_ = self.create_subscription(
                Vector3Stamped, self.sun_sensor_topic_, self.sun_vector_callback, 10
            )
            self.sun_az_sub_ = self.create_subscription(
                Float64, self.sun_azimuth_topic_, self.sun_azimuth_callback, 10
            )
            self.sun_el_sub_ = self.create_subscription(
                Float64, self.sun_elevation_topic_, self.sun_elevation_callback, 10
            )

        self.odom_pub_ = self.create_publisher(Odometry, "/marid/odom", 10)
        self.diag_pub_ = self.create_publisher(DiagnosticArray, "/diagnostics", 10)

        timer_period = 1.0 / max(1e-3, self.publish_rate_)
        self.timer_ = self.create_timer(timer_period, self.publish_odom)

        self.reset_srv_ = self.create_service(
            Empty, "~/reset_odometry", self.reset_odometry_callback
        )

        self.odom_msg_ = Odometry()
        self.odom_msg_.header.frame_id = "odom"
        self.odom_msg_.child_frame_id = self.child_frame_id_

        self.get_logger().info("IMU Odometry Node initialized (fixed math).")
        self.get_logger().info(
            f"IMU specific_force={self.accel_is_sf_}, gravity={self.g_}, "
            f"calibration_required={self.calibration_required_} samples"
        )

    # ------------------------------------------------------------------
    # Utility helpers: validation, quaternion math, attitude gates
    # ------------------------------------------------------------------
    @staticmethod
    def _is_finite(x: float) -> bool:
        return not (math.isnan(x) or math.isinf(x))

    def normalize_quaternion(self, qx, qy, qz, qw):
        """Return normalized quaternion; identity if invalid."""
        if not (self._is_finite(qx) and self._is_finite(qy) and self._is_finite(qz) and self._is_finite(qw)):
            return 0.0, 0.0, 0.0, 1.0
        norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if not self._is_finite(norm) or norm < 1e-9:
            return 0.0, 0.0, 0.0, 1.0
        return qx/norm, qy/norm, qz/norm, qw/norm

    @staticmethod
    def _rpy_from_quaternion(qx, qy, qz, qw):
        roll = math.atan2(
            2.0 * (qw * qx + qy * qz),
            1.0 - 2.0 * (qx * qx + qy * qy),
        )
        sinp = 2.0 * (qw * qy - qz * qx)
        pitch = math.asin(max(-1.0, min(1.0, sinp)))
        yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        return roll, pitch, yaw

    @staticmethod
    def _quaternion_from_rpy(roll, pitch, yaw):
        cr, cp, cy = math.cos(roll / 2.0), math.cos(pitch / 2.0), math.cos(yaw / 2.0)
        sr, sp, sy = math.sin(roll / 2.0), math.sin(pitch / 2.0), math.sin(yaw / 2.0)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        return qx, qy, qz, qw

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def _hybrid_raw_tilt_madgwick_yaw_q(self):
        """Return [x,y,z,w] using raw /imu roll/pitch and best startup yaw.

        On the runway, prefer the startup-aligned magnetometer yaw if available;
        otherwise keep the launch-provided runway yaw prior. Once airborne, the
        continuous heading updates own yaw.
        """
        _, _, yaw_madg = self._rpy_from_quaternion(
            self.madg_qx_, self.madg_qy_, self.madg_qz_, self.madg_qw_)
        if not self.is_airborne_:
            yaw_ref = self.initial_ground_yaw_
        else:
            yaw_ref = yaw_madg
        if self.raw_imu_orientation_ready_:
            roll = self.raw_roll_
            pitch = self.raw_pitch_
        else:
            roll, pitch, _ = self._rpy_from_quaternion(
                self.madg_qx_, self.madg_qy_, self.madg_qz_, self.madg_qw_)
        qx, qy, qz, qw = self._quaternion_from_rpy(roll, pitch, yaw_ref)
        return np.array(self.normalize_quaternion(qx, qy, qz, qw))

    def _sun_yaw_fresh(self, now=None):
        if not (self.use_sun_sensor_ and self.sun_valid_
                and self.last_sun_stamp_ is not None):
            return False
        now = now if now is not None else self.get_clock().now()
        age = (now - self.last_sun_stamp_).nanoseconds / 1e9
        return age <= self.sun_sensor_timeout_

    def _eskf_pitch_ok_for_body_velocity(self):
        if self.eskf_ is None or not self.eskf_seeded_:
            return False
        qx, qy, qz, qw = self.eskf_.q
        _, pitch, _ = self._rpy_from_quaternion(qx, qy, qz, qw)
        return abs(pitch) < self.max_body_velocity_pitch_rad_

    def _guard_body_velocity_for_yaw_jump(self, yaw_meas: float, now=None):
        if self.eskf_ is None or not self.eskf_seeded_:
            return
        _, _, yaw_est = self._rpy_from_quaternion(*self.eskf_.q)
        dyaw = (yaw_meas - yaw_est + math.pi) % (2.0 * math.pi) - math.pi
        if abs(dyaw) > self.body_velocity_yaw_jump_rad_:
            now = now if now is not None else self.get_clock().now()
            self.body_velocity_yaw_guard_until_ = (
                now + rclpy.duration.Duration(seconds=self.body_velocity_yaw_guard_s_)
            )
            # Inflate horizontal velocity covariance so the next airspeed
            # 2D update can quickly re-converge vx/vy after a large heading correction.
            if self.eskf_ is not None and self.eskf_vel_jump_inflate_ > 0.0:
                self.eskf_.P[3:5, 3:5] += np.eye(2) * self.eskf_vel_jump_inflate_

    def _eskf_yaw_ok_for_body_velocity(self):
        if self.body_velocity_yaw_guard_until_ is None:
            return True
        return self.get_clock().now() >= self.body_velocity_yaw_guard_until_

    def _eskf_attitude_ok_for_body_velocity(self):
        return (self._eskf_pitch_ok_for_body_velocity()
                and self._eskf_yaw_ok_for_body_velocity())

    def _apply_coordinated_turn_yaw_rate_input(
        self,
        omega: np.ndarray,
        phi: float,
        theta: float,
        airspeed: float,
        agl: float,
    ) -> np.ndarray:
        """Blend gyro body-z rate toward coordinated-turn body-rate.

        The coordinated-turn equation gives Euler yaw rate ψ̇, not body yaw-rate r.
        Convert it using Euler ZYX kinematics:
          ψ̇ = (q sinφ + r cosφ) / cosθ   where q = ω_y (pitch rate), r = ω_z (yaw rate)
          r_model = (ψ̇_model cosθ - q sinφ) / cosφ

        This is intentionally a soft process input.  It does not observe yaw angle,
        and it stays gated to avoid forcing yaw during uncoordinated or near-stall
        maneuvers.
        """
        self.yaw_rate_assist_ = 0.0
        if not (self.use_coor_turn_yaw_rate_input_ and self.use_coordinated_turn_):
            return omega
        if not (self.is_airborne_ and agl >= self.coor_turn_min_agl_):
            return omega
        bank_abs = abs(phi)
        if not (self.coor_turn_min_bank_rad_ <= bank_abs <= self.coor_turn_max_bank_rad_):
            return omega
        if airspeed < self.coor_turn_min_spd_:
            return omega
        cos_phi = math.cos(phi)
        cos_theta = math.cos(theta)
        sin_phi = math.sin(phi)
        if abs(cos_phi) < 1e-3 or abs(cos_theta) < 1e-3:
            return omega

        psi_dot_model = -(self.g_ / airspeed) * math.tan(phi)
        r_model = (psi_dot_model * cos_theta - omega[1] * sin_phi) / cos_phi
        r_model = max(-2.0, min(2.0, r_model))
        w = self.coor_turn_yaw_rate_weight_
        assisted = omega.copy()
        assisted[2] = (1.0 - w) * omega[2] + w * r_model
        self.yaw_rate_assist_ = assisted[2] - omega[2]
        return assisted

    def body_to_world_R(self):
        """Rotation matrix Body -> World using current quaternion."""
        R = quaternion_matrix([self.qx_, self.qy_, self.qz_, self.qw_])[:3, :3]
        return R

    def _update_z_fused(self):
        """Recompute z_fused_ with altitude-gated source priority.

        Priority order:
          1. Sonar (AGL ≤ max_sonar_alt_) — direct AGL measurement, most accurate at low altitude.
             z_fused = w_sonar * z_sonar + (1 - w_sonar) * z_imu
          2. Barometer — absolute pressure altitude, dominates above sonar range.
             z_fused = (1 - w_baro) * z_imu + w_baro * z_baro
          3. FAST-LIO / IMU z alone — if neither sonar nor baro available.
        """
        sonar_fresh = (
            self.last_sonar_stamp_ is not None
            and (self.get_clock().now() - self.last_sonar_stamp_).nanoseconds / 1e9
            <= self.sonar_timeout_
        )
        sonar_valid = (
            self.use_sonar_ and self.sonar_ready_ and sonar_fresh
            and self.sonar_range_ is not None
            and self.sonar_range_ <= self.max_sonar_alt_
        )
        if sonar_valid:
            # Sonar dominates at low altitude — direct AGL measurement outperforms
            # both baro (1-2 m drift) and LiDAR scan-match Z over flat ground.
            self.z_fused_ = (self.sonar_weight_ * self.sonar_range_
                             + (1.0 - self.sonar_weight_) * self.z_imu_)
        elif self.use_barometer_ and self.baro_ready_:
            self.z_fused_ = ((1.0 - self.baro_weight_) * self.z_imu_
                             + self.baro_weight_ * self.baro_z_)
        else:
            self.z_fused_ = self.z_imu_

    # ------------------------------------------------------------------
    # Reset and environment helpers
    # ------------------------------------------------------------------
    def reset_odometry_callback(self, request, response):
        self.x_ = 0.0
        self.y_ = 0.0
        self.z_imu_ = 0.0
        self.z_fused_ = 0.0

        self.vx_ = 0.0
        self.vy_ = 0.0
        self.vz_imu_ = 0.0
        self.vz_baro_ = 0.0
        self.last_baro_z_for_vz_ = None
        self.last_baro_stamp_for_vz_ = None

        self.integration_count_ = 0
        self.dropped_samples_ = 0
        self.warning_count_ = 0

        # Re-seed ESKF attitude on next Madgwick message. Reset quaternion to identity
        # so the ESKF publishes level attitude until the seed fires, rather than
        # carrying over a wrong attitude from a previous flight's bad final state.
        self.eskf_seeded_ = False
        if self.eskf_ is not None:
            self.eskf_.reset()

        self.start_time_ = self.get_clock().now()
        self.prev_imu_time_ = None
        self.prev_publish_time_ = None

        # Re-capture camera_init rotation on next FAST-LIO message
        self.fastlio_ready_ = False
        self.fastlio_vel_initialized_ = False
        self.camera_init_to_odom_R_ = None
        self.last_fastlio_stamp_ = None

        self.wheel_odom_ready_ = False
        self.wheel_x_    = 0.0
        self.wheel_y_    = 0.0
        self.wheel_vfwd_ = 0.0

        self.last_airspeed_stamp_   = None
        self.last_airspeed_y_stamp_ = None
        self.last_airspeed_z_stamp_ = None
        self.last_of_stamp_         = None
        self.last_fwd_cam_stamp_    = None
        self.last_thrust_stamp_ = None
        self.ground_yaw_ = self.initial_ground_yaw_

        self.get_logger().info("Odometry reset to zero")
        return response


    @staticmethod
    def _isa_density(h: float) -> float:
        """ISA troposphere air density (kg/m³) at altitude h metres AGL.
        Valid 0–11 000 m. Clamps below 0 m to sea-level value."""
        h = max(0.0, h)
        T = 288.15 - 0.0065 * h          # temperature [K]
        return 1.225 * (T / 288.15) ** 4.2559

    # ------------------------------------------------------------------
    # Altitude and pose callbacks: z, FAST-LIO x/y/yaw
    # ------------------------------------------------------------------

    def baro_callback(self, msg: PoseWithCovarianceStamped):
        z = msg.pose.pose.position.z
        if not self._is_finite(z):
            return
        self.baro_z_ = float(z)
        rho = self._isa_density(self.baro_z_)
        self.air_density_ = rho
        if self.eskf_ is not None:
            self.eskf_.rho = rho
        self.baro_ready_ = True

        # Differentiate baro z to get vertical velocity fallback for high-altitude flight
        # where FAST-LIO is gated off.  EMA alpha=0.3 — baro is noisier than FAST-LIO diff.
        now_baro = self.get_clock().now()
        if (self.last_baro_stamp_for_vz_ is not None
                and self.last_baro_z_for_vz_ is not None):
            dt_baro = (now_baro - self.last_baro_stamp_for_vz_).nanoseconds / 1e9
            if 0.005 < dt_baro < 1.0:
                vz_raw = (self.baro_z_ - self.last_baro_z_for_vz_) / dt_baro
                vz_raw = max(-15.0, min(15.0, vz_raw))   # clamp baro noise spikes
                self.vz_baro_ = 0.1 * vz_raw + 0.9 * self.vz_baro_
        self.last_baro_z_for_vz_ = self.baro_z_
        self.last_baro_stamp_for_vz_ = now_baro

        self._update_z_fused()
        if self.eskf_ is not None and self.eskf_seeded_:
            if abs(float(z) - self.eskf_.p[2]) < 30.0:
                self.eskf_.update_altitude(float(z), self.eskf_r_alt_baro_)

    def fastlio_callback(self, msg: Odometry):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation

        if not (self._is_finite(pos.x) and self._is_finite(pos.y) and self._is_finite(pos.z)):
            return

        # Transform FAST-LIO position from its frame (camera_init) to odom.
        # Use TF lookup so the offset is always correct regardless of static transform config.
        try:
            tf = self.tf_buffer_.lookup_transform(
                'odom',
                msg.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
            tx = tf.transform.translation.x
            ty = tf.transform.translation.y
            tz = tf.transform.translation.z
        except Exception:
            # Static transform not yet available — skip this update
            return

        # Use the full TF (rotation + translation) to transform FAST-LIO positions into odom.
        # The static TF defines the true camera_init→odom relationship. Using the IMU body
        # orientation here (the old camera_init_to_odom_R_ approach) was wrong: Madgwick
        # without a magnetometer has arbitrary initial yaw, so it baked a random rotation
        # into every FAST-LIO position and caused the drift/reversal behaviour.
        tf_q = [
            tf.transform.rotation.x,
            tf.transform.rotation.y,
            tf.transform.rotation.z,
            tf.transform.rotation.w,
        ]
        R_tf = quaternion_matrix(tf_q)[:3, :3]
        p_camera = np.array([float(pos.x), float(pos.y), float(pos.z)])
        p_odom = R_tf @ p_camera + np.array([tx, ty, tz])
        x_odom = float(p_odom[0])
        y_odom = float(p_odom[1])
        z_odom = float(p_odom[2])

        now = self.get_clock().now()

        # Velocity from finite-differencing consecutive FAST-LIO positions.
        # FAST-LIO does not populate the twist field of its Odometry message — the
        # velocity in msg.twist.twist is always zero. Differencing positions gives
        # world-frame velocity directly with no frame rotation needed.
        if self.fastlio_ready_ and self.last_fastlio_stamp_ is not None:
            dt_fl = (now - self.last_fastlio_stamp_).nanoseconds / 1e9
            if 0.005 < dt_fl < self.fastlio_timeout_:
                vx_new = (x_odom - self.last_fastlio_x_) / dt_fl
                vy_new = (y_odom - self.last_fastlio_y_) / dt_fl
                vz_fl  = (z_odom - self.last_fastlio_z_) / dt_fl
                # EMA smoothing: v_k = α·v_new + (1-α)·v_{k-1}
                # First valid measurement initialises directly to avoid cold-start bias.
                if not self.fastlio_vel_initialized_:
                    self.vx_ = vx_new
                    self.vy_ = vy_new
                    self.fastlio_vel_initialized_ = True
                else:
                    alpha = self.fastlio_vel_alpha_
                    self.vx_ = alpha * vx_new + (1.0 - alpha) * self.vx_
                    self.vy_ = alpha * vy_new + (1.0 - alpha) * self.vy_
                # Blend FAST-LIO Z velocity with IMU Z velocity (IMU is better for short intervals)
                self.vz_imu_ = 0.5 * vz_fl + 0.5 * self.vz_imu_
                # RLS drag calibration + thrust DR re-seed using FAST-LIO as the velocity reference.
                # FAST-LIO is the best truth available in this GPS-denied node — calibrate while
                # it's active, run dead reckoning when it goes stale.
                if self.use_thrust_dr_:
                    R = self.body_to_world_R()
                    vx_body = float((R.T @ np.array([self.vx_, self.vy_, self.vz_imu_]))[0])
                    now_sec = float(now.nanoseconds) * 1e-9
                    if (self.rls_vx_prev_ is not None and self.rls_time_prev_ is not None
                            and abs(vx_body) >= self.min_speed_drag_id_):
                        dt_rls = now_sec - self.rls_time_prev_
                        if 0.005 < dt_rls < self.fastlio_timeout_:
                            a_fl  = (vx_body - self.rls_vx_prev_) / dt_rls
                            pitch = math.asin(max(-1.0, min(1.0,
                                2.0 * (self.qw_ * self.qy_ - self.qz_ * self.qx_))))
                            y   = (self.thrust_N_ - self.drone_mass_ * a_fl
                                   - self.drone_mass_ * self.g_ * math.sin(pitch))
                            phi = self.air_density_ * vx_body * abs(vx_body)
                            if abs(phi) > 1e-3:
                                lam = self.rls_lambda_
                                K = self.rls_P_ * phi / (lam + phi * self.rls_P_ * phi)
                                self.drag_coeff_ = max(0.0, min(10.0,
                                    self.drag_coeff_ + K * (y - phi * self.drag_coeff_)))
                                self.rls_P_ = max(1e-6, min(100.0,
                                    (self.rls_P_ - K * phi * self.rls_P_) / lam))
                    self.rls_vx_prev_   = vx_body
                    self.rls_time_prev_ = now_sec
                    self.thrust_dr_vx_  = vx_body

        self.last_fastlio_stamp_ = now
        self.last_fastlio_x_ = x_odom
        self.last_fastlio_y_ = y_odom
        self.last_fastlio_z_ = z_odom

        # Position: suppressed on the ground — wheel odometry owns XY there.
        # Above ground, FAST-LIO corrects position when scan matching is valid.
        if not (self.use_wheel_odom_ and self._on_ground()):
            if not self.fastlio_ready_:
                w = self.fastlio_pos_weight_
                self.x_ = w * x_odom + (1.0 - w) * self.x_
                self.y_ = w * y_odom + (1.0 - w) * self.y_
            else:
                self.x_ = x_odom
                self.y_ = y_odom

        # Anchor z_imu_ to FAST-LIO z; sonar and baro priority weighting
        # is handled centrally in _update_z_fused().
        self.z_imu_ = z_odom
        self._update_z_fused()

        # Rotate FAST-LIO orientation from camera_init into odom frame.
        # Positions are already rotated via R_tf; orientation must match.
        # q_odom = q_tf ⊗ q_fl  ([x,y,z,w] quaternion product)
        ori_valid = (self._is_finite(ori.x) and self._is_finite(ori.y)
                     and self._is_finite(ori.z) and self._is_finite(ori.w))
        if ori_valid:
            tx, ty, tz, tw = tf_q[0], tf_q[1], tf_q[2], tf_q[3]
            fx, fy, fz, fw = ori.x, ori.y, ori.z, ori.w
            fl_q_w = tw*fw - tx*fx - ty*fy - tz*fz
            fl_q_x = tw*fx + tx*fw + ty*fz - tz*fy
            fl_q_y = tw*fy - tx*fz + ty*fw + tz*fx
            fl_q_z = tw*fz + tx*fy - ty*fx + tz*fw
            fl_yaw_odom = math.atan2(
                2.0 * (fl_q_w * fl_q_z + fl_q_x * fl_q_y),
                1.0 - 2.0 * (fl_q_y * fl_q_y + fl_q_z * fl_q_z),
            )
        else:
            fl_yaw_odom = None

        # Blend FAST-LIO yaw into current heading when scan-matching is reliable.
        # Skipped on the ground: flat surfaces give LiDAR almost no features to
        # constrain heading, so FAST-LIO yaw can jump 90°+ as scan matching
        # tries to converge. The on-ground yaw-only clamp in publish_odom would
        # then freeze that wrong heading. Madgwick gyro-integrated yaw is more
        # stable during ground runs.
        # Also skipped on the very first message (fastlio_ready_ still False).
        if (self.fastlio_ready_
                and self.fastlio_yaw_weight_ > 0.0
                and self.is_airborne_
                and fl_yaw_odom is not None):
            imu_yaw = math.atan2(
                2.0 * (self.qw_ * self.qz_ + self.qx_ * self.qy_),
                1.0 - 2.0 * (self.qy_ * self.qy_ + self.qz_ * self.qz_)
            )
            dyaw = fl_yaw_odom - imu_yaw
            if dyaw > math.pi:
                dyaw -= 2.0 * math.pi
            elif dyaw < -math.pi:
                dyaw += 2.0 * math.pi
            blended_yaw = imu_yaw + self.fastlio_yaw_weight_ * dyaw
            sinr_cosp = 2.0 * (self.qw_ * self.qx_ + self.qy_ * self.qz_)
            cosr_cosp = 1.0 - 2.0 * (self.qx_ * self.qx_ + self.qy_ * self.qy_)
            imu_roll = math.atan2(sinr_cosp, cosr_cosp)
            sinp = 2.0 * (self.qw_ * self.qy_ - self.qz_ * self.qx_)
            imu_pitch = math.asin(max(-1.0, min(1.0, sinp)))
            cr, cp, cy = math.cos(imu_roll / 2), math.cos(imu_pitch / 2), math.cos(blended_yaw / 2)
            sr, sp, sy = math.sin(imu_roll / 2), math.sin(imu_pitch / 2), math.sin(blended_yaw / 2)
            self.qw_ = cr * cp * cy + sr * sp * sy
            self.qx_ = sr * cp * cy - cr * sp * sy
            self.qy_ = cr * sp * cy + sr * cp * sy
            self.qz_ = cr * cp * sy - sr * sp * cy
            self.qx_, self.qy_, self.qz_, self.qw_ = self.normalize_quaternion(
                self.qx_, self.qy_, self.qz_, self.qw_)

        self.fastlio_ready_ = True

        if self.eskf_ is not None and self.eskf_seeded_:
            z_gate = self.z_fused_
            if z_gate <= self.max_fastlio_alt_:
                self.eskf_.update_position_xy(
                    np.array([x_odom, y_odom]), self.eskf_r_pos_
                )
            if self.is_airborne_ and fl_yaw_odom is not None and z_gate <= self.max_fastlio_alt_:
                self.eskf_.update_heading(fl_yaw_odom, self.eskf_r_yaw_)

    # ------------------------------------------------------------------
    # Velocity callbacks: pitot, optical flow, forward flow
    # ------------------------------------------------------------------

    def airspeed_callback(self, msg: Float64):
        val = float(msg.data)
        if math.isfinite(val) and val >= 0.0:
            self.last_airspeed_ = val
            self.airspeed_ready_ = True
            self.last_airspeed_stamp_ = self.get_clock().now()
            if (self.eskf_ is not None and self.eskf_seeded_
                    and val >= self.min_airspeed_):
                if not self.is_airborne_:
                    # On ground or liftoff debounce: 1D forward constraint only.
                    # Using _on_ground() here would switch to 2D on the very first
                    # airborne sonar tick (before is_airborne_ confirms 5 consecutive
                    # ticks). If the ESKF yaw was wrong from a bad ground clamp, the
                    # 2D zero-sideslip c2=[0,1] fires with a 14+ m/s lateral innovation
                    # and vy snaps to zero in one step. Keep 1D until the airborne
                    # debounce passes so the attitude is stable.
                    self.eskf_.update_body_velocity_1d(val, axis=0,
                                                       r_vel=self.eskf_r_vel_airspeed_,
                                                       gate=True)
                else:
                    # Confirmed airborne: 2D airspeed + zero-sideslip constraint.
                    # Simultaneously constrains body-forward (Va) and body-lateral (0),
                    # making both world-frame vx and vy observable and preventing
                    # lateral velocity divergence during sustained banked turns.
                    self.eskf_.update_airspeed_world_velocity(
                        val,
                        r_fwd=self.eskf_r_vel_airspeed_,
                        r_lat=self.eskf_r_vel_airspeed_lat_,
                    )

    def airspeed_y_callback(self, msg: Float64):
        val = float(msg.data)
        if not math.isfinite(val):
            return
        self.last_airspeed_y_ = val
        self.airspeed_y_ready_ = True
        self.last_airspeed_y_stamp_ = self.get_clock().now()
        if (self.eskf_ is not None and self.eskf_seeded_ and self.is_airborne_
                and self._eskf_attitude_ok_for_body_velocity()):
            # gate=True: unlike Va_x, lateral/vertical sensors have no runaway-divergence
            # history; the gate protects vx from covariance-coupled bad innovations
            # during pitching or banked flight where Va_y/Va_z may be unexpectedly large.
            self.eskf_.update_body_velocity_1d(val, axis=1, r_vel= 0.5, gate=True)

    def airspeed_z_callback(self, msg: Float64):
        val = float(msg.data)
        if not math.isfinite(val):
            return
        self.last_airspeed_z_ = val
        self.airspeed_z_ready_ = True
        self.last_airspeed_z_stamp_ = self.get_clock().now()
        if (self.eskf_ is not None and self.eskf_seeded_ and self.is_airborne_
                and self._eskf_attitude_ok_for_body_velocity()):
            self.eskf_.update_body_velocity_1d(val, axis=2, r_vel=0.5,
                                               gate=True)

    def of_callback(self, msg: TwistWithCovarianceStamped):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        if math.isfinite(vx) and math.isfinite(vy):
            self.of_vx_body_ = vx
            self.of_vy_body_ = vy
            self.of_ready_   = True
            self.last_of_stamp_ = self.get_clock().now()
            if (self.eskf_ is not None and self.eskf_seeded_ and self.is_airborne_
                    and self.min_of_altitude_ <= self.z_fused_ <= self.max_of_altitude_):
                self.eskf_.update_body_velocity_2d(vx, vy, r_vel=self.eskf_r_vel_)

    def fwd_cam_callback(self, msg: TwistWithCovarianceStamped):
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z
        if math.isfinite(vy) and math.isfinite(vz):
            self.fwd_cam_vy_body_ = vy
            self.fwd_cam_vz_body_ = vz
            self.fwd_cam_ready_   = True
            self.last_fwd_cam_stamp_ = self.get_clock().now()
            if (self.eskf_ is not None and self.eskf_seeded_
                    and self.z_fused_ >= self.min_fwd_cam_altitude_
                    and self._eskf_attitude_ok_for_body_velocity()):
                # Forward camera observes body-lateral (vy) and body-up (vz) — doc §6.7.
                # Two sequential 1-D updates reuse the shared H pattern from §6.1 and
                # let each axis be gated independently via Mahalanobis.
                self.eskf_.update_body_velocity_1d(vy, axis=1, r_vel=self.eskf_r_vel_)
                self.eskf_.update_body_velocity_1d(vz, axis=2, r_vel=self.eskf_r_vel_)

    # ------------------------------------------------------------------
    # Ground contact and ground-vehicle callbacks: sonar and wheel odometry
    # ------------------------------------------------------------------

    def sonar_callback(self, msg: Range):
        r = float(msg.range)
        if math.isfinite(r) and msg.min_range <= r <= msg.max_range:
            self.sonar_range_ = r
            self.sonar_ready_ = True
            self.last_sonar_stamp_ = self.get_clock().now()
            self._update_z_fused()
            if self.eskf_ is not None and self.eskf_seeded_:
                self.eskf_.update_altitude(r, self.eskf_r_alt_sonar_)
        # Don't clear sonar_ready_ on a bad packet — use timeout in _on_ground() instead.

    def _on_ground(self) -> bool:
        """True when a recent valid sonar reading confirms AGL <= ground_threshold."""
        if not self.sonar_ready_ or self.sonar_range_ is None or self.last_sonar_stamp_ is None:
            return False
        if (self.get_clock().now() - self.last_sonar_stamp_).nanoseconds / 1e9 > self.sonar_timeout_:
            return False
        return self.sonar_range_ <= self.ground_threshold_

    def wheel_odom_callback(self, msg: Odometry):
        """Consume /wheel/odometry — own XY position and velocity on the ground."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        v = msg.twist.twist.linear.x   # body-frame forward speed
        if not (self._is_finite(x) and self._is_finite(y) and self._is_finite(v)):
            return
        self.wheel_x_    = x
        self.wheel_y_    = y
        self.wheel_vfwd_ = v
        self.wheel_odom_ready_ = True
        if self.on_ground_debounced_:
            self.x_ = x
            self.y_ = y
            if self.eskf_ is not None and self.eskf_seeded_:
                # Position: XY from wheels, keep ESKF altitude for Z.
                # Debounced flag prevents a sonar flicker at liftoff from snapping
                # ESKF position back to the last ground position.
                self.eskf_.update_position_xy(
                    np.array([float(x), float(y)]),
                    self.eskf_r_pos_,
                )
                self.eskf_.update_body_velocity_1d(float(v), axis=0,
                                                   r_vel=self.eskf_r_vel_)
                self.eskf_.update_nhc()

    # ------------------------------------------------------------------
    # Physics/support callbacks: wind, thrust, fuel tank state
    # ------------------------------------------------------------------

    def wind_callback(self, msg: TwistStamped):
        wx = msg.twist.linear.x
        wy = msg.twist.linear.y
        wz = msg.twist.linear.z
        if math.isfinite(wx) and math.isfinite(wy) and math.isfinite(wz):
            self.wind_x_ = wx
            self.wind_y_ = wy
            self.wind_z_ = wz

    def thrust_callback(self, msg: Float64):
        val = float(msg.data)
        if math.isfinite(val) and val >= 0.0:
            self.thrust_N_ = val
            self.last_thrust_stamp_ = self.get_clock().now()

    def tank_pressure_callback(self, msg: Float64):
        val = float(msg.data)
        if math.isfinite(val) and val > 0.0:
            self.tank_pressure_ = val

    def tank_temperature_callback(self, msg: Float64):
        val = float(msg.data)
        if math.isfinite(val) and val > 0.0:
            self.tank_temperature_ = val

    # ------------------------------------------------------------------
    # Heading callbacks: magnetometer and sun sensor
    # ------------------------------------------------------------------

    def _tilt_compensate(self, vx: float, vy: float, vz: float):
        """Project a body-frame vector onto the horizontal plane accounting for roll/pitch.

        Returns (bh_x, bh_y): forward and left horizontal components.
        Derivation: B_level = R_y(pitch) @ R_x(roll) @ B_body; take [0] and [1].
        """
        qx, qy, qz, qw = self.qx_, self.qy_, self.qz_, self.qw_
        roll  = math.atan2(2.0*(qw*qx + qy*qz), 1.0 - 2.0*(qx*qx + qy*qy))
        pitch = math.asin(max(-1.0, min(1.0, 2.0*(qw*qy - qz*qx))))
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        bh_x = vx*cp + (vy*sr + vz*cr)*sp   # forward (East at yaw=0)
        bh_y = vy*cr - vz*sr                  # left (North at yaw=0)
        return bh_x, bh_y

    def _blend_yaw(self, target_yaw: float, weight: float):
        """Blend target_yaw into current attitude at the given weight.

        Only adjusts yaw; roll and pitch are preserved from the current quaternion.
        """
        imu_yaw = math.atan2(
            2.0*(self.qw_*self.qz_ + self.qx_*self.qy_),
            1.0 - 2.0*(self.qy_*self.qy_ + self.qz_*self.qz_)
        )
        dyaw = target_yaw - imu_yaw
        if dyaw > math.pi:    dyaw -= 2.0*math.pi
        elif dyaw < -math.pi: dyaw += 2.0*math.pi
        blended_yaw = imu_yaw + weight * dyaw
        sinr = 2.0*(self.qw_*self.qx_ + self.qy_*self.qz_)
        cosr = 1.0 - 2.0*(self.qx_*self.qx_ + self.qy_*self.qy_)
        roll  = math.atan2(sinr, cosr)
        pitch = math.asin(max(-1.0, min(1.0, 2.0*(self.qw_*self.qy_ - self.qz_*self.qx_))))
        cr, cp, cy = math.cos(roll/2), math.cos(pitch/2), math.cos(blended_yaw/2)
        sr, sp, sy = math.sin(roll/2), math.sin(pitch/2), math.sin(blended_yaw/2)
        self.qw_ = cr*cp*cy + sr*sp*sy
        self.qx_ = sr*cp*cy - cr*sp*sy
        self.qy_ = cr*sp*cy + sr*cp*sy
        self.qz_ = cr*cp*sy - sr*sp*cy
        self.qx_, self.qy_, self.qz_, self.qw_ = self.normalize_quaternion(
            self.qx_, self.qy_, self.qz_, self.qw_)

    def sun_vector_callback(self, msg: Vector3Stamped):
        sx, sy, sz = msg.vector.x, msg.vector.y, msg.vector.z
        if not all(math.isfinite(v) for v in (sx, sy, sz)):
            return
        if sz <= 0.0:
            return  # sun below sensor's local horizon — reject to avoid corrupted yaw
        self.sun_vector_body_ = [sx, sy, sz]
        self.sun_ready_ = True
        self.last_sun_stamp_ = self.get_clock().now()
        if not self.sun_valid_:
            return
        # Tilt-compensated sun heading in ROS ENU (x=fwd, y=left, z=up):
        #   yaw = az_sun_enu - atan2(bh_y, bh_x)
        # Because: az_sun = atan2(N, E) = yaw + atan2(sun_level_y, sun_level_x)
        bh_x, bh_y = self._tilt_compensate(sx, sy, sz)
        sun_yaw = self.sun_azimuth_enu_ - math.atan2(bh_y, bh_x)
        sun_yaw = math.atan2(math.sin(sun_yaw), math.cos(sun_yaw))
        self.sun_yaw_ = sun_yaw
        # ESKF heading update — sun sensor is the most accurate heading source in GPS-denied
        if (self.eskf_ is not None and self.eskf_seeded_
                and self.is_airborne_):
            self.eskf_.update_heading(sun_yaw, self.eskf_r_yaw_sun_)
        elif self.is_airborne_:
            # Non-ESKF path: blend sun heading into complementary filter.
            # Madgwick handles magnetometer; sun provides additional independent correction.
            self._blend_yaw(sun_yaw, self.sun_yaw_weight_)

    def sun_azimuth_callback(self, msg: Float64):
        self.sun_azimuth_enu_ = float(msg.data)

    def sun_elevation_callback(self, msg: Float64):
        el = float(msg.data)
        self.sun_elevation_deg_ = el
        self.sun_valid_ = el >= self.sun_el_min_deg_

    # ------------------------------------------------------------------
    # IMU/ESKF callbacks: attitude update, physics predict, gravity, turn model
    # ------------------------------------------------------------------

    def imu_callback(self, msg: Imu):
        # Orientation and angular velocity only — zero integration.
        # Position is owned entirely by fastlio_callback.
        self.qx_, self.qy_, self.qz_, self.qw_ = self.normalize_quaternion(
            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        )
        self.madg_qx_, self.madg_qy_, self.madg_qz_, self.madg_qw_ = (
            self.qx_, self.qy_, self.qz_, self.qw_
        )
        wx = msg.angular_velocity.x
        wy = msg.angular_velocity.y
        wz = msg.angular_velocity.z
        self.wx_ = wx if self._is_finite(wx) else 0.0
        self.wy_ = wy if self._is_finite(wy) else 0.0
        self.wz_ = wz if self._is_finite(wz) else 0.0
        self.integration_count_ += 1

        # Seed ESKF attitude once from a hybrid attitude reference, then apply
        # continuous split attitude updates:
        #   roll/pitch from raw /imu orientation (high-quality sim tilt)
        #   yaw        from /imu_ekf Madgwick (mag-filtered heading)
        # This keeps yaw observable without letting Madgwick's accel-aided roll
        # flatten true banks.
        # predict_imu / predict_physics run in eskf_raw_imu_callback; the orientation
        # updates here correct δθ using explicit tilt/yaw measurements.
        #
        # Two-stage startup gate:
        #   [0, 400)  : ESKF not seeded — IMU callback returns early, no propagation.
        #   [400,600): ESKF seeded from settled attitude. Gravity corrects any
        #               remaining small spawn error. Attitude updates suppressed.
        #   [600, ∞) : split tilt/yaw updates enabled.
        #
        # Seeding at tick 0 (old behaviour) caught Madgwick mid-convergence (gain=0.3
        # moves ~40° in the first second) → large seeding error → Mahalanobis gate blocked
        # all corrections → HUD showed wrong attitude for tens of seconds.
        if self.eskf_ is not None and not self.eskf_seeded_ and self.integration_count_ >= 400:
            self.eskf_.q = self._hybrid_raw_tilt_madgwick_yaw_q()
            self.eskf_.v = np.array([self.vx_, self.vy_, self.vz_imu_])
            self.eskf_.p = np.array([self.x_,  self.y_,  self.z_fused_])
            self.eskf_seeded_ = True
        elif self.eskf_ is not None and self.eskf_seeded_ and self.integration_count_ >= 600:
            q_meas = np.array([self.qx_, self.qy_, self.qz_, self.qw_])
            # Cache Madgwick roll for diagnostics, but prefer raw /imu roll as the
            # bank reference used to gate accelerometer-as-gravity roll correction.
            C_madg    = MARIDESKF._dcm(q_meas)
            phi_madg  = math.atan2(C_madg[2, 1], C_madg[2, 2])
            self.last_phi_madg_ = phi_madg
            if self.raw_imu_orientation_ready_:
                self.last_phi_tilt_ref_ = self.raw_roll_
                self.eskf_.update_roll_pitch(
                    self.raw_roll_,
                    self.raw_pitch_,
                    r_tilt=self.eskf_r_raw_tilt_,
                    gate=False,
                )
                _, _, yaw_madg = self._rpy_from_quaternion(
                    self.qx_, self.qy_, self.qz_, self.qw_)
                self._guard_body_velocity_for_yaw_jump(yaw_madg)
                # Madgwick yaw is magnetometer-aided and provides continuity even
                # when sun yaw has geometry/dropout issues. Keep it weak so it does
                # not dominate the cleaner sun yaw correction.
                # During banked turns Madgwick's gradient descent mistakes centripetal
                # acceleration for tilt, corrupting the mag projection and drifting yaw.
                # Inflate r by 25× when banked >12° in flight so the bad estimate
                # cannot accumulate into the ESKF; sun sensor (r=0.0085) still dominates.
                r_madg = self.eskf_r_yaw_madgwick_fallback_
                if self.is_airborne_ and abs(self.raw_roll_) > math.radians(12.0):
                    r_madg *= 25.0
                self.eskf_.update_heading(yaw_madg, r_madg)
            else:
                self.last_phi_tilt_ref_ = phi_madg
                # Fallback for real/raw-IMU sources that do not populate orientation.
                # Keep old full-quaternion behavior only when raw tilt is unavailable.
                self.eskf_.update_orientation(q_meas, r_ori=self.eskf_r_ori_)

    def joint_state_callback(self, msg: JointState):
        """Extract surface angles for physics-mode ESKF predict."""
        for i, name in enumerate(msg.name):
            if name == "left_wing_joint":
                self.delta_lw_ = float(msg.position[i])
            elif name == "right_wing_joint":
                self.delta_rw_ = float(msg.position[i])
            elif name == "tail_left_joint":
                self.delta_tl_ = float(msg.position[i])
            elif name == "tail_right_joint":
                self.delta_tr_ = float(msg.position[i])

    def eskf_raw_imu_callback(self, msg: Imu):
        """Drive ESKF prediction from /imu/with_gravity at IMU rate."""
        oqx, oqy, oqz, oqw = (
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        )
        raw_q_norm = math.sqrt(oqx*oqx + oqy*oqy + oqz*oqz + oqw*oqw)
        if (self._is_finite(raw_q_norm) and raw_q_norm > 1e-9
                and all(self._is_finite(v) for v in (oqx, oqy, oqz, oqw))):
            qx, qy, qz, qw = self.normalize_quaternion(oqx, oqy, oqz, oqw)
            # Gazebo's raw IMU orientation is a high-quality simulated attitude signal.
            # Cache it even before ESKF seeding so the seed can use raw roll/pitch.
            self.raw_qx_, self.raw_qy_, self.raw_qz_, self.raw_qw_ = qx, qy, qz, qw
            self.raw_roll_, self.raw_pitch_, self.raw_yaw_ = self._rpy_from_quaternion(
                qx, qy, qz, qw
            )
            self.raw_imu_orientation_ready_ = True

        if not self.eskf_seeded_:
            return
        ox = msg.angular_velocity.x
        oy = msg.angular_velocity.y
        oz = msg.angular_velocity.z
        if not all(math.isfinite(v) for v in (ox, oy, oz)):
            return
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z
        a_sf_valid = all(math.isfinite(v) for v in (ax, ay, az))
        if a_sf_valid:
            self.last_a_norm_ = math.sqrt(ax*ax + ay*ay + az*az)

        now = self.get_clock().now()
        if self.eskf_imu_stamp_ is not None:
            dt = (now - self.eskf_imu_stamp_).nanoseconds / 1e9
            omega = np.array([ox, oy, oz])
            airspeed_fresh = (
                self.airspeed_ready_
                and self.last_airspeed_stamp_ is not None
                and (now - self.last_airspeed_stamp_).nanoseconds / 1e9 < self.airspeed_timeout_
            )
            if airspeed_fresh and self.is_airborne_:
                C_pre = MARIDESKF._dcm(self.eskf_.q)
                phi_pre = math.atan2(C_pre[2, 1], C_pre[2, 2])
                theta_pre = math.asin(max(-1.0, min(1.0, -C_pre[2, 0])))
                agl_pre = (self.sonar_range_ if self.sonar_ready_ and self.sonar_range_ is not None
                           else self.z_fused_)
                omega = self._apply_coordinated_turn_yaw_rate_input(
                    omega, phi_pre, theta_pre, self.last_airspeed_, agl_pre
                )
            else:
                self.yaw_rate_assist_ = 0.0

            if self.eskf_mode_ == "physics":
                self.eskf_.predict_physics(
                    self.thrust_N_,
                    omega,
                    self.delta_lw_,
                    self.delta_rw_,
                    self.delta_tl_,
                    self.delta_tr_,
                    dt,
                )
            else:
                if a_sf_valid:
                    self.eskf_.predict_imu(
                        np.array([ax, ay, az]), omega, dt
                    )
            # Pre-compute bank angle once — shared by the gravity roll-gate and the
            # coordinated-turn estimator below, avoiding a duplicate DCM call.
            phi_now   = 0.0
            theta_now = 0.0
            C_now     = None
            if airspeed_fresh and self.is_airborne_:
                C_now     = MARIDESKF._dcm(self.eskf_.q)
                phi_now   = math.atan2(C_now[2, 1], C_now[2, 2])
                theta_now = math.asin(max(-1.0, min(1.0, -C_now[2, 0])))
            self.last_phi_ = phi_now  # 0.0 when on ground or no airspeed

            # Gravity-alignment: soft tilt correction whenever |a_sf| ≈ g.
            # gate_roll during banked turns or forward acceleration:
            # (1) Banked turn: specific force is lift vector, not world-z gravity —
            #     gravity would pull ESKF toward roll=0, fighting the true bank angle.
            # (2) Forward acceleration (|a_x| > threshold): thrust or dive injects a
            #     body-x component that contaminates the accelerometer gravity reading;
            #     at 2 m/s² (~0.2g) the roll reference error is significant.
            # In both cases gate the roll column so gravity still corrects pitch.
            if a_sf_valid:
                gravity_gate_roll = (
                    self.is_airborne_
                    and abs(self.last_phi_tilt_ref_) > self.gravity_gate_roll_min_bank_rad_
                )
                self.eskf_.update_gravity(
                    np.array([ax, ay, az]),
                    g=self.g_,
                    r_acc=self.eskf_r_acc_,
                    gate_roll=gravity_gate_roll,
                )

            # Coordinated-turn yaw gyro-bias estimator.
            # Uses ψ̇_aero = (g/V)·tan(φ) vs ψ̇_imu to observe and correct b_r.
            # Gated on: airspeed valid + above stall, bank in [min, max], AGL > threshold.
            if self.use_coordinated_turn_ and self.use_eskf_:
                if airspeed_fresh and self.is_airborne_:
                    agl = (self.sonar_range_ if self.sonar_ready_ and self.sonar_range_ is not None
                           else self.z_fused_)
                    bank_abs = abs(phi_now)
                    if (self.last_airspeed_ >= self.coor_turn_min_spd_
                            and self.coor_turn_min_bank_rad_ <= bank_abs <= self.coor_turn_max_bank_rad_
                            and agl >= self.coor_turn_min_agl_):
                        self.eskf_.update_coordinated_turn(
                            phi_now, theta_now, oy, oz,
                            self.last_airspeed_,
                            g=self.g_,
                            r_coor=self.eskf_r_coor_turn_,
                        )

        self.eskf_imu_stamp_ = now

    # ------------------------------------------------------------------
    # Publishing: source priority, final odometry message, diagnostics, TF
    # ------------------------------------------------------------------
    def publish_odom(self):
        now = self.get_clock().now()

        dt_pub = 0.0
        if self.prev_publish_time_ is not None:
            _dt = (now - self.prev_publish_time_).nanoseconds / 1e9
            if 0.0 < _dt < 0.5:
                dt_pub = _dt

        fastlio_stale = (
            not self.fastlio_ready_
            or self.last_fastlio_stamp_ is None
            or (now - self.last_fastlio_stamp_).nanoseconds / 1e9 > self.fastlio_timeout_
            or self.z_fused_ > self.max_fastlio_alt_
        )
        airspeed_stale = (
            not self.airspeed_ready_
            or self.last_airspeed_stamp_ is None
            or (now - self.last_airspeed_stamp_).nanoseconds / 1e9 > self.airspeed_timeout_
        )
        of_stale = (
            not self.of_ready_
            or self.last_of_stamp_ is None
            or (now - self.last_of_stamp_).nanoseconds / 1e9 > self.of_timeout_
        )
        fwd_cam_stale = (
            not self.fwd_cam_ready_
            or self.last_fwd_cam_stamp_ is None
            or (now - self.last_fwd_cam_stamp_).nanoseconds / 1e9 > self.fwd_cam_timeout_
        )

        # If thrust topic has gone silent, treat as glide: thrust_N_ = 0 so drag and
        # gravity still decelerate the estimate naturally via the physics equation.
        if (self.use_thrust_dr_ and self.last_thrust_stamp_ is not None
                and (now - self.last_thrust_stamp_).nanoseconds / 1e9 > self.thrust_timeout_):
            self.thrust_N_ = 0.0

        # H₂ mass tracking: consume fuel proportional to thrust, then correct with
        # tank sensors when available. Updates drone_mass_ for thrust DR accuracy.
        if self.track_h2_mass_ and dt_pub > 0.0:
            # Integrate SFC-based consumption
            dm = self.sfc_ * self.thrust_N_ * dt_pub
            self.h2_remaining_ = max(0.0, self.h2_remaining_ - dm)

            # Ideal gas law correction from tank sensors (slow blend to avoid noise).
            # p·V = m·R_H2·T  →  m = p·V / (R_H2·T)
            if self.tank_pressure_ is not None and self.tank_temperature_ is not None:
                m_sensor = (self.tank_pressure_ * self.tank_volume_) / (self.R_H2_ * self.tank_temperature_)
                m_sensor = max(0.0, min(self.h2_initial_mass_, m_sensor))
                w = self.tank_sensor_w_
                self.h2_remaining_ = (1.0 - w) * self.h2_remaining_ + w * m_sensor

            self.drone_mass_ = self.drone_empty_mass_ + self.h2_remaining_

        # On liftoff, seed thrust_dr_vx_ from airspeed so there's no drop-to-zero
        # when the velocity source switches from wheel odometry to thrust DR.
        # liftoff_seeded_ is a one-shot flag: prevents re-seeding every time sonar
        # noise causes _on_ground() to flicker True→False at the threshold.
        # Guard with use_sonar_: without sonar, _on_ground() is always False so
        # prev_on_ground_ (init=True) would fire a spurious seed on the first tick.
        currently_on_ground = self._on_ground()

        # Debounce for vx source selection only — prevents a single noisy sonar reading
        # below ground_threshold_ from snapping vx_pub to wheel_vfwd_ (typically 0).
        if currently_on_ground:
            self.on_ground_ticks_ = min(self.on_ground_ticks_ + 1, self.on_ground_debounce_ticks_ + 1)
        else:
            self.on_ground_ticks_ = 0
        on_ground_for_vx = (self.on_ground_ticks_ >= self.on_ground_debounce_ticks_)
        self.on_ground_debounced_ = on_ground_for_vx

        # Airborne debounce: require N consecutive not-on-ground ticks before enabling
        # flight sensors (airspeed, heading, roll clamp release).  A single sonar spike
        # above ground_threshold_ (prop wash, vibration) must not permanently corrupt
        # ESKF roll/vy by briefly enabling these sensors.
        if currently_on_ground:
            self.airborne_ticks_ = 0
        else:
            self.airborne_ticks_ = min(self.airborne_ticks_ + 1, self.airborne_debounce_ticks_ + 1)
        self.is_airborne_ = (self.airborne_ticks_ >= self.airborne_debounce_ticks_)

        if (self.use_sonar_ and self.prev_on_ground_ and not currently_on_ground
                and self.use_thrust_dr_ and not self.liftoff_seeded_):
            seed = self.last_airspeed_ if (not airspeed_stale and self.last_airspeed_ >= self.min_airspeed_) \
                   else (self.wheel_vfwd_ if self.wheel_odom_ready_ else self.thrust_dr_vx_)
            self.thrust_dr_vx_ = seed
            self.liftoff_seeded_ = True
        # Reset seed flag only when drone is clearly on the ground (well below threshold),
        # not just at the noisy boundary.
        if self.sonar_range_ is not None and self.sonar_range_ < self.ground_threshold_ * 0.5:
            self.liftoff_seeded_ = False
        self.prev_on_ground_ = currently_on_ground

        # Integrate thrust DR body-frame forward velocity at publish rate.
        if (self.use_thrust_dr_ and self.thrust_dr_ready_
                and self.is_airborne_ and dt_pub > 0.0):
            pitch = math.asin(max(-1.0, min(1.0,
                2.0 * (self.qw_ * self.qy_ - self.qz_ * self.qx_))))
            sinr_cosp = 2.0 * (self.qw_ * self.qx_ + self.qy_ * self.qz_)
            cosr_cosp = 1.0 - 2.0 * (self.qx_ * self.qx_ + self.qy_ * self.qy_)
            roll_dr = math.atan2(sinr_cosp, cosr_cosp)
            cos2_phi_dr = max(math.cos(roll_dr) ** 2, math.cos(math.pi / 6) ** 2)
            bank_factor_dr = 1.0 + (1.0 / cos2_phi_dr - 1.0) * self.induced_drag_frac_
            drag = self.drag_coeff_ * self.air_density_ * self.thrust_dr_vx_ * abs(self.thrust_dr_vx_) * bank_factor_dr
            a_x  = (self.thrust_N_ - drag - self.drone_mass_ * self.g_ * math.sin(pitch)) / self.drone_mass_
            self.thrust_dr_vx_ = max(-self.max_velocity_,
                                      min(self.max_velocity_, self.thrust_dr_vx_ + a_x * dt_pub))

        # Dead-reckon XY position between sensor corrections.
        # Priority: optical flow (FAST-LIO active, low altitude) > thrust DR (FAST-LIO stale, airborne).
        # FAST-LIO hard-resets x_/y_ each scan, capping OF drift to one scan interval (~100 ms).
        if dt_pub > 0.0 and not (self.use_wheel_odom_ and self._on_ground()):
            if (self.fastlio_ready_
                    and self.use_optical_flow_ and not of_stale
                    and self.min_of_altitude_ <= self.z_fused_ <= self.max_of_altitude_):
                R = self.body_to_world_R()
                v_body = np.array([self.of_vx_body_, self.of_vy_body_, 0.0])
                v_world = R @ v_body
                self.x_ += v_world[0] * dt_pub
                self.y_ += v_world[1] * dt_pub
            elif (self.use_thrust_dr_ and self.thrust_dr_ready_ and fastlio_stale):
                R = self.body_to_world_R()
                v_world = R @ np.array([self.thrust_dr_vx_, 0.0, 0.0])
                self.x_ += v_world[0] * dt_pub
                self.y_ += v_world[1] * dt_pub
        self.prev_publish_time_ = now

        # ESKF output takes precedence over complementary filter when active.
        # Altitude (z_fused_, vz_imu_) is intentionally excluded: sonar + baro own
        # altitude and were accurate before ESKF. ESKF z dead-reckons from vz and
        # drifts, corrupting the published altitude and making forward motion appear
        # as altitude gain via the rotated velocity projection.
        if self.eskf_ is not None and self.eskf_seeded_:
            # Keep the internal ESKF vertical state aligned with the altitude owner
            # even though only horizontal ESKF states are published from this block.
            self.eskf_.p[2] = self.z_fused_
            self.eskf_.v[2] = self.vz_imu_ if not fastlio_stale else self.vz_baro_
            ep = self.eskf_.position
            eq = self.eskf_.quaternion   # [x,y,z,w]
            self.x_       = float(ep[0])
            self.y_       = float(ep[1])
            # Divergence gate: only publish ESKF quaternion when it agrees with
            # the hybrid attitude reference to within ~20° (|cos half-angle| > 0.94).
            # When the ESKF
            # diverges (e.g., physics-mode attitude error from corrupted bg) the
            # published attitude would be wrong.  Fall back to raw roll/pitch +
            # Madgwick yaw in that case — position (x,y) from ESKF is still used.
            att_ref_q = self._hybrid_raw_tilt_madgwick_yaw_q()
            dot = abs(float(np.dot(eq, att_ref_q)))
            if dot > 0.94:   # < ~20° divergence
                self.qx_ = float(eq[0]); self.qy_ = float(eq[1])
                self.qz_ = float(eq[2]); self.qw_ = float(eq[3])
            else:
                self.qx_ = float(att_ref_q[0]); self.qy_ = float(att_ref_q[1])
                self.qz_ = float(att_ref_q[2]); self.qw_ = float(att_ref_q[3])

        # On the ground, wheels constrain pitch and roll to zero — only yaw is free.
        # Madgwick yaw drifts from gyro bias without a magnetometer. Instead, derive
        # heading from the direction of FAST-LIO world-frame velocity (atan2(vy, vx)):
        # if FAST-LIO x/y is reliable, so is the motion direction. Hold the last known
        # heading when the drone is stationary (speed too low for a valid direction).
        if self._on_ground():
            ground_speed = math.hypot(self.vx_, self.vy_)
            if ground_speed > 0.3 and not fastlio_stale:
                self.ground_yaw_ = math.atan2(self.vy_, self.vx_)
            cy, sy = math.cos(self.ground_yaw_ / 2), math.sin(self.ground_yaw_ / 2)
            self.qw_ = cy; self.qx_ = 0.0; self.qy_ = 0.0; self.qz_ = sy
            self.qx_, self.qy_, self.qz_, self.qw_ = self.normalize_quaternion(
                self.qx_, self.qy_, self.qz_, self.qw_)
            # Keep ESKF attitude in sync with the ground clamp to prevent pitch/roll
            # from accumulating during the taxi run.  Critically: preserve the ESKF's
            # own yaw (owned by Madgwick/sun sensor, correct even when FAST-LIO is
            # not yet active and ground_yaw_ is still the default initial_ground_yaw_).
            # Forcing the ESKF to ground_yaw_ when FAST-LIO is absent gave yaw=0,
            # which at liftoff caused update_airspeed_world_velocity to fire with
            # c2=[0,1] and a 14+ m/s lateral innovation, snapping vy to zero.
            if self.eskf_ is not None and self.eskf_seeded_:
                _, _, cur_eskf_yaw = self._rpy_from_quaternion(*self.eskf_.q)
                cy2 = math.cos(cur_eskf_yaw / 2)
                sy2 = math.sin(cur_eskf_yaw / 2)
                flat_q = np.array([0.0, 0.0, sy2, cy2])
                self.eskf_.q = flat_q / np.linalg.norm(flat_q)

        # Validate state
        vals = [self.x_, self.y_, self.z_fused_, self.vx_, self.vy_, self.vz_imu_,
                self.qx_, self.qy_, self.qz_, self.qw_]
        if not all(self._is_finite(v) for v in vals):
            return

        # Re-normalize quaternion before publishing
        self.qx_, self.qy_, self.qz_, self.qw_ = self.normalize_quaternion(
            self.qx_, self.qy_, self.qz_, self.qw_
        )

        # Covariances — FAST-LIO bounds drift so variance doesn't grow when it's active
        elapsed_time = (now - self.start_time_).nanoseconds / 1e9
        if self.fastlio_ready_:
            pos_variance = self.base_pos_var_
        else:
            pos_variance = self.base_pos_var_ + self.var_growth_rate_ * max(0.0, elapsed_time)
        vel_variance = self.base_vel_var_

        # Fill odometry
        self.odom_msg_.header.stamp = now.to_msg()

        self.odom_msg_.pose.pose.position.x = self.x_
        self.odom_msg_.pose.pose.position.y = self.y_
        self.odom_msg_.pose.pose.position.z = self.z_fused_

        self.odom_msg_.pose.pose.orientation.x = self.qx_
        self.odom_msg_.pose.pose.orientation.y = self.qy_
        self.odom_msg_.pose.pose.orientation.z = self.qz_
        self.odom_msg_.pose.pose.orientation.w = self.qw_

        # -----------------------------------------------------------------------
        # Velocity estimation
        # -----------------------------------------------------------------------
        # vx — two phases, no elif between them:
        #   Ground : airspeed (primary) → wheel_vfwd (fallback)
        #   Air    : airspeed always; thrust DR when airspeed unavailable
        #
        # vy — layered corrections applied on top of vx phase result:
        #   FAST-LIO lateral finite-diff (active, below max_fastlio_altitude)
        #   → OF vy post-blend (low altitude)
        #   → forward camera vy post-blend
        #   → hold/decay
        #
        # FAST-LIO is used only for position correction (fastlio_callback) and
        # vy; it is NOT a vx source. Optical flow is NOT a vx source — with
        # positive pitch the downward camera cannot cleanly isolate forward motion.
        # -----------------------------------------------------------------------
        decay = self.velocity_decay_per_sec_ ** dt_pub if dt_pub > 0.0 else 1.0
        vx_pub = self.last_vx_pub_ * decay
        vy_pub = self.last_vy_pub_ * decay

        # When sonar is disabled fall back to z_fused_ as ground proxy (vx only).
        ground_for_vx = on_ground_for_vx or (
            not self.use_sonar_ and self.z_fused_ < self.ground_threshold_
        )

        R = self.body_to_world_R()

        if ground_for_vx:
            # Ground: wheel odometry primary (reliable down to 0 m/s),
            # airspeed as fallback (pitot unreliable below ~5 m/s).
            if self.use_wheel_odom_ and self.wheel_odom_ready_:
                v_fwd = self.wheel_vfwd_
            elif self.use_airspeed_ and not airspeed_stale and self.last_airspeed_ >= self.min_airspeed_:
                v_fwd = self.last_airspeed_
            else:
                v_fwd = 0.0
            v_world = R @ np.array([v_fwd, 0.0, 0.0])
            vx_pub = v_world[0]
            vy_pub = v_world[1]

        else:
            # Air vx priority:
            #   1. Airspeed — direct pitot measurement, pitch-independent (preferred).
            #      Note: airspeed = ground speed only in zero wind. Wind correction is
            #      disabled in simulation; enable use_wind_correction for real flight.
            #   2. FAST-LIO finite-diff — position-derived vx, EMA-smoothed.
            #   3. Thrust DR — physics model fallback when both above are unavailable.
            if self.use_airspeed_ and not airspeed_stale and self.last_airspeed_ >= self.min_airspeed_:
                v_world = R @ np.array([self.last_airspeed_, 0.0, 0.0])
                vx_pub = self.airspeed_weight_ * v_world[0] + (1.0 - self.airspeed_weight_) * vx_pub

            elif not fastlio_stale:
                # Airspeed absent or stale: FAST-LIO position-derived vx as fallback.
                vx_pub = self.vx_

            elif (self.use_thrust_dr_ and self.thrust_dr_ready_ and self.fastlio_ready_):
                # No airspeed, no active LiDAR: physics dead reckoning as last resort.
                v_world = R @ np.array([self.thrust_dr_vx_, 0.0, 0.0])
                vx_pub = self.thrust_dr_weight_ * v_world[0] + (1.0 - self.thrust_dr_weight_) * vx_pub
                vy_pub = self.thrust_dr_weight_ * v_world[1] + (1.0 - self.thrust_dr_weight_) * vy_pub

            # OF vx correction — pitch-gated blend on top of the primary vx source.
            # Downward camera is NOT a standalone vx source: at non-zero pitch the camera
            # normal tilts away from vertical, mixing pitch rotation into the flow field
            # and corrupting the forward velocity estimate. The pitch gate (max_of_pitch_deg)
            # limits application to near-level flight where contamination is negligible.
            pitch = math.asin(max(-1.0, min(1.0,
                2.0 * (self.qw_ * self.qy_ - self.qz_ * self.qx_))))
            if (self.use_optical_flow_ and not of_stale
                    and self.min_of_altitude_ <= self.z_fused_ <= self.max_of_altitude_
                    and abs(pitch) <= self.max_of_pitch_rad_):
                vx_of = (R @ np.array([self.of_vx_body_, self.of_vy_body_, 0.0]))[0]
                vx_pub = self.of_weight_ * vx_of + (1.0 - self.of_weight_) * vx_pub

            # vy — layered corrections applied in priority order.
            # 1. FAST-LIO lateral velocity (EMA-smoothed finite-diff) when scan-match active.
            if not fastlio_stale:
                vy_pub = self.vy_

            # 2. OF vy blend at low altitude.  vy is less sensitive to pitch contamination
            #    than vx (lateral flow is orthogonal to the pitch rotation axis), so no
            #    pitch gate is required here.
            if (self.use_optical_flow_ and not of_stale
                    and self.min_of_altitude_ <= self.z_fused_ <= self.max_of_altitude_):
                vy_of = (R @ np.array([self.of_vx_body_, self.of_vy_body_, 0.0]))[1]
                vy_pub = self.of_weight_ * vy_of + (1.0 - self.of_weight_) * vy_pub

            # 3. Forward camera vy above min_fwd_cam_altitude_.
            if (self.use_forward_camera_ and not fwd_cam_stale
                    and self.z_fused_ >= self.min_fwd_cam_altitude_):
                vy_world_fwd = (R @ np.array([0.0, self.fwd_cam_vy_body_, 0.0]))[1]
                vy_pub = self.fwd_cam_weight_ * vy_world_fwd + (1.0 - self.fwd_cam_weight_) * vy_pub

        self.last_vx_pub_ = vx_pub
        self.last_vy_pub_ = vy_pub

        # ESKF velocity takes precedence over complementary filter
        if self.eskf_ is not None and self.eskf_seeded_:
            _ev = self.eskf_.velocity
            vx_pub = float(_ev[0])
            vy_pub = float(_ev[1])
            self.last_vx_pub_ = vx_pub
            self.last_vy_pub_ = vy_pub

        self.odom_msg_.twist.twist.linear.x = vx_pub
        self.odom_msg_.twist.twist.linear.y = vy_pub
        self.odom_msg_.twist.twist.linear.z = self.vz_imu_ if not fastlio_stale else self.vz_baro_
        self.odom_msg_.twist.twist.angular.x = self.wx_
        self.odom_msg_.twist.twist.angular.y = self.wy_
        self.odom_msg_.twist.twist.angular.z = self.wz_

        # Pose covariance (x,y,z,roll,pitch,yaw)
        self.odom_msg_.pose.covariance = [
            pos_variance, 0, 0, 0, 0, 0,
            0, pos_variance, 0, 0, 0, 0,
            0, 0, pos_variance, 0, 0, 0,
            0, 0, 0, 0.01, 0, 0,
            0, 0, 0, 0, 0.01, 0,
            0, 0, 0, 0, 0, 0.01
        ]

        # Twist covariance (vx,vy,vz,wx,wy,wz)
        self.odom_msg_.twist.covariance = [
            vel_variance, 0, 0, 0, 0, 0,
            0, vel_variance, 0, 0, 0, 0,
            0, 0, vel_variance, 0, 0, 0,
            0, 0, 0, 0.1, 0, 0,
            0, 0, 0, 0, 0.1, 0,
            0, 0, 0, 0, 0, 0.1
        ]

        self.odom_pub_.publish(self.odom_msg_)

        # Diagnostics every ~2s
        if self.integration_count_ > 0 and (self.integration_count_ % int(max(1, self.publish_rate_ * 2)) == 0):
            self.publish_diagnostics(now)

        # TF
        if self.publish_tf_:
            t = TransformStamped()
            t.header.stamp = now.to_msg()
            t.header.frame_id = "odom"
            t.child_frame_id = self.child_frame_id_

            t.transform.translation.x = self.x_
            t.transform.translation.y = self.y_
            t.transform.translation.z = self.z_fused_
            t.transform.rotation.x = self.qx_
            t.transform.rotation.y = self.qy_
            t.transform.rotation.z = self.qz_
            t.transform.rotation.w = self.qw_
            self.tf_broadcaster_.sendTransform(t)

    def publish_diagnostics(self, now):
        diag_array = DiagnosticArray()
        diag_array.header.stamp = now.to_msg()

        status = DiagnosticStatus()
        status.name = "IMU Odometry"
        status.hardware_id = "marid_odom_node"

        if self.dropped_samples_ > 100 or self.warning_count_ > 50:
            status.level = DiagnosticStatus.WARN
            status.message = "High error/warning rate detected"
        else:
            status.level = DiagnosticStatus.OK
            status.message = "Operating normally"

        status.values.append(KeyValue(key="Integration Count", value=str(self.integration_count_)))
        status.values.append(KeyValue(key="Dropped Samples", value=str(self.dropped_samples_)))
        status.values.append(KeyValue(key="Warnings", value=str(self.warning_count_)))
        status.values.append(KeyValue(key="Position X", value=f"{self.x_:.3f}"))
        status.values.append(KeyValue(key="Position Y", value=f"{self.y_:.3f}"))
        status.values.append(KeyValue(key="Position Z (fused)", value=f"{self.z_fused_:.3f}"))

        vmag = math.sqrt(self.vx_**2 + self.vy_**2 + self.vz_imu_**2)
        status.values.append(KeyValue(key="Velocity Magnitude", value=f"{vmag:.3f}"))

        status.values.append(KeyValue(key="Accel Bias X", value=f"{self.accel_bias_[0]:.5f}"))
        status.values.append(KeyValue(key="Accel Bias Y", value=f"{self.accel_bias_[1]:.5f}"))
        status.values.append(KeyValue(key="Accel Bias Z", value=f"{self.accel_bias_[2]:.5f}"))

        status.values.append(KeyValue(key="IMU specific force", value=str(self.accel_is_sf_)))
        status.values.append(KeyValue(key="IMU XY integration", value=str(self.use_imu_xy_)))
        status.values.append(KeyValue(key="Barometer ready", value=str(self.baro_ready_)))
        status.values.append(KeyValue(key="Baro weight", value=f"{self.baro_weight_:.2f}"))
        status.values.append(KeyValue(key="FAST-LIO ready", value=str(self.fastlio_ready_)))
        status.values.append(KeyValue(key="FAST-LIO pos weight", value=f"{self.fastlio_pos_weight_:.2f}"))
        status.values.append(KeyValue(key="FAST-LIO yaw weight", value=f"{self.fastlio_yaw_weight_:.2f}"))
        now_diag = self.get_clock().now()
        def _age(stamp):
            if stamp is None:
                return float('inf')
            return (now_diag - stamp).nanoseconds / 1e9

        status.values.append(KeyValue(key="Airspeed ready", value=str(self.airspeed_ready_)))
        status.values.append(KeyValue(key="Airspeed (m/s)", value=f"{self.last_airspeed_:.2f}"))
        status.values.append(KeyValue(key="Airspeed age (s)", value=f"{_age(self.last_airspeed_stamp_):.2f}"))
        status.values.append(KeyValue(key="Yaw-rate assist (rad/s)", value=f"{self.yaw_rate_assist_:.5f}"))
        status.values.append(KeyValue(key="Wind X", value=f"{self.wind_x_:.2f}"))
        status.values.append(KeyValue(key="Wind Y", value=f"{self.wind_y_:.2f}"))
        status.values.append(KeyValue(key="OF ready", value=str(self.of_ready_)))
        status.values.append(KeyValue(key="OF age (s)", value=f"{_age(self.last_of_stamp_):.2f}"))
        status.values.append(KeyValue(key="OF vx_body", value=f"{self.of_vx_body_:.3f}"))
        status.values.append(KeyValue(key="OF vy_body", value=f"{self.of_vy_body_:.3f}"))
        sonar_str = f"{self.sonar_range_:.3f}" if self.sonar_range_ is not None else "N/A"
        status.values.append(KeyValue(key="Sonar AGL (m)", value=sonar_str))
        status.values.append(KeyValue(key="On ground", value=str(self._on_ground())))
        status.values.append(KeyValue(key="Wheel odom ready", value=str(self.wheel_odom_ready_)))
        status.values.append(KeyValue(key="Wheel vfwd (m/s)", value=f"{self.wheel_vfwd_:.3f}"))
        status.values.append(KeyValue(key="Fwd cam ready", value=str(self.fwd_cam_ready_)))
        status.values.append(KeyValue(key="Fwd cam age (s)", value=f"{_age(self.last_fwd_cam_stamp_):.2f}"))
        status.values.append(KeyValue(key="Fwd cam vy_body", value=f"{self.fwd_cam_vy_body_:.3f}"))
        status.values.append(KeyValue(key="Fwd cam vz_body", value=f"{self.fwd_cam_vz_body_:.3f}"))
        status.values.append(KeyValue(key="Thrust DR ready", value=str(self.thrust_dr_ready_)))
        status.values.append(KeyValue(key="Thrust (N)", value=f"{self.thrust_N_:.1f}"))
        status.values.append(KeyValue(key="Thrust DR vx_body", value=f"{self.thrust_dr_vx_:.3f}"))
        status.values.append(KeyValue(key="Drag coeff k_d", value=f"{self.drag_coeff_:.4f}"))
        status.values.append(KeyValue(key="RLS covariance P", value=f"{self.rls_P_:.6f}"))
        status.values.append(KeyValue(key="H2 remaining (kg)", value=f"{self.h2_remaining_:.3f}"))
        status.values.append(KeyValue(key="Drone mass (kg)", value=f"{self.drone_mass_:.2f}"))
        tank_p_str = f"{self.tank_pressure_:.1f}" if self.tank_pressure_ is not None else "N/A"
        tank_t_str = f"{self.tank_temperature_:.2f}" if self.tank_temperature_ is not None else "N/A"
        status.values.append(KeyValue(key="Tank pressure (Pa)", value=tank_p_str))
        status.values.append(KeyValue(key="Tank temperature (K)", value=tank_t_str))
        status.values.append(KeyValue(key="Sun valid", value=str(self.sun_valid_)))
        status.values.append(KeyValue(key="Sun elevation (deg)", value=f"{self.sun_elevation_deg_:.1f}"))
        sun_yaw_str = f"{math.degrees(self.sun_yaw_):.1f}" if self.sun_valid_ else "N/A"
        status.values.append(KeyValue(key="Sun yaw (deg)", value=sun_yaw_str))
        status.values.append(KeyValue(key="Sun age (s)", value=f"{_age(self.last_sun_stamp_):.2f}"))

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
