# MARID Drone Physics Formulas & Equations Reference

**Project:** MARID Drone Simulation  
**Date:** 1/19/2026  
**Gazebo Version:** Sim 8  
**ROS2 Version:** Jazzy

---

## Table of Contents

1. [Aerodynamics - Lift and Drag Forces](#1-aerodynamics---lift-and-drag-forces)
2. [Propulsion - Thruster Forces](#2-propulsion---thruster-forces)
3. [Hydrodynamics - Linear and Angular Damping](#3-hydrodynamics---linear-and-angular-damping)
4. [Inertia and Mass Properties](#4-inertia-and-mass-properties)
5. [Gravity](#5-gravity)
6. [Control Systems](#6-control-systems)
7. [Joint Dynamics](#7-joint-dynamics)
8. [Coordinate Systems](#8-coordinate-systems)
9. [Sensor Models](#9-sensor-models)
10. [Aerodynamic Moments](#10-aerodynamic-moments)
11. [Aerodynamic Center and Stability](#11-aerodynamic-center-and-stability)
12. [Thrust-to-Weight Ratio](#12-thrust-to-weight-ratio)
13. [Parameter Summary](#13-parameter-summary)

---

## 1. Aerodynamics - Lift and Drag Forces

### 1.1 Lift Force (LiftDrag Plugin)

The lift force is calculated using the standard aerodynamic equation:

```
L = ½ × ρ × V² × A × CL(α)
```

**Variables:**
- `L` = Lift force (N)
- `ρ` = Air density = 1.225 kg/m³ (sea level)
- `V` = Airspeed magnitude (m/s)
- `A` = Surface area (m²)
- `CL(α)` = Lift coefficient as a function of angle of attack

### 1.2 Lift Coefficient Calculation

The lift coefficient varies with angle of attack:

```
CL(α) = cla × (α - a0)    for |α - a0| < α_stall
CL(α) = CL_stall × sign(α - a0)    for |α - a0| ≥ α_stall
```

**Parameters:**
- `cla` = Lift curve slope (dimensionless)
- `α` = Angle of attack (radians)
- `a0` = Zero-lift angle of attack (radians) = 0.0
- `α_stall` = Stall angle (radians)

**Current Configuration:**
| Component | cla | Area (m²) | α_stall (rad) |
|-----------|-----|-----------|----------------|
| Left Wing | 3.0 |   0.207    |    0.35       |
| Right Wing | 3.0 |  0.207    |    0.35       |
| Body b_l_f| 0.5  |  1.143    |    1.57       |
| Main Tail | 0.4 |   0.145    |    1.57       |
| Tail Left | 0.8 |   0.057    |    0.35       |
| Tail Right | 0.8 |  0.057    |    0.35       |

**⚠️ Important Note on Lift Coefficients (cla):**
The `cla` values listed above are **estimated/assumed values** based on typical aircraft parameters and empirical approximations. For accurate values, **Computational Fluid Dynamics (CFD) analysis** should be performed on the actual wing geometry to determine the true lift curve slope. These values are set to be as close as possible to reality based on available data, but CFD validation is recommended for production use.



### 1.3 Drag Force

```
D = ½ × ρ × V² × A × CD(α)
```

**Drag Coefficient:**
```
CD(α) = cda × (α - a0)² + CD0
```

**Current Drag Coefficients:**
- Wings: `cda = 0.5`
- Body: `cda = 0.2`
- Tail: `cda = 0.15`

### 1.4 Angle of Attack Calculation

```
α = atan2(V · up, V · forward) - a0
```

**Where:**
- `V` = Velocity vector relative to link frame
- `forward` = Forward direction vector (e.g., `[0, 1, 0]` for Y-axis)
- `up` = Up direction vector (e.g., `[0, 0, 1]` for Z-axis)

---

## 2. Propulsion - Thruster Forces

### 2.1 Thrust Force (Gazebo Thruster Plugin)

The Gazebo Thruster plugin calculates thrust using:

```
T = Ct × ρ × D⁴ × ω²
```

**Variables:**
- `T` = Thrust force (N)
- `Ct` = Thrust coefficient = 0.8
- `ρ` = Fluid density = 1.225 kg/m³ (air)
- `D` = Propeller diameter = 0.2 m
- `ω` = Angular velocity (rad/s)

### 2.2 Thrust to Angular Velocity Conversion (Controller)

**Simplified Model:**
```
ω = K × √T
```

**Where:**
- `ω` = Angular velocity command (rad/s)
- `K` = Thrust-to-angular-velocity gain = 50.0 (default)
- `T` = Desired thrust (N)

**More Accurate Model:**
```
ω = √(T / (Ct × ρ × D⁴))
```

### 2.3 Reaction Torque

Propeller rotation creates reaction torque on the body:

```
τ_reaction = -τ_propeller
```

**Propeller Torque:**
```
τ_propeller ∝ CQ × ρ × D⁵ × ω²
```

Where `CQ` = Torque coefficient (handled internally by plugin)

---

## 3. Hydrodynamics - Linear and Angular Damping

### 3.1 Linear Drag Forces

**X-axis (Forward):**
```
Fx = xDotU × u + xUabsU × |u| × u
```

**Y-axis (Side):**
```
Fy = yDotV × v + yVabsV × |v| × v
```

**Z-axis (Vertical):**
```
Fz = zDotW × w + zWabsW × |w| × w
```

**Variables:**
- `u, v, w` = Linear velocities in body frame (m/s)
- `xDotU, yDotV, zDotW` = Linear damping coefficients
- `xUabsU, yVabsV, zWabsW` = Quadratic drag coefficients

**Current Values:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| xDotU | -0.5 | Forward linear damping |
| yDotV | -1.0 | Side linear damping |
| zDotW | -1.0 | Vertical linear damping |
| xUabsU | -0.1 | Forward quadratic drag |
| yVabsV | -0.2 | Side quadratic drag |
| zWabsW | -0.2 | Vertical quadratic drag |

### 3.2 Angular Damping Torques

**Roll (X-axis):**
```
τx = kDotP × p + kPabsP × |p| × p
```

**Pitch (Y-axis):**
```
τy = mDotQ × q + mQabsQ × |q| × q
```

**Yaw (Z-axis):**
```
τz = nDotR × r + nRabsR × |r| × r
```

**Variables:**
- `p, q, r` = Angular velocities (roll, pitch, yaw) in rad/s
- `kDotP, mDotQ, nDotR` = Angular damping coefficients
- `kPabsP, mQabsQ, nRabsR` = Quadratic angular damping

**Current Values:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| kDotP | -2.0 | Roll linear damping |
| mDotQ | -2.0 | Pitch linear damping |
| nDotR | -2.0 | Yaw linear damping |
| kPabsP | -1.0 | Roll quadratic damping |
| mQabsQ | -1.0 | Pitch quadratic damping |
| nRabsR | -1.0 | Yaw quadratic damping |

---

## 4. Inertia and Mass Properties

### 4.1 Center of Mass (COM)

**Total Mass:**
```
M_total = Σ m_i
```

**COM Position:**
```
r_COM = (1/M_total) × Σ (m_i × r_i)
```

### 4.2 Mass Distribution

| Component | Mass (kg) |
|-----------|-----------|
| base_link_front | 3.090 |
| left_wing | 1.44 |
| right_wing | 1.44 |
| main_tail | 0.691 |
| tail_left | 0.166 |
| tail_right | 0.166 |
| thruster_L | 0.05 |
| thruster_R | 0.05 |
| thruster_center | 0.05 |
| **Total** | **~6.1** |

### 4.3 Inertia Tensor

```
I = [Ixx  Ixy  Ixz]
    [Ixy  Iyy  Iyz]
    [Ixz  Iyz  Izz]
```

**Main Body Inertia (base_link_front):**
- `Ixx = 0.111 kg·m²` (roll inertia)
- `Iyy = 0.025 kg·m²` (pitch inertia)
- `Izz = 0.129 kg·m²` (yaw inertia)
- `Ixy = Ixz = Iyz = 0` (no cross-coupling)

### 4.4 Rotational Dynamics (Euler's Equations)

```
τ = I × α + ω × (I × ω)
```

**Where:**
- `τ` = Torque vector (N·m)
- `I` = Inertia tensor (kg·m²)
- `α` = Angular acceleration (rad/s²)
- `ω` = Angular velocity vector (rad/s)

---

## 5. Gravity

### 5.1 Gravitational Force

```
F_gravity = m × g
```

**Where:**
- `m` = Mass (kg)
- `g` = Gravity vector (m/s²)

**Current Configuration:**
- **Testing:** `g = [0, 0, 0]` m/s² (disabled)
- **Standard:** `g = [0, 0, -9.81]` m/s² (Earth gravity)

---

## 6. Control Systems

### 6.1 Thrust Rate Limiting

```
ΔT_max = rate_limit × Δt
T_new = T_old + clamp(ΔT, -ΔT_max, ΔT_max)
```

**Parameters:**
- `rate_limit` = 50.0 N/s (default)
- `Δt` = Time step (s)

### 6.2 Exponential Smoothing

```
T_smooth = T_old + α × (T_target - T_old)
```

**Where:**
- `α` = Smoothing factor = 0.1 (default)
- `T_target` = Desired thrust (N)
- `T_old` = Previous thrust value (N)

### 6.3 Differential Thrust for Yaw Control

```
T_left = T_base + ΔT_diff
T_right = T_base - ΔT_diff
```

**Differential Calculation:**
```
ΔT_diff = yaw_differential × gain × T_base
```

**Parameters:**
- `yaw_differential` = -1.0 to 1.0 (control input)
- `gain` = 0.3 (30% max differential)
- `T_base` = Base thrust value (N)

---

## 7. Joint Dynamics

### 7.1 Joint Torque (with Damping and Friction)

```
τ_joint = τ_control - b × ω - f × sign(ω)
```

**Where:**
- `b` = Damping coefficient (N·m·s/rad)
- `f` = Friction coefficient (N·m)
- `ω` = Joint angular velocity (rad/s)
- `τ_control` = Control torque (N·m)

### 7.2 Joint Configuration

**Thruster Joints:**
- `damping = 0.0` (zero damping for reaction torque visibility)
- `friction = 0.0` (zero friction)

**Control Surface Joints (wings/tail):**
- `damping = 10.0`
- `friction = 5.0`

---

## 8. Coordinate Systems

### 8.1 Gazebo Frame Convention

- **X-axis**: Right (roll axis)
- **Y-axis**: Forward (pitch axis) - **Primary flight direction**
- **Z-axis**: Up (yaw axis)

### 8.2 Euler Angles (RPY)

**Roll Rotation Matrix:**
```
R(roll) = [1     0        0    ]
          [0  cos(φ) -sin(φ)]
          [0  sin(φ)  cos(φ)]
```

**Pitch Rotation Matrix:**
```
P(pitch) = [ cos(θ)  0  sin(θ)]
           [     0   1      0 ]
           [-sin(θ)  0  cos(θ)]
```

**Yaw Rotation Matrix:**
```
Y(yaw) = [cos(ψ) -sin(ψ)  0]
         [sin(ψ)  cos(ψ)  0]
         [   0       0    1]
```

**Combined Rotation:**
```
R_total = Y(yaw) × P(pitch) × R(roll)
```

---

## 9. Sensor Models

### 9.1 IMU Noise Model (Gaussian)

```
measurement = true_value + N(μ, σ²)
```

**Noise Parameters:**
- Angular velocity: `σ = 2×10⁻⁴` rad/s
- Linear acceleration: `σ = 1.7×10⁻²` m/s²
- Update rate: 100 Hz

### 9.2 GPS Noise Model

```
position_noise = N(0, σ²)
```

**Noise Parameters:**
- Horizontal position: `σ = 2.5` m
- Vertical position: `σ = 2.5` m
- Horizontal velocity: `σ = 1.5` m/s
- Vertical velocity: `σ = 1.5` m/s
- Update rate: 10 Hz

### 9.3 Barometer Noise

```
pressure_noise = N(0, 5.0²)` Pa
```

**Update rate:** 50 Hz

### 9.4 Airspeed Sensor (Pitot)

```
airspeed_noise = N(0, 0.1²)` m/s
```

**Update rate:** 50 Hz

### 9.5 Magnetometer Noise

```
magnetic_field_noise = N(0, 0.01²)` T
```

**Update rate:** 50 Hz

---

## 10. Aerodynamic Moments

### 10.1 Moment from Lift Force

```
M = r × F_lift
```

**Where:**
- `r` = Position vector from COM to center of pressure
- `F_lift` = Lift force vector

### 10.2 Pitch Moment (Critical for Stability)

```
M_pitch = (y_CP - y_COM) × L
```

**Where:**
- `y_CP` = Y-position of center of pressure
- `y_COM` = Y-position of center of mass
- `L` = Lift force magnitude

**Sign Convention:**
- Positive moment = pitch-up (nose up)
- Negative moment = pitch-down (nose down)

### 10.3 Roll Moment

```
M_roll = (z_CP - z_COM) × F_y - (y_CP - y_COM) × F_z
```

### 10.4 Yaw Moment

```
M_yaw = (x_CP - x_COM) × F_y - (y_CP - y_COM) × F_x
```

---

## 11. Aerodynamic Center and Stability

### 11.1 Definition of Aerodynamic Center (AC)

The **Aerodynamic Center (AC)** is the point on an aircraft where the pitching moment is independent of angle of attack. It represents the weighted average position of all lift-generating surfaces relative to their lift contributions.

**Key Properties:**
- The AC is the effective point where all aerodynamic forces can be considered to act
- For pitch stability, the AC position relative to the Center of Gravity (COG) is critical
- The AC is typically located at 25-30% of the mean aerodynamic chord for most airfoils

### 11.2 Calculation of Aerodynamic Center

For a multi-surface aircraft, the AC position is calculated as:

```
Y_AC = Σ(Lift_i × Y_i) / Σ(Lift_i)
```

**Where:**
- `Y_AC` = Y-position of aerodynamic center (m)
- `Lift_i` = Lift contribution from surface i (proportional to `cla_i × Area_i`)
- `Y_i` = Y-position of surface i's center of pressure (m)

**Lift Contribution Calculation:**
```
Lift_contribution_i = cla_i × Area_i
```

### 11.3 Relationship to Center of Gravity (COG)

The relative positions of AC and COG determine **pitch stability**:

#### Stable Configuration (AC behind COG):
```
Y_AC < Y_COG  (AC is behind COG)
```

**Behavior:**
- When nose pitches down → AC generates more lift → creates pitch-up moment → **restores level flight**
- When nose pitches up → AC generates less lift → creates pitch-down moment → **restores level flight**
- **Result:** Aircraft is naturally stable and self-correcting

#### Unstable Configuration (AC ahead of COG):
```
Y_AC > Y_COG  (AC is ahead of COG)
```

**Behavior:**
- When nose pitches down → AC generates more lift → creates pitch-down moment → **amplifies the dive**
- When nose pitches up → AC generates less lift → creates pitch-up moment → **amplifies the climb**
- **Result:** Aircraft is unstable and requires constant control input

### 11.4 Static Margin

The **static margin** quantifies stability:

```
Static_Margin = (Y_COG - Y_AC) / Mean_Aerodynamic_Chord
```

**Typical Values:**
- **Stable:** Static margin = 5-15% (AC 5-15% behind COG)
- **Neutral:** Static margin = 0% (AC at COG)
- **Unstable:** Static margin < 0% (AC ahead of COG)

### 11.5 Current MARID Configuration Analysis

**Component Positions and Lift Contributions:**

| Component | Y Position (m) | cla | Area (m²) | Lift Contribution | Moment Contribution (rel to COG) |
|-----------|----------------|-----|-----------|-------------------|-----------------------------------|
| Left Wing | +0.28 | 3.0 | 0.207 | 0.621 | +0.081 (at +0.13 m) |
| Right Wing | +0.28 | 3.0 | 0.207 | 0.621 | +0.081 (at +0.13 m) |
| Body | 0.0 | 0.5 | 1.143 | 0.572 | -0.086 (at -0.15 m) |
| Main Tail | -0.2115 | 0.4 | 0.145 | 0.058 | -0.021 (at -0.3615 m) |
| Tail Left | -0.2415 | 0.8 | 0.057 | 0.046 | -0.018 (at -0.3915 m) |
| Tail Right | -0.2415 | 0.8 | 0.057 | 0.046 | -0.018 (at -0.3915 m) |
| **Total** | - | - | - | **1.964** | **-0.001** |

**Aerodynamic Center Calculation (relative to COG at 0.15 m):**
```
Y_AC_rel = (0.081 + 0.081 - 0.086 - 0.021 - 0.018 - 0.018) / 1.964
Y_AC_rel = -0.001 / 1.964
Y_AC_rel ≈ -0.0005 m ≈ 0.0 m (nearly at COG)
```

**Center of Gravity:**
```
Y_COG = 0.15 m  (moved forward to account for sensors, batteries, and electronics in front)
```

**Stability Analysis:**
```
Y_AC (absolute) = 0.15 m  (nearly at COG)
Y_COG = 0.15 m
Static Margin ≈ 0% (NEARLY NEUTRAL)
```

**Result:** With main tail `cla = 0.4` and V-tail surfaces set to reasonable `cla = 0.8`, the aerodynamic center is now **nearly at the COG** (within 0.0005 m). This makes the aircraft **nearly neutrally stable**. The configuration is more realistic with lower tail lift coefficients.

**⚠️ Important Note on Aerodynamic Center and Center of Gravity:**
The calculated `Y_AC` and `Y_COG` values are based on:
- **Estimated lift coefficients (cla)** - See note in Section 1.2
- **Assumed center of pressure positions** - Based on geometric centers
- **Estimated mass distribution** - Based on component masses and positions

For accurate values, **CFD analysis** should be performed to:
- Determine true lift coefficients for each surface
- Calculate actual center of pressure locations
- Validate aerodynamic center position

These values are set to be **as close as possible to reality** based on available data, but CFD validation is strongly recommended for production use and flight-critical applications.

### 11.6 Solutions for Potential Instability (Any Aircraft)

#### Option 1: Move Wings Rearward
```
Target: Y_AC < Y_COG
Move wings from Y = +0.28 m to Y ≈ 0.0 m (at COG)
```

#### Option 2: Increase Tail Lift
```
Increase tail cla from 0.4 to 0.8-1.2
Or increase tail area
```

#### Option 3: Move Tail Further Back
```
Extend tail from Y = -0.21 m to Y = -0.4 to -0.5 m
Increases moment arm for tail lift
```

#### Option 4: Reduce Wing Lift Forward
```
Reduce wing cla or area
```

### 11.7 Importance for Flight Control

**For Stable Aircraft:**
- Natural tendency to return to level flight
- Reduced control effort required
- Safer and more predictable behavior

**For Unstable Aircraft:**
- Requires constant control input (fly-by-wire)
- Higher pilot workload
- Can be more maneuverable but less forgiving

**For MARID Drone:**
- Current unstable configuration requires active pitch control
- Consider moving AC behind COG for passive stability
- Or implement robust pitch control system

### 11.8 Modular Control and Agility Benefits

**Modular Wing Control:**
The MARID drone features **fully rotating wings** that can be controlled independently. This modular control capability, combined with adjustable `Y_COG` and `Y_AC` positions, offers significant agility advantages:

**Benefits of Modular Control:**
1. **Dynamic Stability Adjustment:**
   - By rotating wings, the effective `Y_AC` can be shifted in real-time
   - Allows the aircraft to switch between stable and maneuverable configurations
   - Enables adaptive flight characteristics based on mission requirements

2. **Enhanced Maneuverability:**
   - Independent wing control provides roll, pitch, and yaw authority
   - Fully rotating wings can generate forces in multiple directions
   - Enables aggressive maneuvers not possible with fixed-wing aircraft

3. **Agility Through COG/AC Modulation:**
   - Adjustable `Y_COG` (via weight distribution or ballast) allows stability tuning
   - Dynamic `Y_AC` control (via wing rotation) enables real-time stability changes
   - **Coupled together**, these provide exceptional agility:
     - **Stable mode:** COG ahead of AC for steady flight
     - **Maneuverable mode:** AC ahead of COG for high agility
     - **Transition:** Smoothly switch between modes via wing rotation

4. **Flight Regime Optimization:**
   - Different flight phases (takeoff, cruise, landing, maneuvering) can use different stability configurations
   - Modular control allows optimization for each regime
   - Reduces control effort in stable modes, maximizes agility in maneuverable modes

**Implementation Considerations:**
- Wing rotation changes effective angle of attack → changes lift → shifts AC
- Mass redistribution (battery, payload) can shift COG
- Real-time adjustment of both provides unprecedented control authority

---

## 12. Thrust-to-Weight Ratio

### 11.1 Calculation

```
TWR = T_total / W_total
```

**Where:**
- `T_total` = Sum of all thruster forces (N)
- `W_total` = Total weight = `M_total × g` (N)

### 11.2 Flight Requirements

- **Hover:** `TWR = 1.0`
- **Stable Flight:** `TWR > 1.0`
- **Typical Range:** `TWR = 1.5 - 2.5` for drones
- **Current Configuration:** `TWR = 2.5` (default)

---

## 13. Parameter Summary

### 12.1 Physical Constants

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| Air Density (Sea Level) | ρ | 1.225 | kg/m³ |
| Standard Gravity | g | -9.81 | m/s² |
| Propeller Diameter | D | 0.2 | m |
| Thrust Coefficient | Ct | 0.8 | - |

### 12.2 Control Parameters

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| Thrust-to-AngVel Gain | K | 50.0 | rad·s⁻¹/√N |
| Thrust Rate Limit | rate_limit | 50.0 | N/s |
| Smoothing Factor | α | 0.1 | - |
| Differential Gain | gain | 0.3 | - |

### 12.3 Mass Properties

| Component | Mass (kg) |
|-----------|-----------|
| Total Mass | ~6.1 | kg |
| Main Body | 3.090 | kg |
| Wings (each) | 1.44 | kg |
| Main Tail | 0.691 | kg |

### 12.4 Aerodynamic Coefficients

| Component | Lift Coeff (cla) | Drag Coeff (cda) | Area (m²) |
|-----------|------------------|------------------|-----------|
| Wings | 3.0 | 0.5 | 0.207 |
| Body | 0.5 | 0.2 | 1.143 |
| Main Tail | 0.4 | 0.15 | 0.145 |
| Tail Surfaces | 0.8 | 0.5 | 0.057 |

### 12.5 Damping Coefficients

| Type | Linear | Quadratic |
|------|--------|-----------|
| Forward (X) | -0.5 | -0.1 |
| Side (Y) | -1.0 | -0.2 |
| Vertical (Z) | -1.0 | -0.2 |
| Roll (P) | -2.0 | -1.0 |
| Pitch (Q) | -2.0 | -1.0 |
| Yaw (R) | -2.0 | -1.0 |

---

## Appendix A: Unit Conversions

| From | To | Conversion Factor |
|------|-----|------------------|
| RPM | rad/s | × (2π/60) |
| rad/s | RPM | × (60/2π) |
| degrees | radians | × (π/180) |
| radians | degrees | × (180/π) |
| N | lbf | × 0.224809 |
| kg | lb | × 2.20462 |

## Appendix B: Reference Values

### Typical Aircraft Parameters

- **Small UAV Mass:** 1-10 kg
- **Thrust-to-Weight Ratio:** 1.5-3.0
- **Wing Loading:** 50-200 N/m²
- **Stall Speed:** 5-15 m/s
- **Cruise Speed:** 10-30 m/s

### Propeller Parameters

- **Diameter Range:** 0.1-0.5 m
- **Thrust Coefficient:** 0.6-1.0
- **Efficiency:** 0.6-0.8

---

## Document Revision History

| Date | Version | Changes |
|------|---------|---------|
| 2026 | 1.0 | Initial document creation |

---

**End of Document**
