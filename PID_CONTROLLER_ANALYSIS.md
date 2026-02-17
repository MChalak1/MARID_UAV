# PID Controller Analysis Guide

## Overview

This document explains the mathematical foundations of PID controllers and how to use the analysis tool (`marid_controller/scripts/analyze_pid.py`) to understand and tune PID controllers for the MARID drone.

## Table of Contents

1. [PID Controller Mathematics](#pid-controller-mathematics)
2. [Individual Terms Explained](#individual-terms-explained)
3. [Step Response Analysis](#step-response-analysis)
4. [Bode Plot Analysis](#bode-plot-analysis)
5. [Using the Analysis Tool](#using-the-analysis-tool)
6. [Tuning Guidelines](#tuning-guidelines)
7. [Examples](#examples)

---

## PID Controller Mathematics

### Time Domain Equation

The PID controller output in time domain:

```
u(t) = Kp × e(t) + Ki × ∫e(τ)dτ + Kd × de(t)/dt
      └─ P term ─┘  └── I term ───┘  └── D term ───┘
```

Where:
- `u(t)` = controller output (control signal, e.g., thrust adjustment in Newtons)
- `e(t)` = error = setpoint - actual value
- `Kp` = Proportional gain
- `Ki` = Integral gain
- `Kd` = Derivative gain

### Laplace Domain (Transfer Function)

Taking the Laplace transform:

```
U(s) = [Kp + Ki/s + Kd×s] × E(s)
```

Rearranging:

```
U(s)/E(s) = Kp + Ki/s + Kd×s
          = (Kd×s² + Kp×s + Ki) / s
```

**PID Transfer Function:**
```
G(s) = (Kd×s² + Kp×s + Ki) / s
```

This is the form used in the analysis script.

---

## Individual Terms Explained

### 1. Proportional Term (Kp)

**Time Domain:**
```
P(t) = Kp × e(t)
```

**Effect:**
- Responds to **current error** only
- Larger error → larger correction
- No memory (doesn't consider past)
- Provides immediate response

**Transfer Function:**
```
Gp(s) = Kp
```

**In Bode Plot:**
- Constant magnitude: `|Gp(jω)| = Kp` (flat line)
- Constant phase: `∠Gp(jω) = 0°`

**In Step Response:**
- Immediate response proportional to error
- **Cannot eliminate steady-state error** (leaves residual error)

**Example:**
```
Error = 1m, Kp = 2.0
→ P output = 2.0 × 1.0 = 2.0 N

Error = 0.5m, Kp = 2.0
→ P output = 2.0 × 0.5 = 1.0 N
```

### 2. Integral Term (Ki)

**Time Domain:**
```
I(t) = Ki × ∫₀ᵗ e(τ) dτ
```

**Effect:**
- **Accumulates error over time**
- Eliminates steady-state error
- Can cause overshoot if too high
- Provides "memory" of past errors

**Transfer Function:**
```
Gi(s) = Ki/s
```

**In Bode Plot:**
- Magnitude: `|Gi(jω)| = Ki/ω` (decreases as frequency increases)
  - At ω = 0.01 rad/s: `|Gi| = 0.1/0.01 = 10` (20 dB)
  - At ω = 0.1 rad/s: `|Gi| = 0.1/0.1 = 1` (0 dB)
  - At ω = 1 rad/s: `|Gi| = 0.1/1 = 0.1` (-20 dB)
- Phase: `∠Gi(jω) = -90°` (constant)
- **Slope: -20 dB/decade** (characteristic of integrator)

**In Step Response:**
- **Ramp output**: `I(t) = Ki × t` for constant error
- This is why the step response shows a linear increase

**Example:**
```
At t=0s:  Error = 1m → I = Ki × 0 = 0
At t=1s:  Error = 1m → I = Ki × 1 = 0.1 × 1 = 0.1
At t=2s:  Error = 1m → I = Ki × 2 = 0.1 × 2 = 0.2
At t=10s: Error = 1m → I = Ki × 10 = 0.1 × 10 = 1.0
```

### 3. Derivative Term (Kd)

**Time Domain:**
```
D(t) = Kd × de(t)/dt
```

**Effect:**
- Responds to **rate of change** of error
- Provides **damping** (reduces oscillations)
- **Amplifies noise** (can be problematic)
- Provides "anticipation" of future error

**Transfer Function:**
```
Gd(s) = Kd×s
```

**In Bode Plot:**
- Magnitude: `|Gd(jω)| = Kd×ω` (increases as frequency increases)
  - At ω = 0.01 rad/s: `|Gd| = 0.5 × 0.01 = 0.005` (-46 dB)
  - At ω = 0.1 rad/s: `|Gd| = 0.5 × 0.1 = 0.05` (-26 dB)
  - At ω = 1 rad/s: `|Gd| = 0.5 × 1 = 0.5` (-6 dB)
  - At ω = 10 rad/s: `|Gd| = 0.5 × 10 = 5` (14 dB)
- Phase: `∠Gd(jω) = +90°` (constant)
- **Slope: +20 dB/decade** (characteristic of differentiator)

**In Step Response:**
- **Impulse at t=0** (infinite spike, then zero)
- Provides initial "kick" then decays

**Example:**
```
Error changing rapidly: de/dt = 2 m/s
→ D output = 0.5 × 2 = 1.0 N

Error constant: de/dt = 0
→ D output = 0.5 × 0 = 0 N
```

---

## Step Response Analysis

### Step Input

**Step function:**
```
r(t) = { 0,  t < 0
       { 1,  t ≥ 0
```

**Laplace transform:**
```
R(s) = 1/s
```

### Step Response Mathematics

**Controller output:**
```
U(s) = G(s) × R(s)
     = [(Kd×s² + Kp×s + Ki) / s] × (1/s)
     = (Kd×s² + Kp×s + Ki) / s²
```

**Inverse Laplace (time domain):**
```
u(t) = Kd×δ(t) + Kp×u(t) + Ki×t
```

Where:
- `δ(t)` = Dirac delta (impulse at t=0)
- `u(t)` = unit step function
- `t` = time

**For constant error (step input):**
```
u(t) = Kp + Ki×t  (ignoring impulse)
```

**This explains the ramp in the plot:**
```
u(t) = 2.0 + 0.1×t
```

At different times:
- t=0: `u(0) = 2.0 + 0.1×0 = 2.0`
- t=1: `u(1) = 2.0 + 0.1×1 = 2.1`
- t=5: `u(5) = 2.0 + 0.1×5 = 2.5`
- t=10: `u(10) = 2.0 + 0.1×10 = 3.0`

### Interpreting Step Response

**What the plot shows:**
- **Open-loop response** of the PID controller (not the closed-loop system)
- How the controller's output changes over time for a step input
- The ramp is **expected** due to the integral term

**What it means:**
- The amplitude represents the controller's output (control signal)
- For altitude PID: this is the thrust adjustment in Newtons
- The increasing amplitude shows the integral term accumulating error

**Important Note:**
- This is **theoretical** behavior of the controller in isolation
- In a **closed-loop system**, the error decreases as the system responds
- The integral stops accumulating when error reaches zero
- The output stabilizes at the value needed to maintain the setpoint

---

## Bode Plot Analysis

### Frequency Response

**Substitute s = jω:**
```
G(jω) = Kp + Ki/(jω) + Kd×(jω)
      = Kp + Ki/jω + jKd×ω
```

**Magnitude:**
```
|G(jω)| = √[(Kp)² + (Ki/ω - Kd×ω)²]
```

**Phase:**
```
∠G(jω) = atan2(Kd×ω - Ki/ω, Kp)
```

### At Different Frequencies

**Low frequency (ω → 0):**
```
|G(jω)| ≈ Ki/ω  (dominated by integral term)
∠G(jω) ≈ -90°  (from integral term)
```
- **High gain** (good for steady-state accuracy)
- **-20 dB/decade slope** (characteristic of integrator)

**High frequency (ω → ∞):**
```
|G(jω)| ≈ Kd×ω  (dominated by derivative term)
∠G(jω) ≈ +90°  (from derivative term)
```
- **Increasing gain** (can amplify noise)
- **+20 dB/decade slope** (characteristic of differentiator)

**Mid frequency:**
- Proportional term dominates
- `|G(jω)| ≈ Kp`

### Interpreting Bode Plot

**What the plot shows:**
- How the controller's gain changes with input frequency
- Magnitude in dB (higher = stronger response)
- Frequency in rad/s (logarithmic scale)

**What it means:**
- **High gain at low frequencies**: Strong correction of slow, steady errors (good for steady-state accuracy)
- **Decreasing gain at high frequencies**: Filters out high-frequency noise (good for stability)
- **Smooth transition**: Well-balanced controller

**Good characteristics:**
- High gain at low frequencies (eliminates steady-state error)
- Decreasing gain at high frequencies (filters noise)
- Smooth, predictable response

**Bad characteristics:**
- Gain margin < 0 dB (unstable)
- Phase margin < 45° (marginally stable)
- Peak in magnitude response (resonance, causes oscillations)
- Too high gain at all frequencies (amplifies noise)

---

## Using the Analysis Tool

### Location

The analysis tool is located at:
```
marid_ws/src/marid_controller/scripts/analyze_pid.py
```

### Prerequisites

1. **Activate virtual environment:**
   ```bash
   source ~/venv_analysis/bin/activate
   ```

2. **Run the script:**
   ```bash
   python3 ~/marid_ws/src/marid_controller/scripts/analyze_pid.py
   ```

### Output

The script generates:
- **Step Response plot**: Shows controller's theoretical response to a step input
- **Bode Plot**: Shows frequency response (magnitude)
- **PNG files**: Saved in the current working directory

### Customizing Analysis

**To analyze different PID gains:**

Edit the script or modify the function calls:

```python
# Analyze custom PID configuration
analyze_pid(kp=3.0, ki=0.15, kd=0.7, controller_name="Custom PID")
```

### Comparing Configurations

**Example: Compare different altitude PID gains:**

```python
# Current configuration
analyze_pid(kp=2.0, ki=0.1, kd=0.5, controller_name="Current")

# More aggressive
analyze_pid(kp=3.0, ki=0.15, kd=0.7, controller_name="Aggressive")

# More conservative
analyze_pid(kp=1.5, ki=0.08, kd=0.6, controller_name="Conservative")
```

This generates separate plots for each configuration, allowing you to compare:
- Step response characteristics
- Frequency response behavior
- Theoretical performance

---

## Tuning Guidelines

### Understanding What Each Gain Does

| Gain | Effect | Too Low | Too High |
|------|--------|---------|----------|
| **Kp** | Response speed | Slow response, steady-state error | Overshoot, oscillations |
| **Ki** | Steady-state accuracy | Steady-state error persists | Overshoot, slow settling |
| **Kd** | Damping | Oscillations, overshoot | Noise amplification, instability |

### Tuning Process

1. **Start with P only** (Ki=0, Kd=0)
   - Increase Kp until you get oscillations
   - Reduce Kp by 50% (this is your starting Kp)

2. **Add I term** (Ki > 0)
   - Start with small Ki (e.g., 0.1)
   - Increase until steady-state error is eliminated
   - Watch for overshoot

3. **Add D term** (Kd > 0)
   - Start with small Kd (e.g., 0.5)
   - Increase to reduce oscillations
   - Watch for noise amplification

### Common Issues and Solutions

**Problem: Oscillations**
- **Solution**: Reduce Kp or increase Kd
- **Example**: Kp=2.0 → Kp=1.5, or Kd=0.5 → Kd=1.0

**Problem: Too slow response**
- **Solution**: Increase Kp
- **Example**: Kp=2.0 → Kp=2.5 or 3.0

**Problem: Overshoot**
- **Solution**: Reduce Kp or increase Kd
- **Example**: Kp=2.0 → Kp=1.5, or Kd=0.5 → Kd=0.7

**Problem: Steady-state error**
- **Solution**: Increase Ki
- **Example**: Ki=0.1 → Ki=0.15 or 0.2

**Problem: Noise/jitter**
- **Solution**: Increase Kd (but watch for instability)
- **Example**: Kd=0.5 → Kd=0.7 or 1.0

### Good vs Bad Step Response

| Characteristic | Good | Bad |
|---------------|------|-----|
| **Overshoot** | <10% | >30% |
| **Settling time** | Fast (<5s) | Very slow (>20s) |
| **Oscillations** | None or minimal | Large, persistent |
| **Steady-state error** | 0 | Non-zero |
| **Stability** | Stable | Unstable/diverging |

---

## Examples

### Example 1: Current Altitude PID

**Configuration:**
- Kp = 2.0
- Ki = 0.1
- Kd = 0.5

**Step Response:**
```
u(t) = 2.0 + 0.1×t
```
- Starts at 2.0 (P term)
- Increases by 0.1 per second (I term)
- Ramp is expected for open-loop

**Bode Plot:**
- High gain at low frequencies (~60 dB at 0.01 rad/s)
- Decreasing gain at high frequencies
- Good for steady-state accuracy and noise filtering

**Interpretation:**
- Moderate proportional response
- Slow integral accumulation (good for stability)
- Moderate derivative action (good damping)

### Example 2: More Aggressive Configuration

**Configuration:**
- Kp = 3.0
- Ki = 0.15
- Kd = 0.7

**Expected behavior:**
- Faster response (higher Kp)
- Faster error elimination (higher Ki)
- Better damping (higher Kd)
- **Risk**: May overshoot or oscillate

**When to use:**
- System is too slow
- Need faster altitude correction
- Can tolerate some overshoot

### Example 3: More Conservative Configuration

**Configuration:**
- Kp = 1.5
- Ki = 0.08
- Kd = 0.6

**Expected behavior:**
- Slower response (lower Kp)
- Slower error elimination (lower Ki)
- Better damping (higher Kd)
- **Benefit**: More stable, less overshoot

**When to use:**
- System is oscillating
- Need more stability
- Can tolerate slower response

---

## Real-Time Monitoring

### Important Note

The analysis tool provides **theoretical** behavior. To validate actual performance:

1. **Test in simulation:**
   ```bash
   ros2 launch marid_description gazebo.launch.py
   ros2 launch marid_controller option_a_controller.launch.py \
       altitude_kp:=2.5 altitude_ki:=0.15 altitude_kd:=0.7
   ```

2. **Monitor real-time behavior:**
   ```bash
   # Monitor altitude
   ros2 topic echo /odometry/filtered/local --field pose.pose.position.z
   
   # Monitor thrust
   ros2 topic echo /marid/thrust/total
   ```

3. **Use PlotJuggler:**
   - Record ROS2 bag files
   - Analyze actual step response
   - Compare with theoretical predictions

### Workflow

1. **Analyze** with script (predict behavior)
2. **Test** in simulation (validate predictions)
3. **Monitor** real-time (check actual performance)
4. **Iterate** (adjust gains based on results)

---

## References

- **Analysis Tool**: `marid_ws/src/marid_controller/scripts/analyze_pid.py`
- **PID Implementation**: `marid_ws/src/marid_controller/marid_controller/marid_guidance_tracker.py`
- **Python Control Library**: https://python-control.readthedocs.io/

---

## Summary

- **PID Controller**: Combines proportional, integral, and derivative terms
- **Step Response**: Shows theoretical open-loop behavior (ramp due to integral term)
- **Bode Plot**: Shows frequency response (high gain at low frequencies, decreasing at high frequencies)
- **Analysis Tool**: Helps predict behavior before testing in simulation
- **Tuning**: Iterative process of adjusting gains based on theoretical analysis and real-world testing

The theoretical analysis is a **design tool**, not a replacement for real-world testing. Always validate in simulation or hardware.
