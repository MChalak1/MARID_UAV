#!/usr/bin/env python3
"""
PID Controller Analysis Tool
Analyzes PID controller performance using the Python Control Systems Library.
"""

import control
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

def analyze_pid(kp, ki, kd, controller_name="PID Controller", save_plots=True):
    """
    Analyze a PID controller's step response and frequency response.
    
    Args:
        kp: Proportional gain
        ki: Integral gain
        kd: Derivative gain
        controller_name: Name for plot titles
        save_plots: If True, save plots to files instead of displaying
    """
    # Create PID transfer function: Kd*s^2 + Kp*s + Ki / s
    numerator = [kd, kp, ki]
    denominator = [1, 0, 0]  # 1/s (integrator)
    
    sys = control.TransferFunction(numerator, denominator)
    
    # Step response
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    t, y = control.step_response(sys, T=np.linspace(0, 10, 1000))
    plt.plot(t, y)
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'{controller_name} - Step Response')
    
    # Bode plot
    plt.subplot(1, 2, 2)
    mag, phase, omega = control.bode_plot(sys, plot=False)
    plt.semilogx(omega, 20 * np.log10(mag))
    plt.grid(True)
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('Magnitude (dB)')
    plt.title(f'{controller_name} - Bode Plot (Magnitude)')
    
    plt.tight_layout()
    
    if save_plots:
        filename = f'{controller_name.replace(" ", "_")}_analysis.png'
        plt.savefig(filename, dpi=150)
        print(f"Plot saved to: {filename}")
        plt.close()
    else:
        plt.show()
    
    # Print controller info
    print(f"\n{controller_name} Analysis:")
    print(f"  Kp = {kp:.3f}")
    print(f"  Ki = {ki:.3f}")
    print(f"  Kd = {kd:.3f}")

if __name__ == '__main__':
    # Analyze your current altitude PID
    print("Analyzing Altitude PID Controller...")
    analyze_pid(kp=2.0, ki=0.1, kd=0.5, controller_name="Altitude PID", save_plots=True)
    
    # You can analyze other controllers too
    print("\nAnalyzing Speed PID Controller...")
    analyze_pid(kp=1.0, ki=0.05, kd=0.3, controller_name="Speed PID", save_plots=True)
    
    print("\nAnalyzing Heading PID Controller...")
    analyze_pid(kp=1.0, ki=0.1, kd=0.3, controller_name="Heading PID", save_plots=True)