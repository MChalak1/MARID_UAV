# Gradient Descent Optimization Studies

## Overview
This repository contains implementations and experiments exploring gradient descent optimization algorithms, including parameter sensitivity analysis and visualization.

## Files

### `gradient_descent_1D.py`
- **Purpose**: Initial 1D gradient descent implementation - the first file created in this project, establishing the foundational optimization framework
- **Features**:
  - Simple quadratic function: `f(x) = 3x² - 3x + 4` with analytical derivative
  - Basic gradient descent loop with fixed learning rate and training epochs
  - Trajectory tracking: stores parameter values and gradients over iterations
  - Convergence-based stopping: stops when gradient magnitude falls below threshold (0.0001)
  - Visualization of function, derivative, convergence path, and final minimum
- **Key Results**: Demonstrates core gradient descent mechanics - iterative parameter updates, convergence behavior, and the relationship between learning rate and required iterations
- **Note**: This foundational implementation introduced the core concepts (learning rate, epochs, convergence criteria) that were later expanded into systematic parameter studies and 2D implementations

### `gradient_descent_2D.py`
- **Purpose**: Initial implementation of 2D gradient descent/ascent
- **Features**: 
  - Symbolic differentiation using SymPy
  - Visualization of optimization trajectories
  - Tests on two different 2D functions
- **Key Results**: Demonstrates convergence to local minima/maxima with trajectory visualization

### `gradient_descent_2D_refactored.py`
- **Purpose**: Refactored version with reusable functions
- **Features**:
  - `run_gradient()` function for DRY code
  - `plot_gradient_result()` helper for visualization
  - Convergence tolerance (`tol`) for early stopping
- **Key Results**: Same functionality with improved code organization

### `gradient_descent_1D_experiments.py`
- **Purpose**: Parameter sensitivity analysis
- **Experiments**:
  1. **Varying Starting Location**: Shows basin of attraction - different starting points converge to different local minima
  2. **Learning Rate Variation**: Demonstrates optimal learning rate range for convergence
  3. **Learning Rate × Epochs**: Heatmap showing trade-off between learning rate and training epochs
- **Key Insights**: 
  - Low learning rates require more epochs
  - Starting location determines which minimum is found
  - Optimal learning rate range identified from heatmap
 
 ### `parametric_experiments_1D.py` 
- **Purpose**: Compare constant vs adaptive learning rate schedules on 1D gradient descent from a single run
- **Features**:
  - **Basic GD**: Fixed learning rate with trajectory and derivative plots
  - **Step & exponential decay**: Decay every 50 steps using `lr = lr₀ * exp(-j / training_epochs)`
  - **Gradient-based adaptive lr**: Learning rate proportional to |gradient| with cap (e.g. 2× base lr) for stability
  - **Time-based linear decay**: Linear schedule from initial lr down to 0 over training epochs
  - **Comparison**: One run from the same starting point for constant lr and all three schedules; plots for parameter trajectory, derivative, and learning rate
- **Key Results**: Same baseline (constant lr) vs step/exp decay, gradient-based, and time-based schedules; illustrates how schedule choice affects convergence and lr over time
- **Note**: Builds on the basic 1D implementation with explicit lr scheduling and a single-run comparison for fair visualization

## Notes
The mathematical foundation (calculus, optimization theory) underlying these implementations 
helps in understanding gradient descent convergence properties, hyperparameter sensitivity, and 
the relationship between learning rate and training epochs demonstrated in the experiments.
