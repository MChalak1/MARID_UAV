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

## Notes
The mathematical foundation (calculus, optimization theory) underlying these implementations 
helps in understanding gradient descent convergence properties, hyperparameter sensitivity, and 
the relationship between learning rate and training epochs demonstrated in the experiments.
