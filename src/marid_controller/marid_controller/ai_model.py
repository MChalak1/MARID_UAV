#!/usr/bin/env python3
"""
MARID AI Model Wrapper
Wrapper for loading and using trained neural network models.
Supports PyTorch and TensorFlow models.

State Vector Specification (20 dimensions):
  Base state (12): [x, y, z, vx, vy, vz, roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
  Waypoint info (8): [waypoint_x, waypoint_y, distance_to_waypoint, heading_error, altitude_min, altitude_max, target_altitude, target_velocity]
  Note: z (base state index 2) represents altitude from EKF/odom. No redundant altitude dimension.
  
Output Action Vector (2 dimensions):
  [total_thrust, yaw_differential]
"""
import numpy as np
import os

# Try importing ML frameworks
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# State vector dimension constant
# Base state: 12-D (no redundant altitude)
# Extended state: 20-D (12 base + 8 waypoint info)
STATE_DIM = 20
# Action dimension: will be 3 for high-level commands (heading_rate, speed, altitude_target)
# Currently 2 for backward compatibility (thrust, yaw_diff) - will migrate to high-level
ACTION_DIM = 2  # TODO: Migrate to 3-D for high-level: [desired_heading_rate, desired_speed, desired_altitude]


class MaridAIModel:
    """
    Wrapper for AI model (PyTorch or TensorFlow).
    
    State Vector (20 dimensions):
        [0:3]   Position: x, y, z (m) - Note: z is altitude from EKF/odom
        [3:6]   Linear velocity: vx, vy, vz (m/s)
        [6:9]   Attitude: roll, pitch, yaw (rad)
        [9:12]  Angular velocity: roll_rate, pitch_rate, yaw_rate (rad/s)
        [12:13] Waypoint position: waypoint_x, waypoint_y (m)
        [14]    Distance to waypoint (m)
        [15]    Heading error: desired_heading - current_yaw (rad, normalized [-pi, pi])
        [16]    Altitude minimum constraint (m)
        [17]    Altitude maximum constraint (m)
        [18]    Target altitude (m)
        [19]    Target velocity (m/s)
        
    Output Action Vector (currently 2-D, will migrate to 3-D high-level):
        Current (low-level, backward compat):
            [0] Total thrust (N)
            [1] Yaw differential (rad/s or normalized differential)
        Future (high-level, Option A):
            [0] Desired heading rate (rad/s) or desired heading (rad)
            [1] Desired speed (m/s)
            [2] Desired altitude (m) [optional, can use existing PID]
    """
    def __init__(self, model_path=None, model_type='pytorch'):
        self.model_path_ = model_path
        self.model_type_ = model_type
        self.model_ = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Create a simple placeholder model for testing
            self.model_ = self._create_placeholder_model()
            print("Warning: Using placeholder model. Train a real model for actual control.")
    
    def _create_placeholder_model(self):
        """Create a simple placeholder model that returns zero actions"""
        class PlaceholderModel:
            def predict(self, state):
                # Return zero actions (no control)
                return np.array([0.0, 0.0])
        return PlaceholderModel()
    
    def load_model(self, model_path):
        """Load trained model from file"""
        if self.model_type_ == 'pytorch' and PYTORCH_AVAILABLE:
            try:
                self.model_ = torch.load(model_path, map_location='cpu')
                self.model_.eval()
                print(f"Loaded PyTorch model from {model_path}")
            except Exception as e:
                print(f"Failed to load PyTorch model: {e}")
                self.model_ = self._create_placeholder_model()
        elif self.model_type_ == 'tensorflow' and TENSORFLOW_AVAILABLE:
            try:
                self.model_ = tf.keras.models.load_model(model_path)
                print(f"Loaded TensorFlow model from {model_path}")
            except Exception as e:
                print(f"Failed to load TensorFlow model: {e}")
                self.model_ = self._create_placeholder_model()
        else:
            print(f"ML framework not available for {self.model_type_}")
            self.model_ = self._create_placeholder_model()
    
    def predict(self, state):
        """
        Predict control action from state.
        
        Args:
            state: numpy array of shape (20,) with state vector as specified in class docstring
                  Expected order: [pos(3), vel(3), att(3), ang_vel(3), 
                                  waypoint(2), distance(1), heading_err(1), 
                                  alt_min(1), alt_max(1), target_alt(1), target_vel(1)]
                  Indices: [0:11] base state (12-D, z is altitude), [12:13] waypoint, [14] distance, 
                          [15] heading_err, [16] alt_min, [17] alt_max, 
                          [18] target_alt, [19] target_vel
                  Note: Base state z (index 2) is altitude. No redundant altitude dimension.
        
        Returns:
            action: numpy array of shape (2,) - [total_thrust, yaw_differential]
        """
        if self.model_ is None:
            return np.array([0.0, 0.0])
        
        # Convert state to appropriate format and validate
        state = np.array(state, dtype=np.float32)
        
        # Validate state dimension
        if state.shape[0] != STATE_DIM:
            raise ValueError(
                f"State vector dimension mismatch: expected {STATE_DIM}, got {state.shape[0]}. "
                f"Please ensure the controller sends the correct state vector format."
            )
        
        # Check for NaN or Inf values
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            raise ValueError(
                f"State vector contains NaN or Inf values. State: {state}"
            )
        
        if self.model_type_ == 'pytorch' and PYTORCH_AVAILABLE:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).unsqueeze(0)
                action_tensor = self.model_(state_tensor)
                action = action_tensor.squeeze(0).numpy()
        elif self.model_type_ == 'tensorflow' and TENSORFLOW_AVAILABLE:
            state_tensor = tf.expand_dims(state, 0)
            action_tensor = self.model_(state_tensor)
            action = action_tensor.numpy().squeeze()
        else:
            # Placeholder model
            action = self.model_.predict(state)
        
        # Validate action dimension
        if action.shape[0] != ACTION_DIM:
            raise ValueError(
                f"Action vector dimension mismatch: expected {ACTION_DIM}, got {action.shape[0]}"
            )
        
        return action


# Example PyTorch model architecture (for training)
# This class is only available if PyTorch is installed
# Use this as a reference when training your model
if PYTORCH_AVAILABLE:
    class MaridPolicyNetwork(nn.Module):
        """
        Neural network policy for MARID flight control.
        
        Input: 20-dimensional state vector (12 base + 8 waypoint info)
        Output: Currently 2-D [total_thrust, yaw_differential] for backward compatibility
                Will migrate to 3-D high-level: [desired_heading_rate, desired_speed, desired_altitude]
        """
        def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=128):
            super(MaridPolicyNetwork, self).__init__()
            
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, hidden_dim)
            self.fc4 = nn.Linear(hidden_dim, action_dim)
            
            self.relu = nn.ReLU()
            self.tanh = nn.Tanh()
        
        def forward(self, state):
            x = self.relu(self.fc1(state))
            x = self.relu(self.fc2(x))
            x = self.relu(self.fc3(x))
            action = self.tanh(self.fc4(x))  # Output in [-1, 1] range
            
            # Scale outputs: thrust [0, 30], yaw_differential [-0.2, 0.2]
            thrust = (action[:, 0] + 1.0) * 15.0  # Scale to [0, 30]
            yaw_diff = action[:, 1] * 0.2  # Scale to [-0.2, 0.2]
            
            return torch.stack([thrust, yaw_diff], dim=-1)

