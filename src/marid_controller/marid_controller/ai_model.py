#!/usr/bin/env python3
"""
MARID AI Model Wrapper
Wrapper for loading and using trained neural network models.
Supports PyTorch and TensorFlow models.
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


class MaridAIModel:
    """
    Wrapper for AI model (PyTorch or TensorFlow).
    Input: State vector [x, y, z, vx, vy, vz, roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate, altitude, target_altitude, target_velocity]
    Output: Action vector [total_thrust, yaw_differential]
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
            state: numpy array of shape (15,) - [x, y, z, vx, vy, vz, roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate, altitude, target_altitude, target_velocity]
        Returns:
            action: numpy array of shape (2,) - [total_thrust, yaw_differential]
        """
        if self.model_ is None:
            return np.array([0.0, 0.0])
        
        # Convert state to appropriate format
        state = np.array(state, dtype=np.float32)
        
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
        
        return action


# Example PyTorch model architecture (for training)
# This class is only available if PyTorch is installed
# Use this as a reference when training your model
if PYTORCH_AVAILABLE:
    class MaridPolicyNetwork(nn.Module):
        """Neural network policy for MARID flight control"""
        def __init__(self, state_dim=15, action_dim=2, hidden_dim=128):
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

