#!/usr/bin/env python3
"""
MARID State Normalization Module
Computes and applies mean/std normalization for neural network training.
Critical for ML: neural networks perform poorly with unnormalized inputs.
"""
import numpy as np
import json
import os
from pathlib import Path


class StateNormalizer:
    """
    Normalizes state vectors using mean and standard deviation.
    
    For ML training, state values can vary wildly:
        - Position: 0-10000+ meters
        - Velocities: m/s
        - Angles: radians [-pi, pi]
        - Distances: potentially large
    
    Normalization: state_norm = (state - mean) / (std + eps)
    """
    
    def __init__(self, mean=None, std=None, eps=1e-8):
        """
        Initialize normalizer.
        
        Args:
            mean: numpy array of shape (STATE_DIM,) - mean of each dimension
            std: numpy array of shape (STATE_DIM,) - standard deviation of each dimension
            eps: small value to prevent division by zero
        """
        self.mean_ = mean
        self.std_ = std
        self.eps_ = eps
        self.fitted_ = (mean is not None) and (std is not None)
    
    def fit(self, states):
        """
        Compute mean and std from a collection of state vectors.
        
        Args:
            states: numpy array of shape (N, STATE_DIM) where N is number of samples
        """
        if len(states) == 0:
            raise ValueError("Cannot fit normalizer with empty state array")
        
        states = np.array(states)
        if states.ndim != 2:
            raise ValueError(f"Expected 2D array (N, STATE_DIM), got shape {states.shape}")
        
        self.mean_ = np.mean(states, axis=0)
        self.std_ = np.std(states, axis=0)
        
        # Handle zero std (constant features) by setting to 1.0
        self.std_[self.std_ < self.eps_] = 1.0
        
        self.fitted_ = True
        return self
    
    def transform(self, state):
        """
        Normalize a single state vector or batch of state vectors.
        
        Args:
            state: numpy array of shape (STATE_DIM,) or (N, STATE_DIM)
        
        Returns:
            normalized_state: same shape as input
        """
        if not self.fitted_:
            raise ValueError("Normalizer not fitted. Call fit() first or load from file.")
        
        state = np.array(state, dtype=np.float32)
        original_shape = state.shape
        
        # Handle both single vector and batch
        if state.ndim == 1:
            state = state.reshape(1, -1)
            was_1d = True
        else:
            was_1d = False
        
        # Normalize: (x - mean) / (std + eps)
        normalized = (state - self.mean_) / (self.std_ + self.eps_)
        
        if was_1d:
            normalized = normalized.squeeze(0)
        
        return normalized
    
    def inverse_transform(self, normalized_state):
        """
        Denormalize a normalized state vector or batch.
        
        Args:
            normalized_state: numpy array of shape (STATE_DIM,) or (N, STATE_DIM)
        
        Returns:
            original_state: same shape as input
        """
        if not self.fitted_:
            raise ValueError("Normalizer not fitted. Call fit() first or load from file.")
        
        normalized_state = np.array(normalized_state, dtype=np.float32)
        original_shape = normalized_state.shape
        
        # Handle both single vector and batch
        if normalized_state.ndim == 1:
            normalized_state = normalized_state.reshape(1, -1)
            was_1d = True
        else:
            was_1d = False
        
        # Denormalize: x = normalized * (std + eps) + mean
        original = normalized_state * (self.std_ + self.eps_) + self.mean_
        
        if was_1d:
            original = original.squeeze(0)
        
        return original
    
    def save(self, filepath):
        """
        Save normalizer parameters to JSON file.
        
        Args:
            filepath: path to save file (.json)
        """
        if not self.fitted_:
            raise ValueError("Cannot save unfitted normalizer. Call fit() first.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'mean': self.mean_.tolist(),
            'std': self.std_.tolist(),
            'eps': self.eps_,
            'state_dim': len(self.mean_)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved normalizer to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load normalizer parameters from JSON file.
        
        Args:
            filepath: path to load file (.json)
        
        Returns:
            StateNormalizer instance
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Normalizer file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        mean = np.array(data['mean'], dtype=np.float32)
        std = np.array(data['std'], dtype=np.float32)
        eps = data.get('eps', 1e-8)
        
        normalizer = cls(mean=mean, std=std, eps=eps)
        print(f"Loaded normalizer from {filepath} (state_dim={data['state_dim']})")
        return normalizer
    
    def get_stats(self):
        """
        Get normalization statistics for inspection.
        
        Returns:
            dict with 'mean', 'std', 'min', 'max' for each dimension
        """
        if not self.fitted_:
            return None
        
        return {
            'mean': self.mean_,
            'std': self.std_,
            'min': self.mean_ - 2 * self.std_,  # approximate 2-sigma bounds
            'max': self.mean_ + 2 * self.std_
        }


def compute_normalization_from_data(data_file, state_key='states', save_path=None):
    """
    Convenience function to compute normalization from saved data file.
    
    Args:
        data_file: path to .npz file containing state arrays
        state_key: key in .npz file for state data (default: 'states')
        save_path: optional path to save normalizer JSON (if None, saves next to data_file)
    
    Returns:
        StateNormalizer instance
    """
    data = np.load(data_file, allow_pickle=True)
    states = data[state_key]
    
    normalizer = StateNormalizer()
    normalizer.fit(states)
    
    if save_path is None:
        save_path = Path(data_file).with_suffix('.normalizer.json')
    
    normalizer.save(save_path)
    return normalizer
