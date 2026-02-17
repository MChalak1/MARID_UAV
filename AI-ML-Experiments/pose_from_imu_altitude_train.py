# IMU Data Training
## Pose-from-IMU+Altitude Training (MARID)
"""
Train a feedforward network to predict **z and orientation** (z, roll, pitch, yaw) from a single timestep of IMU + altitude.

- **Input (11-D):** quaternion (4), gyro (3), linear acceleration (3), altitude (1)
- **Output (4-D):** z, roll, pitch, yaw — the **targets** the network is trained to predict (ground truth from Gazebo in the .npz).
- **Data:** `.npz` from `pose_estimator_logger` (keys: `imu_inputs`, `pose_targets`)

**Usage:** Run cells in order. Upload your `.npz` when prompted (or set `data_path` to a Drive path). Training uses 80/20 train/val split, MSE loss, and reports per-output validation MSE for z, roll, pitch, yaw.

# ROS: data collection (order of launch files and nodes)

1. **Start simulation and localization** (one launch; provides `/imu_ekf`, `/barometer/altitude`, `/gazebo/odom`):
   `ros2 launch marid_description gazebo.launch.py`

   """

# Import Libraries
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

## Uploading The Data
# Option A: Single .npz file — set path to your file
#data_path = "~/marid_ws/data/marid_pose_imu_altitude_XXXX_chunk0000.npz"
#data_path = Path(data_path).expanduser()

# Note, please make sure to modify the paths to the .npz files to the correct paths on your system.
# It will inlcude your username in the path, so you will need to change it to your own username.
# As well as the run number, so you will need to change it to the correct run number.
# Option B: Multiple .npz files — list all paths to merge (e.g. from two runs)
data_paths = [
     Path("~/marid_ws/data/marid_pose_imu_altitude_run1_chunk0000.npz").expanduser(),
     Path("~/marid_ws/data/marid_pose_imu_altitude_run2_chunk0000.npz").expanduser(),
]

# --- Single file: load once ---
# data = np.load(str(data_path), allow_pickle=True)
# X = data['imu_inputs'].astype(np.float32)
# y = data['pose_targets'].astype(np.float32)

X_list, y_list = [], []
for name in uploaded.keys():
    if not name.endswith('.npz'):
        continue
    data = np.load(name, allow_pickle=True)
    X_list.append(data['imu_inputs'].astype(np.float32))
    y_list.append(data['pose_targets'].astype(np.float32))

X = np.concatenate(X_list, axis=0)
y = np.concatenate(y_list, axis=0)
print(f'Combined: X {X.shape}, y {y.shape}')

## Normalize and Split Data

# Compute mean and std per feature (axis=0) so all 11 inputs have similar scale for training
X_mean, X_std = X.mean(axis=0), X.std(axis=0)

# Avoid division by zero: if a feature never changes (std ≈ 0), treat it as scale 1
X_std[X_std < 1e-8] = 1.0

# Standardize: subtract mean, divide by std (zero mean, unit variance per feature)
X_norm = (X - X_mean) / X_std

# Train/val 80-20 split 
n = len(X_norm)
# Shuffle indices so train and val are random subsets (not first 80% vs last 20%)
idx = np.random.permutation(n)
split = int(0.8 * n)
train_idx, val_idx = idx[:split], idx[split:]

# Convert to PyTorch tensors: train set for learning, val set for checking generalization
X_train = torch.tensor(X_norm[train_idx])
y_train = torch.tensor(y[train_idx])
X_val   = torch.tensor(X_norm[val_idx])
y_val   = torch.tensor(y[val_idx])
print(f'Train: {X_train.shape}, Val: {X_val.shape}')

## Model and Optimizer

model = nn.Sequential(
    nn.Linear(11, 256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 4)
)

# Loss Function
loss_fn = nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Reduce LR when val loss stops improving (factor=0.5 => halve LR, patience=epochs to wait)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=100
)

## Training Loop

epochs = 2000
train_losses, val_losses = [], []

for epoch in range(epochs):
    # Training mode: enables dropout/batch-norm training behavior and allows gradients to be computed
    model.train()

    # Forward pass: run the model on training inputs to get predictions (batch of 4-D vectors)
    pred = model(X_train)

    # Loss: mean squared error between predictions and ground-truth targets (single scalar)
    loss = loss_fn(pred, y_train)

    # Backpropagation: clear old gradients, compute new ones from loss, then update weights
    optimizer.zero_grad()   # Clear gradients from the previous step (PyTorch accumulates by default)
    loss.backward()        # Compute gradients of loss w.r.t. all model parameters
    optimizer.step()       # Update each parameter: param = param - lr * grad

    # Store training MSE for this epoch (for plotting); .item() gets a Python float from the tensor
    train_losses.append(loss.item())

    # Validation: check how well the model generalizes to unseen data (no gradient updates)
    model.eval()           # Eval mode: disables dropout etc., same behavior as at inference
    with torch.no_grad():  # Disable gradient tracking to save memory and speed (we don't need gradients for val)
        v_loss = loss_fn(model(X_val), y_val)
    val_losses.append(v_loss.item())

    scheduler.step(v_loss)

    # Print progress every 20 epochs so we can monitor train vs val MSE without flooding the output
    if (epoch + 1) % 20 == 0:
        print(f'Epoch {epoch+1}/{epochs}  Train MSE: {train_losses[-1]:.6f}  Val MSE: {val_losses[-1]:.6f}')

## Plotting Results

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

ax1.plot(train_losses, 'r-', label='Train MSE', linewidth=0.8)
ax1.plot(val_losses, 'b-', label='Val MSE', linewidth=0.8)
ax1.set_ylabel('MSE')
ax1.set_yscale('log')

ax1.legend()
ax1.grid()
ax1.set_title('Full run')

# Zoom on last part (last 50% of epochs)
start = len(train_losses) // 2
ax2.plot(range(start, len(train_losses)), train_losses[start:], 'r-', label='Train MSE', linewidth=0.8)
ax2.plot(range(start, len(val_losses)), val_losses[start:], 'b-', label='Val MSE', linewidth=0.8)
ax2.set_xlim(start, len(train_losses))
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MSE')
ax2.legend()
ax2.grid()
ax2.set_title('Zoom: second half')

plt.suptitle('Pose-from-IMU+Altitude')
plt.tight_layout()
plt.show()

## Per-output Validation MSE

# Use inference mode: no dropout, batch norm in eval mode (same as when you deploy the model)
model.eval()
# No gradient tracking: we're only computing predictions, not training
with torch.no_grad():
    # Run the model on the full validation set and convert to NumPy for per-output math
    y_val_pred = model(X_val).numpy()

# Ground-truth validation targets as NumPy (same shape as y_val_pred: N x 4)
y_val_np = y_val.numpy()

# One label per output: z (m), roll/pitch/yaw (rad)
labels = ['z', 'roll', 'pitch', 'yaw']

# Per-output MSE: for each of the 4 outputs, average squared error over all validation samples.
# This shows which output (z vs roll/pitch/yaw) the model predicts best or worst.
for i in range(4):
    mse = np.mean((y_val_np[:, i] - y_val_pred[:, i])**2)
    print(f'{labels[i]}: MSE = {mse:.6f}')

