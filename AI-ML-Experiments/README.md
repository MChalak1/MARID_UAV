# AI-ML Experiments

ML/DL experiments and demos for the **MARID** UAV project. This folder holds optimization and learning foundations that will feed into MARID's architecture.

## Contents

| Item | Description |
|------|-------------|
| `gradient_descent_1D` | 1D gradient descent: f(x) = cos(2πx) + x², SymPy derivatives, stop at target gradient. |
| `gradient_descent_2D` | 2D gradient descent: f(x,y) and g(x,y), partial derivatives, trajectory on heatmap. |
| `gradient_descent_2D_refactored` | 2D gradient refactored version using reusable functions. |
| `parametric_experiments_1D` | 1D gradient descent parameter sensitivity experiments & dnamic learning rate. |
| `pytorch_ANN_1D` | PyTorch 1D regression demo: one MLP trained on linear, quadratic, and damped-sinusoid targets with per-task weight reset, loss curves, and correlation plots. |
| `pytorch_parametric_regression` | PyTorch parametric regression slope-sweep experiment |
| `pytorch_binary_classification` | PyTorch binary classification: two 2D clusters, ANN with BCE + Sigmoid, decision boundary and misclassification plot. |
| `ANN_learningrates_binary` | Parametric learning rate sweep + hidden units comparison (h=1,2,5) with meta-experiments. |
| `ANN_multi_in_out` | PyTorch 3-class classification: three 2D Gaussian zones (upper-left, upper-right, bottom), 2→h→3 or 2→h→h→3, CrossEntropyLoss, parametric LR sweep + 50-run meta-experiment. |
| `ANN_multi_in_out_iris` | PyTorch multi-output ANN on IRIS: 4 features → 3 species, 4→64→64→3, CrossEntropyLoss. |
| `pose_from_imu_altitude_train` | Phase 1: Train 4-D pose (z, roll, pitch, yaw) from 11-D IMU+altitude. Notebook + script; local or Colab; 256+dropout+ReduceLROnPlateau. |

## Roadmap

- [x] **1D gradient descent/ascent** — 1-Variable optimization (done).
- [x] **2D gradient descent/ascent** — Multi-variable optimization (done).
- [x] **2D gradient descent refactored** — Refactored version using a reusable run_gradient function, with cleaner convergence handling (done).
- [x] **Parametric Experiments** — Running parametric experiments on Gradient Descent (done).
- [x] **Dynamic Learning Rate** — Applying Dynamic Learning Rate (done).
- [x] **PyTorch Implementation** — Using PyTorch to implement model training 1D (done).
- [x] **PyTorch Parametric Expirementation** — Experimenting model behavior across different regression slopes (done).
- [x] **PyTorch Binary Classification** — 2D cluster classification with BCE + Sigmoid (done).
- [x] **PyTorch Binary Classification: LR & Capacity Analysis** — Parametric LR sweep + hidden units comparison with meta-experiments (done).
- [x] **IMU data logging** — Log IMU from simulation to CSV for ML/training (marid_logging package). (done)
- [x] **Pose estimator data logging** — Log IMU + altitude → pose pairs for EKF training (pose_estimator_logger) (done).
- [x] **Multi-in multi-out ANN (3-class zones)** — Synthetic 3-zone classification with LR sweep and meta-experiment (done).
- [x] **Multi-in multi-out ANN (IRIS)** — IRIS 4→3 classification (done).
- [x] **IMU → pose prediction** — Train ANN to predict pose (orientation) from raw IMU channels. (done)
- [ ] **MARID integration** — Connect these methods to MARID's perception, state estimation, or control pipeline.

## Phase 1: Pose-from-IMU+Altitude results

Feedforward net (11→256→256→4) trained on combined sim data; targets z, roll, pitch, yaw from Gazebo.

![Phase 1 training curves](<img width="1390" height="789" alt="image" src="https://github.com/user-attachments/assets/c1df9f97-ceb7-46c4-9b60-74789b44004e" />
)
*Example run: Val MSE ≈ 0.19; per-output (z, roll, pitch, yaw): 0.600, 0.079, 0.034, 0.066.*

**Observations:** Single-file runs gave lower val MSE; combined data (multiple runs) is harder. With dropout, train MSE can sit above val MSE. Orientation (roll, pitch, yaw) is typically better than z. Next: more flights and multiple seeds for mean±std.


## How to run
- **Pose-from-IMU+Altitude (Phase 1):** Open `pose_from_imu_altitude_train.ipynb` or run `python pose_from_imu_altitude_train.py` (set `.npz` path(s) in the script).
- **Notebook:** Open the `.ipynb` in Jupyter or VS Code.
- **Script:** `python gradient_descent_optimization.py` (if using the `.py` version).

**Dependencies:** **Dependencies:** `numpy`, `matplotlib`, `sympy`, `torch` (and for notebooks: `ipython`, `matplotlib-inline`; for IRIS: `seaborn`).

<img width="1337" height="473" alt="image" src="https://github.com/user-attachments/assets/a7f6d54c-febb-4729-8c74-73aa0990ff93" />

<img width="638" height="532" alt="image" src="https://github.com/user-attachments/assets/654e8894-82e3-4092-889c-68286ee7fa7b" /> <img width="642" height="522" alt="image" src="https://github.com/user-attachments/assets/4511e721-7ce5-428b-9c2e-6c2b24a3f110" />


