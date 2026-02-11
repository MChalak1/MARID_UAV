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
| `pytorch_parametric_regression.py` | PyTorch parametric regression slope-sweep experiment |

## Roadmap

- [x] **1D gradient descent/ascent** — 1-Variable optimization (done).
- [x] **2D gradient descent/ascent** — Multi-variable optimization (done).
- [x] **2D gradient descent refactored** — Refactored version using a reusable run_gradient function, with cleaner convergence handling (done).
- [x] **Parametric Experiments** — Running parametric experiments on Gradient Descent (done).
- [x] **Dynamic Learning Rate** — Applying Dynamic Learning Rate (done).
- [x] **PyTorch Implementation** — Using PyTorch to implement model training 1D (done).
- [x] **PyTorch Parametric Expirementation** — Experimenting model behavior across different regression slopes (done).
- [ ] **MARID integration** — Connect these methods to MARID's perception, state estimation, or control pipeline.

## How to run

- **Notebook:** Open the `.ipynb` in Jupyter or VS Code.
- **Script:** `python gradient_descent_optimization.py` (if using the `.py` version).

**Dependencies:** `numpy`, `matplotlib`, `sympy` (and for the notebook: `ipython`, `matplotlib-inline`).

<img width="1337" height="473" alt="image" src="https://github.com/user-attachments/assets/a7f6d54c-febb-4729-8c74-73aa0990ff93" />

<img width="638" height="532" alt="image" src="https://github.com/user-attachments/assets/654e8894-82e3-4092-889c-68286ee7fa7b" /> <img width="642" height="522" alt="image" src="https://github.com/user-attachments/assets/4511e721-7ce5-428b-9c2e-6c2b24a3f110" />


