# AI-ML Experiments

ML/DL experiments and demos for the **MARID** UAV project. This folder holds optimization and learning foundations that will feed into MARID’s architecture.

## Contents

| Item | Description |
|------|-------------|
| `gradient_descent_optimization` | 1D gradient descent: f(x) = cos(2πx) + x², SymPy derivatives, stop at target gradient. |
| `gradient_descent_2D`           | 2D gradient descent:  f(x,y) and g(x,y), partial derivatives, trajectory on heatmap.            |

## Roadmap
- [x] **1D gradient descent** — implement 1d gradient descent optimization.
- [x] **2D gradient descent** — extend to multi-variable optimization.
- [ ] **Gradient ascent** — implement algorithm for gradient ascent.
- [ ] **MARID integration** — connect these methods to MARID’s perception, state estimation, or control pipeline.

## How to run

- **Notebook:** Open the `.ipynb` in Jupyter or VS Code.
- **Script:** `python gradient_descent_optimization.py` (if using the `.py` version).

**Dependencies:** `numpy`, `matplotlib`, `sympy` (and for the notebook: `ipython`, `matplotlib-inline`).



![Function 3D plot]
<img width="506" height="437" alt="image" src="https://github.com/user-attachments/assets/9298a475-64f8-4054-bf4d-5880b9d77eea" />

![Trajectory on Heatmap]
<img width="652" height="546" alt="image" src="https://github.com/user-attachments/assets/885df2bd-c67c-4f40-911e-8937aeba1bfb" />
