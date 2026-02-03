# AI-ML Experiments

ML/DL experiments and demos for the **MARID** UAV project. This folder holds optimization and learning foundations that will feed into MARID’s architecture.

## Contents

| Item | Description |
|------|-------------|
| `gradient_descent_optimization` | 1D gradient descent: \(f(x) = \cos(2\pi x) + x^2\), SymPy derivatives, stop at target gradient. |

## Roadmap

- **2D gradient descent** — extend to multi-variable optimization.
- **MARID integration** — connect these methods to MARID’s perception, state estimation, or control pipeline.

## How to run

- **Notebook:** Open the `.ipynb` in Jupyter or VS Code.
- **Script:** `python gradient_descent_optimization.py` (if using the `.py` version).

**Dependencies:** `numpy`, `matplotlib`, `sympy` (and for the notebook: `ipython`, `matplotlib-inline`).
