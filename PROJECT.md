# Project: Kalman Filter Fusion Lab & Lynx-07 Mission

## Architecture
- **Language & Runtime**: Python 3.12, Jupyter Notebook (`Lab/kalman_fusion_lab_STUDENT.ipynb`).
- **Target File**: `Lab/kalman_fusion_lab_STUDENT.ipynb` (contains all exercises, simulations, and Section 9 Lynx-07 mission).
- **Core Modules & Components**:
  1. *Linear Kinematics & Kalman Filter*: Constant-velocity state transition $\mathbf{F}(\Delta t)$, observation matrix $\mathbf{H}$, class `KalmanFilter` with `predict(F, Q)` and `update(z, H, R)`.
  2. *Sensor Fusion Pipeline*: Time-ordered event queue processing in `run_fusion(data, kf, q)` with dynamic $\Delta t$ and multi-sensor observation integration.
  3. *Validation & Outlier Rejection*: Chi-Square gating in `gated_update(kf, z, H, R, p)` using Mahalanobis distance $d^2 = y^\top \mathbf{S}^{-1} y$ and $\chi^2$ CDF critical values.
  4. *Nonlinear Estimation (EKF)*: Polar/Radar observation model $h_{rb}(x, s)$, Jacobian matrix $H_{rb}(x, s)$, angle normalization $\in [-\pi, \pi]$, and `ekf_update`.
  5. *Fault Diagnosis & Mitigation (Lynx-07)*: Student ID `2A202602597` seed generation, NIS statistical analysis, fault classification (`underrated_noise` on sensor `UWB`), noise inflation fix (`inflate_R`), and mission reporting.
  6. *Notebook Execution & Serialization*: Sequential headless kernel execution, saving all cell outputs and visualizations.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Dependency Environment | Install `numpy`, `matplotlib`, `scipy`, `ipywidgets`, `nbclient` | M1 | Survey 1 |
| 2 | Exercise 5.1 (`make_F`, `make_H`) | Construct 4x4 state transition and 2x4 observation matrices in Cell 56 | M1 | Survey 1 / Cell 56 |
| 3 | Exercise 5.2 (`KalmanFilter`) | Implement `predict(F, Q)` and `update(z, H, R)` in Cell 59 | M1 | Survey 1 / Cell 59 |
| 4 | Exercise 6.1 (`run_fusion`) | Asynchronous multi-sensor fusion loop with $\Delta t$ handling in Cell 69 | M2 | Survey 2 / Cell 69 |
| 5 | Exercise 7.1 (`gated_update`) | Mahalanobis distance & Chi-square gating outlier rejection in Cell 80 | M2 | Survey 2 / Cell 80 |
| 6 | Exercise 8.1 (`h_rb`, `H_rb`) | Range & bearing radar measurement function and analytic Jacobian in Cell 109 | M3 | Survey 2 / Cell 109 |
| 7 | Section 9 Student ID | Set `STUDENT_ID = "2A202602597"` in Cell 91 | M4 | Survey 3 / Cell 91 |
| 8 | Section 9 Sensor Diagnosis | Set `MY_DIAGNOSIS_SENSOR = "UWB"` and `MY_DIAGNOSIS_TYPE = "underrated_noise"` in Cell 95 | M4 | Survey 3 / Cell 95 |
| 9 | Section 9 Fix Configuration | Set `FIX_SENSOR = "UWB"` and `FIX_METHOD = "inflate_R"` in Cell 98 (pooled NIS < 8.0) | M4 | Survey 3 / Cell 98 |
| 10 | Section 9 Mission Report | Complete Markdown Cell 102 with all 4 required evidence & analytical sections | M4 | Survey 3 / Cell 102 |
| 11 | Sequential Execution & Outputs | Run entire notebook end-to-end, verify assertion outputs and plots saved | M5 | R4 / ORIGINAL_REQUEST |
| 12 | E2E Independent Test Suite | Build independent requirement-driven verification test suite and test runner | E2E Track | ORIGINAL_REQUEST |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Linear Kalman Filter Core | Environment setup + Ex 5.1 & Ex 5.2 implementation & unit checks | none | DONE (Ex 5.1 & 5.2 pass) |
| M2 | Multi-Sensor Fusion & Gating | Ex 6.1 (`run_fusion`) & Ex 7.1 (`gated_update`) implementation & unit checks | M1 | DONE (Ex 6.1 & 7.1 pass) |
| M3 | Extended Kalman Filter (EKF) | Ex 8.1 (`h_rb`, `H_rb`) implementation & Jacobian checks | M1 | DONE (Ex 8.1 passes) |
| M4 | Lynx-07 Diagnosis, Fix & Report | Section 9 configuration, diagnosis, fix, and mission report markdown cell | M1, M2, M3 | DONE (Pooled NIS 2.08, report complete) |
| M5 | Sequential Notebook Execution & Final Gate | Full headless notebook execution, plot verification, and final assertions | M4, E2E Track | DONE (119 cells executed, 0 errors, 19 plots) |
| E2E | Independent E2E Test Suite | Design independent test suite, test runner, and publish `TEST_READY.md` | none | DONE (82/82 tests pass, TEST_READY.md published) |

## Interface Contracts
### `make_F(dt)` & `make_H()`
- `make_F(dt: float) -> np.ndarray`: Shape `(4, 4)`. Diagonal = 1, `F[0, 2] = dt`, `F[1, 3] = dt`.
- `make_H() -> np.ndarray`: Shape `(2, 4)`. `H[0, 0] = 1.0`, `H[1, 1] = 1.0`.

### `KalmanFilter`
- `KalmanFilter(x0: np.ndarray, P0: np.ndarray)`: Initializes `self.x` (4, 1 or 4,), `self.P` (4, 4).
- `predict(F: np.ndarray, Q: np.ndarray) -> None`: `x = F @ x`, `P = F @ P @ F.T + Q`.
- `update(z: np.ndarray, H: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]`: Returns `(y, S, K)`. Updates `self.x` and `self.P` with standard Joseph/Kalman formula.

### `run_fusion(data, kf, q)`
- Input: `data` iterable of `(ts, sensor_name, z, H, R)`, `kf` KalmanFilter, `q` process noise spectral density.
- Predict called with `make_F(dt)` and `make_Q(dt, q)` whenever `dt = ts - t_prev > 0`.
- Update called on every measurement.
- Output: List of `(ts, x_est, P_est)`.

### `gated_update(kf, z, H, R, p=0.99)`
- Innovation $y = z - H x$, Innovation covariance $S = H P H^\top + R$.
- Mahalanobis distance $d^2 = y^\top S^{-1} y$.
- Threshold $\gamma = \chi^2_{df}(\text{len}(z), p)$.
- If $d^2 > \gamma$: rejects measurement, returns `False` without changing `kf`.
- If $d^2 \le \gamma$: calls `kf.update(z, H, R)`, returns `True`.

### `h_rb(x, s)` & `H_rb(x, s)`
- $h_{rb}(x, s) = [\sqrt{(x_0-s_0)^2 + (x_1-s_1)^2}, \text{atan2}(x_1-s_1, x_0-s_0)]^\top$.
- $H_{rb}(x, s) = \begin{bmatrix} dx/r & dy/r & 0 & 0 \\ -dy/r^2 & dx/r^2 & 0 & 0 \end{bmatrix}$.

### Lynx-07 Mission Configuration (Section 9)
- `STUDENT_ID = "2A202602597"`
- `MY_DIAGNOSIS_SENSOR = "UWB"`
- `MY_DIAGNOSIS_TYPE = "underrated_noise"`
- `FIX_SENSOR = "UWB"`
- `FIX_METHOD = "inflate_R"`
- Report Markdown Cell: Must contain pre-fix NIS evidence, applied fix & post-fix NIS, 1-sigma uncertainty, and limitation analysis.

## Code Layout
- Target Notebook: `Lab/kalman_fusion_lab_STUDENT.ipynb` (Owned exclusively by Worker during implementation milestones).
- E2E Test Suite: `tests/test_kalman_e2e.py` (Owned exclusively by E2E Test Writer / Test Orchestrator).
- State & Coordination: `.agents/teamwork/`
