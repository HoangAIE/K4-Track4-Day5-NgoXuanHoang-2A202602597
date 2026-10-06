# Test Infrastructure & Specification (`TEST_INFRA.md`)
**Project:** Kalman Filter Fusion Lab & Lynx-07 Mission Verification  
**Author:** `teamwork_preview_test_writer_e2e_1` (Specialist / QA)  
**Target Notebook:** `Lab/kalman_fusion_lab_STUDENT.ipynb`  
**Test Suite:** `tests/test_kalman_e2e.py`  
**Student ID Seed Target:** `"2A202602597"`  

---

## 1. Test Philosophy: Opaque-Box & Requirement-Driven

This test infrastructure adheres to an **opaque-box (black-box), contract-driven testing philosophy**:
1. **Separation of Verification and Implementation**: The test suite validates observable mathematical and physical behavior against formal interface contracts specified in `PROJECT.md` and `ORIGINAL_REQUEST.md`, independent of internal implementation artifacts or variable namings.
2. **Authoritative Oracle Derivation**: Expected states, transition kinetics, Kalman gains, innovation statistics ($\chi^2$), and analytic Jacobians are derived from first-principles estimation theory (Kalman 1960, Bar-Shalom 2001) and deterministic reference oracles.
3. **Progressive & Multi-Tiered Verification**: Tests are partitioned into distinct validation tiers (Tiers 1 to 4) ranging from fundamental mathematical operations to realistic multi-sensor vehicle missions.
4. **Adversarial Boundary Resistance**: The suite deliberately subjects the estimation algorithms to ill-conditioned covariances, zero time steps ($\Delta t = 0$), angle boundary crossings ($\pm \pi$), extreme coordinate spaces, and heavy sensor fault regimes.
5. **Dual-Layer Validation**:
   - Algorithmic layer: Evaluates mathematical correctness of core functions (`make_F`, `KalmanFilter`, `run_fusion`, `gated_update`, `h_rb`, `H_rb`).
   - Notebook artifact layer: Verifies headless notebook serialization, cell execution completion, absence of execution errors, passing confirmation markers, and completeness of technical mission reporting.

---

## 2. Feature Inventory & Tier Mapping

| Feature # | Feature Name | Interface / Scope | Target Tier | Minimum Tests |
|:---:|:---|:---|:---:|:---:|
| **F-01** | State Transition Matrix `make_F(dt)` | CV kinematic model $F \in \mathbb{R}^{4 \times 4}$ | Tier 1, Tier 2 | $\ge 5$ (T1), $\ge 5$ (T2) |
| **F-02** | Observation Matrix `make_H()` | Cartesian 2D position extractor $H \in \mathbb{R}^{2 \times 4}$ | Tier 1 | $\ge 5$ (T1) |
| **F-03** | Kalman Filter Core `KalmanFilter` | `predict(F, Q)` and `update(z, H, R)` methods | Tier 1, Tier 2 | $\ge 5$ (T1), $\ge 5$ (T2) |
| **F-04** | Asynchronous Multi-Sensor Fusion `run_fusion` | Event-driven fusion queue with dynamic $\Delta t$ | Tier 1, Tier 2, Tier 3 | $\ge 5$ (T1), $\ge 5$ (T2), $\ge 4$ (T3) |
| **F-05** | Chi-Square Gating `gated_update` | Mahalanobis outlier rejection $d^2 \le \chi^2_{p}(df)$ | Tier 1, Tier 2, Tier 3 | $\ge 5$ (T1), $\ge 5$ (T2), $\ge 4$ (T3) |
| **F-06** | Polar EKF Observation `h_rb` & Jacobian `H_rb` | Non-linear radar range & bearing + analytic Jacobian | Tier 1, Tier 2, Tier 3 | $\ge 5$ (T1), $\ge 5$ (T2), $\ge 4$ (T3) |
| **F-07** | Lynx-07 Mission Simulation & Diagnosis | Seed generation (`"2A202602597"`), NIS consistency | Tier 4 | $\ge 6$ (T4) |
| **F-08** | Notebook Execution & Report Audit | `Lab/kalman_fusion_lab_STUDENT.ipynb` run & cell audit | Tier 4 | $\ge 5$ (T4) |

---

## 3. Tier Architecture & Test Categorization

### Tier 1: Feature Coverage (Nominal Behavior)
Focuses on happy-path contract compliance for each independent functional component:
- `test_tier1_make_f_cv_propagation`: Verifies position displacement $\Delta x = v_x \Delta t$ and constant velocity conservation.
- `test_tier1_make_h_extraction`: Verifies linear extraction of $[x, y]^\top$ from state $[x, y, v_x, v_y]^\top$.
- `test_tier1_kf_predict_mean_and_cov`: Verifies $\mathbf{x}^- = \mathbf{F}\mathbf{x}$ and $\mathbf{P}^- = \mathbf{F}\mathbf{P}\mathbf{F}^\top + \mathbf{Q}$.
- `test_tier1_kf_update_equations`: Verifies innovation $\mathbf{y}$, covariance $\mathbf{S}$, Kalman gain $\mathbf{K}$, and state update.
- `test_tier1_run_fusion_nominal_stream`: Verifies chronological processing, state logging, and copy detachment.
- `test_tier1_gated_update_nominal_acceptance`: Verifies acceptance of inlier measurements under 99% chi-square threshold.
- `test_tier1_h_rb_range_bearing_calculation`: Verifies Euclidean distance and bearing angle calculation.
- `test_tier1_H_rb_analytic_jacobian`: Verifies partial derivatives against finite differences.

### Tier 2: Boundary, Extreme & Corner Cases
Stress-tests numeric stability, geometric limits, and mathematical edge conditions:
- **Zero Timestep ($\Delta t = 0$)**: Validates $F(0) = \mathbf{I}$, $Q(0) = \mathbf{0}$, and sequential updates without duplicate prediction.
- **Zero / Extreme Covariance**: Tests deterministic propagation when $P_0 = 0$, near-zero measurement noise $R \to \epsilon$, and infinite uncertainty regimes.
- **Ill-Conditioned Covariance Handling**: Verifies that solving $\mathbf{S}^{-1}\mathbf{y}$ via `np.linalg.solve` withstands poorly scaled matrices without numerical collapse.
- **Negative Coordinates & 4-Quadrant Wrapping**: Tests vehicle motion in all 4 Cartesian quadrants ($[-x, -y]$) and radar positions across quadrants.
- **Angle Wrapping in $[-\pi, \pi]$**: Verifies that angular innovations crossing $\pm \pi$ are correctly normalized to prevent $2\pi$ phase wrap jumps.

### Tier 3: Cross-Feature Interactions
Validates pairwise and multi-stage interactions between interconnected components:
- **Pairwise: Fusion + Gating**: Evaluates `run_fusion` integrated with `gated_update` on an asynchronous stream containing sporadic outlier bursts, verifying outlier rejection while preserving tracking continuity.
- **Pairwise: EKF + Gating**: Evaluates polar radar updates with Chi-square gating, verifying that angular residuals are properly normalized before computing Mahalanobis distance $d^2 = \mathbf{y}^\top \mathbf{S}^{-1} \mathbf{y}$.
- **Sensor Dropout & Gate Widening**: Verifies that when a primary sensor drops out, state uncertainty $\mathbf{P}$ expands via prediction, which dynamically widens acceptance gate $\mathbf{S} = \mathbf{H}\mathbf{P}\mathbf{H}^\top + \mathbf{R}$, avoiding permanent filter lock-out upon sensor reappearance.

### Tier 4: Real-World Application Scenarios (Lynx-07 & Notebook Verification)
- **Deterministic Personalization**: Verifies that `STUDENT_ID="2A202602597"` maps deterministically to seed `1563971285` via SHA-256.
- **Injected Fault Characteristics**: Verifies reproduction of UWB underrated noise ($\times 4.12$) on the 90-second mission log.
- **Mitigation Efficacy**: Verifies that applying `FIX_SENSOR="UWB"` with `FIX_METHOD="inflate_R"` achieves pooled mean NIS $< 8.0$ (target $\approx 2.08$) and preserves 100% of measurements.
- **Headless Notebook Audit**:
  - `Lab/kalman_fusion_lab_STUDENT.ipynb` exists and parses cleanly as valid nbformat v4.
  - All code cells have been executed (`execution_count` is integer, not null).
  - No notebook cell contains error outputs (`ename`, `evalue`, `traceback`).
  - All 5 milestone verification markers are present in cell outputs:
    1. `✅ Exercise 5.1 passed`
    2. `✅ Exercise 5.2 passed`
    3. `✅ Exercise 6.1 passed`
    4. `✅ Exercise 7.1 passed`
    5. `✅ Exercise 8.1 passed`
  - Cell 102 Lynx-07 Mission Report markdown contains all 4 mandatory empirical sections and has zero unresolved placeholders (`TODO`, `chưa điền`, `[...]`, `...`).

---

## 4. Test Execution Architecture & Commands

### Test Execution Runner
All tests are implemented using standard `pytest`:
```bash
# Execute entire test suite
pytest tests/test_kalman_e2e.py -v

# Execute specific tier
pytest tests/test_kalman_e2e.py -k "tier1" -v
pytest tests/test_kalman_e2e.py -k "tier2" -v
pytest tests/test_kalman_e2e.py -k "tier3" -v
pytest tests/test_kalman_e2e.py -k "tier4" -v

# Execute notebook verification exclusively
pytest tests/test_kalman_e2e.py -k "notebook" -v
```

### Self-Contained Extraction Mechanism
The test suite is completely decoupled from the notebook's internal state. It loads the student notebook code cells dynamically via an isolated AST/exec loader into a controlled sandbox namespace. This enables:
1. Strict opaque-box evaluation of algorithmic functions against interface contracts.
2. Direct inspection of notebook metadata, cell execution order, cell outputs, and markdown reports.
3. Isolated test runs that do not pollute or mutate the student's notebook file.

---

## 5. Coverage & Acceptance Thresholds

| Metric | Target Requirement | Strict Enforcement |
|:---|:---:|:---:|
| **Tier 1 Feature Tests** | $\ge 5$ test cases per feature | Enforced |
| **Tier 2 Boundary Tests** | $\ge 5$ test cases per category | Enforced |
| **Tier 3 Interaction Tests** | $\ge 4$ test cases per pair | Enforced |
| **Tier 4 Lynx-07 Tests** | $\ge 6$ simulation validations | Enforced |
| **Pooled Mean NIS** | $< 8.0$ (empirical expectation $\approx 2.08$) | Enforced |
| **Outlier Rejection Rate** | $> 95\%$ on $56$m outliers | Enforced |
| **Inlier Acceptance Rate** | $> 98\%$ on nominal observations | Enforced |
| **Notebook Execution Errors** | 0 errors across all 119 cells | Enforced |
| **Notebook Exercise Checks** | All 5 "✅ Exercise X.X passed" present | Enforced |
| **Report Completeness** | 0 placeholder tokens in Cell 102 | Enforced |

---
