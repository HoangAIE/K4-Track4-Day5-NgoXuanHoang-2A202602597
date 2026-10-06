"""Empirical Adversarial Stress Harness for Kalman Filter Fusion Lab.

Teamwork Preview Challenger 1 (critic, specialist).
Targets:
- Exercise 5.1 & 5.2: Negative dt, high velocities, ill-conditioned P, zero measurements, dimension mismatches.
- Exercise 6.1: Out-of-order timestamps, identical timestamps (dt=0), single-sensor streams, empty streams.
- Exercise 7.1: Extreme p-quantiles, 1D/2D/3D vectors, singular/ill-conditioned S, state preservation.
- Exercise 8.1: Quadrant boundaries (-pi, pi, 0, pi/2), origin singularity, large distances, Jacobian finite-difference oracle.
"""

from __future__ import annotations

import io
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
from scipy.stats import chi2


NOTEBOOK_PATH = (
    Path(__file__).resolve().parent.parent
    / "Lab"
    / "kalman_fusion_lab_STUDENT.ipynb"
)


def _load_lab_environment(notebook_path: Path = NOTEBOOK_PATH) -> SimpleNamespace:
    """Load lab definitions from notebook in an isolated sandbox."""
    assert notebook_path.exists(), f"Notebook file not found at: {notebook_path}"

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    target_cell_indices = [
        3, 54, 56, 59, 67, 69, 80, 87, 88, 92, 95, 98, 107, 109, 111
    ]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.show = lambda *args, **kwargs: None

    sandbox_env: Dict[str, Any] = {
        "__name__": "__main__",
        "plt": plt,
        "np": np,
    }

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    try:
        cells = nb_data.get("cells", [])
        for idx in target_cell_indices:
            if idx < len(cells) and cells[idx].get("cell_type") == "code":
                source_lines = cells[idx].get("source", [])
                cleaned_code = "\n".join(
                    line
                    for line in "".join(source_lines).splitlines()
                    if not line.strip().startswith(("%", "!"))
                )
                exec(cleaned_code, sandbox_env)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    return SimpleNamespace(**sandbox_env)


@pytest.fixture(scope="session")
def lab() -> SimpleNamespace:
    return _load_lab_environment()


# ============================================================================
# EXERCISE 5.1 & 5.2: LINEAR DYNAMICS & KALMAN FILTER STRESS TESTS
# ============================================================================

class TestExercise5Stress:
    """Adversarial stress tests for make_F, make_H, make_Q, and KalmanFilter."""

    def test_negative_dt_matrix_inversion_property(self, lab: SimpleNamespace) -> None:
        """CHALLENGE: Negative dt in make_F acts as an algebraic time-reversal inverse."""
        dt = 0.4
        F_fwd = lab.make_F(dt)
        F_bwd = lab.make_F(-dt)
        # Invertibility check: F(-dt) @ F(dt) must equal Identity
        product = F_bwd @ F_fwd
        np.testing.assert_allclose(product, np.eye(4), atol=1e-14)

    def test_negative_dt_causes_indefinite_q_matrix(self, lab: SimpleNamespace) -> None:
        """CHALLENGE: make_Q with negative dt produces non-positive-definite covariance."""
        dt = -0.5
        q = 1.0
        Q_neg = lab.make_Q(dt, q)
        # Diagonal elements: dt^3/3 and dt are negative!
        diag = np.diag(Q_neg)
        assert np.all(diag < 0), f"Diagonal of Q for negative dt must be negative: {diag}"
        # Eigenvalues must be negative
        eigvals = np.linalg.eigvalsh(Q_neg)
        assert np.any(eigvals < 0), "Q(dt<0) violates positive semi-definiteness!"

    def test_extreme_high_velocities_propagation_and_update(self, lab: SimpleNamespace) -> None:
        """CHALLENGE: High velocities (v ~ 10^7 m/s, relativistic/orbital scale) numerical stability."""
        c_speed = 3e7  # 30,000 km/s
        x0 = np.array([0.0, 0.0, c_speed, -c_speed / 2])
        P0 = np.diag([1e4, 1e4, 1e6, 1e6])
        kf = lab.KalmanFilter(x0, P0)

        dt = 0.1
        F = lab.make_F(dt)
        Q = lab.make_Q(dt, 10.0)
        kf.predict(F, Q)

        # Expected position advancement
        assert np.isclose(kf.x[0], c_speed * dt)
        assert np.isclose(kf.x[1], -c_speed / 2 * dt)
        assert np.all(np.isfinite(kf.x))
        assert np.all(np.isfinite(kf.P))

        # Update with noisy position measurement
        H = lab.make_H()
        R = np.eye(2) * 25.0
        z = np.array([kf.x[0] + 5.0, kf.x[1] - 5.0])
        y, S, K = kf.update(z, H, R)

        assert np.all(np.isfinite(kf.x))
        assert np.all(np.isfinite(kf.P))
        assert np.all(np.linalg.eigvalsh(kf.P) > 0)

    def test_ill_conditioned_p_matrix_symmetry_and_positive_definiteness(self, lab: SimpleNamespace) -> None:
        """CHALLENGE: Stress test standard update form (I - KH)P with condition number 10^12."""
        # Highly anisotropic covariance: 10^8 on x, 10^-4 on y, 10^6 on vx, 10^-6 on vy
        x0 = np.array([100.0, 50.0, 10.0, -5.0])
        P0 = np.diag([1e8, 1e-4, 1e6, 1e-6])
        cond_initial = np.linalg.cond(P0)
        assert cond_initial >= 1e12

        kf = lab.KalmanFilter(x0, P0)
        H = lab.make_H()
        R = np.diag([1.0, 1e-2])  # Precise y measurement, standard x measurement

        z = np.array([105.0, 50.01])
        y, S, K = kf.update(z, H, R)

        # Check for NaN / Inf
        assert np.all(np.isfinite(kf.x))
        assert np.all(np.isfinite(kf.P))

        # Check symmetry: (I - KH)P may lose numerical symmetry!
        asymmetry = np.max(np.abs(kf.P - kf.P.T))
        assert asymmetry < 1e-3, f"Asymmetry too large: {asymmetry}"

        # Eigenvalues must stay strictly positive
        eigvals = np.linalg.eigvalsh(kf.P)
        assert np.all(eigvals > 0), f"P developed non-positive eigenvalues: {eigvals}"

    def test_column_vector_z_broadcasting_vulnerability(self, lab: SimpleNamespace) -> None:
        """CHALLENGE: Passing a 2D column vector (2, 1) to update() causes broadcasting error.

        Demonstrates that update strictly assumes 1D arrays for measurements when state is 1D.
        """
        kf = lab.KalmanFilter(np.zeros(4), np.eye(4))
        H = lab.make_H()
        R = np.eye(2)
        z_col = np.zeros((2, 1))  # Column vector instead of 1D vector

        with pytest.raises(ValueError, match="operands could not be broadcast together"):
            kf.update(z_col, H, R)

    def test_zero_measurements_and_pure_extrapolation(self, lab: SimpleNamespace) -> None:
        """CHALLENGE: Long horizon extrapolation (1000 steps) without any measurement."""
        x0 = np.array([0.0, 0.0, 10.0, 5.0])
        P0 = np.eye(4)
        kf = lab.KalmanFilter(x0, P0)

        dt = 0.1
        F = lab.make_F(dt)
        Q = lab.make_Q(dt, 0.1)

        for _ in range(1000):
            kf.predict(F, Q)

        # Position must equal v * T = 10.0 * 100s = 1000m, 5.0 * 100s = 500m
        assert np.isclose(kf.x[0], 1000.0)
        assert np.isclose(kf.x[1], 500.0)
        assert np.isclose(kf.x[2], 10.0)
        assert np.isclose(kf.x[3], 5.0)

        # Covariance should grow monotonically
        assert kf.P[0, 0] > 1000.0
        assert kf.P[1, 1] > 1000.0
        assert np.all(np.isfinite(kf.P))
        assert np.all(np.linalg.eigvalsh(kf.P) > 0)

    def test_origin_measurement_convergence(self, lab: SimpleNamespace) -> None:
        """CHALLENGE: Repeated zero measurements z=[0, 0] pull state to origin."""
        x0 = np.array([500.0, -300.0, 20.0, -10.0])
        P0 = np.eye(4) * 1000.0
        kf = lab.KalmanFilter(x0, P0)

        dt = 0.1
        F = lab.make_F(dt)
        Q = lab.make_Q(dt, 0.01)
        H = lab.make_H()
        R = np.eye(2) * 1.0

        for _ in range(200):
            kf.predict(F, Q)
            kf.update(np.array([0.0, 0.0]), H, R)

        # State should converge to ~0
        np.testing.assert_allclose(kf.x[:2], [0.0, 0.0], atol=0.1)


# ============================================================================
# EXERCISE 6.1: ASYNCHRONOUS MULTI-SENSOR FUSION STRESS TESTS
# ============================================================================

class TestExercise6Stress:
    """Adversarial stress tests for run_fusion."""

    def test_out_of_order_events_behavioral_characterization(self, lab: SimpleNamespace) -> None:
        """CHALLENGE: Feed out-of-order timestamps to run_fusion and verify vulnerability.

        run_fusion has the loop:
            dt = ts - t_prev
            if dt > 0:
                kf.predict(...)
            kf.update(...)
            t_prev = ts

        When ts < t_prev, dt <= 0, so predict is bypassed, but update is executed
        out of temporal order, and t_prev is regressed.
        """
        x0 = np.array([0.0, 0.0, 1.0, 0.0])
        P0 = np.eye(4)
        H = lab.make_H()
        R = np.eye(2)

        # Out-of-order sequence: t=0.0 -> t=2.0 -> t=1.0 -> t=3.0
        meas = [
            (2.0, "GPS", np.array([2.0, 0.0]), H, R),
            (1.0, "UWB", np.array([1.0, 0.0]), H, R),  # Arrives out of order!
            (3.0, "GPS", np.array([3.0, 0.0]), H, R),
        ]

        # The function executes without raising an exception, but let's inspect the log
        log = lab.run_fusion(meas, q=0.1, x0=x0, P0=P0, t0=0.0)
        assert len(log) == 3
        timestamps = [item[0] for item in log]
        assert timestamps == [2.0, 1.0, 3.0]

        # Note: At t=1.0, dt was (1.0 - 2.0) = -1.0 < 0, so predict was skipped!
        # At t=3.0, dt was (3.0 - 1.0) = 2.0! The step advanced by 2.0s rather than 1.0s.
        # This confirms that run_fusion assumes pre-sorted input and lacks monotonic timestamp verification.

    def test_identical_timestamps_dt_zero_successive_updates(self, lab: SimpleNamespace) -> None:
        """CHALLENGE: Simultaneous sensor arrivals at the exact same millisecond (dt=0)."""
        x0 = np.array([0.0, 0.0, 5.0, 0.0])
        P0 = np.eye(4) * 10.0
        H = lab.make_H()
        R_gps = np.eye(2) * 4.0
        R_uwb = np.eye(2) * 1.0

        # Two sensors fire at t=1.0 simultaneously
        meas = [
            (1.0, "GPS", np.array([5.1, 0.1]), H, R_gps),
            (1.0, "UWB", np.array([4.9, -0.05]), H, R_uwb),
        ]

        log = lab.run_fusion(meas, q=0.5, x0=x0, P0=P0, t0=0.0)
        assert len(log) == 2

        t1, x1, P1 = log[0]
        t2, x2, P2 = log[1]

        assert t1 == 1.0 and t2 == 1.0
        # Second update must further reduce covariance without predicting
        assert np.trace(P2[:2, :2]) < np.trace(P1[:2, :2])
        # State should be refined towards the UWB measurement (smaller R)
        assert np.all(np.isfinite(x2))

    def test_single_sensor_stream(self, lab: SimpleNamespace) -> None:
        """CHALLENGE: Stream containing only one sensor type across all steps."""
        x0 = np.array([0.0, 0.0, 2.0, 1.0])
        P0 = np.eye(4) * 5.0
        H = lab.make_H()
        R = np.eye(2) * 2.0

        single_meas = [
            (float(k) * 0.1, "GPS", np.array([float(k) * 0.2, float(k) * 0.1]), H, R)
            for k in range(1, 21)
        ]

        log = lab.run_fusion(single_meas, q=0.1, x0=x0, P0=P0, t0=0.0)
        assert len(log) == 20
        # Verify monotone timestamps
        ts_log = [item[0] for item in log]
        assert ts_log == sorted(ts_log)
        # Covariance must reach steady-state
        final_P = log[-1][2]
        assert np.all(np.linalg.eigvalsh(final_P) > 0)

    def test_empty_measurement_stream(self, lab: SimpleNamespace) -> None:
        """CHALLENGE: Calling run_fusion with empty measurement stream."""
        x0 = np.array([0.0, 0.0, 1.0, 1.0])
        P0 = np.eye(4)
        log = lab.run_fusion([], q=0.1, x0=x0, P0=P0, t0=0.0)
        assert log == [], "Empty stream must return empty log list without error."


# ============================================================================
# EXERCISE 7.1: CHI-SQUARE GATING ADVERSARIAL STRESS TESTS
# ============================================================================

class TestExercise7Stress:
    """Adversarial stress tests for gated_update."""

    def test_extreme_p_quantiles(self, lab: SimpleNamespace) -> None:
        """CHALLENGE: Extreme p values (p=0.001 vs p=0.999 vs boundary values)."""
        x0 = np.array([0.0, 0.0, 0.0, 0.0])
        P0 = np.eye(4)
        H = lab.make_H()
        R = np.eye(2)

        # Inlier at 1.5 sigma: Mahalanobis d^2 = (1.5)^2 + 0 = 2.25
        z_inlier = np.array([1.5, 0.0])

        # Test at p=0.001 (threshold chi2(2, 0.001) = 0.0020)
        kf_strict = lab.KalmanFilter(x0, P0)
        accepted_strict = lab.gated_update(kf_strict, z_inlier, H, R, p=0.001)
        assert not accepted_strict, "p=0.001 must reject virtually everything (d^2=2.25 > 0.002)"
        np.testing.assert_array_equal(kf_strict.x, x0)

        # Test at p=0.999 (threshold chi2(2, 0.999) = 13.816)
        kf_lenient = lab.KalmanFilter(x0, P0)
        accepted_lenient = lab.gated_update(kf_lenient, z_inlier, H, R, p=0.999)
        assert accepted_lenient, "p=0.999 must accept 1.5 sigma inlier (d^2=2.25 < 13.82)"
        assert not np.array_equal(kf_lenient.x, x0)

    def test_gating_boundary_p_zero_and_one(self, lab: SimpleNamespace) -> None:
        """CHALLENGE: Boundary probabilities p=0.0 and p=1.0."""
        x0 = np.array([0.0, 0.0, 0.0, 0.0])
        P0 = np.eye(4)
        H = lab.make_H()
        R = np.eye(2)

        # p = 1.0 (threshold = inf, accepts all finite measurements)
        kf1 = lab.KalmanFilter(x0, P0)
        z_huge = np.array([10000.0, -10000.0])
        accepted_p1 = lab.gated_update(kf1, z_huge, H, R, p=1.0)
        assert accepted_p1 is True

        # p = 0.0 (threshold = 0.0, rejects any non-exact-zero innovation)
        kf0 = lab.KalmanFilter(x0, P0)
        z_tiny = np.array([1e-5, 0.0])
        accepted_p0 = lab.gated_update(kf0, z_tiny, H, R, p=0.0)
        assert accepted_p0 is False

    def test_multi_dimension_measurements_1d_2d_3d(self, lab: SimpleNamespace) -> None:
        """CHALLENGE: Dimensionality adaptation: 1D scalar, 2D vector, 3D vector."""
        x0 = np.array([10.0, 20.0, 1.0, 2.0])
        P0 = np.eye(4)

        # 1D measurement (e.g. odometer vx)
        H_1d = np.array([[0.0, 0.0, 1.0, 0.0]])
        R_1d = np.array([[0.5]])
        z_1d = np.array([1.2])  # inlier
        kf_1d = lab.KalmanFilter(x0, P0)
        assert lab.gated_update(kf_1d, z_1d, H_1d, R_1d, p=0.99) is True

        # 2D measurement (standard position x, y)
        H_2d = lab.make_H()
        R_2d = np.eye(2) * 0.5
        z_2d = np.array([10.2, 20.1])
        kf_2d = lab.KalmanFilter(x0, P0)
        assert lab.gated_update(kf_2d, z_2d, H_2d, R_2d, p=0.99) is True

        # 3D measurement (x, y, vx)
        H_3d = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        R_3d = np.eye(3) * 0.5
        z_3d = np.array([10.1, 19.9, 1.05])
        kf_3d = lab.KalmanFilter(x0, P0)
        assert lab.gated_update(kf_3d, z_3d, H_3d, R_3d, p=0.99) is True

        # Outlier in 3D (30 sigma spike in vx)
        z_3d_outlier = np.array([10.1, 19.9, 30.0])
        kf_3d_out = lab.KalmanFilter(x0, P0)
        assert lab.gated_update(kf_3d_out, z_3d_outlier, H_3d, R_3d, p=0.99) is False

    def test_scalar_float_z_typeerror_vulnerability(self, lab: SimpleNamespace) -> None:
        """CHALLENGE: Passing a bare Python float scalar to gated_update triggers TypeError.

        Demonstrates that gated_update requires an array-like object with len(z).
        """
        kf = lab.KalmanFilter(np.zeros(4), np.eye(4))
        H = np.array([[1.0, 0.0, 0.0, 0.0]])
        R = np.array([[1.0]])

        with pytest.raises(TypeError, match="has no len"):
            lab.gated_update(kf, 1.5, H, R, p=0.99)

    def test_ill_conditioned_and_singular_s_matrix(self, lab: SimpleNamespace) -> None:
        """CHALLENGE: Ill-conditioned S matrix in gated_update."""
        x0 = np.array([0.0, 0.0, 0.0, 0.0])
        # P0 with 0 variance along x and y, and R = 0
        P0 = np.zeros((4, 4))
        H = lab.make_H()
        R_singular = np.zeros((2, 2))  # Exactly singular S = H P H.T + R = 0

        kf = lab.KalmanFilter(x0, P0)
        z = np.array([1.0, 1.0])

        # np.linalg.solve(S, y) must raise LinAlgError
        with pytest.raises(np.linalg.LinAlgError):
            lab.gated_update(kf, z, H, R_singular, p=0.99)

    def test_exact_state_and_covariance_preservation_on_rejection(self, lab: SimpleNamespace) -> None:
        """CHALLENGE: Gating rejection must preserve state and covariance bit-by-bit."""
        x0 = np.array([12.345, -67.89, 3.14159, -2.71828])
        P0 = np.array([
            [2.0, 0.1, 0.0, 0.0],
            [0.1, 3.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.05],
            [0.0, 0.0, 0.05, 0.8],
        ])
        kf = lab.KalmanFilter(x0, P0)
        H = lab.make_H()
        R = np.eye(2)

        # Glitch outlier
        z_glitch = np.array([1000.0, 1000.0])
        res = lab.gated_update(kf, z_glitch, H, R, p=0.99)
        assert res is False
        np.testing.assert_array_equal(kf.x, x0)
        np.testing.assert_array_equal(kf.P, P0)


# ============================================================================
# EXERCISE 8.1: EXTENDED KALMAN FILTER (EKF) ADVERSARIAL STRESS TESTS
# ============================================================================

class TestExercise8Stress:
    """Adversarial stress tests for h_rb, H_rb, and ekf_update."""

    @pytest.mark.parametrize(
        "x_pos, s_pos, expected_bearing, desc",
        [
            ([10.0, 0.0], [0.0, 0.0], 0.0, "Positive x axis"),
            ([0.0, 10.0], [0.0, 0.0], np.pi / 2, "Positive y axis"),
            ([-10.0, 0.0], [0.0, 0.0], np.pi, "Negative x axis (pi)"),
            ([0.0, -10.0], [0.0, 0.0], -np.pi / 2, "Negative y axis (-pi/2)"),
            ([10.0, 10.0], [0.0, 0.0], np.pi / 4, "Quadrant 1"),
            ([-10.0, 10.0], [0.0, 0.0], 3 * np.pi / 4, "Quadrant 2"),
            ([-10.0, -10.0], [0.0, 0.0], -3 * np.pi / 4, "Quadrant 3"),
            ([10.0, -10.0], [0.0, 0.0], -np.pi / 4, "Quadrant 4"),
        ],
    )
    def test_quadrant_boundaries_bearing_accuracy(
        self, lab: SimpleNamespace, x_pos: List[float], s_pos: List[float], expected_bearing: float, desc: str
    ) -> None:
        """CHALLENGE: Quadrant transitions and axes bearings."""
        x = np.array([x_pos[0], x_pos[1], 0.0, 0.0])
        s = np.array(s_pos)
        meas = lab.h_rb(x, s)
        expected_range = np.hypot(x_pos[0] - s_pos[0], x_pos[1] - s_pos[1])

        assert np.isclose(meas[0], expected_range, atol=1e-12), f"Range mismatch for {desc}"
        assert np.isclose(meas[1], expected_bearing, atol=1e-12), f"Bearing mismatch for {desc}"

    def test_origin_singularity_division_by_zero(self, lab: SimpleNamespace) -> None:
        """CHALLENGE: Target placed exactly at radar origin (r = 0, dx=0, dy=0).

        Mathematical singularity:
            H[0, 0] = dx / r -> 0 / 0 (NaN)
            H[1, 0] = -dy / r^2 -> 0 / 0 (NaN)
        """
        s = np.array([10.0, 20.0])
        x = np.array([10.0, 20.0, 0.0, 0.0])  # Exactly at radar position

        # h_rb produces r = 0, angle = 0
        z_pred = lab.h_rb(x, s)
        assert z_pred[0] == 0.0

        # H_rb produces NaNs due to 0 / 0 division
        with np.errstate(divide="ignore", invalid="ignore"):
            H = lab.H_rb(x, s)
        assert np.any(np.isnan(H)), f"H_rb at origin must contain NaNs due to r=0 singularity: {H}"

        # In ekf_update, updating at origin propagates NaNs
        kf = lab.KalmanFilter(x, np.eye(4))
        R = np.eye(2)
        z = np.array([0.0, 0.0])
        with np.errstate(divide="ignore", invalid="ignore"):
            lab.ekf_update(kf, z, s, R)
        assert np.any(np.isnan(kf.x)) or np.any(np.isnan(kf.P)), "EKF at origin propagates NaNs."

    def test_very_large_distances_deep_space_scale(self, lab: SimpleNamespace) -> None:
        """CHALLENGE: Target at r = 10^8 m (100,000 km) testing underflow in H_rb."""
        s = np.array([0.0, 0.0])
        x = np.array([1e8, 0.0, 0.0, 0.0])

        z_pred = lab.h_rb(x, s)
        assert np.isclose(z_pred[0], 1e8)
        assert np.isclose(z_pred[1], 0.0)

        H = lab.H_rb(x, s)
        # H[0, 0] = dx / r = 1.0
        # H[1, 1] = dx / r^2 = 1e8 / 1e16 = 1e-8
        assert np.isclose(H[0, 0], 1.0)
        assert np.isclose(H[1, 1], 1e-8)
        assert np.all(np.isfinite(H))

        # EKF update with reasonable measurement
        kf = lab.KalmanFilter(x, np.diag([1e6, 1e6, 100.0, 100.0]))
        R = np.diag([10.0**2, (1e-4)**2])
        z = np.array([1e8 + 5.0, 1e-5])
        lab.ekf_update(kf, z, s, R)
        assert np.all(np.isfinite(kf.x))
        assert np.all(np.isfinite(kf.P))

    def test_bearing_residual_angle_wrap_at_branch_cut(self, lab: SimpleNamespace) -> None:
        """CHALLENGE: Innovation wrap across [-pi, pi] branch cut in ekf_update."""
        s = np.array([0.0, 0.0])
        # Vehicle state is at angle +3.14 rad (just below +pi)
        r = 50.0
        theta_pred = np.pi - 0.01  # +3.13159
        x = np.array([r * np.cos(theta_pred), r * np.sin(theta_pred), 0.0, 0.0])

        # Radar measurement observes vehicle at -3.14 rad (just above -pi)
        theta_meas = -np.pi + 0.01  # -3.13159
        z = np.array([r, theta_meas])

        # Without wrapping, raw difference would be:
        # (-pi + 0.01) - (pi - 0.01) = -2*pi + 0.02 ~ -6.26 rad!
        # With correct wrapping: (-6.26 + pi) % (2*pi) - pi = +0.02 rad
        P0 = np.eye(4) * 10.0
        kf = lab.KalmanFilter(x.copy(), P0.copy())
        R = np.diag([1.0, 0.01**2])

        lab.ekf_update(kf, z, s, R)

        # The state should barely move, rather than suffering a catastrophic jump!
        delta_pos = np.linalg.norm(kf.x[:2] - x[:2])
        assert delta_pos < 1.0, f"Catastrophic jump detected across branch cut! delta_pos={delta_pos}"

    def test_jacobian_comprehensive_central_finite_differences(self, lab: SimpleNamespace) -> None:
        """CHALLENGE: Central finite difference check of H_rb against h_rb across all quadrants."""
        eps = 1e-6
        test_points = [
            (np.array([15.0, 25.0, 2.0, -1.0]), np.array([0.0, 0.0])),
            (np.array([-40.0, 30.0, -5.0, 3.0]), np.array([10.0, -10.0])),
            (np.array([-20.0, -60.0, 1.0, 0.0]), np.array([-5.0, 5.0])),
            (np.array([75.0, -15.0, 0.0, 4.0]), np.array([20.0, 20.0])),
            (np.array([100.0, 0.1, 10.0, 10.0]), np.array([0.0, 0.0])),  # Near x axis
            (np.array([0.1, 100.0, 10.0, 10.0]), np.array([0.0, 0.0])),  # Near y axis
        ]

        for x, s in test_points:
            H_analytic = lab.H_rb(x, s)
            H_numeric = np.zeros((2, 4))

            for j in range(4):
                dx_step = np.zeros(4)
                dx_step[j] = eps

                h_plus = lab.h_rb(x + dx_step, s)
                h_minus = lab.h_rb(x - dx_step, s)

                # Angle difference with wrapping
                d_bearing = h_plus[1] - h_minus[1]
                d_bearing = (d_bearing + np.pi) % (2 * np.pi) - np.pi

                H_numeric[0, j] = (h_plus[0] - h_minus[0]) / (2 * eps)
                H_numeric[1, j] = d_bearing / (2 * eps)

            # Assert match within relative tolerance
            np.testing.assert_allclose(
                H_analytic,
                H_numeric,
                rtol=1e-5,
                atol=1e-7,
                err_msg=f"Jacobian mismatch at state {x} with radar {s}",
            )
