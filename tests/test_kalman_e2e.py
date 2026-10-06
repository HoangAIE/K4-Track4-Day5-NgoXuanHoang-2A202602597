"""E2E Test Suite for Kalman Filter Fusion Lab & Lynx-07 Mission.

=============================================================================
Test Philosophy: Opaque-Box, Contract-Driven, Multi-Tier Verification
Student Target ID: "2A202602597"
Target Notebook: Lab/kalman_fusion_lab_STUDENT.ipynb
Test Framework: pytest
=============================================================================

This suite provides complete requirement-driven testing across 4 tiers:
- Tier 1: Feature Coverage (>=5 test cases per feature)
- Tier 2: Boundary & Corner Cases (>=5 test cases per feature category)
- Tier 3: Cross-Feature Interactions (pairwise: fusion + gating, EKF + gating)
- Tier 4: Real-World Scenarios (Lynx-07 simulation + notebook execution audit)
"""

from __future__ import annotations

import io
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
from scipy.stats import chi2


# ============================================================================
# Notebook Loader Sandbox Fixture
# ============================================================================

NOTEBOOK_PATH = (
    Path(__file__).resolve().parent.parent
    / "Lab"
    / "kalman_fusion_lab_STUDENT.ipynb"
)


def _load_lab_environment(notebook_path: Path = NOTEBOOK_PATH) -> SimpleNamespace:
    """Load and execute key notebook definition cells in an isolated sandbox.

    Suppresses GUI plotting and standard output during execution.
    """
    assert notebook_path.exists(), f"Notebook file not found at: {notebook_path}"

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    # Key definition cells in kalman_fusion_lab_STUDENT.ipynb:
    # 3: imports, plot style, rmse, cov_ellipse
    # 54: truth_state
    # 56: make_F, make_H, make_Q
    # 59: KalmanFilter class
    # 67: SENSORS, make_measurements
    # 69: run_fusion
    # 80: gated_update
    # 87: seed_from_id, _true_mission_path, generate_mission_log
    # 88: STUDENT_ID declaration
    # 92: diagnose
    # 95: diagnosis variables
    # 98: apply_mission_fix, run_mission_filter
    # 107: RADAR_POS
    # 109: h_rb, H_rb
    # 111: ekf_update
    target_cell_indices = [
        3, 54, 56, 59, 67, 69, 80, 87, 88, 92, 95, 98, 107, 109, 111
    ]

    # Pre-configure matplotlib in headless non-interactive mode
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
    """Session-scoped fixture providing access to lab functions."""
    return _load_lab_environment()


# ============================================================================
# TIER 1: FEATURE COVERAGE (>=5 test cases per feature)
# ============================================================================


class TestTier1MakeF:
    """Tier 1: Feature coverage for make_F(dt)."""

    def test_tier1_make_f_shape_and_dtype(self, lab: SimpleNamespace) -> None:
        """TC-T1-F-01: make_F returns a 4x4 float array."""
        F = lab.make_F(0.5)
        assert isinstance(F, np.ndarray), "F must be a numpy ndarray"
        assert F.shape == (4, 4), f"F must be of shape (4, 4), got {F.shape}"
        assert np.issubdtype(F.dtype, np.floating), "F must have floating point dtype"

    def test_tier1_make_f_dt_propagation(self, lab: SimpleNamespace) -> None:
        """TC-T1-F-02: F correctly advances position by velocity * dt."""
        dt = 0.5
        x = np.array([10.0, 20.0, 3.0, -2.0])
        x_next = lab.make_F(dt) @ x
        expected = np.array([11.5, 19.0, 3.0, -2.0])
        np.testing.assert_allclose(
            x_next, expected, atol=1e-12,
            err_msg="make_F must advance position by velocity*dt",
        )

    def test_tier1_make_f_velocity_preservation(self, lab: SimpleNamespace) -> None:
        """TC-T1-F-03: F preserves velocities identically (rows 2 and 3)."""
        F = lab.make_F(2.0)
        np.testing.assert_allclose(F[2, :], [0.0, 0.0, 1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(F[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-12)

    def test_tier1_make_f_diagonal_unity(self, lab: SimpleNamespace) -> None:
        """TC-T1-F-04: Diagonal elements of F are all exactly 1.0."""
        for dt_val in [0.01, 0.1, 1.0, 10.0]:
            F = lab.make_F(dt_val)
            np.testing.assert_allclose(np.diag(F), np.ones(4), atol=1e-12)

    def test_tier1_make_f_varying_dt_scaling(self, lab: SimpleNamespace) -> None:
        """TC-T1-F-05: Off-diagonal cross terms scale linearly with dt."""
        for dt_val in [0.05, 0.2, 1.5, 7.3]:
            F = lab.make_F(dt_val)
            assert F[0, 2] == pytest.approx(dt_val), "F[0, 2] must equal dt"
            assert F[1, 3] == pytest.approx(dt_val), "F[1, 3] must equal dt"
            assert F[2, 0] == 0.0, "F[2, 0] must be 0"
            assert F[3, 1] == 0.0, "F[3, 1] must be 0"

    def test_tier1_make_f_semigroup_property(self, lab: SimpleNamespace) -> None:
        """TC-T1-F-06: Semigroup composition F(dt1) @ F(dt2) == F(dt1 + dt2)."""
        dt1, dt2 = 0.3, 0.7
        F1 = lab.make_F(dt1)
        F2 = lab.make_F(dt2)
        F_comp = F1 @ F2
        F_direct = lab.make_F(dt1 + dt2)
        np.testing.assert_allclose(
            F_comp, F_direct, atol=1e-12,
            err_msg="F(dt1) @ F(dt2) must equal F(dt1 + dt2)",
        )


class TestTier1MakeH:
    """Tier 1: Feature coverage for make_H()."""

    def test_tier1_make_h_shape_and_dtype(self, lab: SimpleNamespace) -> None:
        """TC-T1-H-01: make_H returns a 2x4 float array."""
        H = lab.make_H()
        assert isinstance(H, np.ndarray), "H must be a numpy ndarray"
        assert H.shape == (2, 4), f"H must be of shape (2, 4), got {H.shape}"
        assert np.issubdtype(H.dtype, np.floating), "H must have floating point dtype"

    def test_tier1_make_h_extracts_position(self, lab: SimpleNamespace) -> None:
        """TC-T1-H-02: H @ x extracts [x, y] coordinates."""
        x = np.array([12.5, -45.2, 3.14, -9.81])
        z = lab.make_H() @ x
        expected = np.array([12.5, -45.2])
        np.testing.assert_allclose(z, expected, atol=1e-12)

    def test_tier1_make_h_ignores_velocity(self, lab: SimpleNamespace) -> None:
        """TC-T1-H-03: H extracts [0, 0] when state has zero position but huge velocities."""
        x = np.array([0.0, 0.0, 1000.0, -500.0])
        z = lab.make_H() @ x
        np.testing.assert_allclose(z, np.zeros(2), atol=1e-12)

    def test_tier1_make_h_structure(self, lab: SimpleNamespace) -> None:
        """TC-T1-H-04: Non-zero entries are solely H[0, 0]=1 and H[1, 1]=1."""
        H = lab.make_H()
        expected = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        np.testing.assert_allclose(H, expected, atol=1e-12)

    def test_tier1_make_h_linearity(self, lab: SimpleNamespace) -> None:
        """TC-T1-H-05: Linear extraction holds: H @ (a*x1 + b*x2) == a*(H@x1) + b*(H@x2)."""
        H = lab.make_H()
        x1 = np.array([1.0, 2.0, 3.0, 4.0])
        x2 = np.array([-5.0, 6.0, -7.0, 8.0])
        a, b = 2.5, -1.5
        np.testing.assert_allclose(H @ (a * x1 + b * x2), a * (H @ x1) + b * (H @ x2), atol=1e-12)


class TestTier1KalmanFilter:
    """Tier 1: Feature coverage for KalmanFilter class predict/update."""

    def test_tier1_kf_init_attributes(self, lab: SimpleNamespace) -> None:
        """TC-T1-KF-01: Initialization sets x and P with float dtype and matching shapes."""
        x0 = [1.0, 2.0, 3.0, 4.0]
        P0 = np.eye(4) * 5.0
        kf = lab.KalmanFilter(x0, P0)
        assert kf.x.shape == (4,), "State estimate x must be 1D with shape (4,)"
        assert kf.P.shape == (4, 4), "Covariance P must be 2D with shape (4, 4)"
        assert np.issubdtype(kf.x.dtype, np.floating)
        assert np.issubdtype(kf.P.dtype, np.floating)

    def test_tier1_kf_predict_advances_mean(self, lab: SimpleNamespace) -> None:
        """TC-T1-KF-02: predict advances mean: x = F @ x."""
        kf = lab.KalmanFilter([10.0, 20.0, 4.0, -2.0], np.eye(4))
        F = lab.make_F(1.0)
        Q = np.zeros((4, 4))
        kf.predict(F, Q)
        np.testing.assert_allclose(kf.x, [14.0, 18.0, 4.0, -2.0], atol=1e-12)

    def test_tier1_kf_predict_expands_covariance(self, lab: SimpleNamespace) -> None:
        """TC-T1-KF-03: predict expands covariance: P = F @ P @ F.T + Q."""
        P0 = np.diag([1.0, 1.0, 2.0, 2.0])
        kf = lab.KalmanFilter(np.zeros(4), P0)
        F = lab.make_F(0.5)
        Q = lab.make_Q(0.5, 0.1)
        expected_P = F @ P0 @ F.T + Q
        kf.predict(F, Q)
        np.testing.assert_allclose(kf.P, expected_P, atol=1e-12)

    def test_tier1_kf_update_returns_tuple(self, lab: SimpleNamespace) -> None:
        """TC-T1-KF-04: update returns (y, S, K) with correct dimensions."""
        kf = lab.KalmanFilter(np.zeros(4), np.eye(4))
        z = np.array([1.0, 2.0])
        H = lab.make_H()
        R = np.eye(2) * 0.5
        ret = kf.update(z, H, R)
        assert isinstance(ret, tuple) and len(ret) == 3, "update must return a 3-tuple (y, S, K)"
        y, S, K = ret
        assert y.shape == (2,), "y must have shape (2,)"
        assert S.shape == (2, 2), "S must have shape (2, 2)"
        assert K.shape == (4, 2), "K must have shape (4, 2)"

    def test_tier1_kf_update_shifts_mean(self, lab: SimpleNamespace) -> None:
        """TC-T1-KF-05: update shifts mean towards observation proportionally to gain."""
        kf = lab.KalmanFilter([0.0, 0.0, 0.0, 0.0], np.eye(4))
        z = np.array([10.0, 20.0])
        H = lab.make_H()
        R = np.eye(2)
        y, S, K = kf.update(z, H, R)
        # S = H P H.T + R = I_2 + I_2 = 2*I_2; K = P H.T S^-1 = 0.5 * H.T
        # x_new = x_old + K y = [5.0, 10.0, 0.0, 0.0]
        np.testing.assert_allclose(kf.x[:2], [5.0, 10.0], atol=1e-12)

    def test_tier1_kf_update_contracts_covariance(self, lab: SimpleNamespace) -> None:
        """TC-T1-KF-06: update strictly reduces trace of state covariance matrix."""
        kf = lab.KalmanFilter(np.zeros(4), np.eye(4) * 4.0)
        tr_before = np.trace(kf.P)
        kf.update(np.array([1.0, 1.0]), lab.make_H(), np.eye(2))
        tr_after = np.trace(kf.P)
        assert tr_after < tr_before, "Covariance trace must decrease after incorporating measurement"

    def test_tier1_kf_arbitrary_dimension_support(self, lab: SimpleNamespace) -> None:
        """TC-T1-KF-07: KalmanFilter operates correctly on arbitrary dimension n=3, m=1."""
        kf = lab.KalmanFilter(np.array([1.0, 2.0, 3.0]), np.eye(3))
        F = np.eye(3) * 2.0
        Q = np.eye(3) * 0.1
        kf.predict(F, Q)
        assert kf.x.shape == (3,)
        assert kf.P.shape == (3, 3)
        H = np.array([[1.0, 0.0, 0.0]])
        R = np.array([[0.5]])
        y, S, K = kf.update(np.array([5.0]), H, R)
        assert y.shape == (1,)
        assert S.shape == (1, 1)
        assert K.shape == (3, 1)


class TestTier1RunFusion:
    """Tier 1: Feature coverage for run_fusion(meas, q, x0, P0, t0)."""

    def test_tier1_run_fusion_log_length(self, lab: SimpleNamespace) -> None:
        """TC-T1-RF-01: Log length equals measurement sequence length."""
        meas = [
            (0.1, "GPS", np.array([1.0, 2.0]), lab.make_H(), np.eye(2)),
            (0.2, "GPS", np.array([1.5, 2.5]), lab.make_H(), np.eye(2)),
            (0.3, "GPS", np.array([2.0, 3.0]), lab.make_H(), np.eye(2)),
        ]
        log = lab.run_fusion(meas, q=0.1, x0=np.zeros(4), P0=np.eye(4))
        assert len(log) == 3, f"Expected 3 log entries, got {len(log)}"

    def test_tier1_run_fusion_tuple_structure(self, lab: SimpleNamespace) -> None:
        """TC-T1-RF-02: Each log item is a (timestamp, x, P) tuple with matching shapes."""
        meas = [(0.1, "GPS", np.array([1.0, 2.0]), lab.make_H(), np.eye(2))]
        log = lab.run_fusion(meas, q=0.1, x0=np.zeros(4), P0=np.eye(4))
        ts, x, P = log[0]
        assert ts == pytest.approx(0.1)
        assert x.shape == (4,)
        assert P.shape == (4, 4)

    def test_tier1_run_fusion_detached_state_copies(self, lab: SimpleNamespace) -> None:
        """TC-T1-RF-03: Logged states are detached deep copies, not mutating references."""
        meas = [
            (0.1, "GPS", np.array([1.0, 1.0]), lab.make_H(), np.eye(2)),
            (0.2, "GPS", np.array([2.0, 2.0]), lab.make_H(), np.eye(2)),
        ]
        log = lab.run_fusion(meas, q=0.1, x0=np.zeros(4), P0=np.eye(4))
        # Modify the first logged vector
        log[0][1][0] = 9999.0
        assert log[1][1][0] != 9999.0, "Logged states must be independent copies"

    def test_tier1_run_fusion_timestamp_ordering(self, lab: SimpleNamespace) -> None:
        """TC-T1-RF-04: Log timestamps match chronological measurement timestamps."""
        meas = [
            (0.05, "M1", np.array([0.0, 0.0]), lab.make_H(), np.eye(2)),
            (0.12, "M2", np.array([0.1, 0.1]), lab.make_H(), np.eye(2)),
            (0.25, "M3", np.array([0.2, 0.2]), lab.make_H(), np.eye(2)),
        ]
        log = lab.run_fusion(meas, q=0.1, x0=np.zeros(4), P0=np.eye(4))
        log_ts = [entry[0] for entry in log]
        assert log_ts == pytest.approx([0.05, 0.12, 0.25])

    def test_tier1_run_fusion_multi_sensor_handling(self, lab: SimpleNamespace) -> None:
        """TC-T1-RF-05: run_fusion seamlessly processes mixed 2D (position) and 4D (pos+vel) sensors."""
        meas = [
            (0.1, "PosSensor", np.array([1.0, 2.0]), lab.make_H(), np.eye(2)),
            (0.2, "FullSensor", np.array([1.5, 2.5, 5.0, 5.0]), np.eye(4), np.eye(4)),
        ]
        log = lab.run_fusion(meas, q=0.2, x0=np.zeros(4), P0=np.eye(4))
        assert len(log) == 2
        assert np.all(np.isfinite(log[1][1])), "State after multi-sensor update must be finite"

    def test_tier1_run_fusion_convergence(self, lab: SimpleNamespace) -> None:
        """TC-T1-RF-06: run_fusion error decreases on stationary target with noisy observations."""
        rng = np.random.default_rng(42)
        true_pos = np.array([10.0, -10.0])
        meas = [
            (
                float(i * 0.1),
                "Sensor",
                true_pos + rng.normal(0, 1.0, 2),
                lab.make_H(),
                np.eye(2),
            )
            for i in range(1, 50)
        ]
        log = lab.run_fusion(meas, q=0.01, x0=np.zeros(4), P0=np.eye(4) * 100.0)
        final_pos = log[-1][1][:2]
        error = np.linalg.norm(final_pos - true_pos)
        assert error < 0.8, f"Filter should converge near true state, error was {error:.3f} m"


class TestTier1GatedUpdate:
    """Tier 1: Feature coverage for gated_update(kf, z, H, R, p)."""

    def test_tier1_gated_update_accepts_inlier(self, lab: SimpleNamespace) -> None:
        """TC-T1-GU-01: Inlier observation within 1 sigma returns True and updates filter."""
        kf = lab.KalmanFilter(np.zeros(4), np.eye(4))
        accepted = lab.gated_update(
            kf, np.array([0.2, -0.3]), lab.make_H(), np.eye(2) * 0.25, p=0.99
        )
        assert accepted is True, "Inlier observation must be accepted"

    def test_tier1_gated_update_rejects_outlier(self, lab: SimpleNamespace) -> None:
        """TC-T1-GU-02: Outlier observation far beyond gate threshold returns False."""
        kf = lab.KalmanFilter(np.zeros(4), np.eye(4))
        accepted = lab.gated_update(
            kf, np.array([50.0, 50.0]), lab.make_H(), np.eye(2) * 0.25, p=0.99
        )
        assert accepted is False, "Massive outlier (50m) must be rejected"

    def test_tier1_gated_update_rejection_preserves_state(self, lab: SimpleNamespace) -> None:
        """TC-T1-GU-03: Rejected measurement leaves kf.x and kf.P completely unchanged."""
        kf = lab.KalmanFilter([1.0, 2.0, 3.0, 4.0], np.eye(4) * 2.0)
        x_before = kf.x.copy()
        P_before = kf.P.copy()
        lab.gated_update(kf, np.array([100.0, 100.0]), lab.make_H(), np.eye(2), p=0.99)
        np.testing.assert_array_equal(kf.x, x_before, err_msg="State must not mutate on rejection")
        np.testing.assert_array_equal(kf.P, P_before, err_msg="Covariance must not mutate on rejection")

    def test_tier1_gated_update_acceptance_modifies_state(self, lab: SimpleNamespace) -> None:
        """TC-T1-GU-04: Accepted measurement mutates kf.x and kf.P."""
        kf = lab.KalmanFilter(np.zeros(4), np.eye(4))
        x_before = kf.x.copy()
        accepted = lab.gated_update(kf, np.array([0.5, 0.5]), lab.make_H(), np.eye(2), p=0.99)
        assert accepted is True
        assert not np.array_equal(kf.x, x_before), "State must update on acceptance"

    def test_tier1_gated_update_p_quantile_sensitivity(self, lab: SimpleNamespace) -> None:
        """TC-T1-GU-05: Strict probability quantile (e.g. p=0.10) rejects marginal measurements that p=0.99 accepts."""
        # For df=2: chi2.ppf(0.10, 2) = 0.211, chi2.ppf(0.99, 2) = 9.210
        # If y = [1.0, 0.0], S = I -> d^2 = 1.0. Rejected under p=0.10, accepted under p=0.99.
        kf_strict = lab.KalmanFilter(np.zeros(4), np.eye(4) * 0.0)
        kf_loose = lab.KalmanFilter(np.zeros(4), np.eye(4) * 0.0)
        z = np.array([1.0, 0.0])  # d^2 = 1.0 with R=I
        res_strict = lab.gated_update(kf_strict, z, lab.make_H(), np.eye(2), p=0.10)
        res_loose = lab.gated_update(kf_loose, z, lab.make_H(), np.eye(2), p=0.99)
        assert res_strict is False, "d^2=1.0 should exceed chi2_0.10(2) ~ 0.21"
        assert res_loose is True, "d^2=1.0 should fall within chi2_0.99(2) ~ 9.21"

    def test_tier1_gated_update_dimension_adaptive(self, lab: SimpleNamespace) -> None:
        """TC-T1-GU-06: gated_update adapts correctly to 1D and 4D measurement dimensions."""
        # 1D test:
        kf1 = lab.KalmanFilter(np.zeros(1), np.eye(1))
        # threshold for df=1, p=0.99 is chi2.ppf(0.99, 1) = 6.635
        # d^2 = (3.0)^2 / (1+1) = 9.0 / 2.0 = 4.5 < 6.635 -> accept
        acc1 = lab.gated_update(kf1, np.array([3.0]), np.eye(1), np.eye(1), p=0.99)
        assert acc1 is True

        # 4D test:
        kf4 = lab.KalmanFilter(np.zeros(4), np.eye(4))
        # threshold for df=4, p=0.99 is chi2.ppf(0.99, 4) = 13.277
        # huge deviation
        acc4 = lab.gated_update(kf4, np.array([10.0, 10.0, 10.0, 10.0]), np.eye(4), np.eye(4), p=0.99)
        assert acc4 is False


class TestTier1EKFModel:
    """Tier 1: Feature coverage for h_rb(x, s) and H_rb(x, s)."""

    def test_tier1_h_rb_range_euclidean(self, lab: SimpleNamespace) -> None:
        """TC-T1-EKF-01: h_rb range matches exact Euclidean hypotenuse."""
        x = np.array([30.0, 40.0, 1.0, 2.0])
        s = np.array([0.0, 0.0])
        z = lab.h_rb(x, s)
        assert z[0] == pytest.approx(50.0, abs=1e-12)

    def test_tier1_h_rb_bearing_atan2(self, lab: SimpleNamespace) -> None:
        """TC-T1-EKF-02: h_rb bearing matches arctan2(dy, dx)."""
        x = np.array([10.0, 10.0, 0.0, 0.0])
        s = np.array([0.0, 0.0])
        z = lab.h_rb(x, s)
        assert z[1] == pytest.approx(math.pi / 4.0, abs=1e-12)

    def test_tier1_h_rb_output_vector_shape(self, lab: SimpleNamespace) -> None:
        """TC-T1-EKF-03: h_rb returns a 1D array of shape (2,)."""
        z = lab.h_rb(np.array([5.0, 5.0, 0.0, 0.0]), np.array([1.0, 2.0]))
        assert isinstance(z, np.ndarray)
        assert z.shape == (2,)

    def test_tier1_H_rb_dimensions_and_velocity_zeros(self, lab: SimpleNamespace) -> None:
        """TC-T1-EKF-04: H_rb has shape (2, 4) with zero velocity columns."""
        H = lab.H_rb(np.array([10.0, 20.0, 5.0, -5.0]), np.array([0.0, 0.0]))
        assert H.shape == (2, 4)
        np.testing.assert_allclose(H[:, 2:], np.zeros((2, 2)), atol=1e-12)

    def test_tier1_H_rb_first_row_range_gradient(self, lab: SimpleNamespace) -> None:
        """TC-T1-EKF-05: First row of H_rb equals [dx/r, dy/r, 0, 0]."""
        x = np.array([3.0, 4.0, 0.0, 0.0])
        s = np.array([0.0, 0.0])
        H = lab.H_rb(x, s)
        # r = 5.0; dx/r = 3/5 = 0.6; dy/r = 4/5 = 0.8
        np.testing.assert_allclose(H[0, :2], [0.6, 0.8], atol=1e-12)

    def test_tier1_H_rb_second_row_bearing_gradient(self, lab: SimpleNamespace) -> None:
        """TC-T1-EKF-06: Second row of H_rb equals [-dy/r^2, dx/r^2, 0, 0]."""
        x = np.array([3.0, 4.0, 0.0, 0.0])
        s = np.array([0.0, 0.0])
        H = lab.H_rb(x, s)
        # r^2 = 25.0; -dy/r^2 = -4/25 = -0.16; dx/r^2 = 3/25 = 0.12
        np.testing.assert_allclose(H[1, :2], [-0.16, 0.12], atol=1e-12)


# ============================================================================
# TIER 2: BOUNDARY & CORNER CASES (>=5 test cases per feature category)
# ============================================================================


class TestTier2DtZeroHandling:
    """Tier 2: Boundary test cases for zero time step dt = 0."""

    def test_tier2_dt_zero_make_f_identity(self, lab: SimpleNamespace) -> None:
        """TC-T2-DT0-01: make_F(0.0) produces an exact 4x4 identity matrix."""
        F0 = lab.make_F(0.0)
        np.testing.assert_array_equal(F0, np.eye(4))

    def test_tier2_dt_zero_make_q_zeros(self, lab: SimpleNamespace) -> None:
        """TC-T2-DT0-02: make_Q(0.0, q) produces an exact 4x4 zero matrix."""
        Q0 = lab.make_Q(0.0, 0.5)
        np.testing.assert_allclose(Q0, np.zeros((4, 4)), atol=1e-15)

    def test_tier2_dt_zero_kf_predict_identity(self, lab: SimpleNamespace) -> None:
        """TC-T2-DT0-03: predict with dt=0 leaves state mean and covariance invariant."""
        kf = lab.KalmanFilter([1.0, 2.0, 3.0, 4.0], np.eye(4) * 2.5)
        x_orig = kf.x.copy()
        P_orig = kf.P.copy()
        kf.predict(lab.make_F(0.0), lab.make_Q(0.0, 1.0))
        np.testing.assert_allclose(kf.x, x_orig, atol=1e-12)
        np.testing.assert_allclose(kf.P, P_orig, atol=1e-12)

    def test_tier2_dt_zero_run_fusion_simultaneous_meas(self, lab: SimpleNamespace) -> None:
        """TC-T2-DT0-04: run_fusion processes simultaneous measurements at identical timestamp."""
        meas = [
            (1.0, "SensorA", np.array([2.0, 3.0]), lab.make_H(), np.eye(2)),
            (1.0, "SensorB", np.array([2.1, 2.9]), lab.make_H(), np.eye(2)),
        ]
        log = lab.run_fusion(meas, q=0.5, x0=np.zeros(4), P0=np.eye(4))
        assert len(log) == 2
        assert log[0][0] == 1.0 and log[1][0] == 1.0
        assert np.all(np.isfinite(log[1][1]))

    def test_tier2_dt_zero_consecutive_updates_decrease_covariance(self, lab: SimpleNamespace) -> None:
        """TC-T2-DT0-05: Consecutive updates at dt=0 successively shrink covariance without noise inflation."""
        meas = [
            (0.5, "S1", np.array([1.0, 1.0]), lab.make_H(), np.eye(2)),
            (0.5, "S2", np.array([1.0, 1.0]), lab.make_H(), np.eye(2)),
            (0.5, "S3", np.array([1.0, 1.0]), lab.make_H(), np.eye(2)),
        ]
        log = lab.run_fusion(meas, q=0.5, x0=np.zeros(4), P0=np.eye(4) * 10.0)
        tr0 = np.trace(log[0][2])
        tr1 = np.trace(log[1][2])
        tr2 = np.trace(log[2][2])
        assert tr2 < tr1 < tr0, "Covariance trace must strictly decrease across dt=0 updates"


class TestTier2ZeroCovariance:
    """Tier 2: Boundary test cases for zero and extreme covariance regimes."""

    def test_tier2_zero_cov_predict_propagation(self, lab: SimpleNamespace) -> None:
        """TC-T2-ZCOV-01: Zero covariance P=0 with Q=0 remains zero under predict."""
        kf = lab.KalmanFilter([1.0, 2.0, 3.0, 4.0], np.zeros((4, 4)))
        kf.predict(lab.make_F(1.0), np.zeros((4, 4)))
        np.testing.assert_allclose(kf.P, np.zeros((4, 4)), atol=1e-15)

    def test_tier2_zero_cov_with_q_predict(self, lab: SimpleNamespace) -> None:
        """TC-T2-ZCOV-02: Zero covariance P=0 with non-zero Q predicts exactly Q."""
        kf = lab.KalmanFilter(np.zeros(4), np.zeros((4, 4)))
        Q = lab.make_Q(0.5, 0.2)
        kf.predict(lab.make_F(0.5), Q)
        np.testing.assert_allclose(kf.P, Q, atol=1e-12)

    def test_tier2_near_zero_r_kalman_gain(self, lab: SimpleNamespace) -> None:
        """TC-T2-ZCOV-03: Near-zero measurement noise R -> 0 snaps state estimate to measurement."""
        kf = lab.KalmanFilter(np.zeros(4), np.eye(4))
        z = np.array([42.0, -84.0])
        R_tiny = np.eye(2) * 1e-10
        kf.update(z, lab.make_H(), R_tiny)
        np.testing.assert_allclose(kf.x[:2], z, atol=1e-5)

    def test_tier2_huge_r_negligible_update(self, lab: SimpleNamespace) -> None:
        """TC-T2-ZCOV-04: Huge measurement noise R = 1e12 yields gain K ~ 0, state stays unchanged."""
        kf = lab.KalmanFilter([5.0, 10.0, 1.0, -1.0], np.eye(4))
        x_prior = kf.x.copy()
        R_huge = np.eye(2) * 1e12
        kf.update(np.array([1000.0, 1000.0]), lab.make_H(), R_huge)
        np.testing.assert_allclose(kf.x, x_prior, atol=1e-6)

    def test_tier2_rank_deficient_cov_propagation(self, lab: SimpleNamespace) -> None:
        """TC-T2-ZCOV-05: Rank-deficient initial covariance (rank 2) propagates validly without crash."""
        P_singular = np.diag([1.0, 1.0, 0.0, 0.0])  # Rank 2
        kf = lab.KalmanFilter(np.zeros(4), P_singular)
        kf.predict(lab.make_F(0.5), lab.make_Q(0.5, 0.1))
        assert np.linalg.matrix_rank(kf.P) == 4, "Process noise restores full rank"


class TestTier2NegativeCoordinates:
    """Tier 2: Boundary test cases for negative coordinates and 4 quadrants."""

    def test_tier2_negative_coords_cv_propagation(self, lab: SimpleNamespace) -> None:
        """TC-T2-NEG-01: Vehicle in third quadrant (-x, -y) with negative velocities propagates correctly."""
        x0 = np.array([-50.0, -30.0, -4.0, -2.0])
        dt = 2.0
        x_next = lab.make_F(dt) @ x0
        expected = np.array([-58.0, -34.0, -4.0, -2.0])
        np.testing.assert_allclose(x_next, expected, atol=1e-12)

    def test_tier2_h_rb_quadrant_2(self, lab: SimpleNamespace) -> None:
        """TC-T2-NEG-02: Target in Quadrant 2 (dx < 0, dy > 0) has bearing in (pi/2, pi]."""
        radar = np.array([0.0, 0.0])
        target = np.array([-10.0, 10.0, 0.0, 0.0])
        z = lab.h_rb(target, radar)
        assert z[0] == pytest.approx(math.hypot(10.0, 10.0))
        assert z[1] == pytest.approx(3.0 * math.pi / 4.0)

    def test_tier2_h_rb_quadrant_3(self, lab: SimpleNamespace) -> None:
        """TC-T2-NEG-03: Target in Quadrant 3 (dx < 0, dy < 0) has bearing in (-pi, -pi/2)."""
        radar = np.array([0.0, 0.0])
        target = np.array([-10.0, -10.0, 0.0, 0.0])
        z = lab.h_rb(target, radar)
        assert z[0] == pytest.approx(math.hypot(10.0, 10.0))
        assert z[1] == pytest.approx(-3.0 * math.pi / 4.0)

    def test_tier2_h_rb_quadrant_4(self, lab: SimpleNamespace) -> None:
        """TC-T2-NEG-04: Target in Quadrant 4 (dx > 0, dy < 0) has bearing in (-pi/2, 0)."""
        radar = np.array([0.0, 0.0])
        target = np.array([10.0, -10.0, 0.0, 0.0])
        z = lab.h_rb(target, radar)
        assert z[0] == pytest.approx(math.hypot(10.0, 10.0))
        assert z[1] == pytest.approx(-math.pi / 4.0)

    def test_tier2_linear_kf_negative_state_convergence(self, lab: SimpleNamespace) -> None:
        """TC-T2-NEG-05: Linear Kalman filter accurately tracks negative trajectory."""
        kf = lab.KalmanFilter(np.array([-100.0, -100.0, 0.0, 0.0]), np.eye(4) * 20.0)
        for i in range(1, 20):
            kf.predict(lab.make_F(0.1), lab.make_Q(0.1, 0.05))
            kf.update(np.array([-50.0, -60.0]), lab.make_H(), np.eye(2))
        np.testing.assert_allclose(kf.x[:2], [-50.0, -60.0], atol=1.0)


class TestTier2AngleWrapping:
    """Tier 2: Boundary test cases for angle wrapping in [-pi, pi]."""

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Helper standard angle normalizer to [-pi, pi]."""
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def test_tier2_angle_wrap_positive_overflow(self) -> None:
        """TC-T2-ANG-01: Angle > pi wraps to (-pi, pi]."""
        val = 3.5  # rad (~ 200 deg)
        wrapped = self._wrap_angle(val)
        expected = val - 2.0 * math.pi
        assert wrapped == pytest.approx(expected)
        assert -math.pi <= wrapped <= math.pi

    def test_tier2_angle_wrap_negative_underflow(self) -> None:
        """TC-T2-ANG-02: Angle < -pi wraps to (-pi, pi]."""
        val = -3.5  # rad
        wrapped = self._wrap_angle(val)
        expected = val + 2.0 * math.pi
        assert wrapped == pytest.approx(expected)
        assert -math.pi <= wrapped <= math.pi

    def test_tier2_angle_wrap_multi_turn(self) -> None:
        """TC-T2-ANG-03: Multi-revolution angles wrap within [-pi, pi]."""
        for k in [-5, -3, 3, 5]:
            angle = k * math.pi + 0.2
            wrapped = self._wrap_angle(angle)
            assert -math.pi <= wrapped <= math.pi

    def test_tier2_ekf_update_angle_residual_wrapping(self, lab: SimpleNamespace) -> None:
        """TC-T2-ANG-04: ekf_update wraps bearing innovation y[1] so it never jumps by 2*pi."""
        # True target near +pi, measurement observed at -pi
        radar = np.array([0.0, 0.0])
        # Target at x=-10, y=0.01 -> bearing ~ +3.1406
        kf = lab.KalmanFilter(np.array([-10.0, 0.01, 0.0, 0.0]), np.eye(4))
        # Measurement reported at bearing -3.1406 (difference ~ 2*pi if unwrapped)
        z_meas = np.array([10.0, -3.1406])
        R_rb = np.diag([0.1**2, 0.02**2])
        # Update should normalize innovation and remain finite
        lab.ekf_update(kf, z_meas, radar, R_rb)
        assert np.all(np.isfinite(kf.x)), "State must remain finite after branch-cut crossing"
        assert -15.0 < kf.x[0] < -5.0, "State x-position must remain bounded near -10"

    def test_tier2_h_rb_atan2_range_boundaries(self, lab: SimpleNamespace) -> None:
        """TC-T2-ANG-05: h_rb bearing is strictly within [-pi, pi] across test grid."""
        radar = np.array([5.0, -5.0])
        for x_coord in [-20.0, 0.0, 25.0]:
            for y_coord in [-30.0, 0.0, 30.0]:
                if x_coord == 5.0 and y_coord == -5.0:
                    continue
                z = lab.h_rb(np.array([x_coord, y_coord, 0.0, 0.0]), radar)
                assert -math.pi <= z[1] <= math.pi


class TestTier2IllConditionedCovariance:
    """Tier 2: Boundary test cases for ill-conditioned covariance matrices."""

    def test_tier2_ill_conditioned_s_solve_stability(self, lab: SimpleNamespace) -> None:
        """TC-T2-ILL-01: gated_update uses solve without singular matrix error on ill-conditioned S."""
        kf = lab.KalmanFilter(np.zeros(4), np.diag([1e6, 1e-4, 1.0, 1.0]))
        H = lab.make_H()
        R = np.diag([1e-4, 1e6])
        # S condition number is very high (~ 1e10)
        accepted = lab.gated_update(kf, np.array([0.0, 0.0]), H, R, p=0.99)
        assert accepted is True

    def test_tier2_ill_conditioned_anisotropic_r(self, lab: SimpleNamespace) -> None:
        """TC-T2-ILL-02: Highly anisotropic measurement noise updates only the precise axis."""
        kf = lab.KalmanFilter(np.zeros(4), np.eye(4) * 10.0)
        H = lab.make_H()
        R_aniso = np.diag([1e-6, 1e8])  # x is ultra-precise, y is completely uninformative
        kf.update(np.array([5.0, 100.0]), H, R_aniso)
        assert kf.x[0] == pytest.approx(5.0, abs=1e-3), "Precise axis x must snap to measurement"
        assert kf.x[1] < 1.0, "Noisy axis y must barely shift from 0"

    def test_tier2_ill_conditioned_p_stays_symmetric(self, lab: SimpleNamespace) -> None:
        """TC-T2-ILL-03: State covariance P remains symmetric within numerical tolerance."""
        kf = lab.KalmanFilter(np.zeros(4), np.eye(4) * 2.0)
        kf.predict(lab.make_F(0.5), lab.make_Q(0.5, 0.5))
        kf.update(np.array([3.0, -2.0]), lab.make_H(), np.diag([0.2, 0.5]))
        np.testing.assert_allclose(kf.P, kf.P.T, atol=1e-10)

    def test_tier2_ill_conditioned_p_eigenvalues_positive(self, lab: SimpleNamespace) -> None:
        """TC-T2-ILL-04: Covariance eigenvalues remain strictly non-negative after multiple updates."""
        kf = lab.KalmanFilter(np.zeros(4), np.eye(4))
        for _ in range(10):
            kf.predict(lab.make_F(0.1), lab.make_Q(0.1, 0.1))
            kf.update(np.array([1.0, 1.0]), lab.make_H(), np.eye(2) * 0.1)
        eigvals = np.linalg.eigvalsh(kf.P)
        assert np.all(eigvals >= -1e-12), f"Covariance eigenvalues must be >= 0, got {eigvals}"

    def test_tier2_extreme_gain_finite_state(self, lab: SimpleNamespace) -> None:
        """TC-T2-ILL-05: State remains finite under singular-approaching updates."""
        kf = lab.KalmanFilter(np.zeros(4), np.eye(4) * 1e-8)
        kf.update(np.array([1.0, 1.0]), lab.make_H(), np.eye(2) * 1e-8)
        assert np.all(np.isfinite(kf.x))
        assert np.all(np.isfinite(kf.P))


# ============================================================================
# TIER 3: CROSS-FEATURE INTERACTIONS (Pairwise interactions)
# ============================================================================


class TestTier3CrossFeatureInteractions:
    """Tier 3: Pairwise interactions between fusion, gating, and EKF."""

    def test_tier3_fusion_gating_rejects_outliers_preserves_inliers(
        self, lab: SimpleNamespace
    ) -> None:
        """TC-T3-PAIR-01: Gated fusion rejects sporadic outlier spikes while tracking inlier stream."""
        rng = np.random.default_rng(123)
        kf = lab.KalmanFilter(np.zeros(4), np.eye(4) * 5.0)
        rejected_count = 0
        accepted_count = 0

        # Stream of 50 steps: step 15 and 35 are huge 40m outliers
        for k in range(50):
            ts = k * 0.1
            kf.predict(lab.make_F(0.1), lab.make_Q(0.1, 0.1))
            if k in (15, 35):
                z = np.array([40.0, 40.0])  # outlier
            else:
                z = np.array([ts * 1.0, ts * 0.5]) + rng.normal(0, 0.2, 2)

            acc = lab.gated_update(kf, z, lab.make_H(), np.eye(2) * 0.04, p=0.99)
            if acc:
                accepted_count += 1
            else:
                rejected_count += 1

        assert rejected_count == 2, f"Both 40m outliers must be rejected, rejected={rejected_count}"
        assert accepted_count >= 46, "All or nearly all inliers must be accepted"

    def test_tier3_fusion_gating_state_continuity(self, lab: SimpleNamespace) -> None:
        """TC-T3-PAIR-02: Outlier rejection ensures smooth state continuity without position spikes."""
        kf = lab.KalmanFilter(np.array([10.0, 10.0, 1.0, 1.0]), np.eye(4))
        # Nominal step
        kf.predict(lab.make_F(0.1), lab.make_Q(0.1, 0.05))
        lab.gated_update(kf, np.array([10.1, 10.1]), lab.make_H(), np.eye(2) * 0.1)
        pos_before_outlier = kf.x[:2].copy()

        # Outlier arrival
        kf.predict(lab.make_F(0.1), lab.make_Q(0.1, 0.05))
        lab.gated_update(kf, np.array([100.0, -100.0]), lab.make_H(), np.eye(2) * 0.1)
        pos_after_outlier = kf.x[:2].copy()

        # Displacement should only be due to CV prediction (~0.1m), not 100m outlier
        disp = np.linalg.norm(pos_after_outlier - pos_before_outlier)
        assert disp < 0.5, f"Displacement was {disp:.2f} m; outlier corrupted track!"

    def test_tier3_fusion_gating_blackout_widens_gate(self, lab: SimpleNamespace) -> None:
        """TC-T3-PAIR-03: Uncertainty growth during sensor blackout expands gate, preventing lockout."""
        kf = lab.KalmanFilter(np.zeros(4), np.eye(4) * 0.1)
        # Simulate 5 seconds without updates (50 predict steps)
        for _ in range(50):
            kf.predict(lab.make_F(0.1), lab.make_Q(0.1, 0.5))

        # Covariance P has expanded; gate S = H P H.T + R is wide
        # Measurement arrived with 3m offset
        z_reconnect = np.array([3.0, 3.0])
        accepted = lab.gated_update(kf, z_reconnect, lab.make_H(), np.eye(2) * 0.25, p=0.99)
        assert accepted is True, "Widened gate after blackout must accept valid reconnect measurement"

    def test_tier3_fusion_gating_order_commutativity(self, lab: SimpleNamespace) -> None:
        """TC-T3-PAIR-04: Simultaneous independent measurements produce consistent state regardless of update order."""
        H = lab.make_H()
        R1 = np.eye(2) * 0.5
        R2 = np.eye(2) * 0.8
        z1 = np.array([2.0, 3.0])
        z2 = np.array([2.1, 2.9])

        # Filter A: update z1 then z2
        kf_a = lab.KalmanFilter(np.zeros(4), np.eye(4) * 2.0)
        lab.gated_update(kf_a, z1, H, R1)
        lab.gated_update(kf_a, z2, H, R2)

        # Filter B: update z2 then z1
        kf_b = lab.KalmanFilter(np.zeros(4), np.eye(4) * 2.0)
        lab.gated_update(kf_b, z2, H, R2)
        lab.gated_update(kf_b, z1, H, R1)

        np.testing.assert_allclose(kf_a.x, kf_b.x, atol=1e-10)
        np.testing.assert_allclose(kf_a.P, kf_b.P, atol=1e-10)

    def test_tier3_ekf_gating_range_outlier(self, lab: SimpleNamespace) -> None:
        """TC-T3-PAIR-05: Radar range outlier (false multi-path reflection) rejected by Chi-square gate."""
        radar = np.array([0.0, 0.0])
        kf = lab.KalmanFilter(np.array([20.0, 0.0, 0.0, 0.0]), np.eye(4) * 0.5)
        # Expected range is 20.0, measurement reports 120.0
        z_outlier = np.array([120.0, 0.0])
        R_rb = np.diag([0.5**2, 0.05**2])

        # Manual chi-square check on EKF innovation
        y = z_outlier - lab.h_rb(kf.x, radar)
        y[1] = (y[1] + math.pi) % (2.0 * math.pi) - math.pi
        H = lab.H_rb(kf.x, radar)
        S = H @ kf.P @ H.T + R_rb
        d2 = float(y.T @ np.linalg.solve(S, y))
        threshold = chi2.ppf(0.99, df=2)
        assert d2 > threshold, f"d2={d2:.1f} should exceed threshold {threshold:.2f}"

    def test_tier3_ekf_gating_bearing_outlier(self, lab: SimpleNamespace) -> None:
        """TC-T3-PAIR-06: Radar bearing outlier (false strobe) rejected by Chi-square gate."""
        radar = np.array([0.0, 0.0])
        kf = lab.KalmanFilter(np.array([30.0, 0.0, 0.0, 0.0]), np.eye(4) * 0.5)
        # Expected bearing is 0.0, measurement reports 1.5 rad (86 degrees off)
        z_outlier = np.array([30.0, 1.5])
        R_rb = np.diag([0.5**2, 0.02**2])
        y = z_outlier - lab.h_rb(kf.x, radar)
        y[1] = (y[1] + math.pi) % (2.0 * math.pi) - math.pi
        H = lab.H_rb(kf.x, radar)
        S = H @ kf.P @ H.T + R_rb
        d2 = float(y.T @ np.linalg.solve(S, y))
        threshold = chi2.ppf(0.99, df=2)
        assert d2 > threshold, f"d2={d2:.1f} should exceed threshold {threshold:.2f}"

    def test_tier3_ekf_curved_trajectory_tracking(self, lab: SimpleNamespace) -> None:
        """TC-T3-PAIR-07: EKF successfully tracks turning vehicle using range-bearing radar."""
        radar = np.array([-30.0, -20.0])
        R_rb = np.diag([0.5**2, 0.03**2])
        # Start filter at true position
        kf = lab.KalmanFilter(np.array([0.0, 10.0, 5.0, 0.0]), np.eye(4) * 1.0)
        dt = 0.1
        for k in range(30):
            # Curved motion
            theta = k * 0.05
            p_true = np.array([k * 0.5, 10.0 + 5.0 * math.sin(theta), 5.0, 5.0 * 0.05 * math.cos(theta)])
            kf.predict(lab.make_F(dt), lab.make_Q(dt, 0.2))
            z_meas = lab.h_rb(p_true, radar)
            lab.ekf_update(kf, z_meas, radar, R_rb)

        error = np.linalg.norm(kf.x[:2] - p_true[:2])
        assert error < 1.5, f"EKF tracking error {error:.2f} m should be under 1.5 m"

    def test_tier3_fusion_gating_varying_frequencies(self, lab: SimpleNamespace) -> None:
        """TC-T3-PAIR-08: Gated fusion over mixed 10Hz/5Hz asynchronous stream with intermittent spikes."""
        kf = lab.KalmanFilter(np.zeros(4), np.eye(4) * 2.0)
        meas: List[Tuple[float, str, np.ndarray, np.ndarray, np.ndarray]] = []
        for i in range(20):
            t = i * 0.1
            meas.append((t, "UWB", np.array([t * 2.0, t * 1.0]), lab.make_H(), np.eye(2) * 0.25))
            if i % 2 == 0:
                meas.append((t + 0.02, "GPS", np.array([t * 2.0, t * 1.0]), lab.make_H(), np.eye(2) * 0.5))
        # Insert outlier at step 10
        meas.append((1.05, "UWB_Outlier", np.array([80.0, -80.0]), lab.make_H(), np.eye(2) * 0.25))
        meas.sort(key=lambda m: m[0])

        t_prev = 0.0
        accepted_cnt = 0
        rejected_cnt = 0
        for ts, name, z, H, R in meas:
            dt = ts - t_prev
            if dt > 0:
                kf.predict(lab.make_F(dt), lab.make_Q(dt, 0.1))
                t_prev = ts
            if lab.gated_update(kf, z, H, R, p=0.99):
                accepted_cnt += 1
            else:
                rejected_cnt += 1

        assert rejected_cnt >= 1, "The 80m outlier must be rejected"
        assert accepted_cnt >= 28, "All valid asynchronous measurements must be accepted"

    def test_tier3_ekf_hybrid_cartesian_fusion(self, lab: SimpleNamespace) -> None:
        """TC-T3-PAIR-09: Collaborative estimation interleaving Cartesian linear updates and polar EKF updates."""
        radar = np.array([-20.0, -20.0])
        R_rb = np.diag([0.4**2, 0.02**2])
        H_lin = lab.make_H()
        R_lin = np.eye(2) * 0.5
        kf = lab.KalmanFilter(np.array([10.0, 10.0, 2.0, 1.0]), np.eye(4) * 5.0)

        for step in range(15):
            dt = 0.1
            kf.predict(lab.make_F(dt), lab.make_Q(dt, 0.1))
            true_pos = np.array([10.0 + step * 0.2, 10.0 + step * 0.1])
            # Alternate between linear Cartesian measurement and polar Radar measurement
            if step % 2 == 0:
                kf.update(true_pos, H_lin, R_lin)
            else:
                z_polar = lab.h_rb(np.array([*true_pos, 2.0, 1.0]), radar)
                lab.ekf_update(kf, z_polar, radar, R_rb)

        pos_err = np.linalg.norm(kf.x[:2] - true_pos)
        assert pos_err < 0.6, f"Hybrid Cartesian/polar fusion error was {pos_err:.2f} m"

    def test_tier3_ekf_gating_alternating_quadrant_sightings(self, lab: SimpleNamespace) -> None:
        """TC-T3-PAIR-10: EKF gating validates sightings when vehicle rapidly transitions across radar boresight."""
        radar = np.array([0.0, 0.0])
        R_rb = np.diag([0.2**2, 0.01**2])
        kf = lab.KalmanFilter(np.array([5.0, 0.1, 0.0, 1.0]), np.eye(4))

        # Vehicle moves from +y to -y crossing x-axis
        for k in range(10):
            y_val = 0.5 - k * 0.1
            target = np.array([5.0, y_val, 0.0, -1.0])
            kf.predict(lab.make_F(0.1), lab.make_Q(0.1, 0.05))
            z_meas = lab.h_rb(target, radar)
            # Check gate
            y = z_meas - lab.h_rb(kf.x, radar)
            y[1] = (y[1] + math.pi) % (2.0 * math.pi) - math.pi
            H = lab.H_rb(kf.x, radar)
            S = H @ kf.P @ H.T + R_rb
            d2 = float(y.T @ np.linalg.solve(S, y))
            assert d2 < chi2.ppf(0.99, df=2), f"Normal transition should pass gate, got d2={d2:.2f}"
            lab.ekf_update(kf, z_meas, radar, R_rb)



# ============================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS (Lynx-07 & Notebook Verification)
# ============================================================================


class TestTier4Lynx07Mission:
    """Tier 4: Mission simulation and diagnosis for STUDENT_ID='2A202602597'."""

    STUDENT_ID = "2A202602597"

    def test_tier4_lynx07_student_id_deterministic_seed(self, lab: SimpleNamespace) -> None:
        """TC-T4-LX-01: seed_from_id generates deterministic seed 1563971285."""
        seed = lab.seed_from_id(self.STUDENT_ID)
        assert seed == 1563971285, f"Seed must equal 1563971285, got {seed}"

    def test_tier4_lynx07_mission_log_dimensions(self, lab: SimpleNamespace) -> None:
        """TC-T4-LX-02: Mission log contains 450 GPS points, 900 UWB points, T=90s."""
        mission = lab.generate_mission_log(self.STUDENT_ID)
        assert mission["T"] == 90.0
        assert len(mission["t_gps"]) == 450, f"Expected 450 GPS points, got {len(mission['t_gps'])}"
        assert len(mission["t_uwb"]) == 900, f"Expected 900 UWB points, got {len(mission['t_uwb'])}"
        assert mission["z_gps"].shape == (450, 2)
        assert mission["z_uwb"].shape == (900, 2)

    def test_tier4_lynx07_injected_fault_identification(self, lab: SimpleNamespace) -> None:
        """TC-T4-LX-03: Injected ground-truth fault is UWB noise underrated (~4.12x)."""
        mission, truth = lab.generate_mission_log(self.STUDENT_ID, _reveal_truth=True)
        assert truth["fault_sensor"] == "UWB"
        assert truth["fault_type"] == "underrated_noise"
        assert truth["noise_mult"] == pytest.approx(4.1174, abs=0.01)

    def test_tier4_lynx07_pre_fix_diagnosis_elevated_nis(self, lab: SimpleNamespace) -> None:
        """TC-T4-LX-04: Baseline diagnosis without fix exhibits severe UWB mean NIS >> 2.0."""
        mission = lab.generate_mission_log(self.STUDENT_ID)
        diag = lab.diagnose(mission)
        uwb_mean_nis = float(diag["UWB"]["nis"].mean())
        assert uwb_mean_nis > 25.0, f"Pre-fix UWB mean NIS should be ~33.7, got {uwb_mean_nis:.2f}"

    def test_tier4_lynx07_mitigation_pooled_mean_nis_below_threshold(
        self, lab: SimpleNamespace
    ) -> None:
        """TC-T4-LX-05: Applying FIX_SENSOR='UWB' and FIX_METHOD='inflate_R' achieves pooled mean NIS < 8.0."""
        mission = lab.generate_mission_log(self.STUDENT_ID)
        diag = lab.diagnose(mission)
        fix_kwargs = lab.apply_mission_fix(diag, fix_sensor="UWB", fix_method="inflate_R")
        log, nis_all = lab.run_mission_filter(mission, **fix_kwargs)
        pooled_mean_nis = float(nis_all.mean())
        assert pooled_mean_nis < 8.0, (
            f"Pooled mean NIS must be < 8.0 for acceptance, got {pooled_mean_nis:.3f}"
        )
        # Empirical expected value for inflate_R on UWB is ~2.08
        assert 1.5 <= pooled_mean_nis <= 3.0, (
            f"Pooled mean NIS should converge near theoretical 2.0, got {pooled_mean_nis:.3f}"
        )
        assert len(nis_all) == 1350, "inflate_R must retain 100% of measurements (1350/1350)"

    def test_tier4_lynx07_final_position_uncertainty_bounded(
        self, lab: SimpleNamespace
    ) -> None:
        """TC-T4-LX-06: Final state position 1-sigma uncertainty is bounded under 1.0 m."""
        mission = lab.generate_mission_log(self.STUDENT_ID)
        diag = lab.diagnose(mission)
        fix_kwargs = lab.apply_mission_fix(diag, fix_sensor="UWB", fix_method="inflate_R")
        log, nis_all = lab.run_mission_filter(mission, **fix_kwargs)
        final_P = log[-1][2]
        sigma_pos = math.sqrt(final_P[0, 0] + final_P[1, 1])
        assert sigma_pos < 1.0, f"Final 1-sigma uncertainty {sigma_pos:.2f} m exceeds 1.0 m bound"
        # Empirical value is ~ 0.47 m
        assert sigma_pos == pytest.approx(0.47, abs=0.1)


class TestTier4NotebookExecutionVerification:
    """Tier 4: Headless notebook execution completeness and quality audit."""

    @pytest.fixture(scope="module")
    def notebook_json(self) -> Dict[str, Any]:
        """Loads raw JSON of the target notebook."""
        with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_tier4_notebook_valid_json_structure(self, notebook_json: Dict[str, Any]) -> None:
        """TC-T4-NB-01: Notebook file exists, parses as valid JSON with nbformat v4."""
        assert notebook_json.get("nbformat") == 4
        assert "cells" in notebook_json
        assert len(notebook_json["cells"]) == 119

    def test_tier4_notebook_all_code_cells_executed(self, notebook_json: Dict[str, Any]) -> None:
        """TC-T4-NB-02: Every code cell has a non-null integer execution_count."""
        code_cells = [c for c in notebook_json["cells"] if c.get("cell_type") == "code"]
        unexecuted = [
            i for i, c in enumerate(code_cells) if c.get("execution_count") is None
        ]
        assert not unexecuted, (
            f"{len(unexecuted)} code cells were not executed! Indices: {unexecuted}"
        )

    def test_tier4_notebook_zero_error_outputs(self, notebook_json: Dict[str, Any]) -> None:
        """TC-T4-NB-03: Zero cells contain error outputs or unhandled tracebacks."""
        error_cells: List[Tuple[int, str]] = []
        for i, cell in enumerate(notebook_json["cells"]):
            if cell.get("cell_type") == "code":
                for out in cell.get("outputs", []):
                    if out.get("output_type") == "error":
                        error_cells.append(
                            (i, f"{out.get('ename')}: {out.get('evalue')}")
                        )
        assert not error_cells, f"Cells produced execution errors: {error_cells}"

    def test_tier4_notebook_all_exercise_passes_present(
        self, notebook_json: Dict[str, Any]
    ) -> None:
        """TC-T4-NB-04: All 5 milestone pass strings are present in the cell outputs."""
        pass_markers = [
            "✅ Exercise 5.1 passed",
            "✅ Exercise 5.2 passed",
            "✅ Exercise 6.1 passed",
            "✅ Exercise 7.1 passed",
            "✅ Exercise 8.1 passed",
        ]
        all_text_outputs = "".join(
            str(out)
            for cell in notebook_json["cells"]
            if cell.get("cell_type") == "code"
            for out in cell.get("outputs", [])
        )

        missing_markers = [m for m in pass_markers if m not in all_text_outputs]
        assert not missing_markers, (
            f"Missing required pass assertion output markers: {missing_markers}"
        )

    def test_tier4_notebook_cell_102_report_completeness(
        self, notebook_json: Dict[str, Any]
    ) -> None:
        """TC-T4-NB-05: Cell 102 markdown report has no remaining placeholders and covers all 4 sections."""
        cell_102 = notebook_json["cells"][102]
        assert cell_102.get("cell_type") == "markdown"
        content = "".join(cell_102.get("source", []))

        # Check absence of unreplaced blanks or template markers
        placeholder_patterns = [
            r"\bTODO\b",
            r"chưa điền",
            r"\?\?\?",
            r"\[\.\.\.\]",
            r"_{3,}",
            r"\(STUDENT_ID của bạn\)",
        ]
        found_placeholders = [
            p for p in placeholder_patterns if re.search(p, content, re.IGNORECASE)
        ]
        assert not found_placeholders, (
            f"Cell 102 contains unresolved placeholders matching: {found_placeholders}"
        )

        # Check required analytical sections
        assert "2A202602597" in content, "Engineer student ID must be included"
        assert "1. Bằng chứng" in content, "Section 1 (Bằng chứng) must be present"
        assert "2. Cách sửa" in content, "Section 2 (Cách sửa) must be present"
        assert "3. Độ tin cậy cuối" in content, "Section 3 (Độ tin cậy) must be present"
        assert "4. Hạn chế" in content, "Section 4 (Hạn chế) must be present"

        # Check empirical figures presence
        assert "inflate_R" in content, "Must explain inflate_R mitigation"
        assert "2.08" in content or "2.0" in content, "Must reference post-fix pooled NIS value"
