"""Empirical Verification & Adversarial Stress Testing Suite for Section 9 (Lynx-07).

Author: teamwork_preview_challenger_2 (Empirical Challenger)
Student Target: "2A202602597"
Target Notebook: Lab/kalman_fusion_lab_STUDENT.ipynb
"""

import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from scipy.stats import chi2

from tests.test_kalman_e2e import _load_lab_environment

NOTEBOOK_PATH = Path("Lab/kalman_fusion_lab_STUDENT.ipynb")
STUDENT_ID = "2A202602597"


@pytest.fixture(scope="module")
def lab_env() -> SimpleNamespace:
    return _load_lab_environment(NOTEBOOK_PATH)


class TestChallenger2Section9Verification:
    """Rigorous empirical challenge and verification of Section 9 Lynx-07."""

    def test_ch2_01_seed_reproducibility_and_invariance(self, lab_env: SimpleNamespace):
        """Verify deterministic seed derivation and case/whitespace invariance."""
        raw_hash = hashlib.sha256(STUDENT_ID.strip().lower().encode()).hexdigest()
        expected_seed = int(raw_hash[:8], 16)
        computed_seed = lab_env.seed_from_id(STUDENT_ID)
        assert computed_seed == 1563971285
        assert computed_seed == expected_seed

        # Test invariance
        for variant in ["2a202602597", " 2A202602597 ", "  2a202602597\n"]:
            assert lab_env.seed_from_id(variant) == 1563971285

    def test_ch2_02_ground_truth_fault_injection(self, lab_env: SimpleNamespace):
        """Verify ground truth injected fault parameters."""
        mission, truth = lab_env.generate_mission_log(STUDENT_ID, _reveal_truth=True)
        assert truth["fault_sensor"] == "UWB"
        assert truth["fault_type"] == "underrated_noise"
        assert np.allclose(truth["bias_vec"], [0.0, 0.0])
        assert truth["noise_mult"] == pytest.approx(4.117438, abs=1e-5)
        # Dimensions
        assert mission["T"] == 90.0
        assert len(mission["t_gps"]) == 450
        assert len(mission["t_uwb"]) == 900

    def test_ch2_03_baseline_provisional_filter_diagnostics(self, lab_env: SimpleNamespace):
        """Verify provisional filter statistics and match Cell 102 reported numbers."""
        mission = lab_env.generate_mission_log(STUDENT_ID)
        diag = lab_env.diagnose(mission)

        gps_mean_nis = float(np.mean(diag["GPS"]["nis"]))
        gps_med_nis = float(np.median(diag["GPS"]["nis"]))
        gps_res = np.array(diag["GPS"]["resid_mean"])

        uwb_mean_nis = float(np.mean(diag["UWB"]["nis"]))
        uwb_med_nis = float(np.median(diag["UWB"]["nis"]))
        uwb_res = np.array(diag["UWB"]["resid_mean"])

        # Check Cell 102 reported values
        assert round(gps_mean_nis, 2) == 4.57
        assert round(gps_med_nis, 2) == 3.25
        assert np.allclose(np.round(gps_res, 2), [-0.01, 0.04])

        assert round(uwb_mean_nis, 2) == 33.72
        assert round(uwb_med_nis, 2) == 25.06
        assert np.allclose(np.round(uwb_res, 2), [0.00, -0.09])

    def test_ch2_04_inflate_r_effectiveness(self, lab_env: SimpleNamespace):
        """Verify inflate_R achieves pooled mean NIS < 8.0 with 100% data retention."""
        mission = lab_env.generate_mission_log(STUDENT_ID)
        diag = lab_env.diagnose(mission)
        kw = lab_env.apply_mission_fix(diag, "UWB", "inflate_R")
        log, nis_all = lab_env.run_mission_filter(mission, **kw)

        pooled_mean_nis = float(np.mean(nis_all))
        pooled_med_nis = float(np.median(nis_all))

        assert pooled_mean_nis < 8.0
        assert round(pooled_mean_nis, 2) == 2.08
        assert round(pooled_med_nis, 2) == 1.51
        assert len(nis_all) == 1350

    def test_ch2_05_alternative_bias_fails(self, lab_env: SimpleNamespace):
        """Verify that bias correction on UWB fails threshold (pooled NIS ~ 24.00 >> 8.0)."""
        mission = lab_env.generate_mission_log(STUDENT_ID)
        diag = lab_env.diagnose(mission)
        kw = lab_env.apply_mission_fix(diag, "UWB", "bias")
        log, nis_all = lab_env.run_mission_filter(mission, **kw)

        pooled_mean_nis = float(np.mean(nis_all))
        assert pooled_mean_nis > 8.0
        assert round(pooled_mean_nis, 2) == 24.00

    def test_ch2_06_alternative_gate_data_starvation(self, lab_env: SimpleNamespace):
        """Verify that chi-square gating causes severe data starvation (618 rejected = 68.7%)."""
        mission = lab_env.generate_mission_log(STUDENT_ID)
        diag = lab_env.diagnose(mission)
        kw = lab_env.apply_mission_fix(diag, "UWB", "gate")
        log, nis_all = lab_env.run_mission_filter(mission, **kw, gate_p=0.999)

        rejected = 1350 - len(nis_all)
        assert rejected == 618
        assert round((rejected / 900.0) * 100.0, 1) == 68.7

    def test_ch2_07_final_uncertainty_figures(self, lab_env: SimpleNamespace):
        """Verify final state covariance and 1-sigma uncertainty match Cell 102."""
        mission = lab_env.generate_mission_log(STUDENT_ID)
        diag = lab_env.diagnose(mission)
        kw = lab_env.apply_mission_fix(diag, "UWB", "inflate_R")
        log, nis_all = lab_env.run_mission_filter(mission, **kw)

        final_P = log[-1][2]
        Pxx = float(final_P[0, 0])
        Pyy = float(final_P[1, 1])
        sigma_pos = math.sqrt(Pxx + Pyy)
        radius_95 = 2.0 * sigma_pos

        assert round(Pxx, 4) == 0.1088
        assert round(Pyy, 4) == 0.1088
        assert round(sigma_pos, 4) == 0.4664
        assert round(radius_95, 4) == 0.9329

    def test_ch2_08_process_noise_q_robustness_sweep(self, lab_env: SimpleNamespace):
        """Stress-test filter across a wide span of process noise strengths q."""
        mission = lab_env.generate_mission_log(STUDENT_ID)
        diag = lab_env.diagnose(mission)
        kw = lab_env.apply_mission_fix(diag, "UWB", "inflate_R")

        for q_test in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 3.0]:
            log_q, nis_q = lab_env.run_mission_filter(mission, q=q_test, **kw)
            mean_nis = float(np.mean(nis_q))
            assert mean_nis < 8.0, f"Failed at q={q_test} with mean NIS={mean_nis}"

    def test_ch2_09_accuracy_gain_over_gps_only(self, lab_env: SimpleNamespace):
        """Adversarial check: verify that inflating R beats completely dropping UWB."""
        mission, truth = lab_env.generate_mission_log(STUDENT_ID, _reveal_truth=True)
        diag = lab_env.diagnose(mission)
        true_t = truth["t"]
        true_xy = truth["path"][:, :2]

        def get_rmse(filter_log):
            est_t = np.array([e[0] for e in filter_log])
            est_xy = np.array([e[1][:2] for e in filter_log])
            tx = np.interp(est_t, true_t, true_xy[:, 0])
            ty = np.interp(est_t, true_t, true_xy[:, 1])
            return float(np.sqrt(np.mean(np.sum((est_xy - np.stack([tx, ty], axis=1))**2, axis=1))))

        # inflate_R run
        kw = lab_env.apply_mission_fix(diag, "UWB", "inflate_R")
        log_inflate, _ = lab_env.run_mission_filter(mission, **kw)
        rmse_inflate = get_rmse(log_inflate)

        # GPS only run (uwb_r_mult -> infinity)
        log_gps_only, _ = lab_env.run_mission_filter(mission, uwb_r_mult=1e12)
        rmse_gps_only = get_rmse(log_gps_only)

        assert rmse_inflate <= rmse_gps_only
        assert rmse_inflate < 0.45


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
