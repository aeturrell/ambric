"""Tests for the simulator's truth outputs and the recovery benchmark."""

import numpy as np
import pytest

from ambric.benchmark import run_recovery_benchmark
from ambric.utilities import (
    SIM_SIGMA_UK,
    generate_realistic_simulated_data,
    simulate_data,
)


class TestSimulatorTruth:
    def test_return_truth_shapes_align_with_dataframe(self):
        T, R = 40, 4

        df, truth = generate_realistic_simulated_data(T=T, R=R, J=2, return_truth=True)

        assert truth["y_reg_true"].shape == (T, R)
        assert truth["y_annual_no_lag"].shape == (T, R)
        assert list(truth["datetime"]) == sorted(df["datetime"].unique())
        assert truth["region_names"] == sorted(
            x for x in df["region"].unique() if x != "uk"
        )
        np.testing.assert_allclose(truth["w_true"], np.full(R, 1 / R))

    def test_default_output_is_unchanged_dataframe(self):
        df = generate_realistic_simulated_data(T=40, R=4, J=2)
        df_with_truth, _ = generate_realistic_simulated_data(
            T=40, R=4, J=2, return_truth=True
        )

        assert df.equals(df_with_truth)

    def test_national_growth_uses_supplied_weights(self):
        w_true = np.array([0.7, 0.2, 0.1])

        y_uk, _, y_reg_true, _, _ = simulate_data(T=200, R=3, w_true=w_true)

        residual = y_uk - y_reg_true @ w_true
        assert residual.std() == pytest.approx(SIM_SIGMA_UK, rel=0.2)
        # ...and not the equal-weighted mean
        assert (y_uk - y_reg_true.mean(axis=1)).std() > 3 * SIM_SIGMA_UK

    def test_wrong_weight_shape_raises(self):
        with pytest.raises(ValueError, match="w_true must have shape"):
            simulate_data(T=20, R=3, w_true=np.array([0.5, 0.5]))


@pytest.mark.slow
def test_recovery_benchmark():
    """Full-length fit on simulated data, in decimal growth rates. Thresholds
    sit a little below the values measured when they were last updated
    (quarterly correlation 0.84, noise scales 3-4.5x truth, bridge R² 0.55 on
    out-of-fold XGBoost predictions) and are tightened as the model
    improves. Run with ``pytest -m slow``."""
    metrics = run_recovery_benchmark(n_model_fit_iterations=200000)

    assert metrics["quarterly_corr"] > 0.8
    assert metrics["quarterly_rmse"] < 0.65 * metrics["quarterly_sd_truth"]
    assert metrics["annual_in_sample_rmse"] < 0.5 * metrics["annual_sd_truth"]
    assert metrics["national_rmse"] < 0.001
    assert metrics["bridge_r2"] > 0.3
    for name in ("sigma_eps_ratio", "sigma_uk_ratio", "sigma_annual_ratio"):
        assert metrics[name] < 6


@pytest.mark.slow
def test_recovery_benchmark_with_supplied_unequal_weights():
    """When regions differ in size and their shares are supplied, the
    posterior weights stay on those shares (measured largest error 0.003)
    and sum to one. Run with ``pytest -m slow``."""
    w_true = np.array(
        [0.24, 0.15, 0.10, 0.09, 0.08, 0.08, 0.07, 0.06, 0.05, 0.04, 0.02, 0.02]
    )

    metrics = run_recovery_benchmark(
        w_true=w_true, supply_weights=True, n_model_fit_iterations=200000
    )

    assert metrics["w_sum"] == pytest.approx(1.0)
    assert metrics["w_max_abs_error"] < 0.02
    assert metrics["quarterly_corr"] > 0.8
    assert metrics["national_rmse"] < 0.001
