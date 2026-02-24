"""Unit tests for ambric helper functions and validation logic.

Each test is small, isolated, and follows the Arrange-Act-Assert pattern.
No model fitting is performed — only the deterministic helpers and
validation paths are exercised.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from ambric import (
    Ambric,
    _aggregate_quarterly_to_annual,
    _almon_weights,
    _apply_midas_weights,
    extract_factors_from_panel,
    impute_panel,
    quarter_differences,
    train_xgboost_annual,
)
from ambric.diagnostics import (
    recession_indicator,
    rmse_national_quarterly,
    rmse_regions_annual,
    rmse_regions_quarterly,
)
from ambric.utilities import generate_realistic_simulated_data, prep_data_for_model_run


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture()
def simulated_df() -> pd.DataFrame:
    """Minimal simulated dataframe for validation tests."""
    return generate_realistic_simulated_data(T=40, R=4, J=2, n_factors=2, n_macro=2)


# ── quarter_differences ─────────────────────────────────────────────────────


class TestQuarterDifferences:
    def test_same_quarter_returns_zero(self):
        ts = pd.Series([pd.Timestamp("2020-03-31")])
        ref = pd.Timestamp("2020-03-31")

        result = quarter_differences(ts, ref)

        assert result.iloc[0] == 0

    def test_one_quarter_ahead(self):
        ts = pd.Series([pd.Timestamp("2020-06-30")])
        ref = pd.Timestamp("2020-03-31")

        result = quarter_differences(ts, ref)

        assert result.iloc[0] == 1

    def test_one_quarter_behind(self):
        ts = pd.Series([pd.Timestamp("2020-03-31")])
        ref = pd.Timestamp("2020-06-30")

        result = quarter_differences(ts, ref)

        assert result.iloc[0] == -1

    def test_multiple_values(self):
        ts = pd.Series(pd.date_range("2020-03-31", periods=5, freq="QE"))
        ref = pd.Timestamp("2020-03-31")

        result = quarter_differences(ts, ref)

        np.testing.assert_array_equal(result.values, [0, 1, 2, 3, 4])


# ── _aggregate_quarterly_to_annual ──────────────────────────────────────────


class TestAggregateQuarterlyToAnnual:
    def test_basic_average(self):
        # 8 quarters, 1 column → 2 annual averages
        X = np.arange(8, dtype=np.float64).reshape(8, 1)

        result = _aggregate_quarterly_to_annual(X)

        assert result.shape == (2, 1)
        np.testing.assert_allclose(result[0, 0], np.mean([0, 1, 2, 3]))
        np.testing.assert_allclose(result[1, 0], np.mean([4, 5, 6, 7]))

    def test_drops_trailing_quarters(self):
        # 9 quarters → only 2 complete years used (first 8)
        X = np.ones((9, 2))

        result = _aggregate_quarterly_to_annual(X)

        assert result.shape == (2, 2)

    def test_multiple_columns(self):
        X = np.ones((4, 3))

        result = _aggregate_quarterly_to_annual(X)

        assert result.shape == (1, 3)
        np.testing.assert_allclose(result, 1.0)


# ── _almon_weights ──────────────────────────────────────────────────────────


class TestAlmonWeights:
    def test_zero_params_give_equal_weights(self):
        weights = _almon_weights(0.0, 0.0, n_lags=4)

        np.testing.assert_allclose(weights, np.ones(4) / 4)

    def test_weights_sum_to_one(self):
        weights = _almon_weights(0.5, -0.1, n_lags=4)

        np.testing.assert_allclose(weights.sum(), 1.0)

    def test_positive_theta1_weights_increase(self):
        weights = _almon_weights(1.0, 0.0, n_lags=4)

        # With positive theta1 and theta2=0, later lags get more weight
        assert weights[-1] > weights[0]

    def test_all_weights_nonnegative(self):
        weights = _almon_weights(-2.0, 0.3, n_lags=4)

        assert np.all(weights >= 0)

    def test_custom_n_lags(self):
        weights = _almon_weights(0.0, 0.0, n_lags=6)

        assert len(weights) == 6


# ── _apply_midas_weights ───────────────────────────────────────────────────


class TestApplyMidasWeights:
    def test_equal_weights_equal_mean(self):
        X = np.arange(8, dtype=np.float64).reshape(8, 1)
        weights = np.ones(4) / 4.0

        result = _apply_midas_weights(X, weights)

        # Equal weights → same as simple average
        expected = _aggregate_quarterly_to_annual(X)
        np.testing.assert_allclose(result, expected)

    def test_single_weight_selects_quarter(self):
        X = np.array([[10], [20], [30], [40]], dtype=np.float64)
        # All weight on Q1
        weights = np.array([1.0, 0.0, 0.0, 0.0])

        result = _apply_midas_weights(X, weights)

        np.testing.assert_allclose(result, [[10.0]])

    def test_shape_correct(self):
        X = np.ones((12, 3))
        weights = np.array([0.1, 0.2, 0.3, 0.4])

        result = _apply_midas_weights(X, weights)

        assert result.shape == (3, 3)


# ── impute_panel ────────────────────────────────────────────────────────────


class TestImputePanel:
    def test_no_nans_returns_original(self):
        panels = [np.ones((10, 3)), np.ones((10, 3)) * 2]

        result = impute_panel(panels)

        # Should return the exact same object (no imputation needed)
        assert result is panels

    def test_nans_are_filled(self):
        panel = np.ones((10, 3))
        panel[2, 1] = np.nan

        result = impute_panel([panel])

        assert not np.any(np.isnan(result[0]))

    def test_all_nan_panel_filled_with_zeros(self):
        panel = np.full((5, 2), np.nan)

        result = impute_panel([panel])

        np.testing.assert_array_equal(result[0], np.zeros((5, 2)))


# ── extract_factors_from_panel ──────────────────────────────────────────────


class TestExtractFactorsFromPanel:
    def test_output_shape(self):
        panels = [np.random.default_rng(0).normal(size=(20, 4)) for _ in range(3)]

        factors = extract_factors_from_panel(panels, n_factors=2)

        assert factors.shape == (20, 2)

    def test_too_many_factors_raises(self):
        # 2 panels × 3 regions = 6 features; requesting 10 factors is invalid
        panels = [np.random.default_rng(0).normal(size=(20, 3)) for _ in range(2)]

        with pytest.raises(ValueError, match="n_factors.*exceeds"):
            extract_factors_from_panel(panels, n_factors=10)


# ── train_xgboost_annual ───────────────────────────────────────────────────


class TestTrainXgboostAnnual:
    def test_all_nan_annual_raises(self):
        T, R, J, M = 20, 2, 2, 1
        Z_panel = [np.random.default_rng(0).normal(size=(T, R)) for _ in range(J)]
        macro = np.random.default_rng(1).normal(size=(T, M))
        y_annual = np.full((T, R), np.nan)  # no observed data

        with pytest.raises(ValueError, match="No observed annual regional data"):
            train_xgboost_annual(Z_panel, macro, y_annual)


# ── prep_data_for_model_run ─────────────────────────────────────────────────


class TestPrepDataForModelRun:
    def test_missing_measure_raises_keyerror(self, simulated_df):
        df = simulated_df
        region_names = [x for x in df["region"].unique() if x != "uk"]

        with pytest.raises(KeyError, match="Measures not found"):
            prep_data_for_model_run(
                df,
                macro_names=["nonexistent_macro"],
                region_names=region_names,
                region_covariate_names=["regional_covar_00"],
            )

    def test_missing_region_raises_keyerror(self, simulated_df):
        df = simulated_df
        macro_names = [x for x in df["measure"].unique() if "macro" in x]

        with pytest.raises(KeyError, match="Regions not found"):
            prep_data_for_model_run(
                df,
                macro_names=macro_names,
                region_names=["nonexistent_region"],
                region_covariate_names=[
                    x for x in df["measure"].unique() if "regional_covar" in x
                ],
            )


# ── Ambric.__init__ validation ──────────────────────────────────────────────


class TestAmbricInitValidation:
    def test_missing_column_raises(self):
        df = pd.DataFrame({"datetime": [1], "measure": [1], "value": [1]})

        with pytest.raises(ValueError, match="column.*region"):
            Ambric(df, macro_names=[], region_names=[], region_covariate_names=[])

    def test_non_quarter_end_dates_raises(self):
        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2020-01-15"]),
                "measure": ["m"],
                "region": ["r"],
                "value": [1.0],
            }
        )

        with pytest.raises(ValueError, match="quarter-end"):
            Ambric(df, macro_names=["m"], region_names=["r"], region_covariate_names=[])

    def test_missing_measure_raises(self):
        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2020-03-31"]),
                "measure": ["some_measure"],
                "region": ["r"],
                "value": [1.0],
            }
        )

        with pytest.raises(ValueError, match="Missing measures"):
            Ambric(
                df,
                macro_names=["nonexistent"],
                region_names=["r"],
                region_covariate_names=[],
            )

    def test_missing_region_raises(self, simulated_df):
        df = simulated_df
        macro_names = [x for x in df["measure"].unique() if "macro" in x]
        region_covariate_names = [
            x for x in df["measure"].unique() if "regional_covar" in x
        ]

        with pytest.raises(ValueError, match="Missing regions"):
            Ambric(
                df,
                macro_names=macro_names,
                region_names=["region_that_does_not_exist"],
                region_covariate_names=region_covariate_names,
            )


# ── RMSE functions ──────────────────────────────────────────────────────────


class TestRMSE:
    def test_rmse_national_quarterly_zero_error(self):
        y = np.array([1.0, 2.0, 3.0])

        result = rmse_national_quarterly(y, y)

        assert result == pytest.approx(0.0, abs=1e-12)

    def test_rmse_national_quarterly_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])  # error of 1 on last element

        result = rmse_national_quarterly(y_true, y_pred)

        expected = np.sqrt(1.0 / 3.0)
        assert result == pytest.approx(expected)

    def test_rmse_regions_quarterly_perfect(self):
        y = np.ones((10, 3))

        result = rmse_regions_quarterly(y, y)

        assert all(r == pytest.approx(0.0, abs=1e-12) for r in result)

    def test_rmse_regions_annual_ignores_nans(self):
        y_true = np.array([[np.nan, 1.0], [2.0, np.nan], [3.0, 3.0]])
        y_est = y_true.copy()

        result = rmse_regions_annual(y_true, y_est)

        assert all(r == pytest.approx(0.0, abs=1e-12) for r in result)


# ── recession_indicator ─────────────────────────────────────────────────────


class TestRecessionIndicator:
    @staticmethod
    def _make_datetime_series(n: int) -> pd.Series:
        """Create a pd.Series mimicking Ambric.datetime_ts."""
        return pd.Series(
            pd.date_range("2020-03-31", periods=n, freq="QE"), name="datetime"
        )

    def test_two_positive_periods_is_growth(self):
        dates = self._make_datetime_series(3)
        y = np.array([[0.01], [0.02], [0.03]])

        result = recession_indicator(y, dates, ["region_a"])

        classifications = result["classification"].tolist()
        assert all(c == "growth" for c in classifications)

    def test_two_negative_periods_is_recession(self):
        dates = self._make_datetime_series(3)
        y = np.array([[-0.01], [-0.02], [-0.03]])

        result = recession_indicator(y, dates, ["region_a"])

        classifications = result["classification"].tolist()
        assert all(c == "recession" for c in classifications)

    def test_alternating_signs_undetermined(self):
        dates = self._make_datetime_series(3)
        y = np.array([[0.01], [-0.01], [0.01]])

        result = recession_indicator(y, dates, ["region_a"])

        classifications = result["classification"].tolist()
        assert all(c == "undetermined" for c in classifications)

    def test_multiple_regions(self):
        dates = self._make_datetime_series(3)
        # Region A: growth, Region B: recession
        y = np.array([[0.01, -0.01], [0.02, -0.02], [0.03, -0.03]])

        result = recession_indicator(y, dates, ["region_a", "region_b"])

        region_a = result.loc[result["region"] == "region_a", "classification"].tolist()
        region_b = result.loc[result["region"] == "region_b", "classification"].tolist()
        assert all(c == "growth" for c in region_a)
        assert all(c == "recession" for c in region_b)


# ── generate_realistic_simulated_data ───────────────────────────────────────


class TestGenerateRealisticSimulatedData:
    def test_output_columns(self):
        df = generate_realistic_simulated_data(T=40, R=4, J=2)

        assert set(df.columns) >= {"datetime", "measure", "region", "value"}

    def test_expected_regions_present(self):
        df = generate_realistic_simulated_data(T=40, R=3, J=2)

        regions = df["region"].unique()
        assert "uk" in regions
        for i in range(3):
            assert f"region_{str(i).zfill(2)}" in regions

    def test_expected_measures_present(self):
        df = generate_realistic_simulated_data(T=40, R=3, J=2, n_macro=2)

        measures = set(df["measure"].unique())
        assert "gva_q_on_q" in measures
        assert "gva_q_on_4q" in measures
        assert "macro_0" in measures
        assert "macro_1" in measures
        assert "regional_covar_00" in measures
        assert "regional_covar_01" in measures
