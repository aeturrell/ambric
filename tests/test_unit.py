"""Unit tests for ambric helper functions and validation logic.

Each test is small, isolated, and follows the Arrange-Act-Assert pattern.
No model fitting is performed — only the deterministic helpers and
validation paths are exercised.
"""

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
from loguru import logger

from ambric import (
    Ambric,
    _aggregate_quarterly_to_annual,
    _almon_weights,
    _apply_midas_weights,
    build_ambric_model,
    extract_factors_from_panel,
    fit_bridge_equation,
    impute_panel,
    quarter_differences,
    train_xgboost_annual,
)
from ambric.diagnostics import (
    AdviProgressLogger,
    assemble_loadings_data,
    assess_advi_convergence,
    recession_indicator,
    rmse_national_quarterly,
    rmse_regions_annual,
    rmse_regions_quarterly,
    trace_to_series,
)
from ambric.utilities import (
    first_q1_offset,
    generate_realistic_simulated_data,
    prep_data_for_model_run,
)

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

    def test_region_column_order_matches_region_names(self, simulated_df):
        """When one region has extra data (no lag), the columns of
        y_a_r_extracted must still match the order of region_names, not
        the alphabetical order that pandas pivot produces by default."""
        df = simulated_df.copy()
        region_names = sorted([x for x in df["region"].unique() if x != "uk"])
        macro_names = [x for x in df["measure"].unique() if "macro" in x]
        region_covariate_names = [
            x for x in df["measure"].unique() if "regional_covar" in x
        ]

        # Reverse the region_names so they differ from alphabetical order
        region_names = list(reversed(region_names))

        # Extend the *first* region in our list with extra data (no lag)
        target_region = region_names[0]
        uk_dates = sorted(df.loc[df["measure"] == "gva_q_on_q", "datetime"].unique())
        existing_dates = sorted(
            df.loc[df["measure"] == "gva_q_on_4q", "datetime"].unique()
        )
        extra_dates = [d for d in uk_dates if d not in existing_dates]
        extra_rows = pd.DataFrame(
            {
                "datetime": extra_dates,
                "measure": "gva_q_on_4q",
                "region": target_region,
                "value": np.random.default_rng(99).normal(
                    0.05, 0.01, size=len(extra_dates)
                ),
            }
        )
        df = pd.concat([df, extra_rows], ignore_index=True)

        y_uk, y_a_r, Z_panel, macro, lag_qtrs, y_qoq_r = prep_data_for_model_run(
            df,
            macro_names=macro_names,
            region_names=region_names,
            region_covariate_names=region_covariate_names,
        )

        # Count trailing NaNs per column.  The target region (col 0) should
        # have strictly fewer trailing NaNs than every other region, proving
        # the extra data landed in the right column.
        def _trailing_nan_count(col: np.ndarray) -> int:
            if np.all(np.isnan(col)):
                return len(col)
            return int(np.argmax(~np.isnan(col[::-1])))

        target_trailing = _trailing_nan_count(y_a_r[:, 0])
        for col in range(1, y_a_r.shape[1]):
            other_trailing = _trailing_nan_count(y_a_r[:, col])
            assert target_trailing < other_trailing, (
                f"{target_region} (col 0) has {target_trailing} trailing NaNs "
                f"but {region_names[col]} (col {col}) has only "
                f"{other_trailing}; extra data appears mis-ordered"
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


# ── assemble_loadings_data scaling ──────────────────────────────────────────


class TestAssembleLoadingsDataScaling:
    """Verify that stds scale loadings correctly in assemble_loadings_data."""

    @pytest.fixture()
    def mock_trace(self):
        """Minimal ArviZ InferenceData with deterministic posterior values."""
        import arviz as az
        import xarray as xr

        R, K, M = 2, 2, 1
        # Use a single chain / single draw so mean == the value itself.
        lambda_vals = np.array([[[0.5, -1.0], [2.0, 0.3]]])  # (1, R, K)
        gamma_vals = np.array([[[0.4], [0.8]]])  # (1, R, M)
        delta_vals = np.array([[1.5, -0.6]])  # (1, R)

        posterior = xr.Dataset(
            {
                "Lambda": (["chain", "region", "factor"], lambda_vals),
                "Gamma": (["chain", "region", "macro"], gamma_vals),
                "delta_r": (["chain", "region"], delta_vals),
            },
            coords={
                "chain": [0],
                "region": list(range(R)),
                "factor": list(range(K)),
                "macro": list(range(M)),
            },
        )
        # ArviZ expects a "draw" dimension; rename the single-element chain
        # trick: expand with a draw dimension of length 1.
        posterior = posterior.expand_dims("draw")

        return az.InferenceData(posterior=posterior)

    def test_unscaled_returns_raw_values(self, mock_trace):
        df = assemble_loadings_data(
            mock_trace,
            region_names=["A", "B"],
            macro_names=["gdp"],
        )
        # No scaling applied — factor_0 for region A should be raw 0.5
        row = df[(df["region"] == "A") & (df["loading_name"] == "factor_0")]
        assert np.isclose(row["mean"].iloc[0], 0.5)
        assert "scaled" in df.columns
        assert not df["scaled"].any()

    def test_scaled_multiplies_by_std(self, mock_trace):
        factor_stds = np.array([2.0, 3.0])
        macro_stds = np.array([0.5])
        bridge_signal_stds = np.array([4.0, 5.0])

        df = assemble_loadings_data(
            mock_trace,
            region_names=["A", "B"],
            macro_names=["gdp"],
            factor_stds=factor_stds,
            macro_stds=macro_stds,
            bridge_signal_stds=bridge_signal_stds,
        )

        # Factor 0 for region A: 0.5 * 2.0 = 1.0
        row = df[(df["region"] == "A") & (df["loading_name"] == "factor_0")]
        assert np.isclose(row["mean"].iloc[0], 1.0)

        # Factor 1 for region A: -1.0 * 3.0 = -3.0
        row = df[(df["region"] == "A") & (df["loading_name"] == "factor_1")]
        assert np.isclose(row["mean"].iloc[0], -3.0)

        # Macro "gdp" for region B: 0.8 * 0.5 = 0.4
        row = df[(df["region"] == "B") & (df["loading_name"] == "gdp")]
        assert np.isclose(row["mean"].iloc[0], 0.4)

        # Bridge signal for region B: -0.6 * 5.0 = -3.0
        row = df[(df["region"] == "B") & (df["loading_name"] == "boost_signal")]
        assert np.isclose(row["mean"].iloc[0], -3.0)

        assert df["scaled"].all()


# ── Data alignment (date spine, ordering, calendar) ─────────────────────────


def _names(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """(region_names, macro_names, region_covariate_names) for a simulated df."""
    region_names = sorted([x for x in df["region"].unique() if x != "uk"])
    macro_names = sorted([x for x in df["measure"].unique() if "macro" in x])
    covariate_names = sorted(
        [x for x in df["measure"].unique() if "regional_covar" in x]
    )
    return region_names, macro_names, covariate_names


class TestDataAlignment:
    def test_row_order_does_not_change_arrays(self, simulated_df):
        region_names, macro_names, covariate_names = _names(simulated_df)
        shuffled = simulated_df.sample(frac=1, random_state=0)

        expected = prep_data_for_model_run(
            simulated_df, macro_names, region_names, covariate_names
        )
        result = prep_data_for_model_run(
            shuffled, macro_names, region_names, covariate_names
        )

        np.testing.assert_array_equal(result[0], expected[0])  # y_uk
        np.testing.assert_array_equal(result[1], expected[1])  # y_annual
        np.testing.assert_array_equal(result[3], expected[3])  # macro
        for z_result, z_expected in zip(result[2], expected[2], strict=True):
            np.testing.assert_array_equal(z_result, z_expected)

    def test_shuffled_rows_give_sorted_datetime_ts(self, simulated_df):
        region_names, macro_names, covariate_names = _names(simulated_df)
        shuffled = simulated_df.sample(frac=1, random_state=0)

        amb = Ambric(shuffled, macro_names, region_names, covariate_names)

        assert amb.datetime_ts.is_monotonic_increasing
        assert amb.datetime_ts.is_unique
        assert len(amb.datetime_ts) == amb.y_uk.shape[0]

    def test_macro_columns_follow_macro_names_order(self, simulated_df):
        df = simulated_df.copy()
        # Names whose requested order is the reverse of alphabetical order
        df["measure"] = df["measure"].replace(
            {"macro_0": "unemployment", "macro_1": "cpi"}
        )
        region_names, _, covariate_names = _names(df)
        macro_names = ["unemployment", "cpi"]

        macro = prep_data_for_model_run(df, macro_names, region_names, covariate_names)[
            3
        ]

        for col, name in enumerate(macro_names):
            expected = df.loc[df["measure"] == name].sort_values("datetime")["value"]
            np.testing.assert_array_equal(macro[:, col], expected.values)

    def test_annual_rows_only_at_year_end_align_to_spine(self, simulated_df):
        """Regional data supplied without NaN padding rows must land on the
        same quarters as the padded version."""
        region_names, macro_names, covariate_names = _names(simulated_df)
        is_padding = (simulated_df["measure"] == "gva_q_on_4q") & simulated_df[
            "value"
        ].isna()
        df_no_padding = simulated_df.loc[~is_padding]

        expected = prep_data_for_model_run(
            simulated_df, macro_names, region_names, covariate_names
        )
        result = prep_data_for_model_run(
            df_no_padding, macro_names, region_names, covariate_names
        )

        assert is_padding.any()
        np.testing.assert_array_equal(result[1], expected[1])
        assert result[4] == expected[4]  # same publication lag

    def test_duplicate_rows_raise(self, simulated_df):
        region_names, macro_names, covariate_names = _names(simulated_df)
        df = pd.concat([simulated_df, simulated_df.iloc[[0]]], ignore_index=True)

        with pytest.raises(ValueError, match="Duplicate rows"):
            prep_data_for_model_run(df, macro_names, region_names, covariate_names)

    def test_gap_in_aggregate_series_raises(self, simulated_df):
        region_names, macro_names, covariate_names = _names(simulated_df)
        dates = sorted(simulated_df["datetime"].unique())
        is_gap = (simulated_df["measure"] == "gva_q_on_q") & (
            simulated_df["datetime"] == dates[10]
        )

        with pytest.raises(ValueError, match="missing quarters"):
            prep_data_for_model_run(
                simulated_df.loc[~is_gap], macro_names, region_names, covariate_names
            )

    def test_incomplete_macro_series_raises(self, simulated_df):
        region_names, macro_names, covariate_names = _names(simulated_df)
        dates = sorted(simulated_df["datetime"].unique())
        is_missing = (simulated_df["measure"] == macro_names[0]) & (
            simulated_df["datetime"] == dates[-1]
        )

        with pytest.raises(ValueError, match="Macro series with missing values"):
            prep_data_for_model_run(
                simulated_df.loc[~is_missing],
                macro_names,
                region_names,
                covariate_names,
            )


class TestCalendarOffset:
    @pytest.mark.parametrize(
        ("first_date", "expected"),
        [
            ("2000-03-31", 0),
            ("2000-06-30", 3),
            ("2000-09-30", 2),
            ("2000-12-31", 1),
        ],
    )
    def test_first_q1_offset(self, first_date, expected):
        spine = pd.date_range(first_date, periods=12, freq="QE")

        offset = first_q1_offset(spine)

        assert offset == expected
        assert spine[offset].quarter == 1

    def test_aggregate_skips_rows_before_first_q1(self):
        X = np.arange(11, dtype=np.float64).reshape(-1, 1)  # rows 0,1 are Q3,Q4

        result = _aggregate_quarterly_to_annual(X, q1_offset=2)

        np.testing.assert_allclose(result[:, 0], [3.5, 7.5])

    def test_xgboost_start_quarter_does_not_change_predictions(self):
        """Dropping the first two quarters (so the sample starts in Q3) must
        give the same annual predictions for the years that remain."""
        rng = np.random.default_rng(0)
        T, R, J, M = 40, 3, 2, 1
        Z_panel = [rng.normal(size=(T, R)) for _ in range(J)]
        macro = rng.normal(size=(T, M))
        y_annual = np.full((T, R), np.nan)
        y_annual[7::4] = rng.normal(size=(len(range(7, T, 4)), R))  # year 0 unobserved

        _, expected = train_xgboost_annual(Z_panel, macro, y_annual)
        _, result = train_xgboost_annual(
            [z[2:] for z in Z_panel], macro[2:], y_annual[2:], q1_offset=2
        )

        np.testing.assert_allclose(result, expected[1:])

    def test_bridge_signal_sums_to_annual_fit_when_sample_starts_mid_year(self):
        rng = np.random.default_rng(1)
        T, R, J, offset = 38, 3, 2, 2
        Z_panel = [rng.normal(size=(T, R)) for _ in range(J)]
        y_uk = rng.normal(scale=0.5, size=T)
        macro = rng.normal(size=(T, 1))
        y_annual = np.full((T, R), np.nan)
        q4_rows = np.arange(offset + 3, T, 4)
        y_annual[q4_rows] = rng.normal(size=(len(q4_rows), R))
        xgb_preds = rng.normal(size=(len(q4_rows), R))

        signal, info = fit_bridge_equation(
            y_annual, xgb_preds, Z_panel, y_uk, macro, use_almon=False, q1_offset=offset
        )

        # Summing the quarterly signal over a calendar year reproduces the
        # bridge regression's annual fitted value.
        for a, q4 in enumerate(q4_rows):
            for r in range(R):
                indicators = np.concatenate(
                    [[y_uk[q4 - 3 : q4 + 1].mean()]]
                    + [[Z_panel[j][q4 - 3 : q4 + 1, r].mean()] for j in range(J)]
                )
                fitted = info["bridge_reg"].predict(
                    np.concatenate([[xgb_preds[a, r]], indicators]).reshape(1, -1)
                )[0]
                assert signal[q4 - 3 : q4 + 1, r].sum() == pytest.approx(fitted)

    def test_ambric_detects_mid_year_start(self, simulated_df):
        region_names, macro_names, covariate_names = _names(simulated_df)
        dates = sorted(simulated_df["datetime"].unique())
        df = simulated_df.loc[simulated_df["datetime"] >= dates[2]]  # starts in Q3

        amb = Ambric(df, macro_names, region_names, covariate_names)

        assert amb.q1_offset == 2
        assert amb.datetime_ts.iloc[amb.q1_offset].quarter == 1

    def test_sample_starts_six_quarters_before_first_annual_observation(
        self, simulated_df
    ):
        """With year-end-only regional rows, the sample keeps the six earlier
        quarters that the first annual observation depends on."""
        region_names, macro_names, covariate_names = _names(simulated_df)
        is_padding = (simulated_df["measure"] == "gva_q_on_4q") & simulated_df[
            "value"
        ].isna()
        df = simulated_df.loc[~is_padding]
        first_obs = df.loc[df["measure"] == "gva_q_on_4q", "datetime"].min()

        amb = Ambric(df, macro_names, region_names, covariate_names)

        assert amb.datetime_ts.iloc[6] == first_obs
        assert not np.any(np.isnan(amb.y_annual[6]))


class TestEarlyAnnualObservations:
    def test_observations_without_full_lag_window_are_excluded(self):
        rng = np.random.default_rng(0)
        T, R, K, M = 16, 2, 1, 1
        y_annual = np.full((T, R), np.nan)
        y_annual[3::4] = 0.02  # rows 3, 7, 11, 15; row 3 lacks a full window

        model = build_ambric_model(
            y_uk=rng.normal(scale=0.005, size=T),
            y_annual=y_annual,
            factors=rng.normal(size=(T, K)),
            macro=rng.normal(size=(T, M)),
            bridge_signal=rng.normal(scale=0.005, size=(T, R)),
        )

        n_observed = model["obs_annual_observed"].eval().size
        assert n_observed == 3 * R

    def test_annual_estimates_are_nan_before_full_lag_window(self):
        rng = np.random.default_rng(0)
        T, R = 12, 2
        trace = az.from_dict(
            posterior={
                "y_reg": rng.normal(size=(1, 20, T, R)),
                "w": np.full((1, 20, R), 0.5),
            }
        )

        _, _, y_annual_est = trace_to_series(trace)

        assert np.all(np.isnan(y_annual_est[:6]))
        assert not np.any(np.isnan(y_annual_est[6:]))


# ── ADVI convergence reporting ──────────────────────────────────────────────


class TestAdviConvergence:
    def test_falling_loss_is_not_converged(self):
        losses = np.linspace(1000.0, 0.0, 2000)

        result = assess_advi_convergence(losses, tolerance=1.0)

        assert result["converged"] is False
        assert result["improvement"] == pytest.approx(50.0, rel=0.01)
        assert result["window"] == 100
        assert result["n_iterations"] == 2000

    def test_flat_noisy_loss_is_converged(self):
        rng = np.random.default_rng(0)
        losses = np.concatenate(
            [np.linspace(1000.0, 5.0, 500), 5.0 + rng.normal(scale=2.0, size=20000)]
        )

        result = assess_advi_convergence(losses, tolerance=1.0)

        assert result["converged"] is True
        assert result["final_loss"] == pytest.approx(5.0, abs=0.2)

    def test_noisy_short_run_is_not_converged_by_chance(self):
        """A loss too noisy to measure must not pass just because the last
        window happened to be no lower than the one before."""
        rng = np.random.default_rng(0)
        losses = 30000.0 + rng.normal(scale=15000.0, size=400)

        result = assess_advi_convergence(losses, tolerance=1.0)

        assert result["converged"] is False
        assert result["standard_error"] > 1.0

    def test_non_finite_loss_is_not_converged(self):
        losses = np.full(1000, np.inf)

        result = assess_advi_convergence(losses)

        assert result["converged"] is False

    def test_progress_logger_reports_every_n_iterations(self):
        messages: list[str] = []
        sink_id = logger.add(messages.append, format="{message}", level="INFO")
        progress = AdviProgressLogger(every=100, n_total=300)
        losses = np.linspace(300.0, 0.0, 300)

        try:
            for i in range(1, 301):
                progress(None, losses[:i], i)
        finally:
            logger.remove(sink_id)

        reports = [m for m in messages if "ADVI" in m]
        assert len(reports) == 3
        assert "100/300" in reports[0]
        assert "improved by 100.33" in reports[1]


# ── Scale handling (decimal growth rates) ───────────────────────────────────


def _bridge_inputs(seed: int = 0):
    """Small synthetic inputs for the XGBoost and bridge steps, in decimals."""
    rng = np.random.default_rng(seed)
    T, R, J = 60, 4, 2
    Z_panel = [rng.normal(size=(T, R)) for _ in range(J)]
    macro = rng.normal(size=(T, 1))
    y_uk = rng.normal(scale=0.005, size=T)
    y_annual = np.full((T, R), np.nan)
    q4_rows = np.arange(3, T, 4)
    signal = sum(z.reshape(T // 4, 4, R).mean(axis=1) for z in Z_panel)
    y_annual[q4_rows] = 0.02 * signal + rng.normal(scale=0.005, size=signal.shape)
    return Z_panel, macro, y_uk, y_annual


class TestScaleHandling:
    def test_xgboost_learns_from_decimal_growth_rates(self):
        """With targets of order 0.02 the regularisation must not flatten the
        fit: predictions should vary and track the target."""
        Z_panel, macro, _, y_annual = _bridge_inputs()

        _, preds = train_xgboost_annual(Z_panel, macro, y_annual)

        target = y_annual[3::4]
        assert np.corrcoef(preds.ravel(), target.ravel())[0, 1] > 0.8
        assert preds.std() > 0.5 * target.std()

    def test_xgboost_predictions_scale_with_growth_units(self):
        Z_panel, macro, _, y_annual = _bridge_inputs()

        _, preds_decimal = train_xgboost_annual(Z_panel, macro, y_annual)
        _, preds_percent = train_xgboost_annual(Z_panel, macro, y_annual * 100)

        np.testing.assert_allclose(preds_percent, preds_decimal * 100, rtol=1e-3)

    def test_bridge_fit_does_not_depend_on_growth_units(self):
        Z_panel, macro, y_uk, y_annual = _bridge_inputs()
        _, preds = train_xgboost_annual(Z_panel, macro, y_annual)

        signal_decimal, info_decimal = fit_bridge_equation(
            y_annual, preds, Z_panel, y_uk, macro, use_almon=False
        )
        signal_percent, info_percent = fit_bridge_equation(
            y_annual * 100, preds * 100, Z_panel, y_uk * 100, macro, use_almon=False
        )

        assert info_percent["bridge_r2"] == pytest.approx(info_decimal["bridge_r2"])
        np.testing.assert_allclose(
            signal_percent, signal_decimal * 100, rtol=1e-4, atol=1e-6
        )
        # The ridge penalty must not wipe out a real relationship
        assert info_decimal["bridge_r2"] > 0.5

    def test_bridge_coefficients_reproduce_regression_predictions(self):
        Z_panel, macro, y_uk, y_annual = _bridge_inputs()
        _, preds = train_xgboost_annual(Z_panel, macro, y_annual)
        _, info = fit_bridge_equation(
            y_annual, preds, Z_panel, y_uk, macro, use_almon=False
        )
        rng = np.random.default_rng(1)
        X_new = rng.normal(size=(5, 1 + 1 + len(Z_panel)))

        by_hand = (
            X_new @ np.concatenate([[info["delta"]], info["betas"]]) + info["intercept"]
        )

        np.testing.assert_allclose(info["bridge_reg"].predict(X_new), by_hand)

    def test_model_does_not_depend_on_macro_units(self):
        """Rescaling and shifting a macro series must leave the model's
        log-probability unchanged, since macro is z-scored internally."""
        rng = np.random.default_rng(0)
        T, R, K, M = 16, 2, 1, 2
        y_annual = np.full((T, R), np.nan)
        y_annual[7::4] = 0.02
        macro = rng.normal(size=(T, M))
        common = dict(
            y_uk=rng.normal(scale=0.005, size=T),
            y_annual=y_annual,
            factors=rng.normal(size=(T, K)),
            bridge_signal=rng.normal(scale=0.005, size=(T, R)),
        )

        model = build_ambric_model(macro=macro, **common)
        model_rescaled = build_ambric_model(macro=macro * 1000 + 50, **common)

        point = model.initial_point()
        assert model_rescaled.compile_logp()(point) == pytest.approx(
            model.compile_logp()(point)
        )

    def test_growth_parameters_are_reported_in_decimal_units(self):
        rng = np.random.default_rng(0)
        T, R, K, M = 16, 2, 1, 1
        y_annual = np.full((T, R), np.nan)
        y_annual[7::4] = 0.02

        model = build_ambric_model(
            y_uk=rng.normal(scale=0.005, size=T),
            y_annual=y_annual,
            factors=rng.normal(size=(T, K)),
            macro=rng.normal(size=(T, M)),
            bridge_signal=rng.normal(scale=0.005, size=(T, R)),
        )

        deterministic_names = {v.name for v in model.deterministics}
        assert {
            "y_reg",
            "sigma_eps",
            "sigma_uk",
            "sigma_ann",
            "Lambda",
            "Gamma",
            "alpha_r",
        }.issubset(deterministic_names)
        prior = pm.sample_prior_predictive(
            draws=500, model=model, var_names=["sigma_eps", "y_reg"], random_seed=0
        ).prior
        # Prior quarterly growth and its innovation scale are of the order of
        # a few percent at most, not tens of percent.
        assert float(prior["sigma_eps"].mean()) < 0.02
        assert float(np.abs(prior["y_reg"]).mean()) < 0.1


# ── Out-of-fold XGBoost predictions (no in-sample leakage) ──────────────────

# Settings under which XGBoost can memorise its training data
_MEMORISING_XGB = dict(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.3,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_alpha=0.0,
    reg_lambda=0.0,
    min_child_weight=0,
)


def _pure_noise_inputs(seed: int = 0, T: int = 80, R: int = 5):
    """Annual targets that are unrelated to the features."""
    rng = np.random.default_rng(seed)
    Z_panel = [rng.normal(size=(T, R)) for _ in range(3)]
    macro = rng.normal(size=(T, 1))
    y_annual = np.full((T, R), np.nan)
    y_annual[3::4] = rng.normal(scale=0.02, size=(T // 4, R))
    return Z_panel, macro, y_annual


class TestOutOfFoldPredictions:
    def test_observed_years_are_not_predicted_by_a_model_that_saw_them(self):
        """The target is noise, so a model can only match it by memorising.
        The fitted model does memorise its training data, but the returned
        predictions for observed years must not benefit from that."""
        Z_panel, macro, y_annual = _pure_noise_inputs()

        model, preds = train_xgboost_annual(
            Z_panel, macro, y_annual, xgb_params=_MEMORISING_XGB
        )

        target = y_annual[3::4]
        assert model.in_sample_rmse_ < 0.1 * target.std()  # it can memorise
        assert abs(np.corrcoef(preds.ravel(), target.ravel())[0, 1]) < 0.3
        assert model.out_of_fold_rmse_ > 0.8 * target.std()
        assert model.out_of_fold_ is True

    def test_unobserved_years_are_predicted_by_the_full_model(self):
        Z_panel, macro, y_annual = _pure_noise_inputs()
        y_annual[-8:] = np.nan  # last two years unpublished

        model, preds = train_xgboost_annual(Z_panel, macro, y_annual)

        assert not np.any(np.isnan(preds))
        features_last_year = np.column_stack(
            [z[-4:].mean(axis=0) for z in Z_panel]
            + [np.full(y_annual.shape[1], macro[-4:].mean())]
        )
        expected = (
            model.predict(features_last_year) * model.target_scale_ + model.target_mean_
        )
        np.testing.assert_allclose(preds[-1], expected, rtol=1e-5)

    def test_single_observed_year_falls_back_to_in_sample(self):
        Z_panel, macro, y_annual = _pure_noise_inputs(T=20)
        y_annual[:] = np.nan
        y_annual[7] = 0.02 + 0.01 * np.arange(y_annual.shape[1])

        model, preds = train_xgboost_annual(Z_panel, macro, y_annual)

        assert model.out_of_fold_ is False
        assert not np.any(np.isnan(preds))

    def test_incomplete_final_year_gets_its_own_prediction(self):
        Z_panel, macro, y_annual = _pure_noise_inputs(T=80)
        Z_panel = [np.vstack([z, z[:2] + 5.0]) for z in Z_panel]  # 2 extra quarters
        macro = np.vstack([macro, macro[:2]])
        y_annual = np.vstack([y_annual, np.full((2, y_annual.shape[1]), np.nan)])

        model, preds = train_xgboost_annual(Z_panel, macro, y_annual)
        first_edge, last_edge = model.edge_year_preds_

        assert preds.shape[0] == 20
        assert first_edge is None
        assert last_edge is not None and last_edge.shape == (y_annual.shape[1],)

    def test_bridge_signal_uses_incomplete_year_prediction(self):
        rng = np.random.default_rng(2)
        T, R = 42, 3  # ten complete years and two further quarters
        Z_panel = [rng.normal(size=(T, R))]
        macro = rng.normal(size=(T, 1))
        y_uk = rng.normal(scale=0.005, size=T)
        y_annual = np.full((T, R), np.nan)
        y_annual[3:40:4] = rng.normal(scale=0.02, size=(10, R))
        xgb_preds = y_annual[3:40:4] + rng.normal(scale=0.005, size=(10, R))
        last_edge = np.array([0.5, -0.5, 0.25])

        signal_nearest, info = fit_bridge_equation(
            y_annual, xgb_preds, Z_panel, y_uk, macro, use_almon=False
        )
        signal_edge, _ = fit_bridge_equation(
            y_annual,
            xgb_preds,
            Z_panel,
            y_uk,
            macro,
            use_almon=False,
            xgb_edge_preds=(None, last_edge),
        )

        np.testing.assert_allclose(signal_edge[:40], signal_nearest[:40])
        expected_shift = (info["delta"] / 4.0) * (last_edge - xgb_preds[-1])
        np.testing.assert_allclose(
            signal_edge[40:] - signal_nearest[40:], np.tile(expected_shift, (2, 1))
        )
