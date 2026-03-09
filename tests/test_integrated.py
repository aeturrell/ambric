from unittest.mock import patch

import numpy as np
import pandas as pd
from ambric import Ambric, run_out_of_sample_exercise
from ambric.diagnostics import (
    out_of_sample_classification_performance_table,
    plot_out_of_sample_nowcasts,
    plot_out_of_sample_rmse,
)
from ambric.utilities import generate_realistic_simulated_data
from loguru import logger


@patch("matplotlib.pyplot.show")
def test_vanilla_run(mock_show) -> None:
    n_factors = 2

    df_as_if_real: pd.DataFrame = generate_realistic_simulated_data(n_factors=n_factors)

    # Now we try out the data ingestion process
    # user sets the below
    aggregation_region = "uk"
    region_names = [
        x for x in df_as_if_real["region"].unique() if x != aggregation_region
    ]
    macro_names = [x for x in df_as_if_real["measure"].unique() if "macro" in x]
    region_covariate_names = [
        x for x in df_as_if_real["measure"].unique() if "regional_covar" in x
    ]

    # Then users runs the following:

    amb = Ambric(
        df_as_if_real,
        macro_names,
        region_names,
        region_covariate_names,
        n_factors=n_factors,
    )

    logger.info("Ambric model created with ID: " + amb.model_id)

    n_its = 1000
    n_posterior_samples = 3000

    logger.info(
        f"Fitting AMBRIC model with {n_its} iterations and {n_posterior_samples} posterior samples"
    )

    amb.fit(
        n_model_fit_iterations=n_its,
        n_posterior_samples=n_posterior_samples,
    )

    logger.info("AMBRIC model fit complete")

    amb.plot_national_quarterly_vs_implied()
    amb.plot_regional_annual_estimate()
    amb.plot_single_region_annual_estimate(region_name="region_00")
    amb.plot_estimated_regional_quarterly()
    amb.plot_current_nowcast()
    amb.bands_indicator()
    amb.live_recession_indicator()
    amb.point_estimates_q_on_4q()
    amb.point_estimates_q_on_q()
    amb.to_index_q_on_q()

    # Loadings diagnostics: verify data assembly and both plot variants.
    loadings_df = amb.assemble_loadings_data()
    assert not loadings_df.empty
    expected_cols = {
        "region",
        "loading_name",
        "broad_type",
        "mean",
        "hdi_low",
        "hdi_high",
    }
    assert expected_cols.issubset(loadings_df.columns)
    expected_broad_types = {"factors", "macro", "boost_signal"}
    assert expected_broad_types == set(loadings_df["broad_type"].unique())
    assert set(loadings_df["region"].unique()) == set(region_names)
    amb.plot_loadings_by_region()
    amb.plot_loadings_aggregate()

    mock_show.assert_called()


# ----------------------------------------------
## Model run simulated out-of-sample
# NB: doesn't test model prep
# ----------------------------------------------
@patch("matplotlib.pyplot.show")
def test_vanilla_run_one_region_no_lag(mock_show) -> None:
    """Like test_vanilla_run but one region has valid data up to the latest
    quarterly period (i.e. no publication lag for that region).  The model
    should still identify the modal lag across regions and work correctly."""
    n_factors = 2
    lag_qtrs = 6

    df_as_if_real: pd.DataFrame = generate_realistic_simulated_data(
        n_factors=n_factors, lag_qtrs=lag_qtrs
    )

    aggregation_region = "uk"
    region_names = [
        x for x in df_as_if_real["region"].unique() if x != aggregation_region
    ]
    macro_names = [x for x in df_as_if_real["measure"].unique() if "macro" in x]
    region_covariate_names = [
        x for x in df_as_if_real["measure"].unique() if "regional_covar" in x
    ]

    # Extend one region's gva_q_on_4q data to cover the lag period.
    # The simulated data has all regional annual data ending `lag_qtrs`
    # quarters before the UK quarterly data.  We fill in the missing
    # quarters for region_00 with plausible values so it has no lag.
    uk_dates = sorted(
        df_as_if_real.loc[df_as_if_real["measure"] == "gva_q_on_q", "datetime"].unique()
    )
    existing_regional_dates = sorted(
        df_as_if_real.loc[
            df_as_if_real["measure"] == "gva_q_on_4q", "datetime"
        ].unique()
    )
    extra_dates = [d for d in uk_dates if d not in existing_regional_dates]
    target_region = region_names[0]

    extra_rows = pd.DataFrame(
        {
            "datetime": extra_dates,
            "measure": "gva_q_on_4q",
            "region": target_region,
            "value": np.random.default_rng(42).normal(
                0.02, 0.01, size=len(extra_dates)
            ),
        }
    )
    df_as_if_real = pd.concat([df_as_if_real, extra_rows], ignore_index=True)

    amb = Ambric(
        df_as_if_real,
        macro_names,
        region_names,
        region_covariate_names,
        n_factors=n_factors,
    )

    # The modal lag should be the original lag_qtrs, not 0
    assert (
        amb.lag_qtrs == lag_qtrs
    ), f"Expected modal lag_qtrs={lag_qtrs}, got {amb.lag_qtrs}"

    logger.info("Ambric model created with ID: " + amb.model_id)

    n_its = 1000
    n_posterior_samples = 3000

    amb.fit(
        n_model_fit_iterations=n_its,
        n_posterior_samples=n_posterior_samples,
    )

    logger.info("AMBRIC model fit complete")

    amb.plot_national_quarterly_vs_implied()
    amb.plot_regional_annual_estimate()
    amb.plot_single_region_annual_estimate(region_name=target_region)
    amb.plot_estimated_regional_quarterly()
    amb.plot_current_nowcast()
    amb.live_recession_indicator()
    amb.point_estimates_q_on_4q()

    mock_show.assert_called()


@patch("matplotlib.pyplot.show")
def test_pseudo_realtime_directly(mock_show) -> None:
    """Direct function to run realtime simulation."""

    T = 100
    R = 10
    J = 3
    n_factors = 2
    n_macro = 2
    lag_qtrs = 6
    df = generate_realistic_simulated_data(
        T=T, R=R, J=J, n_factors=n_factors, n_macro=n_macro, lag_qtrs=lag_qtrs
    )
    aggregate_measure: str = "gva_q_on_q"
    aggregation_region: str = "uk"
    region_measure: str = "gva_q_on_4q"
    n_its = 1000
    n_posterior_samples = 1000
    region_names = [x for x in df["region"].unique() if x != aggregation_region]
    macro_names = [x for x in df["measure"].unique() if "macro" in x]
    region_covariate_names = [
        x for x in df["measure"].unique() if "regional_covar" in x
    ]

    aggregation_region = "uk"

    df_results = run_out_of_sample_exercise(
        df,
        macro_names,
        region_names,
        region_covariate_names,
        n_factors,
        aggregate_measure,
        aggregation_region,
        region_measure,
        n_its=n_its,
        n_posterior_samples=n_posterior_samples,
        lag_qtrs=lag_qtrs,
        step_size=20,
    )

    plot_out_of_sample_rmse(df_results, region_measure=region_measure)
    plot_out_of_sample_nowcasts(df_results, region_measure=region_measure)
    out_of_sample_classification_performance_table(df_results, region_measure)
    mock_show.assert_called()


@patch("matplotlib.pyplot.show")
def test_pseudo_realtime_with_early_nan_covariates(mock_show) -> None:
    """Like test_pseudo_realtime_directly but with regional covariates NaN'd
    out for the first 20 periods to exercise the reindex/NaN-handling fixes
    in the XGBoost and bridge equation paths."""

    T = 100
    R = 10
    J = 3
    n_factors = 2
    n_macro = 2
    lag_qtrs = 6
    df = generate_realistic_simulated_data(
        T=T, R=R, J=J, n_factors=n_factors, n_macro=n_macro, lag_qtrs=lag_qtrs
    )
    aggregate_measure: str = "gva_q_on_q"
    aggregation_region: str = "uk"
    region_measure: str = "gva_q_on_4q"
    n_its = 1000
    n_posterior_samples = 1000
    region_names = [x for x in df["region"].unique() if x != aggregation_region]
    macro_names = [x for x in df["measure"].unique() if "macro" in x]
    region_covariate_names = [
        x for x in df["measure"].unique() if "regional_covar" in x
    ]

    # NaN out the first n periods of regional covariate data
    first_n_periods_to_nan = 16
    sorted_dates = sorted(df["datetime"].unique())
    early_dates = sorted_dates[:first_n_periods_to_nan]
    mask = df["measure"].isin(region_covariate_names) & df["datetime"].isin(early_dates)
    df.loc[mask, "value"] = np.nan

    df_results = run_out_of_sample_exercise(
        df,
        macro_names,
        region_names,
        region_covariate_names,
        n_factors,
        aggregate_measure,
        aggregation_region,
        region_measure,
        n_its=n_its,
        n_posterior_samples=n_posterior_samples,
        lag_qtrs=lag_qtrs,
        step_size=20,
    )

    plot_out_of_sample_rmse(df_results, region_measure=region_measure)
    plot_out_of_sample_nowcasts(df_results, region_measure=region_measure)
    out_of_sample_classification_performance_table(df_results, region_measure)
    mock_show.assert_called()
