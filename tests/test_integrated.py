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
        f"Fitting Ambric model with {n_its} iterations and {n_posterior_samples} posterior samples"
    )

    amb.fit(
        n_model_fit_iterations=n_its,
        n_posterior_samples=n_posterior_samples,
    )

    logger.info("Ambric model fit complete")

    amb.plot_national_quarterly_vs_implied()
    amb.plot_regional_annual_estimate()
    amb.plot_single_region_annual_estimate(region_name="region_00")
    amb.plot_estimated_regional_quarterly()
    amb.plot_current_nowcast()
    amb.live_recession_indicator()
    amb.live_point_estimates()
    mock_show.assert_called()


# ----------------------------------------------
## Model run simulated out-of-sample
# NB: doesn't test model prep
# ----------------------------------------------
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

    df_results, df_annual_regional, df_quarterly_national = run_out_of_sample_exercise(
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
    )

    plot_out_of_sample_rmse(df_annual_regional)
    plot_out_of_sample_nowcasts(df_annual_regional, df, region_measure=region_measure)
    out_of_sample_classification_performance_table(df_annual_regional)
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

    df_results, df_annual_regional, df_quarterly_national = run_out_of_sample_exercise(
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
    )

    plot_out_of_sample_rmse(df_annual_regional)
    plot_out_of_sample_nowcasts(df_annual_regional, df, region_measure=region_measure)
    out_of_sample_classification_performance_table(df_annual_regional)
    mock_show.assert_called()
