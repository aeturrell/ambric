import uuid

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger

# =============================================================================
# Global Constants & Weights
# =============================================================================
# Inter-temporal weights for annual-to-quarterly aggregation
OMEGA = np.array([1 / 4, 1 / 2, 3 / 4, 1, 3 / 4, 1 / 2, 1 / 4])


# =============================================================================
# Functions
# =============================================================================
def generate_realistic_simulated_data(
    T=130, R=12, J=4, n_factors=2, n_macro=2, lag_qtrs=6
) -> pd.DataFrame:
    """Generate simulated data and run the Ambric model on it."""
    # Simulate data

    logger.info(
        f"Simulating Data: T={T}, R={R}, J={J}, Macro={n_macro}, Factors={n_factors}, lag={lag_qtrs}"
    )

    y_uk, y_annual, y_reg_true, Z_panel, macro, y_annual_no_lag = (
        simulate_real_time_data(
            T=T,
            R=R,
            J=J,
            n_factors=n_factors,
            n_macro=n_macro,
            annnual_regional_lag_qrtrs=lag_qtrs,
            seed=34,
        )
    )

    # Turn data into real-world style data here, then turn it back
    # later within the model
    # to emulate loading real-world data
    # Incoming data structure:
    # quarter-end datetime | region | measure | value
    # chop the end off the annual data
    y_annual_sim_data = y_annual[:-lag_qtrs].copy()
    quarters = pd.date_range(start="1990-01-30", periods=len(y_uk), freq="QE")
    df_uk_q = pd.DataFrame(y_uk, index=quarters, columns=pd.Index(["uk_q_gva"]))
    df_macro = pd.DataFrame(
        macro,
        index=quarters,
        columns=pd.Index(["macro_" + str(i) for i in range(macro.shape[1])]),
    )
    df_Z_panel = pd.DataFrame()
    for j in range(J):
        df_regional_covars = pd.DataFrame(
            Z_panel[j],
            index=quarters,
            columns=pd.Index(["region_" + str(i).zfill(2) for i in range(R)]),
        )
        df_regional_covars["measure"] = "regional_covar_" + str(j).zfill(2)
        df_Z_panel = pd.concat([df_Z_panel, df_regional_covars], axis=0)

    df_regional_annual = pd.DataFrame(
        data=y_annual_sim_data,
        index=quarters[: len(y_annual_sim_data)],
        columns=pd.Index(["region_" + str(r).zfill(2) for r in range(R)]),
    )
    # stitch this together
    df_uk_q["region"] = "uk"
    df_uk_q["measure"] = "gva_q_on_q"
    df_uk_q = df_uk_q.rename(columns={"uk_q_gva": "value"})
    df_uk_q = df_uk_q.reset_index(names="datetime")
    df_macro = pd.melt(
        df_macro.reset_index(), id_vars=["index"], var_name="measure"
    ).rename(columns={"index": "datetime"})
    df_macro["region"] = "uk"
    df_Z_panel = pd.melt(
        df_Z_panel.reset_index(), id_vars=["measure", "index"], var_name="region"
    ).rename(columns={"index": "datetime"})
    df_regional_annual = pd.melt(
        df_regional_annual.reset_index(), id_vars=["index"], var_name="region"
    ).rename(columns={"index": "datetime"})
    df_regional_annual["measure"] = "gva_q_on_4q"
    df_as_if_real = pd.concat(
        [df_uk_q, df_macro, df_Z_panel, df_regional_annual], axis=0
    )
    return df_as_if_real


def gen_unique_id() -> str:
    """Creates a unique, time-based ID for a model run.

    Returns:
        str: ID string in format UUID
    """
    run_str = str(uuid.uuid4()).split("-")[0]
    return run_str


def prep_data_for_model_run(
    df: pd.DataFrame,
    macro_names: list[str],
    region_names: list[str],
    region_covariate_names: list[str],
    aggregate_measure: str = "gva_q_on_q",
    aggregation_region: str = "uk",
    region_measure: str = "gva_q_on_4q",
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    list[npt.NDArray[np.float64]],
    npt.NDArray[np.float64],
    int,
]:
    """Expects a data frame in following format:
    datetime | measure | region | value


    Args:
        df (pd.DataFrame): Long format dataframe with all data in
        macro_names (list[str]): Names of macro UK series
        region_names (list[str]): Names of (local) regions
        region_covariate_names (list[str]): Names of regional level series. These will be absorbed into exogenous factors.
        aggregate_measure (str, optional): UK-wide measure in q-on-q growth rate. Defaults to "gva_q_on_q".
        aggregation_region (str, optional): Highest level geography, which other regions sum to. Defaults to "uk".
        region_measure (str, optional): Regional measure, q-on-4q growth rate. Defaults to "gva_q_on_4q".

    Returns:
        (npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]): y_uk_extracted, y_a_r_extracted, Z_panel_extract, macro_extracted
        lag_qtrs (int): Number of quarters lag in annual regional data compared to quarterly UK data
    """
    logger.info("Prepping data for model run.")
    # Data checks
    available_measures = set(df["measure"].unique())
    missing_measures = [
        x
        for x in macro_names
        + region_covariate_names
        + [aggregate_measure]
        + [region_measure]
        if x not in available_measures
    ]
    if missing_measures:
        raise KeyError(f"Measures not found in the data: {missing_measures}")

    available_regions = set(df["region"].unique())
    missing_regions = [
        x for x in region_names + [aggregation_region] if x not in available_regions
    ]
    if missing_regions:
        raise KeyError(f"Regions not found in the data: {missing_regions}")

    y_uk_extracted: npt.NDArray[np.float64] = df.loc[
        ((df["measure"] == aggregate_measure) & (df["region"] == aggregation_region)),
        "value",
    ].values
    macro_extracted: npt.NDArray[np.float64] = (
        df.loc[
            ((df["measure"].isin(macro_names)) & (df["region"] == aggregation_region)),
            :,
        ]
        .pivot(index="datetime", columns="measure", values="value")
        .values
    )
    y_a_r_extracted: npt.NDArray[np.float64] = (
        df.loc[
            (df["measure"] == region_measure) & (df["region"].isin(region_names)),
            :,
        ]
        .pivot(index="datetime", columns="region", values="value")
        .values
    )
    # If len(y_a_r_extracted) < len(y_uk_extracted), extend the latter with nans
    # this is to enable the model to fill in the (given) nan gaps
    missing_annual = len(y_uk_extracted) - len(y_a_r_extracted)
    if missing_annual > 0:
        nan_array_to_concat = np.full(
            (missing_annual, y_a_r_extracted.shape[1]), np.nan
        )
        y_a_r_extracted = np.concatenate([y_a_r_extracted, nan_array_to_concat])
        logger.info(
            f"Adding {str(missing_annual)} extra rows of nans to q-on-4q regional data to match number of rows in quarterly data; these extra rows will be estimated by the model."
        )

    all_datetimes = sorted(df["datetime"].unique())
    Z_panel_extract: list[npt.NDArray[np.float64]] = [
        df.loc[
            ((df["region"].isin(region_names)) & (df["measure"] == curr_measure)),
            :,
        ]
        .pivot(index="datetime", columns="region", values="value")
        .reindex(index=all_datetimes, columns=region_names)
        .values
        for curr_measure in region_covariate_names
    ]
    logger.info("Input lengths after data prep:")
    logger.info(f"  Quarterly {aggregate_measure}: T_max = {len(y_uk_extracted)}")
    logger.info(f"  Annual {region_measure}: T_max = {y_a_r_extracted.shape[0]}")
    logger.info(f"  Macro series: {macro_extracted.shape[1]}")
    logger.info(f"  Regional covariate series: {len(Z_panel_extract)} per region")
    return (
        y_uk_extracted,
        y_a_r_extracted,
        Z_panel_extract,
        macro_extracted,
        missing_annual,
    )


def simulate_data(T=80, R=6, J=3, n_factors=2, n_macro=2, seed=42):
    """Simulate mixed-frequency data with regional panels and macro indicators."""
    np.random.seed(seed)

    # True factors (AR(1) processes)
    f_true = np.zeros((T, n_factors))
    for t in range(1, T):
        f_true[t] = 0.7 * f_true[t - 1] + 0.3 * np.random.randn(n_factors)

    # True macro indicators
    macro = np.zeros((T, n_macro))
    for m in range(n_macro):
        macro[:, m] = 0.4 * f_true[:, 0] + 0.6 * np.cumsum(0.1 * np.random.randn(T))
        macro[:, m] = (macro[:, m] - macro[:, m].mean()) / macro[:, m].std()

    # Loadings and Growth
    Lambda_true = np.random.randn(R, n_factors) * 0.3
    Gamma_true = np.random.randn(R, n_macro) * 0.2
    sigma_eps = 0.15
    y_reg_true = (
        f_true @ Lambda_true.T
        + macro @ Gamma_true.T
        + sigma_eps * np.random.randn(T, R)
    )

    # UK aggregate
    w_true = np.ones(R) / R
    y_uk = y_reg_true @ w_true + 0.01 * np.random.randn(T)

    # Regional indicator panels
    Z_panel = [
        f_true @ (np.random.randn(R, n_factors) * 0.5).T + 0.3 * np.random.randn(T, R)
        for _ in range(J)
    ]

    # Annual regional growth
    y_annual = np.full((T, R), np.nan)
    for t in range(6, T):
        if (t + 1) % 4 == 0:
            for r in range(R):
                y_annual[t, r] = (
                    sum(OMEGA[j] * y_reg_true[t - j, r] for j in range(7))
                    + 0.05 * np.random.randn()
                )

    return y_uk, y_annual, y_reg_true, Z_panel, macro


def simulate_real_time_data(
    T=80, R=6, J=3, n_factors=2, n_macro=2, annnual_regional_lag_qrtrs=6, seed=42
):
    y_uk, y_annual, y_reg_true, Z_panel, macro = simulate_data(
        T=T, R=R, J=J, n_factors=n_factors, n_macro=n_macro, seed=seed
    )
    # Now knock out any annual regional data that would not have been available in real-time, starting from the end
    y_annual_no_lags = y_annual.copy()
    for t in range(T - 1, T - annnual_regional_lag_qrtrs, -1):
        y_annual[t, :] = np.nan
    return y_uk, y_annual, y_reg_true, Z_panel, macro, y_annual_no_lags
