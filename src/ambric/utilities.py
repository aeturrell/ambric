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

# Noise scales of the simulated data generating process (decimal growth rates)
SIM_SIGMA_EPS = 0.002  # regional quarterly innovation
SIM_SIGMA_UK = 0.0005  # national aggregation noise
SIM_SIGMA_ANNUAL = 0.005  # annual regional measurement noise


# =============================================================================
# Functions
# =============================================================================
def generate_realistic_simulated_data(
    T: int = 130,
    R: int = 12,
    J: int = 4,
    n_factors: int = 2,
    n_macro: int = 2,
    lag_qtrs: int = 6,
    seed: int = 34,
    w_true: npt.NDArray[np.float64] | None = None,
    return_truth: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Generate simulated mixed-frequency regional data in long format.

    Produces a ``pd.DataFrame`` resembling real-world data suitable for
    initialising an :class:`~ambric.Ambric` model, including UK quarterly
    growth, macro series, regional covariates, and lagged annual regional
    growth.

    Args:
        T (int): Number of quarterly time periods.
        R (int): Number of regions.
        J (int): Number of regional covariate panels.
        n_factors (int): Number of latent factors in the DGP.
        n_macro (int): Number of macro indicator series.
        lag_qtrs (int): Publication lag in quarters for annual regional data.
        seed (int): Random seed for reproducibility.
        w_true (npt.NDArray[np.float64] | None): Regional weights, shape (R,),
            used to form national growth. Defaults to equal weights.
        return_truth (bool): If True, also return the quantities the model
            is trying to recover.

    Returns:
        pd.DataFrame | tuple[pd.DataFrame, dict]: Long-format frame with
            columns ``datetime``, ``measure``, ``region``, ``value``. With
            ``return_truth=True``, a tuple of that frame and a dict with keys
            ``datetime`` (T,), ``region_names``, ``y_reg_true`` (T, R) true
            quarterly regional growth, ``y_annual_no_lag`` (T, R) annual
            regional growth including the unpublished recent values,
            ``w_true`` (R,), and the noise scales ``sigma_eps``,
            ``sigma_uk``, ``sigma_annual``.
    """
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
            seed=seed,
            w_true=w_true,
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
    if return_truth:
        truth = {
            "datetime": quarters,
            "region_names": ["region_" + str(r).zfill(2) for r in range(R)],
            "y_reg_true": y_reg_true,
            "y_annual_no_lag": y_annual_no_lag,
            "w_true": np.ones(R) / R if w_true is None else np.asarray(w_true),
            "sigma_eps": SIM_SIGMA_EPS,
            "sigma_uk": SIM_SIGMA_UK,
            "sigma_annual": SIM_SIGMA_ANNUAL,
        }
        return df_as_if_real, truth
    return df_as_if_real


def gen_unique_id() -> str:
    """Creates a unique, time-based ID for a model run.

    Returns:
        str: ID string in format UUID
    """
    run_str = str(uuid.uuid4()).split("-")[0]
    return run_str


def build_datetime_spine(
    df: pd.DataFrame,
    aggregate_measure: str = "gva_q_on_q",
    aggregation_region: str = "uk",
) -> pd.DatetimeIndex:
    """Build the sorted quarterly datetime spine that every input is aligned to.

    The spine is the set of datetimes at which the aggregate measure is
    present for the aggregation region. It must be a gap-free run of
    quarters.

    Args:
        df (pd.DataFrame): Long format dataframe with all data in.
        aggregate_measure (str, optional): UK-wide measure in q-on-q growth rate. Defaults to "gva_q_on_q".
        aggregation_region (str, optional): Highest level geography. Defaults to "uk".

    Raises:
        ValueError: If there are no rows for the aggregate measure, or if
            the quarters are not contiguous.

    Returns:
        pd.DatetimeIndex: Sorted, unique, contiguous quarterly datetimes.
    """
    datetimes = pd.to_datetime(
        df.loc[
            (df["measure"] == aggregate_measure) & (df["region"] == aggregation_region),
            "datetime",
        ]
    )
    spine = pd.DatetimeIndex(sorted(datetimes.unique()), name="datetime")
    if len(spine) == 0:
        raise ValueError(
            f"No rows found for measure '{aggregate_measure}' in region "
            f"'{aggregation_region}'."
        )
    quarter_numbers = spine.year * 4 + spine.quarter
    gaps = np.flatnonzero(np.diff(quarter_numbers) != 1)
    if len(gaps) > 0:
        first_gap = gaps[0]
        raise ValueError(
            f"Measure '{aggregate_measure}' has missing quarters: gap between "
            f"{spine[first_gap].strftime('%Y-%m-%d')} and "
            f"{spine[first_gap + 1].strftime('%Y-%m-%d')}. A gap-free "
            f"quarterly series is required."
        )
    return spine


def first_q1_offset(spine: pd.DatetimeIndex) -> int:
    """Row index of the first calendar Q1 in a contiguous quarterly spine.

    Args:
        spine (pd.DatetimeIndex): Contiguous quarterly datetimes.

    Returns:
        int: Offset in ``0..3`` such that row ``offset + 4 * a + q`` is
            quarter ``q`` (0 = Q1) of the ``a``-th complete calendar year.
    """
    return int((4 - (spine[0].quarter - 1)) % 4)


def _check_no_duplicates(df: pd.DataFrame) -> None:
    """Raise if any (datetime, measure, region) combination is repeated."""
    keys = ["datetime", "measure", "region"]
    dupes = df.loc[df.duplicated(subset=keys, keep=False), keys]
    if not dupes.empty:
        example = dupes.iloc[0]
        raise ValueError(
            f"Duplicate rows found for {len(dupes.drop_duplicates())} "
            f"(datetime, measure, region) combinations, e.g. "
            f"({pd.Timestamp(example['datetime']).strftime('%Y-%m-%d')}, "
            f"{example['measure']}, {example['region']}). Each combination "
            f"must appear at most once."
        )


def prep_data_for_model_run(
    df: pd.DataFrame,
    macro_names: list[str],
    region_names: list[str],
    region_covariate_names: list[str],
    aggregate_measure: str = "gva_q_on_q",
    aggregation_region: str = "uk",
    region_measure: str = "gva_q_on_4q",
    region_q_on_q_measure: str | None = None,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    list[npt.NDArray[np.float64]],
    npt.NDArray[np.float64],
    int,
    npt.NDArray[np.float64] | None,
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
        region_q_on_q_measure (str | None, optional): Optional measure name in ``df`` supplying
            published quarterly (q-on-q) growth rates for any subset of regions and quarters.
            Rows live in the same long dataframe as every other input, with the standard
            columns ``datetime | measure | region | value``:

                * ``datetime``: quarter-end timestamp.
                * ``measure``: equal to the string passed here.
                * ``region``: one of the names in ``region_names``.
                * ``value``: decimal q-on-q growth rate (e.g. ``0.005`` for 0.5% —
                  **not** a percentage), matching the convention used for
                  ``aggregate_measure`` / ``region_measure``.

            Partial coverage is fully supported: rows may be supplied for only a
            subset of regions and a subset of quarters. After pivoting, absent
            region/quarter pairs become NaN and are automatically masked out of
            the downstream likelihood, so uncovered regions and quarters
            contribute nothing. Defaults to ``None`` (no q-on-q observations —
            backwards compatible).

    Every returned array is aligned to one sorted quarterly spine: the
    quarters at which ``aggregate_measure`` is present for
    ``aggregation_region``. The row order of ``df`` does not matter, macro
    columns follow ``macro_names``, and regional columns follow
    ``region_names``. Regional measures need rows only where a value exists;
    other quarters become NaN.

    Raises:
        KeyError: If a required measure or region is absent from ``df``.
        ValueError: If a (datetime, measure, region) combination is repeated,
            if ``aggregate_measure`` has missing quarters or values, or if a
            macro series is incomplete over the spine.

    Returns:
        tuple: ``(y_uk_extracted, y_a_r_extracted, Z_panel_extract, macro_extracted, lag_qtrs, y_qoq_r_extracted)``.
        ``y_qoq_r_extracted`` is a ``(T, R)`` array aligned to ``region_names`` with NaN in
        unobserved cells, or ``None`` when ``region_q_on_q_measure`` is ``None``.
    """
    logger.info("Prepping data for model run.")
    # Data checks
    available_measures = set(df["measure"].unique())
    required_measures = (
        macro_names + region_covariate_names + [aggregate_measure] + [region_measure]
    )
    if region_q_on_q_measure is not None:
        required_measures = required_measures + [region_q_on_q_measure]
    missing_measures = [x for x in required_measures if x not in available_measures]
    if missing_measures:
        raise KeyError(f"Measures not found in the data: {missing_measures}")

    available_regions = set(df["region"].unique())
    missing_regions = [
        x for x in region_names + [aggregation_region] if x not in available_regions
    ]
    if missing_regions:
        raise KeyError(f"Regions not found in the data: {missing_regions}")

    # Only the measures and regions the model uses are considered from here on.
    df = df.loc[
        df["measure"].isin(required_measures)
        & df["region"].isin(region_names + [aggregation_region]),
        ["datetime", "measure", "region", "value"],
    ].copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    _check_no_duplicates(df)

    # Every array is aligned to one sorted quarterly spine, so results do
    # not depend on the row order of the incoming dataframe.
    spine = build_datetime_spine(
        df, aggregate_measure=aggregate_measure, aggregation_region=aggregation_region
    )
    n_outside_spine = int((~df["datetime"].isin(spine)).sum())
    if n_outside_spine > 0:
        logger.info(
            f"Dropping {n_outside_spine} rows dated outside the range of "
            f"'{aggregate_measure}' ({spine[0].strftime('%Y-%b')} to "
            f"{spine[-1].strftime('%Y-%b')})."
        )

    def _regional_panel(measure: str) -> npt.NDArray[np.float64]:
        """(T, R) array for a regional measure, aligned to spine and region_names."""
        return (
            df.loc[(df["measure"] == measure) & (df["region"].isin(region_names)), :]
            .pivot(index="datetime", columns="region", values="value")
            .reindex(index=spine, columns=region_names)
            .to_numpy(dtype=np.float64)
        )

    df_aggregate = df.loc[df["region"] == aggregation_region, :]
    y_uk_extracted: npt.NDArray[np.float64] = (
        df_aggregate.loc[df_aggregate["measure"] == aggregate_measure, :]
        .set_index("datetime")["value"]
        .reindex(spine)
        .to_numpy(dtype=np.float64)
    )
    if np.any(np.isnan(y_uk_extracted)):
        raise ValueError(
            f"Measure '{aggregate_measure}' contains "
            f"{int(np.isnan(y_uk_extracted).sum())} missing values; a complete "
            f"quarterly series is required."
        )
    # Columns follow macro_names (pivot alone would sort them alphabetically).
    macro_frame = (
        df_aggregate.loc[df_aggregate["measure"].isin(macro_names), :]
        .pivot(index="datetime", columns="measure", values="value")
        .reindex(index=spine, columns=macro_names)
    )
    macro_extracted: npt.NDArray[np.float64] = macro_frame.to_numpy(dtype=np.float64)
    macro_with_nans = [name for name in macro_names if macro_frame[name].isna().any()]
    if macro_with_nans:
        raise ValueError(
            f"Macro series with missing values over "
            f"{spine[0].strftime('%Y-%b')} to {spine[-1].strftime('%Y-%b')}: "
            f"{macro_with_nans}. Macro series must be complete over the "
            f"range of '{aggregate_measure}'."
        )

    # Quarters with no row (e.g. non-year-end quarters, or the unpublished
    # recent period) become NaN and are estimated by the model.
    y_a_r_extracted = _regional_panel(region_measure)

    # Compute lag_qtrs from per-region trailing NaN counts.
    # One region may have data up to the latest period while most have a
    # publication lag — use the modal trailing-NaN count so lag_qtrs reflects
    # the typical lag rather than being driven by a single outlier region.
    n_rows = y_a_r_extracted.shape[0]
    trailing_nans_per_region = np.array(
        [
            n_rows
            if np.all(np.isnan(y_a_r_extracted[:, r]))
            else int(np.argmax(~np.isnan(y_a_r_extracted[::-1, r])))
            for r in range(y_a_r_extracted.shape[1])
        ]
    )
    values, counts = np.unique(trailing_nans_per_region, return_counts=True)
    lag_qtrs = int(values[np.argmax(counts)])
    logger.info(
        f"Per-region trailing NaN counts: {trailing_nans_per_region}; "
        f"using modal lag_qtrs={lag_qtrs}"
    )

    y_qoq_r_extracted: npt.NDArray[np.float64] | None
    if region_q_on_q_measure is not None:
        y_qoq_r_extracted = _regional_panel(region_q_on_q_measure)
        n_obs = int(np.sum(~np.isnan(y_qoq_r_extracted)))
        logger.info(
            f"Regional q-on-q measure '{region_q_on_q_measure}': "
            f"{n_obs} observed cells across shape {y_qoq_r_extracted.shape}"
        )
    else:
        y_qoq_r_extracted = None

    Z_panel_extract: list[npt.NDArray[np.float64]] = [
        _regional_panel(curr_measure) for curr_measure in region_covariate_names
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
        lag_qtrs,
        y_qoq_r_extracted,
    )


def simulate_data(
    T: int = 80,
    R: int = 6,
    J: int = 3,
    n_factors: int = 2,
    n_macro: int = 2,
    seed: int = 42,
    w_true: npt.NDArray[np.float64] | None = None,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    list[npt.NDArray[np.float64]],
    npt.NDArray[np.float64],
]:
    """Simulate mixed-frequency data with regional panels and macro indicators.

    Args:
        T (int): Number of quarterly time periods.
        R (int): Number of regions.
        J (int): Number of regional covariate panels.
        n_factors (int): Number of latent factors.
        n_macro (int): Number of macro indicator series.
        seed (int): Random seed for reproducibility.
        w_true (npt.NDArray[np.float64] | None): Regional weights, shape (R,),
            used to form national growth. Defaults to equal weights.

    Raises:
        ValueError: If ``w_true`` does not have shape (R,).

    Returns:
        tuple: ``(y_uk, y_annual, y_reg_true, Z_panel, macro)`` — national
            quarterly growth (T,), annual regional growth (T, R) with NaNs,
            true quarterly regional growth (T, R), regional covariate
            panels (list of J arrays each (T, R)), and macro series (T, M).
    """
    np.random.seed(seed)

    # True factors (AR(1) processes).
    f_true = np.zeros((T, n_factors))
    for t in range(1, T):
        f_true[t] = 0.4 * f_true[t - 1] + 0.022 * np.random.randn(n_factors)

    # True macro indicators (z-scored, so std = 1)
    macro = np.zeros((T, n_macro))
    for m in range(n_macro):
        macro[:, m] = 0.4 * f_true[:, 0] + 0.6 * np.cumsum(0.02 * np.random.randn(T))
        macro[:, m] = (macro[:, m] - macro[:, m].mean()) / macro[:, m].std()

    # Loadings: NON-ZERO MEAN so factors and macro drive *common* movement
    # across regions (i.e. a shared business cycle). Zero-mean loadings
    # produce regions with random + and – signs that cancel and leave no
    # common cycle for the model to identify.
    Lambda_true = 0.09 + np.random.randn(R, n_factors) * 0.04
    Gamma_true = 0.002 + np.random.randn(R, n_macro) * 0.002

    # Regional growth: AR(1) around an exogenous mean (mirrors the AMBRIC model DGP).
    # phi_r=0.7 gives strongly persistent quarterly regional growth, matching
    # the smooth dynamics seen in real regional GVA / GHDI series.
    phi_r = 0.7 * np.ones(R)
    sigma_eps = SIM_SIGMA_EPS
    mu_exog = f_true @ Lambda_true.T + macro @ Gamma_true.T
    y_reg_true = np.zeros((T, R))
    y_reg_true[0] = mu_exog[0] + sigma_eps * np.random.randn(R)
    for t in range(1, T):
        y_reg_true[t] = (
            phi_r * y_reg_true[t - 1]
            + (1 - phi_r) * mu_exog[t]
            + sigma_eps * np.random.randn(R)
        )

    # UK aggregate
    if w_true is None:
        w_true = np.ones(R) / R
    w_true = np.asarray(w_true, dtype=np.float64)
    if w_true.shape != (R,):
        raise ValueError(f"w_true must have shape ({R},), got {w_true.shape}.")
    y_uk = y_reg_true @ w_true + SIM_SIGMA_UK * np.random.randn(T)

    # Regional indicator panels: strong factor loading + low noise so the
    # extracted factors are a clean view of the latent f_true.
    Z_panel = [
        f_true @ (np.random.randn(R, n_factors) * 1.0).T + 0.15 * np.random.randn(T, R)
        for _ in range(J)
    ]

    # Annual regional growth
    y_annual = np.full((T, R), np.nan)
    for t in range(6, T):
        if (t + 1) % 4 == 0:
            for r in range(R):
                y_annual[t, r] = (
                    sum(OMEGA[j] * y_reg_true[t - j, r] for j in range(7))
                    + SIM_SIGMA_ANNUAL * np.random.randn()
                )

    return y_uk, y_annual, y_reg_true, Z_panel, macro


def simulate_real_time_data(
    T: int = 80,
    R: int = 6,
    J: int = 3,
    n_factors: int = 2,
    n_macro: int = 2,
    annnual_regional_lag_qrtrs: int = 6,
    seed: int = 42,
    w_true: npt.NDArray[np.float64] | None = None,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    list[npt.NDArray[np.float64]],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Simulate mixed-frequency data with a realistic publication lag.

    Wraps :func:`simulate_data` and removes the most recent
    ``annnual_regional_lag_qrtrs`` quarters of annual regional data to
    mimic real-time data availability.

    Args:
        T (int): Number of quarterly time periods.
        R (int): Number of regions.
        J (int): Number of regional covariate panels.
        n_factors (int): Number of latent factors.
        n_macro (int): Number of macro indicator series.
        annnual_regional_lag_qrtrs (int): Quarters of annual data to mask.
        seed (int): Random seed for reproducibility.
        w_true (npt.NDArray[np.float64] | None): Regional weights, shape (R,),
            used to form national growth. Defaults to equal weights.

    Returns:
        tuple: ``(y_uk, y_annual, y_reg_true, Z_panel, macro, y_annual_no_lags)``
            — same as :func:`simulate_data` with an additional copy of the
            annual data before the lag was applied.
    """
    y_uk, y_annual, y_reg_true, Z_panel, macro = simulate_data(
        T=T, R=R, J=J, n_factors=n_factors, n_macro=n_macro, seed=seed, w_true=w_true
    )
    # Now knock out any annual regional data that would not have been available in real-time, starting from the end
    y_annual_no_lags = y_annual.copy()
    for t in range(T - 1, T - annnual_regional_lag_qrtrs, -1):
        y_annual[t, :] = np.nan
    return y_uk, y_annual, y_reg_true, Z_panel, macro, y_annual_no_lags
