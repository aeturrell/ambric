# -----------------------------------------------------------------------------
# Plot Settings
# -----------------------------------------------------------------------------
import re
from importlib.resources import files
from pathlib import Path

import arviz as az
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from matplotlib.ticker import AutoLocator, AutoMinorLocator

from ambric.utilities import OMEGA

style_file = str(files("ambric").joinpath("plot_style.txt"))
plt.style.use(style_file)  # passes the path-like object

colour_wheel = plt.rcParams["axes.prop_cycle"].by_key()["color"]
true_settings = {
    "label": "True",
    "color": colour_wheel[1],
    "s": 40,
    "alpha": 0.9,
    "zorder": 0,
    "edgecolor": "k",
}

estimate_settings_scatter = {
    "label": "Estimated",
    "color": colour_wheel[0],
    "s": 50,
    "alpha": 0.7,
    "zorder": 1,
    "marker": "X",
}

estimate_settings = {
    "label": "Estimated",
    "color": colour_wheel[0],
    "lw": 3,
    "alpha": 0.9,
    "zorder": 2,
}

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------


def rmse_regions_quarterly(
    y_reg_true: npt.NDArray[np.float64], y_reg_est: npt.NDArray[np.float64]
) -> list[float]:
    """Compute the RMSE for each region on quarterly data.

    Args:
        y_reg_true (npt.NDArray[np.float64]): True regional quarterly growth rates
        y_reg_est (npt.NDArray[np.float64]): Estimated regional quarterly growth rates

    Returns:
        list[float]: List of RMSE values for each region
    """
    R = y_reg_true.shape[1]
    rmses: list[float] = [
        np.sqrt(np.mean((y_reg_true[:, r] - y_reg_est[:, r]) ** 2)) for r in range(R)
    ]
    logger.info("RMSE (true vs estimated) per region (quarterly):")
    for r in range(R):
        logger.info(f"  Region {r + 1}: {rmses[r]:.4f}")
    logger.info(f"  Average: {np.mean(rmses):.4f}")
    return rmses


def rmse_regions_annual(
    y_annual_true: npt.NDArray[np.float64], annual_est_growth: npt.NDArray[np.float64]
) -> list[float]:
    """Compute the RMSE for each region on annual data.

    Args:
        y_annual_true (npt.NDArray[np.float64]): True regional annual growth rates
        annual_est_growth (npt.NDArray[np.float64]): Estimated regional annual growth rates
    Returns:
        list[float]: List of RMSE values for each region
    """
    R = y_annual_true.shape[1]
    # create mask of nans from y_annual to use on annual_est_growth
    mask_for_nans = np.isnan(y_annual_true)
    annual_est_growth_masked = annual_est_growth.copy()
    annual_est_growth_masked[mask_for_nans] = np.nan
    rmses = [
        np.sqrt(np.nanmean((annual_est_growth_masked[:, r] - y_annual_true[:, r]) ** 2))
        for r in range(R)
    ]
    logger.info("RMSE (true vs estimated) per region (annual):")
    for r in range(R):
        logger.info(f"  Region {r + 1}: {rmses[r]:.4f}")
    logger.info(f"  Average: {np.mean(rmses):.4f}")
    return rmses


def rmse_national_quarterly(
    y_uk_true: npt.NDArray[np.float64], y_uk_implied: npt.NDArray[np.float64]
) -> float:
    """Compute the RMSE for the UK quarterly growth rates.

    Args:
        y_uk_true (npt.NDArray[np.float64]): True UK quarterly growth rates
        y_uk_implied (npt.NDArray[np.float64]): Implied UK quarterly growth rates

    Returns:
        float: The RMSE of the UK quarterly growth rates (true vs implied)
    """
    rmse = np.sqrt(np.mean((y_uk_true - y_uk_implied) ** 2))
    logger.info(f"UK Quarterly RMSE (true vs implied): {rmse:.4f}")
    return rmse


def trace_to_series(
    trace: az.InferenceData,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Convert a trace to the relevant estimated series coming out of the model.

    Args:
        trace (az.InferenceData): Trace containing posterior.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]: Estimated UK quarterly, regional quarterly, and annual growth rates
    """
    posterior = trace.posterior  # ty: ignore[unresolved-attribute]

    # Quarterly regional estimates
    y_q_r_est_point = posterior["y_reg"].mean(dim=("chain", "draw")).values

    # UK quarterly estimates
    y_q_uk_dist_est = (posterior["y_reg"] * posterior["w"]).sum(dim="w_dim_0")
    y_q_uk_est_point = y_q_uk_dist_est.mean(dim=("chain", "draw")).values

    # Now take the mean over the regional axis to get the national growth rates
    y_q_uk_est_point = y_q_uk_est_point.mean(axis=1)

    # Annual regional estimates

    # 2. Extract latent quarterly growth: (chain, draw, quarter, region)
    y_q_dist_posterior = posterior["y_reg"]

    # 3. Compute the weighted sum across lags
    # We shift the time dimension (y_reg_dim_0) for each weight
    annual_parts = []
    for j in range(7):
        # Shift time by j, fill with 0, and multiply by weight
        shifted = y_q_dist_posterior.shift(y_reg_dim_0=j, fill_value=0)
        annual_parts.append(shifted * OMEGA[j])

    # 4. Sum the parts to get the full annual distribution
    # Shape: (chain, draw, quarter, region)
    y_a_r_dist = sum(annual_parts)

    # 5. Calculate mean for each region (no bands as ADVI)
    y_a_r_est_point = y_a_r_dist.mean(dim=("chain", "draw")).to_numpy()

    return (
        y_q_uk_est_point,
        y_q_r_est_point,
        y_a_r_est_point,
    )


def plot_national_quarterly_vs_implied(
    y_uk: npt.NDArray[np.float64],
    y_uk_implied: npt.NDArray[np.float64],
    datetime_ts: pd.Series,
    path: str | Path | None = None,
) -> None:
    """Plot national quarterly growth rates: observed vs implied.

    Args:
        y_uk (npt.NDArray[np.float64]): True UK quarterly growth rates.
        y_uk_implied (npt.NDArray[np.float64]): Implied UK quarterly growth rates.
        datetime_ts (pd.Series): Time series of quarterly dates.
        path (str | Path | None): Directory to save the plot. When ``None``
            the figure is displayed interactively.
    """
    rmse_national_q = rmse_national_quarterly(y_uk, y_uk_implied)
    fig, ax = plt.subplots(figsize=(15, 6))
    y_lim = np.max(y_uk) * 1.30
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.3)
    ax.plot(
        datetime_ts,
        y_uk,
        label="Observed National Growth",
        alpha=0.7,
        color=colour_wheel[1],
    )
    ax.plot(
        datetime_ts,
        y_uk_implied,
        label=f"Implied National Growth; RMSE {rmse_national_q:.3f}",
        alpha=0.7,
        lw=3,
        color=colour_wheel[0],
    )
    ax.set_title(
        "National Quarterly Growth: Observed vs Implied from Regional Estimates"
    )
    min_point_implied = np.argmin(y_uk_implied)
    max_point_true = np.argmax(y_uk)
    ax.set_ylabel("Growth Rate")
    ax.set_ylim(-y_lim, y_lim)
    ax.xaxis.set_minor_locator(mdates.YearLocator())
    ax.annotate(
        f"Implied National Growth; RMSE {rmse_national_q:.3f}",
        xy=(datetime_ts.iloc[min_point_implied], y_uk_implied[min_point_implied]),
        xytext=(-10, -50),
        textcoords="offset points",
        ha="center",
        fontsize=10,
        arrowprops=dict(
            arrowstyle="->",
            color="0.5",
            shrinkA=5,
            shrinkB=5,
            patchA=None,
            patchB=None,
            connectionstyle="arc3,rad=0.3",
        ),
    )
    ax.annotate(
        "National Growth",
        xy=(datetime_ts.iloc[max_point_true], y_uk[max_point_true]),
        xytext=(30, 30),
        textcoords="offset points",
        ha="center",
        fontsize=10,
        arrowprops=dict(
            arrowstyle="->",
            color="0.5",
            shrinkA=5,
            shrinkB=5,
            patchA=None,
            patchB=None,
            connectionstyle="arc3,rad=0.3",
        ),
    )
    plt.tight_layout()
    if path is not None:
        plt.savefig(Path(path) / "AMBRIC_quarterly_national.svg")
    else:
        plt.show()
    plt.close()


def plot_regional_annual_estimate(
    y_annual_true: npt.NDArray[np.float64],
    y_annual_est: npt.NDArray[np.float64],
    datetime_ts: pd.Series,
    region_names: list[str],
    path: str | Path | None = None,
) -> None:
    """Plot annual regional growth rates: observed vs estimated with recession bands.

    One subplot per region in a grid layout.  Background bands are coloured
    by the recession indicator (growth / recession / undetermined).

    Args:
        y_annual_true (npt.NDArray[np.float64]): True annual growth, shape (T, R).
        y_annual_est (npt.NDArray[np.float64]): Estimated annual growth, shape (T, R).
        datetime_ts (pd.Series): Quarterly datetime index.
        region_names (list[str]): Region names for subplot labels.
        path (str | Path | None): Directory to save the figure. When ``None``
            the figure is displayed interactively.
    """
    rmse_regional_a = rmse_regions_annual(y_annual_true, y_annual_est)

    # Get recession indicator
    recessions = recession_indicator(
        y_annual_est,
        datetime_ts,
        region_names,
    )

    recessions["colours"] = recessions["classification"].map(
        {
            "growth": colour_wheel[2],
            "recession": colour_wheel[3],
            "undetermined": colour_wheel[4],
        }
    )

    recessions["first"] = recessions.groupby(
        [
            (
                recessions["classification"].shift() != recessions["classification"]
            ).cumsum(),
            "region",
        ]
    )["datetime"].transform("first")
    recessions["last"] = recessions.groupby(
        [
            (
                recessions["classification"].shift() != recessions["classification"]
            ).cumsum(),
            "region",
        ]
    )["datetime"].transform("last")

    colour_bands = (
        recessions.groupby(["region", "first", "last"])["colours"].first().reset_index()
    )
    colour_bands["alphas"] = colour_bands["colours"].map(
        {colour_wheel[2]: 0.35, colour_wheel[3]: 0.5, colour_wheel[4]: 0.3}
    )

    R: int = np.shape(y_annual_est)[1]
    n_cols = int(np.ceil(np.sqrt(R)))
    n_rows = int(np.ceil(R / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        ncols=n_cols,
        figsize=(20, 10),
        sharex=True,
        sharey=True,
    )
    axes = axes.flatten()
    y_lim = np.max(y_annual_est) * 1.2
    y_lim = float(f"{float(f'{y_lim * 1.3:.{2}g}'):g}")
    for i, r in enumerate(range(R)):
        axes[i].axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.3)
        colours_here = colour_bands.loc[colour_bands["region"] == region_names[i]]
        for index, row in colours_here.iterrows():
            axes[i].fill_between(
                x=pd.date_range(
                    row["first"] - pd.tseries.offsets.QuarterEnd(1), row["last"]
                ),
                y1=-20,
                y2=20,
                color=row["colours"],
                alpha=row["alphas"],
                zorder=0,
            )
        axes[i].scatter(datetime_ts, y_annual_true[:, r], **true_settings)
        axes[i].plot(datetime_ts, y_annual_est[:, r], **estimate_settings)
        axes[i].set_ylabel(f"{region_names[i]}")
        if i == 0:
            axes[i].legend(loc="best")
        axes[i].set_ylim(-y_lim, y_lim)
        axes[i].xaxis.set_minor_locator(mdates.YearLocator())
    for j in range(R, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle(
        f"Annual q-on-4q Regional Growth: True vs Estimated (mean RMSE: {np.mean(rmse_regional_a):.3f})"
    )
    fig.autofmt_xdate()
    plt.tight_layout()
    if path is not None:
        plt.savefig(Path(path) / "AMBRIC_annual_regional.svg")
    else:
        plt.show()
    plt.close()


def plot_single_region_annual_estimate(
    y_annual_true: npt.NDArray[np.float64],
    y_annual_est: npt.NDArray[np.float64],
    datetime_ts: pd.Series,
    region_idx: int,
    region_names: list[str],
    lag_qtrs: int,
    path: str | Path | None = None,
) -> None:
    """Plot a single region's annual growth: observed vs estimated.

    Draws the observed annual growth as scatter points and the estimated
    series as a line, with a vertical marker indicating the nowcast period.

    Args:
        y_annual_true (npt.NDArray[np.float64]): True annual growth, shape (T, R).
        y_annual_est (npt.NDArray[np.float64]): Estimated annual growth, shape (T, R).
        datetime_ts (pd.Series): Quarterly datetime index.
        region_idx (int): Column index of the region to plot.
        region_names (list[str]): Region names (used for labels and filename).
        lag_qtrs (int): Number of quarters of publication lag.
        path (str | Path | None): Directory to save the figure. When ``None``
            the figure is displayed interactively.
    """
    nowcast_period_start = (
        datetime_ts.iloc[-lag_qtrs] if lag_qtrs > 0 else datetime_ts.iloc[-1]
    )
    fig, ax = plt.subplots(
        figsize=(14, 6),
    )
    y_lim = np.nanmax(y_annual_est) * 1.2
    y_lim = float(f"{float(f'{y_lim * 1.3:.{2}g}'):g}")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.3)
    ax.scatter(
        datetime_ts,
        y_annual_true[:, region_idx],
        **true_settings,  # ty: ignore
    )
    ax.plot(datetime_ts, y_annual_est[:, region_idx], **estimate_settings)  # ty: ignore
    ax.xaxis.set_minor_locator(mdates.YearLocator())
    ax.set_ylabel(region_names[region_idx])
    ax.set_ylim(-y_lim, y_lim)
    ax.axvline(
        nowcast_period_start,
        color="red",
        linestyle="--",
        lw=0.5,
        zorder=0,
    )
    ax.annotate(
        "Nowcast period",
        xy=(nowcast_period_start, 0.04),
        xycoords=("data", "axes fraction"),
    )
    fig.autofmt_xdate()
    min_point_implied = np.argmin(y_annual_est[:, region_idx])
    max_point_true = np.argmax(y_annual_true[:, region_idx])
    ax.annotate(
        "Estimated",
        xy=(
            datetime_ts.iloc[min_point_implied],
            y_annual_est[:, region_idx][min_point_implied],
        ),
        xytext=(-10, -50),
        textcoords="offset points",
        ha="center",
        fontsize=10,
        arrowprops=dict(
            arrowstyle="->",
            color="0.5",
            shrinkA=5,
            shrinkB=5,
            patchA=None,
            patchB=None,
            connectionstyle="arc3,rad=0.3",
        ),
    )
    ax.annotate(
        "Growth",
        xy=(
            datetime_ts.iloc[max_point_true - 2],
            y_annual_true[:, region_idx][max_point_true - 2],
        ),
        xytext=(30, 30),
        textcoords="offset points",
        ha="center",
        fontsize=10,
        arrowprops=dict(
            arrowstyle="->",
            color="0.5",
            shrinkA=5,
            shrinkB=5,
            patchA=None,
            patchB=None,
            connectionstyle="arc3,rad=0.3",
        ),
    )
    plt.suptitle("Annual q-on-4q Regional Growth: True vs Estimated")
    plt.tight_layout()
    if path is not None:
        clean_region = re.sub(r"\s+", "", region_names[region_idx])
        plt.savefig(Path(path) / f"AMBRIC_annual_{clean_region}.svg")
    else:
        plt.show()
    plt.close()


def plot_estimated_regional_quarterly(
    y_reg_est: npt.NDArray[np.float64],
    datetime_ts: pd.Series,
    region_names: list[str],
    lag_qtrs: int,
    path: str | Path | None = None,
) -> None:
    """Plot estimated regional quarterly growth rates in a grid layout.

    One subplot per region showing the q-on-q growth estimate with a
    vertical line marking the start of the nowcast period.

    Args:
        y_reg_est (npt.NDArray[np.float64]): Estimated quarterly growth, shape (T, R).
        datetime_ts (pd.Series): Quarterly datetime index.
        region_names (list[str]): Region names for subplot labels.
        lag_qtrs (int): Number of quarters of publication lag.
        path (str | Path | None): Directory to save the figure. When ``None``
            the figure is displayed interactively.
    """
    nowcast_period_start = (
        datetime_ts.iloc[-lag_qtrs] if lag_qtrs > 0 else datetime_ts.iloc[-1]
    )
    R: int = np.shape(y_reg_est)[1]
    n_cols = int(np.ceil(np.sqrt(R)))
    n_rows = int(np.ceil(R / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        ncols=n_cols,
        figsize=(18, 10),
        sharex=True,
        sharey=True,
    )
    axes = axes.flatten()
    ylim = np.max(y_reg_est) * 1.05
    for i, r in enumerate(range(R)):
        axes[i].axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.3)
        axes[i].plot(datetime_ts, y_reg_est[:, r], **estimate_settings)
        axes[i].set_ylabel(f"{region_names[i]}")
        axes[i].set_ylim(-ylim, ylim)
        axes[i].xaxis.set_major_locator(mdates.YearLocator(20))
        axes[i].axvline(
            nowcast_period_start,
            color="red",
            linestyle="--",
            lw=0.5,
            zorder=0,
        )
    for j in range(R, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle(
        "Regional Q-on-Q Growth Estimates",
        fontsize=14,
    )
    fig.autofmt_xdate()
    plt.tight_layout()
    if path is not None:
        plt.savefig(Path(path) / "AMBRIC_quarterly_regional.svg")
    else:
        plt.show()
    plt.close()


# -----------------------------------------------------------------------------
# Loadings Diagnostics
# -----------------------------------------------------------------------------

_BROAD_TYPE_ORDER: list[str] = ["factors", "macro", "boost_signal"]
_BROAD_TYPE_COLOURS: dict[str, str] = {
    "factors": colour_wheel[5],
    "macro": colour_wheel[3],
    "boost_signal": colour_wheel[7],
}


def assemble_loadings_data(
    trace: az.InferenceData,
    region_names: list[str],
    macro_names: list[str] | None = None,
    factor_stds: npt.NDArray[np.float64] | None = None,
    macro_stds: npt.NDArray[np.float64] | None = None,
    bridge_signal_stds: npt.NDArray[np.float64] | None = None,
) -> pd.DataFrame:
    """Assemble estimated loadings from the posterior into a long-format DataFrame.

    Extracts factor loadings (Lambda), macro loadings (Gamma), and bridge
    signal loadings (delta_r) from the posterior, returning the posterior
    mean and 94 % HDI bounds for every loading-region combination.

    When variable standard deviations are supplied the raw loadings are
    scaled by ``loading × std(variable)`` so that all three signal types
    are expressed on a comparable *contribution* scale (impact per typical
    move in the input).

    Three broad loading types are distinguished:

    - ``"factors"``: Lambda loadings mapping common factors to each region.
    - ``"macro"``: Gamma loadings mapping macro series to each region.
    - ``"boost_signal"``: delta_r loadings scaling the XGBoost bridge signal
      per region.

    Args:
        trace (az.InferenceData): Posterior samples as returned by
            :meth:`~ambric.Ambric.fit`.
        region_names (list[str]): Names of the R regions in the same order
            used when fitting the model.
        macro_names (list[str] | None): Names of the M macro series in model
            order.  When ``None``, generic labels ``"macro_0"``,
            ``"macro_1"``, … are used.
        factor_stds (npt.NDArray | None): Standard deviations of the K
            factor series, shape ``(K,)``.  When provided, factor loadings
            are multiplied by the corresponding std.
        macro_stds (npt.NDArray | None): Standard deviations of the M macro
            series, shape ``(M,)``.  When provided, macro loadings are
            multiplied by the corresponding std.
        bridge_signal_stds (npt.NDArray | None): Standard deviations of the
            R bridge signal series, shape ``(R,)``.  When provided, bridge
            loadings are multiplied by the corresponding std.

    Returns:
        pd.DataFrame: Long-format frame with columns:

            - ``region`` (str) – region name.
            - ``loading_name`` (str) – individual loading label, e.g.
              ``"factor_0"``, ``"macro_gdp"``, ``"boost_signal"``.
            - ``broad_type`` (str) – one of ``"factors"``, ``"macro"``,
              ``"boost_signal"``.
            - ``mean`` (float) – posterior mean of the loading.
            - ``hdi_low`` (float) – lower bound of the 94 % HDI.
            - ``hdi_high`` (float) – upper bound of the 94 % HDI.
            - ``scaled`` (bool) – whether the values have been scaled by
              the variable std.
    """
    rows: list[dict] = []
    scaled = (
        factor_stds is not None
        or macro_stds is not None
        or bridge_signal_stds is not None
    )

    posterior = trace.posterior  # ty: ignore[unresolved-attribute]

    # --- Factor loadings: Lambda has shape (R, K) in the model ---
    lambda_mean: npt.NDArray[np.float64] = (
        posterior["Lambda"].mean(dim=("chain", "draw")).values
    )  # (R, K)
    lambda_hdi: npt.NDArray[np.float64] = az.hdi(
        trace, var_names=["Lambda"], hdi_prob=0.94
    )["Lambda"].values  # (R, K, 2)
    K: int = lambda_mean.shape[1]

    for r, region in enumerate(region_names):
        for k in range(K):
            s = float(factor_stds[k]) if factor_stds is not None else 1.0
            rows.append(
                {
                    "region": region,
                    "loading_name": f"factor_{k}",
                    "broad_type": "factors",
                    "mean": float(lambda_mean[r, k]) * s,
                    "hdi_low": float(lambda_hdi[r, k, 0]) * s,
                    "hdi_high": float(lambda_hdi[r, k, 1]) * s,
                    "scaled": scaled,
                }
            )

    # --- Macro loadings: Gamma has shape (R, M) in the model ---
    gamma_mean: npt.NDArray[np.float64] = (
        posterior["Gamma"].mean(dim=("chain", "draw")).values
    )  # (R, M)
    gamma_hdi: npt.NDArray[np.float64] = az.hdi(
        trace, var_names=["Gamma"], hdi_prob=0.94
    )["Gamma"].values  # (R, M, 2)
    M: int = gamma_mean.shape[1]

    resolved_macro_names: list[str] = (
        macro_names if macro_names is not None else [f"macro_{m}" for m in range(M)]
    )
    for r, region in enumerate(region_names):
        for m, name in enumerate(resolved_macro_names):
            s = float(macro_stds[m]) if macro_stds is not None else 1.0
            rows.append(
                {
                    "region": region,
                    "loading_name": name,
                    "broad_type": "macro",
                    "mean": float(gamma_mean[r, m]) * s,
                    "hdi_low": float(gamma_hdi[r, m, 0]) * s,
                    "hdi_high": float(gamma_hdi[r, m, 1]) * s,
                    "scaled": scaled,
                }
            )

    # --- Bridge signal loadings: delta_r has shape (R,) in the model ---
    delta_mean: npt.NDArray[np.float64] = (
        posterior["delta_r"].mean(dim=("chain", "draw")).values
    )  # (R,)
    delta_hdi: npt.NDArray[np.float64] = az.hdi(
        trace, var_names=["delta_r"], hdi_prob=0.94
    )["delta_r"].values  # (R, 2)

    for r, region in enumerate(region_names):
        s = float(bridge_signal_stds[r]) if bridge_signal_stds is not None else 1.0
        rows.append(
            {
                "region": region,
                "loading_name": "boost_signal",
                "broad_type": "boost_signal",
                "mean": float(delta_mean[r]) * s,
                "hdi_low": float(delta_hdi[r, 0]) * s,
                "hdi_high": float(delta_hdi[r, 1]) * s,
                "scaled": scaled,
            }
        )

    df = pd.DataFrame(rows)
    logger.info(
        f"Assembled loadings data: {len(df)} rows across "
        f"{len(region_names)} regions, "
        f"broad types: {df['broad_type'].unique().tolist()}"
        f"{', scaled by variable stds' if scaled else ''}"
    )
    return df


def plot_loadings_by_region(
    loadings_df: pd.DataFrame,
    region_names: list[str],
    path: Path | None = None,
) -> None:
    """Plot estimated loadings for each region as a horizontal dot chart.

    One panel per region is arranged in a square grid.  Within each panel
    every loading—factor, macro, and boost-signal—is drawn as a dot with
    94 % HDI error bars and coloured by its broad loading type, so the
    three signal categories can be compared at a glance within and across
    regions.

    Args:
        loadings_df (pd.DataFrame): Output of :func:`assemble_loadings_data`.
            Must contain columns ``region``, ``loading_name``,
            ``broad_type``, ``mean``, ``hdi_low``, ``hdi_high``.
        region_names (list[str]): Ordered list of region names that
            determines the panel layout.
        path (Path | None): Directory in which to save the figure as an
            SVG.  When ``None`` the figure is shown interactively.
    """
    R = len(region_names)

    # Canonical order: factors → macro → boost_signal.
    loading_order: list[str] = []
    for bt in _BROAD_TYPE_ORDER:
        names = (
            loadings_df.loc[loadings_df["broad_type"] == bt, "loading_name"]
            .unique()
            .tolist()
        )
        loading_order.extend(names)

    n_loadings = len(loading_order)
    n_cols = int(np.ceil(np.sqrt(R)))
    n_rows = int(np.ceil(R / n_cols))
    panel_height = max(3.5, n_loadings * 0.55 + 1.0)

    fig, axes = plt.subplots(
        n_rows,
        ncols=n_cols,
        figsize=(5 * n_cols, panel_height * n_rows),
        sharex=True,
        sharey=True,
    )
    axes = axes.flatten()

    y_positions = np.arange(n_loadings)

    for i, region in enumerate(region_names):
        ax = axes[i]
        region_df = (
            loadings_df[loadings_df["region"] == region]
            .copy()
            .assign(
                loading_name=lambda d: pd.Categorical(
                    d["loading_name"], categories=loading_order, ordered=True
                )
            )
            .sort_values("loading_name")
            .reset_index(drop=True)
        )

        ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.3)

        for j, (_, row) in enumerate(region_df.iterrows()):
            colour = _BROAD_TYPE_COLOURS.get(row["broad_type"], colour_wheel[0])
            ax.errorbar(
                x=row["mean"],
                y=y_positions[j],
                xerr=[
                    [row["mean"] - row["hdi_low"]],
                    [row["hdi_high"] - row["mean"]],
                ],
                fmt="o",
                color=colour,
                markersize=5,
                capsize=3,
                lw=1.2,
                alpha=0.85,
            )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(region_df["loading_name"].tolist(), fontsize=8)
        ax.set_title(region, fontsize=9, pad=2)

        if i == 0:
            handles = [
                ax.scatter(
                    [],
                    [],
                    color=_BROAD_TYPE_COLOURS[bt],
                    s=30,
                    label=bt.replace("_", " ").title(),
                )
                for bt in _BROAD_TYPE_ORDER
                if bt in loadings_df["broad_type"].values
            ]
            ax.legend(handles=handles, fontsize=7, loc="best")

    # Hide unused grid cells when R is not a perfect square.
    for j in range(R, len(axes)):
        axes[j].set_visible(False)

    is_scaled = "scaled" in loadings_df.columns and loadings_df["scaled"].any()
    if is_scaled:
        plt.suptitle(
            "Scaled Contributions by Region (loading \u00d7 std, 94 % HDI)",
            fontsize=13,
        )
        for ax in axes[:R]:
            ax.set_xlabel("Contribution (loading \u00d7 std)", fontsize=8)
    else:
        plt.suptitle("Estimated Loadings by Region (94 % HDI)", fontsize=13)

    plt.tight_layout()
    if path is not None:
        plt.savefig(Path(path) / "AMBRIC_loadings_by_region.svg")
    else:
        plt.show()
    plt.close()


def plot_loadings_aggregate(
    loadings_df: pd.DataFrame,
    path: Path | None = None,
) -> None:
    """Plot loading distributions across regions, grouped by broad type.

    One panel per broad loading type (factors, macro, boost_signal) appears
    side-by-side.  Within each panel every loading name has one dot per
    region, arranged with a small vertical jitter for legibility.  A
    prominently outlined dot marks the cross-region mean, making it easy to
    compare the relative magnitude of different signal types and to spot
    regions that deviate from the consensus.

    Args:
        loadings_df (pd.DataFrame): Output of :func:`assemble_loadings_data`.
            Must contain columns ``region``, ``loading_name``,
            ``broad_type``, ``mean``, ``hdi_low``, ``hdi_high``.
        path (Path | None): Directory in which to save the figure as an
            SVG.  When ``None`` the figure is shown interactively.
    """
    broad_types_present = [
        bt for bt in _BROAD_TYPE_ORDER if bt in loadings_df["broad_type"].values
    ]
    n_panels = len(broad_types_present)
    n_loadings_max = max(
        loadings_df[loadings_df["broad_type"] == bt]["loading_name"].nunique()
        for bt in broad_types_present
    )
    panel_height = max(4.0, n_loadings_max * 0.7 + 1.0)

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(6 * n_panels, panel_height),
        sharey=False,
        sharex=False,
    )
    # Ensure axes is always a list for uniform indexing.
    axes_list: list = [axes] if n_panels == 1 else list(axes)

    n_regions = loadings_df["region"].nunique()

    for panel_idx, broad_type in enumerate(broad_types_present):
        ax = axes_list[panel_idx]
        subset = loadings_df[loadings_df["broad_type"] == broad_type].copy()

        # Order loading names by descending absolute cross-region mean so the
        # most influential loadings appear at the top.
        loading_names: list[str] = (
            subset.groupby("loading_name")["mean"]
            .mean()
            .abs()
            .sort_values(ascending=False)
            .index.tolist()
        )

        ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.3)

        jitter = np.linspace(-0.3, 0.3, n_regions) if n_regions > 1 else np.array([0.0])
        colour = _BROAD_TYPE_COLOURS[broad_type]

        for y_pos, loading_name in enumerate(loading_names):
            rows = subset[subset["loading_name"] == loading_name].reset_index(drop=True)
            # Individual region dots, semi-transparent.
            for reg_idx, (_, row) in enumerate(rows.iterrows()):
                ax.scatter(
                    row["mean"],
                    y_pos + jitter[reg_idx],
                    color=colour,
                    alpha=0.4,
                    s=22,
                    zorder=2,
                )

            # Cross-region mean as a prominent outlined dot.
            cross_mean = float(rows["mean"].mean())
            ax.scatter(
                cross_mean,
                y_pos,
                color=colour,
                s=80,
                edgecolor="k",
                lw=0.8,
                zorder=4,
                alpha=0.95,
            )

        ax.set_yticks(range(len(loading_names)))
        ax.set_yticklabels(loading_names, fontsize=9)
        ax.set_title(broad_type.replace("_", " ").title(), fontsize=11)
        is_scaled = "scaled" in loadings_df.columns and loadings_df["scaled"].any()
        ax.set_xlabel(
            "Contribution (loading \u00d7 std)" if is_scaled else "Loading value"
        )

    if is_scaled:
        plt.suptitle(
            "Scaled Contributions by Broad Type\n"
            "(small dots = individual regions,  large dot = cross-region mean)",
            fontsize=12,
        )
    else:
        plt.suptitle(
            "Comparative Loadings by Broad Type\n"
            "(small dots = individual regions,  large dot = cross-region mean)",
            fontsize=12,
        )
    plt.tight_layout()
    if path is not None:
        plt.savefig(Path(path) / "AMBRIC_loadings_aggregate.svg")
    else:
        plt.show()
    plt.close()


# Out of sample results diagnostics from here


def rmse(series_in: pd.Series) -> float:
    """Compute root-mean-square error of a Series.

    Args:
        series_in (pd.Series): Error values.

    Returns:
        float: RMSE value.
    """
    return float(np.sqrt(np.mean(np.power(series_in, 2))))


def out_of_sample_rmse(df_oos_reg_a: pd.DataFrame) -> pd.DataFrame:
    """Compute out-of-sample RMSE by region and quarters-to-publication.

    Args:
        df_oos_reg_a (pd.DataFrame): Out-of-sample results with ``error``,
            ``quarters_to_publication``, and ``region`` columns.

    Returns:
        pd.DataFrame: RMSE grouped by quarters-to-publication and region.
    """
    rmse_region_quarters = (
        df_oos_reg_a.dropna(subset="error")
        .groupby(["quarters_to_publication", "region"])["error"]
        .apply(rmse)
        .reset_index()
    )
    return rmse_region_quarters


def plot_out_of_sample_rmse(
    df_results: pd.DataFrame, region_measure: str, path: Path | None = None
) -> None:
    """Plot out-of-sample RMSEs by quarters-to-publication for each region.

    One subplot per region in a grid layout showing how forecast accuracy
    improves as publication approaches.

    Args:
        df_results (pd.DataFrame): Out-of-sample results from
            :func:`~ambric.run_out_of_sample_exercise`.
        region_measure (str): Regional measure name to filter on.
        path (Path | None): Directory to save the figure. When ``None``
            the figure is displayed interactively.
    """
    df_regions = df_results.loc[(df_results["measure"] == region_measure), :].copy()
    df_regions["error"] = df_regions.groupby(["datetime", "region", "measure"])[
        "value"
    ].diff(-1)
    rmses_by_pub_gap = (
        df_regions.dropna(subset="error")
        .set_index("datetime")
        .groupby(["region", "quarters_to_publication"])["error"]
        .apply(lambda x: np.pow(x, 2))
        .unstack()
        .mean(axis=1)
        .apply(np.sqrt)
        .reset_index()
        .rename(columns={0: "error"})
    )

    R: int = len(rmses_by_pub_gap["region"].unique())
    n_cols = int(np.ceil(np.sqrt(R)))
    n_rows = int(np.ceil(R / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        ncols=n_cols,
        figsize=(18, 10),
        sharex=True,
        sharey=True,
    )
    axes = axes.flatten()
    for i, region in enumerate(rmses_by_pub_gap["region"].unique()):
        res_here = rmses_by_pub_gap.loc[rmses_by_pub_gap["region"] == region]
        axes[i].scatter(res_here["quarters_to_publication"], res_here["error"])
        axes[i].set_ylabel(f"{region}")
        axes[i].set_xlim(7, 0)
        axes[i].set_ylim(0, rmses_by_pub_gap["error"].max() * 1.3)
        axes[i].xaxis.set_minor_locator(AutoMinorLocator(2))
        axes[i].yaxis.set_major_locator(AutoLocator())
        axes[i].yaxis.set_minor_locator(AutoMinorLocator(2))
    for j in range(R, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle(
        "Out-of-sample regional growth RMSEs: lower is better but we care most about the first estimate",
    )
    plt.xlabel("Quarters to publication", loc="center")
    plt.tight_layout()
    fig.autofmt_xdate()
    if path is not None:
        plt.savefig(Path(path / "out_of_sample_rmse.svg"))
    else:
        plt.show()
    plt.close()


def plot_out_of_sample_nowcasts(
    df_results: pd.DataFrame,
    region_measure: str,
    path: Path | None = None,
) -> None:
    """Plot out-of-sample nowcasts vs outturns for each region.

    Produces one figure per region showing observed outturns as dots and
    nowcasts at varying horizons with transparency indicating proximity to
    publication.

    Args:
        df_results (pd.DataFrame): Out-of-sample results from
            :func:`~ambric.run_out_of_sample_exercise`.
        region_measure (str): Regional measure name to filter on.
        path (Path | None): Directory to save the figures. When ``None``
            the figures are displayed interactively.
    """
    df_regions = df_results.loc[(df_results["measure"] == region_measure), :].copy()
    df_regions["error"] = df_regions.groupby(["datetime", "region", "measure"])[
        "value"
    ].diff(-1)
    rmses_by_region = (
        df_regions.dropna(subset="error")
        .set_index("datetime")
        .groupby(["region"])["error"]
        .apply(lambda x: np.pow(x, 2))
        .reset_index()
        .groupby(["region"])["error"]
        .mean()
        .apply(np.sqrt)
    )
    y_max = np.max(df_regions["value"]) * 1.1 * 100
    for region in df_regions["region"].unique():
        reg_df = df_regions.loc[(df_regions["region"] == region), :].copy()
        reg_df["value"] = 100 * reg_df["value"]
        reg_df_now = (
            reg_df.loc[reg_df["type"] == "nowcast", :].sort_values(by="datetime").copy()
        )
        reg_df_out = reg_df.loc[reg_df["type"] == "outturn", :].copy()
        reg_df_out = (
            reg_df_out.loc[~reg_df_out["value"].isna(), :]
            .sort_values(by="datetime")
            .copy()
        )
        datetimes_to_use = reg_df_out["datetime"].unique()
        reg_df_now = reg_df_now.loc[
            reg_df_now["datetime"].isin(datetimes_to_use), :
        ].copy()

        fig, ax = plt.subplots()

        # Only plot year end nowcasts
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.3, zorder=0)
        if not reg_df_out.empty:
            ax.scatter(
                reg_df_out["datetime"], reg_df_out["value"], color=colour_wheel[1]
            )
            ax.annotate(
                "Outturn",
                xy=(reg_df_out["datetime"].iloc[0], reg_df_out["value"].iloc[0]),
                xytext=(10, 60),
                textcoords="offset points",
                fontsize=7,
                arrowprops=dict(
                    arrowstyle="->",
                    color="0.5",
                    shrinkA=5,
                    shrinkB=5,
                    patchA=None,
                    patchB=None,
                    connectionstyle="arc3,rad=0.3",
                ),
            )
        # Do a loop for the nowcasts with the lag-quarters out nowcast being more visible
        if not reg_df_now.empty:
            num_qtrs_to_go = int(reg_df_now["quarters_to_publication"].max() + 1)
            full_alpha = 0.9
            for nowcast_idx in range(num_qtrs_to_go):
                this_nowcast = reg_df_now.loc[
                    reg_df_now["quarters_to_publication"] == nowcast_idx
                ].sort_values(by="datetime")
                if not this_nowcast.empty:
                    ax.scatter(
                        this_nowcast["datetime"],
                        this_nowcast["value"],
                        color=colour_wheel[0],
                        marker="x",
                        alpha=full_alpha * ((nowcast_idx + 1) / num_qtrs_to_go),
                    )
            if not this_nowcast.empty:
                ax.annotate(
                    "Nowcasts\n(closer to publication = more transparent)",
                    xy=(
                        this_nowcast["datetime"].iloc[0],
                        this_nowcast["value"].iloc[0],
                    ),
                    xytext=(20, -50),
                    va="top",
                    ha="left",
                    textcoords="offset points",
                    fontsize=7,
                    arrowprops=dict(
                        arrowstyle="->",
                        color="0.5",
                        shrinkA=5,
                        shrinkB=5,
                        patchA=None,
                        patchB=None,
                        connectionstyle="arc3,rad=0.3",
                    ),
                )
        ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
        ax.xaxis.set_minor_locator(mdates.YearLocator(base=1))
        ax.yaxis.set_major_locator(AutoLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.set_title(
            f"Out-of-sample nowcast for {region}, % growth\n(RMSE: {rmses_by_region[region]:.2f})",
            fontsize=11,
            loc="left",
        )
        ax.set_ylim(-y_max, y_max)
        if path is not None:
            clean_region = re.sub(r"\s+", "", region)
            plt.savefig(path / f"out_of_sample_nowcast_{clean_region}.svg")
        else:
            plt.show()
        plt.close()


def plot_current_nowcast(
    y_nowcast: npt.NDArray[np.float64],
    y_annual: npt.NDArray[np.float64],
    datetime_ts: pd.Series,
    region_names: list[str],
    lag_qtrs: int,
    backlook_qtrs: int = 6,
    path: Path | None = None,
) -> None:
    """Plot the latest nowcast vs observed annual growth for each region.

    Shows a truncated window of the most recent quarters with nowcast
    estimates (scatter) overlaid on observed annual data.

    Args:
        y_nowcast (npt.NDArray[np.float64]): Nowcast annual estimates, shape (T, R).
        y_annual (npt.NDArray[np.float64]): Observed annual growth, shape (T, R).
        datetime_ts (pd.Series): Quarterly datetime index.
        region_names (list[str]): Region names for subplot labels.
        lag_qtrs (int): Number of quarters of publication lag.
        backlook_qtrs (int): Number of recent quarters to display.
        path (Path | None): Directory to save the figure. When ``None``
            the figure is displayed interactively.
    """
    nowcast_period_start = (
        datetime_ts.iloc[-lag_qtrs] if lag_qtrs > 0 else datetime_ts.iloc[-1]
    )
    # truncate all the arrays:
    t_y_nowcast = y_nowcast[-backlook_qtrs:, :].copy()
    t_y_annual = y_annual[-backlook_qtrs:, :].copy()
    t_datetime_ts = datetime_ts.iloc[-backlook_qtrs:].copy()
    R: int = np.shape(y_nowcast)[1]
    n_cols = int(np.ceil(np.sqrt(R)))
    n_rows = int(np.ceil(R / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        ncols=n_cols,
        figsize=(18, 10),
        sharex=True,
        sharey=True,
    )
    axes = axes.flatten()
    y_lim = np.nanmax(t_y_annual) * 1.2
    for i, r in enumerate(range(R)):
        if i == 0:
            axes[i].annotate(
                "Nowcast period",
                xy=(nowcast_period_start, 0.05),
                xycoords=("data", "axes fraction"),
                xytext=(1, 0),
                textcoords="offset points",
            )
        axes[i].axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.3)
        axes[i].axvline(
            nowcast_period_start,
            color="red",
            linestyle="--",
            lw=0.5,
            zorder=0,
        )
        axes[i].scatter(t_datetime_ts, t_y_annual[:, r], **true_settings)
        axes[i].scatter(t_datetime_ts, t_y_nowcast[:, r], **estimate_settings_scatter)
        axes[i].set_ylabel(f"{region_names[i]}")
        axes[i].set_ylim(-y_lim, y_lim)
        axes[i].xaxis.set_minor_locator(mdates.YearLocator())
    for j in range(R, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Nowcast: q-on-4q growth vs annual outturns")
    fig.autofmt_xdate()
    plt.tight_layout()
    if path is not None:
        plt.savefig(
            path / f"AMBRIC_nowcast_{t_datetime_ts.iloc[-1].strftime('%Y_%m')}.svg"
        )
    else:
        plt.show()
    plt.close()


def bands_indicator(
    y_nowcast: npt.NDArray[np.float64],
    datetime_ts: pd.Series,
    region_names: list[str],
    bands: list[float] = [-0.8, -0.1, 0.1, 0.8],  # noqa: B006
) -> pd.DataFrame:
    """Classify growth rates into band labels for each region.

    Band thresholds (defaults):

    * ``< -0.8`` -- strong contraction
    * ``>= -0.8`` and ``< -0.1`` -- contraction
    * ``>= -0.1`` and ``< 0.1`` -- indeterminate
    * ``>= 0.1`` and ``< 0.8`` -- growth
    * ``>= 0.8`` -- strong growth

    Args:
        y_nowcast (npt.NDArray[np.float64]): Nowcast values, shape (T, R).
        datetime_ts (pd.Series): Quarterly datetime index.
        region_names (list[str]): Region names used as column headers.
        bands (list[float]): Interior bin edges.
            Defaults to ``[-0.8, -0.1, 0.1, 0.8]``.

    Returns:
        pd.DataFrame: Long-format frame with ``datetime``, ``region``, and
            ``classification`` columns.
    """
    band_names = [
        "strong contraction",
        "contraction",
        "indeterminate",
        "growth",
        "strong growth",
    ]
    bins = [-float("inf")] + bands + [float("inf")]

    df = pd.DataFrame(data=y_nowcast, columns=pd.Index(region_names), index=datetime_ts)
    df = df.reset_index().melt(
        id_vars="datetime", var_name="region", value_name="value"
    )
    df["classification"] = pd.cut(df["value"], bins=bins, labels=band_names)

    return df[["datetime", "region", "classification"]]


def recession_indicator(
    y_nowcast: npt.NDArray[np.float64],
    datetime_ts: pd.Series,
    region_names: list[str],
) -> pd.DataFrame:
    """Classify each time period as growth, recession, or undetermined.

    Growth: part of two or more successive time periods of positive growth.
    Recession: part of two or more successive time periods of negative growth.
    Undetermined: all other cases.

    Args:
        y_nowcast (npt.NDArray[np.float64]): Nowcast values, shape (T, R).
        datetime_ts (pd.Series): Quarterly datetime index.
        region_names (list[str]): Region names used as column headers.

    Returns:
        pd.DataFrame: Long-format frame with ``datetime``, ``region``, and
            ``classification`` columns.
    """
    results = []

    df = pd.DataFrame(data=y_nowcast, columns=pd.Index(region_names), index=datetime_ts)
    df = df.reset_index().melt(
        id_vars="datetime", var_name="region", value_name="value"
    )

    for region, group in df.groupby("region"):
        group_sorted = group.sort_values("datetime").reset_index(drop=True)
        values = group_sorted["value"].values
        datetimes = group_sorted["datetime"].values
        n = len(values)

        for i in range(n):
            current_val = values[i]
            prev_val = values[i - 1] if i > 0 else None
            next_val = values[i + 1] if i < n - 1 else None

            # Check if part of 2+ consecutive positive
            is_growth = (prev_val is not None and current_val > 0 and prev_val > 0) or (
                next_val is not None and current_val > 0 and next_val > 0
            )

            # Check if part of 2+ consecutive negative
            is_recession = (
                prev_val is not None and current_val < 0 and prev_val < 0
            ) or (next_val is not None and current_val < 0 and next_val < 0)

            if is_growth:
                classification = "growth"
            elif is_recession:
                classification = "recession"
            else:
                classification = "undetermined"

            results.append(
                {
                    "datetime": datetimes[i],
                    "region": region,
                    "classification": classification,
                }
            )

    return pd.DataFrame(results)


def live_recession_indicator(
    y_nowcast: npt.NDArray[np.float64],
    datetime_ts: pd.Series,
    region_names: list[str],
) -> pd.DataFrame:
    """Return recession indicator for nowcast growth q-on-4q, pivoted wide.

    Args:
        y_nowcast (npt.NDArray[np.float64]): Nowcast values, shape (T, R).
        datetime_ts (pd.Series): Quarterly datetime index.
        region_names (list[str]): Region names.

    Returns:
        pd.DataFrame: Wide-format frame with ``datetime`` as index and one
            column per region containing the classification.
    """

    df = recession_indicator(
        y_nowcast,
        datetime_ts,
        region_names,
    )
    df = df.pivot(
        index="datetime", columns="region", values="classification"
    ).reset_index()
    return df


def out_of_sample_classification_performance_table(
    df_results: pd.DataFrame, region_measure: str, path: Path | None = None
) -> pd.DataFrame:
    """Compute up/down classification accuracy by region and horizon.

    Compares the sign of each nowcast to its corresponding outturn.
    Returns a pivot table of percentage accuracy grouped by region and
    quarters-to-publication.

    Args:
        df_results (pd.DataFrame): Out-of-sample results from
            :func:`~ambric.run_out_of_sample_exercise`.
        region_measure (str): Regional measure name to filter on.
        path (Path | None): Directory to save a CSV of the table. When
            ``None`` no file is written.

    Returns:
        pd.DataFrame: Pivot table of classification accuracy (%) with
            regions as rows and quarters-to-publication as columns.
    """
    df_region = df_results.loc[df_results["measure"] == region_measure, :].copy()
    # Want to compare every nowcast to its original outturn
    df_outturns_only = df_region.loc[df_region["type"] == "outturn"].drop(
        ["measure", "quarters_to_publication", "nowcast_index"], axis=1
    )
    df_outturns_only = df_outturns_only.loc[~df_outturns_only["value"].isna(), :]

    df_merge = pd.merge(
        df_region.loc[df_region["type"] != "outturn"],
        df_outturns_only,
        on=["datetime", "region"],
        how="inner",
        suffixes=("_nowcast", "_outturn"),
    )
    df_merge = df_merge.drop_duplicates(
        subset=["region", "datetime", "quarters_to_publication"]
    ).copy()

    df_merge["agree"] = df_merge["value_nowcast"].apply(np.sign) == df_merge[
        "value_outturn"
    ].apply(np.sign)

    df_merge["quarters_to_publication"] = df_merge["quarters_to_publication"].astype(
        int
    )

    summary_df = df_merge.groupby(["region", "quarters_to_publication"])["agree"].agg(
        ["sum", "count"]
    )
    summary_df["pct_accuracy"] = (100 * summary_df["sum"] / summary_df["count"]).round(
        1
    )

    summary_df = summary_df["pct_accuracy"].unstack()
    if path:
        summary_df.to_csv(path / "oos_classification_performance.csv")
    return summary_df


def plot_seasonally_adjusted_q_on_q_growth(
    df_sa_trend_orig: pd.DataFrame, path: Path | None
) -> None:
    """_summary_

    Args:
        df_sa_trend_orig (pd.DataFrame): _description_
    """
    for region in df_sa_trend_orig["region"].unique():
        cut_q_on_q_region = df_sa_trend_orig.loc[
            df_sa_trend_orig["region"] == region, :
        ].copy()
        cut_q_on_q_region = cut_q_on_q_region.pivot(
            columns="type", values="q_on_q"
        ).copy()
        cut_q_on_q_region.index = cut_q_on_q_region.index.to_timestamp()
        fig, ax = plt.subplots()
        ax.plot(
            cut_q_on_q_region["estimate"], color="k", zorder=1, lw=2.3, label="estimate"
        )
        ax.plot(
            cut_q_on_q_region["seasonally_adjusted"],
            alpha=0.8,
            lw=1.3,
            ls="dashed",
            label="seasonally adjusted",
        )
        ax.plot(cut_q_on_q_region["trend"], lw=1.8, alpha=0.8, label="trend")
        ax.axhline(0, color="k", alpha=0.4, zorder=0, lw=0.4)
        ax.set_title(f"Growth for {region}, % quarter-on-quarter")
        ax.legend(frameon=False)
        if path:
            clean_region = re.sub(r"\s+", "", region)
            plt.savefig(path / f"{clean_region}_sa_q_on_q.svg")
        plt.close()
