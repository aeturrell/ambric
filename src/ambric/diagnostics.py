# -----------------------------------------------------------------------------
# Plot Settings
# -----------------------------------------------------------------------------
from importlib.resources import files
from pathlib import Path
from textwrap import fill

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
        logger.info(f"  Region {r+1}: {rmses[r]:.4f}")
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
        logger.info(f"  Region {r+1}: {rmses[r]:.4f}")
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
    # Quarterly regional estimates
    y_q_r_est_point = trace.posterior["y_reg"].mean(dim=("chain", "draw")).values

    # UK quarterly estimates
    y_q_uk_dist_est = (trace.posterior["y_reg"] * trace.posterior["w"]).sum(
        dim="w_dim_0"
    )
    y_q_uk_est_point = y_q_uk_dist_est.mean(dim=("chain", "draw")).values

    # Now take the mean over the regional axis to get the national growth rates
    y_q_uk_est_point = y_q_uk_est_point.mean(axis=1)

    # Annual regional estimates

    # 2. Extract latent quarterly growth: (chain, draw, quarter, region)
    y_q_dist_posterior = trace.posterior["y_reg"]  # ty: ignore

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
    path: Path | None = None,
):
    """Plot national quarterly growth rates: observed vs implied.

    Args:
        y_uk (npt.NDArray[np.float64]): True UK quarterly growth rates
        y_uk_implied (npt.NDArray[np.float64]): Implied UK quarterly growth rates
        datetime_ts (pd.Series): Time series of quarterly dates
        path (Path | str): Path to save the plot
    """
    rmse_national_q = rmse_national_quarterly(y_uk, y_uk_implied)
    fig, ax = plt.subplots(figsize=(15, 6))
    y_lim = np.max(y_uk) * 1.15
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
        plt.savefig(Path(path / "AMBRIC_quarterly_national.svg"))
    else:
        plt.show()
    plt.close()


def plot_regional_annual_estimate(
    y_annual_true: npt.NDArray[np.float64],
    y_annual_est: npt.NDArray[np.float64],
    datetime_ts: pd.Series,
    region_names: list[str],
    path: Path | None = None,
):
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
    fig, axes = plt.subplots(
        int(np.floor(np.sqrt(R))),
        ncols=int(np.ceil(np.sqrt(R))),
        figsize=(20, 10),
        sharex=True,
        sharey=True,
    )
    axes = axes.flatten()
    y_lim = np.max(y_annual_est) * 1.2
    y_lim = float(f"{float(f'{y_lim*1.3:.{2}g}'):g}")
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
        if r == 4:
            axes[i].legend(loc="best")
        axes[i].set_ylim(-y_lim, y_lim)
        axes[i].xaxis.set_minor_locator(mdates.YearLocator())
    plt.suptitle(
        f"Annual q-on-4q Regional Growth: True vs Estimated (mean RMSE: {rmse_regional_a[i]:.3f})"
    )
    fig.autofmt_xdate()
    plt.tight_layout()
    if path is not None:
        plt.savefig(Path(path / "AMBRIC_annual_regional.svg"))
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
    path: Path | None = None,
):
    nowcast_period_start = datetime_ts.iloc[-lag_qtrs]
    fig, ax = plt.subplots(
        figsize=(14, 6),
    )
    y_lim = np.nanmax(y_annual_est) * 1.2
    y_lim = float(f"{float(f'{y_lim*1.3:.{2}g}'):g}")
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
        nowcast_period_start,  # ty: ignore
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
        plt.savefig(Path(path / f"AMBRIC_annual_{region_names[region_idx]}.svg"))
    else:
        plt.show()
    plt.close()


def plot_estimated_regional_quarterly(
    y_reg_est: npt.NDArray[np.float64],
    datetime_ts: pd.Series,
    region_names: list[str],
    lag_qtrs: int,
    path: Path | None = None,
):
    nowcast_period_start = datetime_ts.iloc[-lag_qtrs]
    R: int = np.shape(y_reg_est)[1]
    fig, axes = plt.subplots(
        int(np.floor(np.sqrt(R))),
        ncols=int(np.ceil(np.sqrt(R))),
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
            nowcast_period_start,  # ty: ignore
            color="red",
            linestyle="--",
            lw=0.5,
            zorder=0,
        )
    plt.suptitle(
        "Regional Q-on-Q Growth Estimates",
        fontsize=14,
    )
    fig.autofmt_xdate()
    plt.tight_layout()
    if path is not None:
        plt.savefig(Path(path / "AMBRIC_quarterly_regional.svg"))
    else:
        plt.show()


# Out of sample results diagnostics from here


def rmse(series_in: pd.Series):
    return np.sqrt(np.mean(np.power(series_in, 2)))


def out_of_sample_rmse(df_oos_reg_a: pd.DataFrame) -> pd.DataFrame:
    rmse_region_quarters = (
        df_oos_reg_a.dropna(subset="error")
        .groupby(["quarters_to_publication", "region"])["error"]
        .apply(rmse)
        .reset_index()
    )
    return rmse_region_quarters


def plot_out_of_sample_rmse(df_a_r_oos: pd.DataFrame, path: Path | None = None):
    df_prepped = prep_out_of_sample_data(df_a_r_oos)
    rmse_region_quarters = out_of_sample_rmse(df_prepped)
    R: int = len(rmse_region_quarters["region"].unique())
    fig, axes = plt.subplots(
        int(np.floor(np.sqrt(R))),
        ncols=int(np.ceil(np.sqrt(R))),
        figsize=(18, 10),
        sharex=True,
        sharey=True,
    )
    axes = axes.flatten()
    for i, region in enumerate(rmse_region_quarters["region"].unique()):
        res_here = rmse_region_quarters.loc[rmse_region_quarters["region"] == region]
        axes[i].scatter(res_here["quarters_to_publication"], res_here["error"])
        axes[i].set_ylabel(f"{region}")
        axes[i].set_xlim(7, 0)
        axes[i].set_ylim(0, rmse_region_quarters["error"].max() * 1.3)
        axes[i].xaxis.set_minor_locator(AutoMinorLocator(2))
        axes[i].yaxis.set_major_locator(AutoLocator())
        axes[i].yaxis.set_minor_locator(AutoMinorLocator(2))
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


def prep_out_of_sample_data(df: pd.DataFrame) -> pd.DataFrame:
    df_oos_reg_a_here = df.copy()
    df_oos_reg_a_here = pd.melt(
        df_oos_reg_a_here.reset_index(),
        id_vars=["datetime", "nowcast_index", "type", "quarters_to_publication"],
        var_name="region",
        value_name="value",
    )
    df_oos_reg_a_here["error"] = df_oos_reg_a_here.groupby(
        ["quarters_to_publication", "region", "datetime", "nowcast_index"]
    )["value"].diff()
    return df_oos_reg_a_here


def plot_out_of_sample_nowcasts(
    df_ests: pd.DataFrame,
    df_original: pd.DataFrame,
    region_measure: str,
    path: Path | None = None,
):
    df_oos_reg_a = prep_out_of_sample_data(df_ests)
    datetimes_to_use = pd.Series(
        df_original.loc[df_original["measure"] == region_measure, :]
        .dropna(subset="value")["datetime"]
        .unique()
    )
    for region in df_oos_reg_a["region"].unique():
        examine_df = (
            df_oos_reg_a.loc[df_oos_reg_a["region"] == region, :]
            .dropna(subset="value")
            .copy()
        )
        examine_df = examine_df.loc[
            examine_df["datetime"].isin(datetimes_to_use)
        ].copy()
        fig, ax = plt.subplots()
        outturn = df_original.loc[
            (
                (df_original["measure"] == region_measure)
                & (df_original["region"] == region)
            ),
            :,
        ].dropna(subset="value")
        examine_df["value"] = examine_df["value"] * 100.0
        y_max = np.max(outturn["value"]) * 1.05
        nowcast = examine_df.loc[examine_df["type"] == "nowcast", :]
        # Only plot year end nowcasts
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.3, zorder=0)
        ax.scatter(outturn["datetime"], outturn["value"] * 100, color=colour_wheel[1])
        ax.annotate(
            "Outturn",
            xy=(outturn["datetime"].iloc[1], outturn["value"].iloc[1] * 100),
            xytext=(10, 70),
            textcoords="offset points",
            fontsize=8,
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
        num_qtrs_to_go = int(nowcast["quarters_to_publication"].max() + 1)
        full_alpha = 0.9
        for nowcast_idx in range(num_qtrs_to_go):
            this_nowcast = nowcast.loc[
                nowcast["quarters_to_publication"] == nowcast_idx
            ]
            ax.scatter(
                this_nowcast["datetime"],
                this_nowcast["value"],
                color=colour_wheel[0],
                marker="x",
                alpha=full_alpha * ((nowcast_idx + 1) / num_qtrs_to_go),
            )
        ax.annotate(
            fill("Nowcasts\n(closer to publication = more transparent)", 30),
            xy=(this_nowcast["datetime"].iloc[0], this_nowcast["value"].iloc[0]),
            xytext=(20, -50),
            va="top",
            ha="right",
            textcoords="offset points",
            fontsize=8,
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
            f"Annual Growth Nowcast for {region}, %",
            fontsize=14,
            loc="left",
        )
        ax.set_ylim(-y_max, y_max)
        if path is not None:
            plt.savefig(
                path / f"out_of_sample_nowcast_{region.lower().replace(' ', '_')}.svg"
            )
        else:
            plt.show()
        plt.close()


def plot_current_nowcast(
    y_nowcast,
    y_annual,
    datetime_ts,
    region_names,
    lag_qtrs,
    backlook_qtrs: int = 6,
    path: Path | None = None,
):
    nowcast_period_start = datetime_ts.iloc[-lag_qtrs]
    # truncate all the arrays:
    t_y_nowcast = y_nowcast[-backlook_qtrs:, :].copy()
    t_y_annual = y_annual[-backlook_qtrs:, :].copy()
    t_datetime_ts = datetime_ts.iloc[-backlook_qtrs:].copy()
    R: int = np.shape(y_nowcast)[1]
    fig, axes = plt.subplots(
        int(np.floor(np.sqrt(R))),
        ncols=int(np.ceil(np.sqrt(R))),
        figsize=(18, 10),
        sharex=True,
        sharey=True,
    )
    axes = axes.flatten()
    y_lim = np.nanmax(y_annual) * 1.3
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
    plt.suptitle("Nowcast: Regional Annual Growth")
    fig.autofmt_xdate()
    plt.tight_layout()
    if path is not None:
        plt.savefig(
            path / f"AMBRIC_nowcast_{t_datetime_ts.iloc[-1].strftime('%Y_%m')}.svg"
        )
    else:
        plt.show()
    plt.close()


def recession_indicator(
    y_nowcast,
    datetime_ts,
    region_names,
):
    """Classify each time period as growth, recession, or undetermined.

    Growth: part of two or more successive time periods of positive growth.
    Recession: part of two or more successive time periods of negative growth.
    Undetermined: all other cases.

    Args:
        df: DataFrame with datetime, region, and value columns.

    Returns:
        DataFrame with region, datetime, and classification columns.
    """
    results = []

    df = pd.DataFrame(data=y_nowcast, columns=region_names, index=datetime_ts)
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
    y_nowcast,
    datetime_ts,
    region_names,
    lag_qtrs: int,
) -> pd.DataFrame:
    """Return recession indicator for nowcast growth q on 4q."""

    df = recession_indicator(
        y_nowcast[-2:, :],
        datetime_ts.iloc[-2:],
        region_names,
    )
    df = df.pivot(
        index="datetime", columns="region", values="classification"
    ).reset_index()
    return df


def out_of_sample_classification_performance_table(df: pd.DataFrame) -> pd.DataFrame:
    # Up or down classification performance
    # Compare the signs
    df_oos_reg_a = prep_out_of_sample_data(df)
    df_oos_reg_a["sign"] = df_oos_reg_a.groupby(
        ["datetime", "region", "quarters_to_publication", "nowcast_index"]
    )["value"].transform(np.sign)

    df_oos_reg_a["correct"] = (
        df_oos_reg_a.groupby(
            ["quarters_to_publication", "region", "datetime", "nowcast_index"]
        )["sign"]
        .diff()
        .map({0.0: True, 2.0: False})
    )

    # analyse signs compare
    df_signs = df_oos_reg_a.dropna(subset="correct").copy()

    # This one but as a table.
    classifications = (
        df_signs.dropna(subset="value")
        .groupby(["quarters_to_publication", "region"])["correct"]
        .agg(["sum", len])
    )
    classifications["sum"] = classifications["sum"].astype(int)

    classifications["accuracy"] = classifications["sum"] / classifications["len"]
    out_table = (
        (classifications["accuracy"].unstack() * 100)
        .astype("double")
        .round(0)
        # .astype(int)
        .iloc[::-1, :]
    )
    out_table.index = out_table.index.astype(int).astype(str)
    out_table.index.name = "Qs to publication"
    out_table = out_table.T
    out_table.index.name = "Qs to publication"
    return out_table
