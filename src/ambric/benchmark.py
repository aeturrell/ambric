"""Recovery benchmark: fit AMBRIC to simulated data and compare with the truth.

The simulator knows the true quarterly regional growth, the annual values
hidden by the publication lag, the regional weights, and the noise scales.
:func:`run_recovery_benchmark` fits the model to the simulated inputs and
reports how well each of those is recovered, giving a fixed set of numbers
to compare before and after any change to the model.
"""

import numpy as np
import numpy.typing as npt
from loguru import logger

from ambric.diagnostics import trace_to_series
from ambric.utilities import generate_realistic_simulated_data


def run_recovery_benchmark(
    T: int = 130,
    R: int = 12,
    J: int = 4,
    n_factors: int = 2,
    n_macro: int = 2,
    lag_qtrs: int = 6,
    seed: int = 34,
    w_true: npt.NDArray[np.float64] | None = None,
    n_model_fit_iterations: int = 200000,
    n_posterior_samples: int = 1000,
) -> dict[str, float]:
    """Fit AMBRIC to simulated data and measure recovery of the truth.

    Args:
        T (int): Number of quarterly time periods.
        R (int): Number of regions.
        J (int): Number of regional covariate panels.
        n_factors (int): Number of latent factors, in the simulation and the model.
        n_macro (int): Number of macro indicator series.
        lag_qtrs (int): Publication lag in quarters for annual regional data.
        seed (int): Random seed for the simulated data.
        w_true (npt.NDArray[np.float64] | None): True regional weights, shape
            (R,). Defaults to equal weights.
        n_model_fit_iterations (int): Number of ADVI iterations.
        n_posterior_samples (int): Number of posterior samples to draw.

    Returns:
        dict[str, float]: Recovery metrics, all in decimal growth units:

            - ``quarterly_corr``, ``quarterly_rmse``: estimated vs true
              quarterly regional growth, pooled over regions and quarters.
            - ``quarterly_sd_truth``: standard deviation of true quarterly
              regional growth, for scale.
            - ``annual_in_sample_rmse``: estimated vs published annual growth.
            - ``annual_held_out_rmse``: estimated vs true annual growth for
              values hidden by the publication lag (the nowcast target).
            - ``annual_sd_truth``: standard deviation of true annual growth.
            - ``national_rmse``: implied vs observed national growth.
            - ``sigma_eps_ratio``, ``sigma_uk_ratio``, ``sigma_annual_ratio``:
              posterior mean of each noise scale divided by its true value.
            - ``w_sum``: sum of posterior mean weights.
            - ``w_max_abs_error``: largest absolute error in a weight.
            - ``bridge_r2``: R² reported by the bridge equation.
            - ``advi_converged``: 1.0 if the fit was judged converged, else 0.0.
            - ``advi_final_improvement``, ``advi_standard_error``: fall in
              mean loss between the last two windows, and its standard error.
    """
    # Imported here because ambric/__init__.py imports from this package's
    # other modules at import time.
    from ambric import Ambric

    df, truth = generate_realistic_simulated_data(  # ty: ignore[not-iterable]
        T=T,
        R=R,
        J=J,
        n_factors=n_factors,
        n_macro=n_macro,
        lag_qtrs=lag_qtrs,
        seed=seed,
        w_true=w_true,
        return_truth=True,
    )
    region_names: list[str] = truth["region_names"]
    macro_names = sorted(x for x in df["measure"].unique() if "macro" in x)
    region_covariate_names = sorted(
        x for x in df["measure"].unique() if "regional_covar" in x
    )
    amb = Ambric(
        df, macro_names, region_names, region_covariate_names, n_factors=n_factors
    )
    amb.fit(
        n_model_fit_iterations=n_model_fit_iterations,
        n_posterior_samples=n_posterior_samples,
    )
    assert amb.trace is not None and amb.bridge_info is not None
    assert amb.advi_convergence is not None

    # Align the truth to the model's (possibly truncated) sample.
    rows = np.flatnonzero(np.isin(truth["datetime"], amb.datetime_ts))
    y_reg_true = truth["y_reg_true"][rows]
    y_annual_true = truth["y_annual_no_lag"][rows]

    y_uk_est, y_reg_est, y_annual_est = trace_to_series(amb.trace)
    posterior = amb.trace.posterior  # ty: ignore[unresolved-attribute]
    w_est = posterior["w"].mean(dim=("chain", "draw")).values

    published = ~np.isnan(amb.y_annual) & ~np.isnan(y_annual_est)
    held_out = np.isnan(amb.y_annual) & ~np.isnan(y_annual_true)
    held_out &= ~np.isnan(y_annual_est)

    def _rmse(a: npt.NDArray, b: npt.NDArray) -> float:
        return float(np.sqrt(np.mean((a - b) ** 2)))

    metrics = {
        "quarterly_corr": float(
            np.corrcoef(y_reg_est.ravel(), y_reg_true.ravel())[0, 1]
        ),
        "quarterly_rmse": _rmse(y_reg_est, y_reg_true),
        "quarterly_sd_truth": float(y_reg_true.std()),
        "annual_in_sample_rmse": _rmse(
            y_annual_est[published], amb.y_annual[published]
        ),
        "annual_held_out_rmse": _rmse(y_annual_est[held_out], y_annual_true[held_out]),
        "annual_sd_truth": float(np.nanstd(y_annual_true)),
        "national_rmse": _rmse(y_uk_est, amb.y_uk),
        "sigma_eps_ratio": float(posterior["sigma_eps"].mean()) / truth["sigma_eps"],
        "sigma_uk_ratio": float(posterior["sigma_uk"].mean()) / truth["sigma_uk"],
        "sigma_annual_ratio": float(posterior["sigma_ann"].mean())
        / truth["sigma_annual"],
        "w_sum": float(w_est.sum()),
        "w_max_abs_error": float(np.max(np.abs(w_est - truth["w_true"]))),
        "bridge_r2": float(amb.bridge_info["bridge_r2"]),
        "advi_converged": float(amb.advi_convergence["converged"]),
        "advi_final_improvement": float(amb.advi_convergence["improvement"]),
        "advi_standard_error": float(amb.advi_convergence["standard_error"]),
    }
    logger.info("Recovery benchmark:")
    for name, value in metrics.items():
        logger.info(f"   {name}: {value:.5f}")
    return metrics
