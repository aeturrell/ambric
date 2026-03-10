"""
AMBRIC — Augmented Mixed-frequency Bayesian Regional Inference with Constraints
-------------------------------------------
A Bayesian state-space model for estimating latent regional growth from
sparse and temporally misaligned observations. Integrates
XGBoost-driven annual regional predictions disaggregated to quarterly frequency via a MIDAS bridge equation as a core signal in the state-space dynamics.

The model pipeline:
    1. Extract factors from regional indicator panel (dimensionality reduction
       for the Bayesian state-space component).
    2. Train XGBoost on annually-aggregated raw regional indicators to predict
       annual regional growth.
    3. Fit a MIDAS bridge equation to disaggregate the XGBoost annual
       predictions to quarterly frequency, producing s_{t,r}.
    4. Build the Bayesian state-space model with factors, macro, and the
       bridge signal jointly informing latent quarterly regional growth.
    5. Estimate via variational inference.

The core equation becomes:

    y_{t,r} = phi_r * y_{t-1,r}
              + (1 - phi_r) * (Lambda_r @ F_t + Gamma_r @ X_t + delta_r * s_{t,r})
              + epsilon_{t,r}

where delta_r is estimated with a hierarchical shrinkage prior, allowing
the model to learn the value of the XGBoost bridge signal per region.
"""

import logging
import math
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ambric")
except PackageNotFoundError:
    __version__ = "unknown"

from pathlib import Path

import arviz as az
import numpy as np
import numpy.typing as npt
import pandas as pd
import pydemetra as jd
import pymc as pm
import pytensor.tensor as pt
from loguru import logger
from scipy.optimize import minimize
from sklearn.decomposition import FactorAnalysis
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from xgboost import XGBRegressor

from ambric.diagnostics import (
    assemble_loadings_data,
    bands_indicator,
    live_recession_indicator,
    plot_current_nowcast,
    plot_estimated_regional_quarterly,
    plot_loadings_aggregate,
    plot_loadings_by_region,
    plot_national_quarterly_vs_implied,
    plot_regional_annual_estimate,
    plot_seasonally_adjusted_q_on_q_growth,
    plot_single_region_annual_estimate,
    trace_to_series,
)
from ambric.utilities import OMEGA, gen_unique_id, prep_data_for_model_run


# ---------------------------------------------------------------------------
# Forward loguru logs to stdlib logging so that frameworks like Hydra,
# which configure the stdlib logging module, can capture ambric's output.
# ---------------------------------------------------------------------------
class _StdlibHandler(logging.Handler):
    """Receive records from loguru and re-emit via stdlib logging."""

    def emit(self, record: logging.LogRecord) -> None:
        logging.getLogger(record.name).handle(record)


def _propagate_loguru_to_stdlib() -> None:
    """Add a loguru sink that forwards to stdlib ``logging``."""
    logger.add(
        _StdlibHandler(),
        format="{message}",
        level="DEBUG",
    )


_propagate_loguru_to_stdlib()


def quarter_differences(ts_one: pd.Series, time: pd.Timestamp) -> pd.Series:
    """Generates difference in number of full quarters between a time series and a time.

    Args:
        ts_one (pd.Series): Series of datetime-like values.
        time (pd.Timestamp): Reference timestamp.

    Returns:
        pd.Series: Difference in number of full quarters between the two series.
    """
    quarters_ts = ts_one.dt.year * 4 + (ts_one.dt.month - 1) // 3
    quarter_time = time.year * 4 + (time.month - 1) // 3
    return quarters_ts - quarter_time


# =============================================================================
# Helper Functions: Imputation & Factor Extraction
# =============================================================================


def impute_panel(
    Z_panel: list[npt.NDArray[np.float64]],
) -> list[npt.NDArray[np.float64]]:
    """Impute missing values in regional indicator panels.

    Uses iterative imputation with a BayesianRidge estimator per panel.
    Returns a copy; the original panel is not modified.

    Args:
        Z_panel: List of J arrays, each (T, R). May contain NaNs.

    Returns:
        List of J arrays with NaNs filled by imputation, or the
        original list unchanged if no NaNs are present.
    """
    if not np.any(np.isnan(Z_panel)):
        return Z_panel

    logger.info("Missing values (NaNs) found in Z_panel. Applying imputation.")
    Z_imputed = []
    for i in range(len(Z_panel)):
        panel = Z_panel[i]
        # If the entire panel is NaN, imputation can't help — fill with zeros
        if np.all(np.isnan(panel)):
            Z_imputed.append(np.zeros_like(panel))
            continue
        estimator = make_pipeline(RobustScaler(), BayesianRidge())
        imputer = IterativeImputer(
            random_state=0,
            estimator=estimator,
            max_iter=40,
            keep_empty_features=True,
        )
        Z_imputed.append(imputer.fit_transform(panel))
    return Z_imputed


def extract_factors_from_panel(
    Z_panel: list[npt.NDArray], n_factors: int, standardise: bool = True
) -> npt.NDArray[np.float64]:
    """Extract common factors from regional indicator panels via Factor Analysis.

    Expects pre-imputed data (no NaNs).

    Args:
        Z_panel: List of regional indicator panels, each (T, R).
        n_factors: Number of factors to extract.
        standardise: Whether to standardise data before FA.

    Returns:
        Extracted common factors, shape (T, n_factors).
    """
    logger.info("Creating factors:")
    logger.info(f"   From {len(Z_panel)} series per region, creating...")
    logger.info(f"   ...{n_factors} factors")

    Z_stacked = np.hstack(Z_panel) if isinstance(Z_panel, list) else Z_panel

    n_features = Z_stacked.shape[1]
    if n_factors > n_features:
        raise ValueError(
            f"n_factors ({n_factors}) exceeds the number of available features "
            f"({n_features}). Reduce n_factors or add more regional indicators."
        )

    if standardise:
        Z_stacked = StandardScaler().fit_transform(Z_stacked)

    fa = FactorAnalysis(n_components=n_factors, random_state=42)
    factors = fa.fit_transform(Z_stacked)

    reconstructed = fa.transform(Z_stacked) @ fa.components_
    explained_var = 1 - ((Z_stacked - reconstructed) ** 2).sum() / (Z_stacked**2).sum()

    logger.info(
        f"FA: Extracted {n_factors} factors. Approximate explained variance: {explained_var:.1%}"
    )
    logger.debug(
        f"Noise variances range: {fa.noise_variance_.min():.3f} - {fa.noise_variance_.max():.3f}"
    )

    return factors


# =============================================================================
# XGBoost Annual Prediction
# =============================================================================


def _aggregate_quarterly_to_annual(
    X_quarterly: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Average quarterly data to annual (groups of 4 quarters).

    Args:
        X_quarterly: Array of shape (T, P) where T is divisible by 4.

    Returns:
        Array of shape (T//4, P) of annual averages.
    """
    T, P = X_quarterly.shape
    n_complete_years = T // 4
    T_use = n_complete_years * 4
    X_trimmed = X_quarterly[:T_use]
    return X_trimmed.reshape(n_complete_years, 4, P).mean(axis=1)


def train_xgboost_annual(
    Z_panel: list[npt.NDArray[np.float64]],
    macro: npt.NDArray[np.float64],
    y_annual: npt.NDArray[np.float64],
    xgb_params: dict | None = None,
) -> tuple[XGBRegressor, npt.NDArray[np.float64]]:
    """Train XGBoost to predict annual regional growth from annually-aggregated features.

    The model is trained on a pooled panel: all region-year pairs where
    y_annual is observed (non-NaN at Q4 indices). Features are the raw
    regional indicators Z_{j,r} and macro variables, aggregated to annual
    frequency by averaging across quarters.

    Args:
        Z_panel: List of J arrays, each (T, R). Regional indicator panels.
        macro: (T, M) national macro series.
        y_annual: (T, R) annual growth, observed only at Q4 (rest NaN).
        xgb_params: Optional XGBoost hyperparameters.

    Returns:
        Tuple of (fitted XGBRegressor, annual predictions (n_years, R)).
    """
    T, R = y_annual.shape
    J = len(Z_panel)
    M = macro.shape[1]
    n_years = T // 4

    macro_annual = _aggregate_quarterly_to_annual(macro)

    # Build pooled training panel
    X_rows = []
    y_rows = []

    for r in range(R):
        Z_r = np.column_stack([Z_panel[j][:, r] for j in range(J)])
        Z_r_annual = _aggregate_quarterly_to_annual(Z_r)

        for a in range(n_years):
            q4_idx = 4 * a + 3
            if q4_idx < T and not np.isnan(y_annual[q4_idx, r]):
                features = np.concatenate([Z_r_annual[a], macro_annual[a]])
                X_rows.append(features)
                y_rows.append(y_annual[q4_idx, r])

    if not X_rows:
        raise ValueError(
            "No observed annual regional data available for XGBoost training. "
            "Ensure y_annual contains at least one non-NaN value at a Q4 index."
        )

    X_train = np.array(X_rows)
    y_train = np.array(y_rows)

    logger.info(
        f"XGBoost: Training on {len(y_train)} region-year observations "
        f"({J + M} features: {J} regional indicators + {M} macro)."
    )

    default_params = dict(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        min_child_weight=3,
        random_state=42,
    )
    if xgb_params is not None:
        default_params.update(xgb_params)

    model = XGBRegressor(**default_params)
    model.fit(X_train, y_train)

    # Predict for all region-years
    xgb_annual_preds = np.full((n_years, R), np.nan)
    for r in range(R):
        Z_r = np.column_stack([Z_panel[j][:, r] for j in range(J)])
        Z_r_annual = _aggregate_quarterly_to_annual(Z_r)
        for a in range(n_years):
            features = np.concatenate([Z_r_annual[a], macro_annual[a]])
            xgb_annual_preds[a, r] = model.predict(features.reshape(1, -1))[0]

    train_preds = model.predict(X_train)
    rmse = np.sqrt(np.mean((y_train - train_preds) ** 2))
    logger.info(f"XGBoost: In-sample RMSE = {rmse:.6f}")

    return model, xgb_annual_preds


# =============================================================================
# MIDAS Bridge Equation
# =============================================================================


def _almon_weights(theta1: float, theta2: float, n_lags: int = 4) -> npt.NDArray:
    """Compute normalised exponential Almon lag polynomial weights.

    w_q = exp(theta1 * q + theta2 * q^2) / sum(...)

    Args:
        theta1: Linear Almon parameter.
        theta2: Quadratic Almon parameter.
        n_lags: Number of quarterly lags (default 4).

    Returns:
        Normalised weight vector of length n_lags.
    """
    q = np.arange(n_lags, dtype=np.float64)
    log_w = theta1 * q + theta2 * q**2
    log_w -= log_w.max()
    w = np.exp(log_w)
    return w / w.sum()


def _apply_midas_weights(
    X_quarterly: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Apply MIDAS weights to quarterly data to get annual weighted aggregates.

    For each year a with quarters q=0..3 (Q1..Q4):
        X_annual[a] = sum_{q=0}^{3} w_q * X_{4a+q}

    Args:
        X_quarterly: (T, P) quarterly data.
        weights: (4,) MIDAS weights.

    Returns:
        (n_years, P) weighted annual aggregates.
    """
    T, P = X_quarterly.shape
    n_years = T // 4
    T_use = n_years * 4
    X_reshaped = X_quarterly[:T_use].reshape(n_years, 4, P)
    return np.sum(X_reshaped * weights[np.newaxis, :, np.newaxis], axis=1)


def fit_bridge_equation(
    y_annual: npt.NDArray[np.float64],
    xgb_annual_preds: npt.NDArray[np.float64],
    Z_panel: list[npt.NDArray[np.float64]],
    y_uk: npt.NDArray[np.float64],
    macro: npt.NDArray[np.float64],
    use_almon: bool = True,
    ridge_alpha: float = 1.0,
) -> tuple[npt.NDArray[np.float64], dict]:
    """Fit bridge equation and produce quarterly signal s_{t,r}.

    The bridge equation (pooled across regions):

        y_{a,r}^A = alpha + delta * g_{a,r}^XGB
                     + beta_1 * sum_q w_q * y_{4a+q}^UK
                     + sum_j beta_{j+1} * sum_q w_q * Z_{j,r,4a+q}
                     + epsilon_{a,r}

    Quarterly signal:
        s_{t,r} = (alpha / 4) + (delta / 4) * g_{a(t),r}^XGB
                  + beta_1 * w_{q(t)} * y_t^UK
                  + sum_j beta_{j+1} * w_{q(t)} * Z_{j,r,t}

    Args:
        y_annual: (T, R) annual growth, NaN except at Q4.
        xgb_annual_preds: (n_years, R) XGBoost annual predictions.
        Z_panel: List of J arrays, each (T, R).
        y_uk: (T,) UK quarterly growth.
        macro: (T, M) macro series.
        use_almon: Estimate Almon polynomial weights vs equal weights.
        ridge_alpha: Ridge regularisation strength.

    Returns:
        Tuple of (quarterly_signal (T, R), bridge_info dict).
    """
    T, R = y_annual.shape
    J = len(Z_panel)
    n_years = T // 4
    # P = 1 + J  # y_uk + J regional indicators

    def _build_quarterly_indicators_for_region(r: int) -> npt.NDArray:
        """(T, P) matrix of quarterly indicators for region r."""
        cols = [y_uk.reshape(-1, 1)]
        for j in range(J):
            cols.append(Z_panel[j][:, r].reshape(-1, 1))
        return np.hstack(cols)

    # --- Estimate MIDAS weights ---
    if use_almon:

        def _bridge_objective(theta_vec: npt.NDArray) -> float:
            theta1, theta2 = theta_vec
            weights = _almon_weights(theta1, theta2, n_lags=4)

            X_rows = []
            y_rows = []
            for r in range(R):
                Q_r = _build_quarterly_indicators_for_region(r)
                Q_r_annual = _apply_midas_weights(Q_r, weights)
                for a in range(n_years):
                    q4_idx = 4 * a + 3
                    if q4_idx < T and not np.isnan(y_annual[q4_idx, r]):
                        row = np.concatenate([[xgb_annual_preds[a, r]], Q_r_annual[a]])
                        X_rows.append(row)
                        y_rows.append(y_annual[q4_idx, r])

            X = np.array(X_rows)
            y = np.array(y_rows)

            reg = Ridge(alpha=ridge_alpha, fit_intercept=True)
            reg.fit(X, y)
            y_hat = reg.predict(X)
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            return -r2

        result = minimize(
            _bridge_objective,
            x0=np.array([0.0, 0.0]),
            method="Nelder-Mead",
            options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-6},
        )
        theta_opt = result.x
        midas_weights = _almon_weights(theta_opt[0], theta_opt[1], n_lags=4)
        logger.debug(
            f"Bridge equation: Estimated Almon params theta=({theta_opt[0]:.4f}, {theta_opt[1]:.4f})"
        )
        logger.debug(
            f"Bridge equation: MIDAS weights = [{', '.join(f'{w:.4f}' for w in midas_weights)}] "
            f"(Q1->Q4)"
        )
    else:
        theta_opt = None
        midas_weights = np.ones(4) / 4.0
        logger.debug("Bridge equation: Using equal (U-MIDAS) weights.")

    # --- Fit bridge regression with chosen weights ---
    X_bridge_rows = []
    y_bridge_rows = []

    for r in range(R):
        Q_r = _build_quarterly_indicators_for_region(r)
        Q_r_annual = _apply_midas_weights(Q_r, midas_weights)
        for a in range(n_years):
            q4_idx = 4 * a + 3
            if q4_idx < T and not np.isnan(y_annual[q4_idx, r]):
                row = np.concatenate([[xgb_annual_preds[a, r]], Q_r_annual[a]])
                X_bridge_rows.append(row)
                y_bridge_rows.append(y_annual[q4_idx, r])

    if not X_bridge_rows:
        raise ValueError(
            "No observed annual regional data available for bridge equation. "
            "Ensure y_annual contains at least one non-NaN value at a Q4 index."
        )

    X_bridge = np.array(X_bridge_rows)
    y_bridge = np.array(y_bridge_rows)

    bridge_reg = Ridge(alpha=ridge_alpha, fit_intercept=True)
    bridge_reg.fit(X_bridge, y_bridge)

    y_hat_bridge = bridge_reg.predict(X_bridge)
    bridge_rmse = np.sqrt(np.mean((y_bridge - y_hat_bridge) ** 2))
    ss_res = np.sum((y_bridge - y_hat_bridge) ** 2)
    ss_tot = np.sum((y_bridge - y_bridge.mean()) ** 2)
    bridge_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    delta = bridge_reg.coef_[0]
    betas = bridge_reg.coef_[1:]
    intercept = bridge_reg.intercept_

    logger.info(f"Bridge: R² = {bridge_r2:.4f}, RMSE = {bridge_rmse:.6f}")
    logger.debug(f"Bridge: delta (XGBoost loading) = {delta:.4f}")
    logger.debug(f"Bridge: intercept = {intercept:.4f}")

    # --- Construct quarterly signal s_{t,r} ---
    quarterly_signal = np.zeros((T, R))

    for r in range(R):
        Q_r = _build_quarterly_indicators_for_region(r)
        for t in range(T):
            a = min(t // 4, n_years - 1)
            q = t % 4

            xgb_component = (delta / 4.0) * xgb_annual_preds[a, r]
            indicator_component = np.sum(betas * midas_weights[q] * Q_r[t])
            quarterly_signal[t, r] = (
                (intercept / 4.0) + xgb_component + indicator_component
            )

    logger.debug(
        f"Bridge equation: Quarterly signal shape = {quarterly_signal.shape}, "
        f"mean = {np.nanmean(quarterly_signal):.6f}, "
        f"std = {np.nanstd(quarterly_signal):.6f}"
    )

    bridge_info = {
        "bridge_reg": bridge_reg,
        "midas_weights": midas_weights,
        "almon_theta": theta_opt,
        "delta": delta,
        "betas": betas,
        "intercept": intercept,
        "bridge_r2": bridge_r2,
        "bridge_rmse": bridge_rmse,
        "xgb_annual_preds": xgb_annual_preds,
    }

    return quarterly_signal, bridge_info


# =============================================================================
# PyMC Model Definition
# =============================================================================


def build_ambric_model(
    y_uk: npt.NDArray[np.float64],
    y_annual: npt.NDArray[np.float64],
    factors: npt.NDArray[np.float64],
    macro: npt.NDArray[np.float64],
    bridge_signal: npt.NDArray[np.float64],
) -> pm.model.core.Model:
    """Build the AMBRIC Bayesian state-space model.

    The exogenous mean for latent regional growth is:

        mu_exog[t,r] = Lambda[r] @ F[t] + Gamma[r] @ X[t] + delta_r * s[t,r]

    where s[t,r] is the quarterly bridge signal from XGBoost + MIDAS.
    delta_r has a hierarchical shrinkage prior centred at zero.

    Args:
        y_uk: UK quarterly growth rates, shape (T,).
        y_annual: Regional annual growth rates, shape (T, R).
        factors: Extracted factors, shape (T, K).
        macro: Macro UK series, shape (T, M).
        bridge_signal: Quarterly bridge signal, shape (T, R).

    Returns:
        PyMC model object.
    """
    T, K = factors.shape
    _, M = macro.shape
    _, R = y_annual.shape

    logger.info("Building AMBRIC model:")
    logger.debug(f"   {K} factors, {M} macro series, {T} quarters, {R} regions")
    logger.debug(f"   Bridge signal: shape {bridge_signal.shape}")

    with pm.Model() as model:
        # --- Hierarchical factor loadings ---
        Lambda_mu = pm.Normal("Lambda_mu", mu=0, sigma=0.5, shape=K)
        Lambda_sigma = pm.HalfNormal("Lambda_sigma", sigma=0.2, shape=K)
        Lambda = pm.Normal("Lambda", mu=Lambda_mu, sigma=Lambda_sigma, shape=(R, K))

        # --- Hierarchical macro loadings ---
        Gamma_mu = pm.Normal("Gamma_mu", mu=0, sigma=0.5, shape=M)
        Gamma_sigma = pm.HalfNormal("Gamma_sigma", sigma=0.15, shape=M)
        Gamma = pm.Normal("Gamma", mu=Gamma_mu, sigma=Gamma_sigma, shape=(R, M))

        # --- Hierarchical bridge signal loading ---
        delta_mu = pm.Normal("delta_mu", mu=0, sigma=0.3)
        delta_sigma = pm.HalfNormal("delta_sigma", sigma=0.15)
        delta_r = pm.Normal("delta_r", mu=delta_mu, sigma=delta_sigma, shape=R)

        bridge_data = pm.Data("bridge_signal", bridge_signal)

        # --- Noise parameters ---
        sigma_eps = pm.HalfNormal("sigma_eps", sigma=0.03, shape=R)
        sigma_uk = pm.HalfNormal("sigma_uk", sigma=0.01)
        sigma_ann = pm.HalfNormal("sigma_ann", sigma=0.2, shape=R)

        # --- Factor AR(1) dynamics ---
        phi_f = pm.Normal("phi_f", 0.7, 0.1, shape=K)
        sigma_f = pm.HalfNormal("sigma_f", 0.1, shape=K)

        factors_latent = pm.Normal("factors_latent", mu=0, sigma=1, shape=(T, K))

        f_curr = factors_latent[1:]  # ty: ignore[not-subscriptable]
        f_prev = factors_latent[:-1]  # ty: ignore[not-subscriptable]
        pm.Potential(
            "factor_ar_prior",
            -0.5
            * pt.sum(((f_curr - phi_f * f_prev) / sigma_f) ** 2 + 2 * pt.log(sigma_f)),
        )

        # --- Factor observation model ---
        sigma_exog = pm.HalfNormal("sigma_exog", sigma=0.2, shape=K)
        pm.Normal("obs_factors", mu=factors_latent, sigma=sigma_exog, observed=factors)

        # --- Exogenous mean: factors + macro + bridge ---
        mu_exog = (
            pt.dot(factors_latent, Lambda.T)  # ty: ignore[unresolved-attribute]
            + pt.dot(macro, Gamma.T)  # ty: ignore[unresolved-attribute]
            + bridge_data * delta_r  # ty: ignore[unsupported-operator]
        )

        # --- Regional weights for UK aggregation ---
        prior_weights = [1 / R for _ in range(R)]
        w = pm.Normal("w", mu=prior_weights, sigma=0.01, shape=R)

        # --- Degrees of freedom ---
        nu_uk = pm.Gamma("nu_uk", alpha=6, beta=1)
        nu_ann = pm.Gamma("nu_ann", alpha=3, beta=0.5)

        # --- Regional AR(1) dynamics ---
        phi_r = pm.Normal("phi_r", mu=0.5, sigma=0.15, shape=R)
        y_reg = pm.Normal("y_reg", mu=0, sigma=1, shape=(T, R))

        pm.Potential(
            "regional_ar_prior",
            -0.5
            * pt.sum(
                (
                    (
                        y_reg[1:]  # ty: ignore[not-subscriptable]
                        - phi_r * y_reg[:-1]  # ty: ignore[not-subscriptable]
                        - (1 - phi_r)  # ty: ignore[unsupported-operator]
                        * mu_exog[1:]
                    )
                    / sigma_eps
                )
                ** 2
                + 2 * pt.log(sigma_eps)
            ),
        )

        pm.Potential(
            "regional_init",
            -0.5
            * pt.sum(
                ((y_reg[0] - mu_exog[0]) / sigma_eps)  # ty: ignore[not-subscriptable]
                ** 2
            ),
        )

        # --- Constraint: UK Quarterly Growth ---
        mu_uk = pt.sum(y_reg * w, axis=1)  # ty: ignore[unsupported-operator]
        pm.StudentT("obs_uk", nu=nu_uk, mu=mu_uk, sigma=sigma_uk, observed=y_uk)

        # --- Constraint: Annual Regional Growth (temporal convolution) ---
        y_lags = [y_reg * OMEGA[0]]
        for j in range(1, 7):
            y_reg_trimmed = y_reg[:-j, :]  # ty: ignore[not-subscriptable]
            shifted = pt.concatenate([pt.zeros((j, R)), y_reg_trimmed], axis=0)
            y_lags.append(shifted * OMEGA[j])

        mu_annual = pt.sum(pt.stack(y_lags), axis=0)
        pm.StudentT(
            "obs_annual", nu=nu_ann, mu=mu_annual, sigma=sigma_ann, observed=y_annual
        )

    return model


# =============================================================================
# ambric Model Class
# =============================================================================


class Ambric:
    """Augmented Mixed-frequency Bayesian Regional Inference with Constraints.

    Combines factor-analytic Bayesian state-space inference with XGBoost-driven
    predictions via a MIDAS bridge equation for regional nowcasting.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        macro_names: list[str],
        region_names: list[str],
        region_covariate_names: list[str],
        n_factors: int = 4,
        aggregate_measure: str = "gva_q_on_q",
        aggregation_region: str = "uk",
        region_measure: str = "gva_q_on_4q",
    ):
        """Initialise the ambric model.

        Expects a dataframe in long format: datetime | measure | region | value

        Args:
            df: Long format dataframe with all data.
            macro_names: Names of macro UK series.
            region_names: Names of (local) regions.
            region_covariate_names: Names of regional level series (absorbed by
                factor analysis for the Bayesian model; used raw for XGBoost).
            n_factors: Number of factors to extract.
            aggregate_measure: UK-wide measure in q-on-q growth rate.
            aggregation_region: Highest level geography.
            region_measure: Regional measure, q-on-4q growth rate.
        """
        logger.info("Initialising ambric model.")

        required_columns = ["datetime", "measure", "region", "value"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Dataframe must contain column: {col}")

        if not pd.to_datetime(df["datetime"]).dt.is_quarter_end.all():
            raise ValueError("Datetime column must only contain quarter-end dates.")

        missing_measures: list[str] = [
            x
            for x in macro_names
            + region_covariate_names
            + [aggregate_measure]
            + [region_measure]
            if x not in df["measure"].unique()
        ]
        if missing_measures:
            raise ValueError(f"Missing measures from dataframe: {missing_measures}")

        missing_regions: list[str] = [
            x
            for x in region_names + [aggregation_region]
            if x not in df["region"].unique()
        ]
        if missing_regions:
            raise ValueError(f"Missing regions from dataframe: {missing_regions}")

        (y_uk, y_annual, Z_panel, macro, lag_qrtrs) = prep_data_for_model_run(
            df,
            macro_names=macro_names,
            region_names=region_names,
            region_covariate_names=region_covariate_names,
            aggregate_measure=aggregate_measure,
            aggregation_region=aggregation_region,
            region_measure=region_measure,
        )
        self.aggregate_measure = aggregate_measure
        self.aggregation_region = aggregation_region
        self.region_measure = region_measure
        self.macro_names: list[str] = macro_names
        self.df = df.copy()
        self.y_uk: npt.NDArray[np.float64] = y_uk
        self.y_annual: npt.NDArray[np.float64] = y_annual
        self.lag_qtrs = lag_qrtrs
        self.Z_panel: list[npt.NDArray[np.float64]] = Z_panel
        self.macro: npt.NDArray[np.float64] = macro
        self.n_factors = n_factors
        self.trace: az.InferenceData | None = None
        self.model: pm.model.core.Model | None = None
        self.factors: npt.NDArray[np.float64] | None = None
        self.bridge_signal: npt.NDArray[np.float64] | None = None
        self.bridge_info: dict | None = None
        self.xgb_model: XGBRegressor | None = None
        self.n_model_fit_iterations: int | None = None
        self.model_id: str = gen_unique_id()
        self.datetime_ts: pd.Series = df.loc[
            df["measure"] == aggregate_measure, "datetime"
        ]
        self.region_names = region_names

        self._validate_arrays()

    def __repr__(self) -> str:
        out_string = (
            "------------ambric Model-------------\n"
            f"Model ID: {self.model_id}\n"
            f"Parameters:\n"
            f"   Time periods (quarters), T={self.y_uk.shape[0]}\n"
            f"   Earliest: {self.datetime_ts.min().strftime('%Y-%B')}; Latest: {self.datetime_ts.max().strftime('%Y-%B')}\n"
            f"   Regions, R={self.y_annual.shape[1]}\n"
            f"   Macroeconomic series, M={self.macro.shape[1]}\n"
            f"   Regional indicators, J={len(self.Z_panel)}\n"
            f"   n_factors={self.n_factors}\n"
        )
        if self.bridge_info is not None:
            out_string += f"   Bridge R²={self.bridge_info['bridge_r2']:.4f}, "
        if self.trace:
            out_string += (
                f"Model fitted: Yes\n"
                f"   Posterior samples: {self.trace.posterior.draw.shape[0]}\n"  # ty: ignore[unresolved-attribute]
                f"   ADVI iterations: {self.n_model_fit_iterations}\n"
            )
        else:
            out_string += "Model fitted: No\n"
        out_string += "-------------------------------------"
        return out_string

    def _validate_arrays(self) -> None:
        """Validate input array dimensions."""
        T = self.y_uk.shape[0]
        R = self.y_annual.shape[1]
        if R != self.Z_panel[0].shape[1]:
            raise ValueError(
                f"Region mismatch: y_annual has {R}, Z_panel[0] has {self.Z_panel[0].shape[1]}"
            )
        if T != self.Z_panel[0].shape[0]:
            raise ValueError(
                f"Time mismatch: y_uk has {T}, Z_panel[0] has {self.Z_panel[0].shape[0]}"
            )
        if T != self.macro.shape[0]:
            raise ValueError(
                f"Time mismatch: y_uk has {T}, macro has {self.macro.shape[0]}"
            )

    def fit(
        self,
        n_model_fit_iterations: int = 200000,
        n_posterior_samples: int = 3000,
        xgb_params: dict | None = None,
        bridge_use_almon: bool = True,
        bridge_ridge_alpha: float = 1.0,
    ) -> "Ambric":
        """Fit the ambric model.

        Pipeline:
            1. Extract factors from regional indicator panel.
            2. Train XGBoost on annually-aggregated raw indicators to predict
               annual regional growth.
            3. Fit MIDAS bridge equation to disaggregate XGBoost annual
               predictions to quarterly frequency.
            4. Build Bayesian state-space model with factors, macro, and
               bridge signal.
            5. Run variational inference.

        Args:
            n_model_fit_iterations: Number of ADVI iterations.
            n_posterior_samples: Number of posterior samples to draw.
            xgb_params: Optional XGBoost hyperparameters override.
            bridge_use_almon: Use Almon polynomial for MIDAS weights.
            bridge_ridge_alpha: Ridge regularisation for bridge equation.

        Returns:
            Self for method chaining.
        """
        self.n_model_fit_iterations = n_model_fit_iterations

        # Step 0: Impute missing values in Z_panel for factor extraction & bridge
        Z_panel_clean = impute_panel(self.Z_panel)

        # Step 1: Factor extraction (needs complete data)
        logger.info(
            f"\n[1/5] Extracting {self.n_factors} factors from indicator panel..."
        )
        self.factors = extract_factors_from_panel(
            Z_panel_clean, n_factors=self.n_factors
        )

        # Step 2: Train XGBoost on annual data (uses raw panels; XGBoost handles NaN natively)
        logger.info("\n[2/5] Training XGBoost on annual regional growth...")
        self.xgb_model, xgb_annual_preds = train_xgboost_annual(
            Z_panel=self.Z_panel,
            macro=self.macro,
            y_annual=self.y_annual,
            xgb_params=xgb_params,
        )

        # Step 3: Bridge equation -> quarterly signal
        logger.info("\n[3/5] Fitting MIDAS bridge equation...")
        self.bridge_signal, self.bridge_info = fit_bridge_equation(
            y_annual=self.y_annual,
            xgb_annual_preds=xgb_annual_preds,
            Z_panel=Z_panel_clean,
            y_uk=self.y_uk,
            macro=self.macro,
            use_almon=bridge_use_almon,
            ridge_alpha=bridge_ridge_alpha,
        )

        # Step 4: Build PyMC model
        logger.info("\n[4/5] Building AMBRIC Bayesian model...")
        self.model = build_ambric_model(
            self.y_uk,
            self.y_annual,
            self.factors,
            self.macro,
            self.bridge_signal,
        )

        # Step 5: Variational inference
        logger.info(
            f"\n[5/5] Running variational inference ({n_model_fit_iterations} iterations)..."
        )
        with self.model:
            inference = pm.fit(
                n=n_model_fit_iterations,
                method="advi",
                callbacks=[pm.callbacks.CheckParametersConvergence()],
            )
            self.trace: az.InferenceData = inference.sample(n_posterior_samples)

        # For devs: can save model diagram using the below
        # (depends on graphviz)
        # graph = pm.model_to_graphviz(self.model)
        # graph.render("model_diagram", format="pdf", cleanup=True)

        return self

    def save_trace(self, path: str | Path) -> None:
        """Saves the model trace to a NetCDF file.

        Args:
            path (str | Path): Path to save the trace file
        """
        if self.trace is None:
            raise ValueError(
                "Model trace is not available. Fit the model before saving the trace."
            )

        save_path = Path(path) / f"model_trace_{self.model_id}.nc"
        self.trace.to_netcdf(str(save_path))
        logger.info(f"Model trace saved to {save_path}")

    def populate_results(self) -> pd.DataFrame:
        """Returns results from model estimation, and original data, in format:

        | datetime | region | value | measure | type

        where type can be "outturn" or "nowcast"

        Raises:
            ValueError: If model not fitted

        Returns:
            pd.DataFrame: Dataframe of results
        """
        if self.trace is None:
            raise ValueError(
                "Model trace is not available. Fit the model before populating results."
            )
        y_q_uk_est_point, _, y_a_r_est_point = trace_to_series(self.trace)

        annual_regional_ests = pd.DataFrame(
            index=self.datetime_ts,
            columns=pd.Index(self.region_names),
            data=y_a_r_est_point,
        )
        annual_regional_long_est = pd.melt(
            annual_regional_ests.reset_index(), id_vars="datetime", var_name="region"
        )
        annual_regional_long_est["type"] = "nowcast"
        annual_regional_long_est["measure"] = self.region_measure
        annual_national_ests = pd.DataFrame(
            index=self.datetime_ts,
            data=y_q_uk_est_point,
            columns=pd.Index([self.aggregation_region]),
        )
        annual_national_ests_long = pd.melt(
            annual_national_ests.reset_index(), id_vars="datetime", var_name="region"
        )
        annual_national_ests_long["type"] = "nowcast"
        annual_national_ests_long["measure"] = self.aggregate_measure
        xf = self.df.loc[
            self.df["measure"].isin([self.aggregate_measure, self.region_measure]), :
        ].copy()
        xf["type"] = "outturn"
        results_df = pd.concat(
            [xf, annual_regional_long_est, annual_national_ests_long], axis=0
        )
        return results_df

    def plot_national_quarterly_vs_implied(
        self, path: str | Path | None = None
    ) -> None:
        """Plot national quarterly growth rates vs implied estimates from the model.

        Args:
            path (str | Path | None, optional): Save dir for image. Defaults to None.

        Raises:
            ValueError: If model not fitted.
        """
        if self.trace is None:
            raise ValueError(
                "Model trace is not available. Fit the model before plotting."
            )

        y_q_uk_est_point, y_q_r_est_point, y_a_r_est_point = trace_to_series(self.trace)
        plot_national_quarterly_vs_implied(
            self.y_uk, y_q_uk_est_point, self.datetime_ts, path=path
        )
        logger.info(
            "Plotted national quarterly growth rates vs implied estimates from the model."
        )

    def plot_regional_annual_estimate(self, path: str | Path | None = None) -> None:
        """Plot regional annual growth rates vs estimated from the model.

        Args:
            path (str | Path | None, optional): Dir to save fig to. Defaults to None.

        Raises:
            ValueError: If model not fitted.
        """
        if self.trace is None:
            raise ValueError(
                "Model trace is not available. Fit the model before plotting."
            )

        _, _, y_a_r_est_point = trace_to_series(self.trace)
        plot_regional_annual_estimate(
            self.y_annual,
            y_a_r_est_point,
            self.datetime_ts,
            self.region_names,
            path=path,
        )
        logger.info("Plotted regional annual growth rates vs estimated from the model.")

    def plot_single_region_annual_estimate(
        self, region_name: str, path: str | Path | None = None
    ) -> None:
        """Plot a single region's annual growth rates vs estimated from the model.

        Args:
            region_name (str): The region to plot.
            path (str | Path | None, optional): Dir to save fig to. Defaults to None.

        Raises:
            ValueError: If model not fitted.
        """
        if self.trace is None:
            raise ValueError(
                "Model trace is not available. Fit the model before plotting."
            )

        region_idx = self.region_names.index(region_name)
        _, _, y_a_r_est_point = trace_to_series(self.trace)
        plot_single_region_annual_estimate(
            self.y_annual,
            y_a_r_est_point,
            self.datetime_ts,
            region_idx,
            self.region_names,
            self.lag_qtrs,
            path=path,
        )
        logger.info(f"Plotted annual vs estimated for region: {region_name}.")

    def plot_estimated_regional_quarterly(self, path: str | Path | None = None) -> None:
        """Plot estimated regional quarterly growth rates from the model.

        Args:
            path (str | Path | None, optional): Dir to save fig to. Defaults to None.

        Raises:
            ValueError: If model not fitted.
        """
        if self.trace is None:
            raise ValueError(
                "Model trace is not available. Fit the model before plotting."
            )

        _, y_q_r_est_point, _ = trace_to_series(self.trace)
        # Implement plotting function for regional quarterly estimates
        plot_estimated_regional_quarterly(
            y_q_r_est_point,
            self.datetime_ts,
            self.region_names,
            self.lag_qtrs,
            path=path,
        )

    def plot_current_nowcast(self, path: Path | None = None) -> None:
        """Plot the latest nowcast (ie the period for which no annual regional observations are available.)

        Args:
            path (Path | None, optional): Dir to save figure to. Defaults to None.

        Raises:
            ValueError: If model not fitted.
        """
        if self.trace is None:
            raise ValueError(
                "Model trace is not available. Fit the model before plotting."
            )
        _, _, y_a_r_est_point = trace_to_series(self.trace)
        # Look back 4 units more than the nowcast is for:
        backlook = self.lag_qtrs + 6
        plot_current_nowcast(
            y_nowcast=y_a_r_est_point,
            y_annual=self.y_annual,
            datetime_ts=self.datetime_ts,
            region_names=self.region_names,
            lag_qtrs=self.lag_qtrs,
            backlook_qtrs=backlook,
            path=path,
        )

    def assemble_loadings_data(self) -> pd.DataFrame:
        """Assemble estimated loadings from the model posterior.

        Separates data assembly from plotting so the returned frame can be
        inspected, exported, or passed to the companion plot methods.  The
        frame contains one row per (region, loading) combination with the
        posterior mean and 94 % HDI bounds.  Loadings are scaled by the
        standard deviation of their corresponding input variable so that
        the three signal types are on a comparable *contribution* scale.

        Raises:
            ValueError: If the model has not been fitted yet.

        Returns:
            pd.DataFrame: Long-format loadings frame; see
                :func:`~ambric.diagnostics.assemble_loadings_data` for
                column details.
        """
        if self.trace is None:
            raise ValueError(
                "Model trace is not available. Fit the model before assembling loadings."
            )
        if self.factors is None:
            raise ValueError(
                "Factors are not available. Fit the model before assembling loadings."
            )
        factor_stds = np.std(self.factors, axis=0)
        macro_stds = np.std(self.macro, axis=0)
        bridge_signal_stds = (
            np.std(self.bridge_signal, axis=0)
            if self.bridge_signal is not None
            else None
        )
        return assemble_loadings_data(
            self.trace,
            region_names=self.region_names,
            macro_names=self.macro_names,
            factor_stds=factor_stds,
            macro_stds=macro_stds,
            bridge_signal_stds=bridge_signal_stds,
        )

    def plot_loadings_by_region(self, path: Path | None = None) -> None:
        """Plot estimated loadings for each region, coloured by broad type.

        Assembles loadings from the posterior and passes them to
        :func:`~ambric.diagnostics.plot_loadings_by_region`.  One panel per
        region shows all factor, macro, and bridge-signal loadings as a
        horizontal dot chart with 94 % HDI bars, enabling within-region
        comparison of the three signal categories.

        Args:
            path (Path | None): Directory in which to save the figure as
                SVG.  When ``None`` the figure is displayed interactively.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if self.trace is None:
            raise ValueError(
                "Model trace is not available. Fit the model before plotting."
            )
        loadings_df = self.assemble_loadings_data()
        plot_loadings_by_region(loadings_df, self.region_names, path=path)
        logger.info("Plotted loadings by region.")

    def plot_loadings_aggregate(self, path: Path | None = None) -> None:
        """Plot loading distributions across regions, grouped by broad type.

        Assembles loadings from the posterior and passes them to
        :func:`~ambric.diagnostics.plot_loadings_aggregate`.  One panel per
        broad loading type (factors, macro, boost_signal) compares individual
        region estimates against the cross-region mean, enabling assessment
        of which signal category dominates model dynamics and how consistently
        loadings behave across regions.

        Args:
            path (Path | None): Directory in which to save the figure as
                SVG.  When ``None`` the figure is displayed interactively.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if self.trace is None:
            raise ValueError(
                "Model trace is not available. Fit the model before plotting."
            )
        loadings_df = self.assemble_loadings_data()
        plot_loadings_aggregate(loadings_df, path=path)
        logger.info("Plotted loadings in aggregate by broad type.")

    def live_recession_indicator(self, path: Path | None = None) -> pd.DataFrame:
        """Produce a table indicating growth vs recession by region.

        Uses q-on-4q annual growth estimates at quarterly frequency.

        Args:
            path (Path | None): Directory to save the table as Parquet. When
                ``None`` no file is written.

        Raises:
            ValueError: If model not fitted.

        Returns:
            pd.DataFrame: Wide-format table with datetime index and one
                column per region containing the classification.
        """
        if self.trace is None:
            raise ValueError(
                "Model trace is not available. Fit the model before calling this method."
            )
        _, _, y_a_r_est_point = trace_to_series(self.trace)
        out_table = live_recession_indicator(
            y_a_r_est_point,
            region_names=self.region_names,
            datetime_ts=self.datetime_ts,
        )
        if path:
            out_table.to_parquet(path / "recession_indicator.parquet")
        return out_table

    def bands_indicator(self, path: Path | None = None) -> pd.DataFrame:
        """Produce a table indicating bands

        Uses q-on-4q annual growth estimates at quarterly frequency.

        Args:
            path (Path | None): Directory to save the table as Parquet. When
                ``None`` no file is written.

        Raises:
            ValueError: If model not fitted.

        Returns:
            pd.DataFrame: Wide-format table with datetime index and one
                column per region containing the classification.
        """
        if self.trace is None:
            raise ValueError(
                "Model trace is not available. Fit the model before calling this method."
            )
        _, _, y_a_r_est_point = trace_to_series(self.trace)
        out_table = bands_indicator(
            y_a_r_est_point * 100,
            region_names=self.region_names,
            datetime_ts=self.datetime_ts,
        )
        if path:
            out_table.to_parquet(path / "bands_indicator.parquet")
        return out_table

    def point_estimates_q_on_4q(self, path: Path | None = None) -> pd.DataFrame:
        """Produce a table of nowcast point estimates by region.

        Returns q-on-4q annual growth estimates (in percentage points,
        rounded to 2 d.p.) at quarterly frequency.

        Args:
            path (Path | None): Directory to save the table as Parquet. When
                ``None`` no file is written.

        Raises:
            ValueError: If model not fitted.

        Returns:
            pd.DataFrame: Wide-format table with datetime index and one
                column per region containing the point estimate.
        """
        if self.trace is None:
            raise ValueError(
                "Model trace is not available. Fit the model before calling this method."
            )
        _, _, y_a_r_est_point = trace_to_series(self.trace)
        df = (
            pd.DataFrame(
                y_a_r_est_point,
                index=self.datetime_ts,
                columns=pd.Index(self.region_names),
            )
            * 100
        ).round(2)
        df.index.name = "datetime"
        if path:
            df.to_parquet(path / "point_estimates_q_on_4q.parquet")
        return df

    def point_estimates_q_on_q(self, path: Path | None = None) -> pd.DataFrame:
        """Produce a table of nowcast point estimates by region.

        Returns q-on-q annual growth estimates (in percentage points,
        rounded to 2 d.p.) at quarterly frequency.

        Args:
            path (Path | None): Directory to save the table as Parquet. When
                ``None`` no file is written.

        Raises:
            ValueError: If model not fitted.

        Returns:
            pd.DataFrame: Wide-format table with datetime index and one
                column per region containing the point estimate.
        """
        if self.trace is None:
            raise ValueError(
                "Model trace is not available. Fit the model before calling this method."
            )
        _, y_q_r_est_point, _ = trace_to_series(self.trace)
        df = (
            pd.DataFrame(
                y_q_r_est_point,
                index=self.datetime_ts,
                columns=pd.Index(self.region_names),
            )
            * 100
        ).round(2)
        df.index.name = "datetime"
        if path:
            df.to_parquet(path / "point_estimates_q_on_q.parquet")
        return df

    def to_index_q_on_q(self, path: Path | None = None) -> pd.DataFrame:
        """Produce a table of nowcast index.

        Returns index to earliest data point (rounded to 2 d.p.) at quarterly frequency.

        Args:
            path (Path | None): Directory to save the table as Parquet. When
                ``None`` no file is written.

        Raises:
            ValueError: If model not fitted.

        Returns:
            pd.DataFrame: Wide-format table with datetime index and one
                column per region containing the point estimate.
        """
        if self.trace is None:
            raise ValueError(
                "Model trace is not available. Fit the model before calling this method."
            )
        _, y_q_r_est_point, _ = trace_to_series(self.trace)
        df_q_on_q = (
            pd.DataFrame(
                y_q_r_est_point,
                index=self.datetime_ts,
                columns=pd.Index(self.region_names),
            )
            * 100
        ).round(2)
        df_q_on_q.index.name = "datetime"
        start_value = 100.0
        # Convert percentage growth rates to an index:
        # growth factors = 1 + rate/100, then cumulative product scaled by start_value
        df_index = start_value * (1 + df_q_on_q / 100).cumprod()
        df_index = df_index.round(2)

        if path:
            df_index.to_parquet(path / "index_estimates_q_on_q.parquet")
        return df_index

    def seasonally_adjusted_index_and_growth_by_region(
        self, path: Path | None = None
    ) -> pd.DataFrame:
        """Produce seasonally adjusted index and q-on-q growth rates

        Returns data to earliest data point (rounded to 2 d.p.) at quarterly frequency.

        Args:
            path (Path | None): Directory to save the table as Parquet. When
                ``None`` no file is written.

        Raises:
            ValueError: If model not fitted.

        Returns:
            pd.DataFrame: Wide-format table with datetime index and one
                column per region containing the point estimate.
        """
        indices = self.to_index_q_on_q()
        # seasonal adjustment needs a period index
        indices.index = pd.to_datetime(indices.index).to_period()
        # Seasonal adjustment here
        spec = jd.x13_spec("rsa5c")
        # Force log transformation (no automatic detection)
        spec["regarima"]["transform"]["fn"] = "LOG"

        # Disable transitory component (TC) outlier detection
        spec["regarima"]["outlier"]["outliers"] = [
            o for o in spec["regarima"]["outlier"]["outliers"] if o["type"] != "TC"
        ]
        # Disable trading day regressors (suitable for quarterly data)
        spec["regarima"]["regression"]["td"]["td"] = "TD_NONE"
        spec["regarima"]["regression"]["td"]["auto"] = "AUTO_NO"
        spec = jd.x13_spec("rsa5c")
        # Force log transformation (no automatic detection)
        spec["regarima"]["transform"]["fn"] = "LOG"

        # Disable transitory component (TC) outlier detection
        spec["regarima"]["outlier"]["outliers"] = [
            o for o in spec["regarima"]["outlier"]["outliers"] if o["type"] != "TC"
        ]
        # Disable trading day regressors (suitable for quarterly data)
        spec["regarima"]["regression"]["td"]["td"] = "TD_NONE"
        spec["regarima"]["regression"]["td"]["auto"] = "AUTO_NO"
        spec = jd.x13_spec("rsa5c")
        # Force log transformation (no automatic detection)
        spec["regarima"]["transform"]["fn"] = "LOG"

        # Disable transitory component (TC) outlier detection
        spec["regarima"]["outlier"]["outliers"] = [
            o for o in spec["regarima"]["outlier"]["outliers"] if o["type"] != "TC"
        ]

        # Disable trading day regressors (suitable for quarterly data)
        spec["regarima"]["regression"]["td"]["td"] = "TD_NONE"
        spec["regarima"]["regression"]["td"]["auto"] = "AUTO_NO"

        def extract_into_df(ts_in: pd.Series, region: str, type: str):
            new_df = pd.DataFrame(ts_in).copy()
            new_df["region"] = region
            new_df["type"] = type
            return new_df

        df_sa_trend_orig = pd.DataFrame()
        for region in list(indices.columns):
            ts_here = indices.loc[:, region].copy()
            result = jd.x13(indices.loc[:, region], spec)
            res = result["result"]  # ty: ignore[not-subscriptable]
            sa = res["final"]["d11final"]  # seasonally adjusted series
            trend = res["final"]["d12final"]  # trend
            # seasonal = result["result"]["final"]["d16"]  # seasonal component
            sa = extract_into_df(sa, region, "seasonally_adjusted").rename(
                columns={0: "value"}
            )
            trend = extract_into_df(trend, region, "trend").rename(columns={0: "value"})
            # seasonal = extract_into_df(seasonal, region, "seasonal").rename(columns={0:"value"})
            ts_here = extract_into_df(ts_here, region, "estimate").rename(
                columns={region: "value"}
            )
            df_sa_trend_orig = pd.concat([df_sa_trend_orig, ts_here, sa, trend], axis=0)
        df_sa_trend_orig["q_on_q"] = 100 * df_sa_trend_orig.groupby(
            ["region", "type"]
        ).transform("pct_change")
        if path:
            df_sa_trend_orig.to_parquet(path / "sa_q_on_q.parquet")

        plot_seasonally_adjusted_q_on_q_growth(df_sa_trend_orig, path)
        return df_sa_trend_orig


def run_out_of_sample_exercise(
    df: pd.DataFrame,
    macro_names: list[str],
    region_names: list[str],
    region_covariate_names: list[str],
    n_factors: int = 4,
    aggregate_measure: str = "gva_q_on_q",
    aggregation_region: str = "uk",
    region_measure: str = "gva_q_on_4q",
    step_size: int = 1,
    init_chunk_size: int = 20,
    lag_qtrs: int = 6,
    n_its: int = 100000,
    n_posterior_samples: int = 3000,
) -> pd.DataFrame:
    """Run out-of-sample exercise to evaluate model performance.

    Masks the most recent annual regional data by ``lag_qtrs`` quarters
    and fits the model in rolling chunks.  Out-of-sample nowcasts and
    outturns for each step are collected and returned.

    Args:
        df (pd.DataFrame): Dataframe containing relevant columns.
        macro_names (list[str]): Names of macro series.
        region_names (list[str]): Names of regions.
        region_covariate_names (list[str]): Names of by-region covariates.
        n_factors (int): Number of factors. Defaults to 4.
        aggregate_measure (str): Nation-wide measure. Defaults to "gva_q_on_q".
        aggregation_region (str): Top level geography. Defaults to "uk".
        region_measure (str): Growth measure regional. Defaults to "gva_q_on_4q".
        step_size (int): Quarters to advance per OOS step. Defaults to 1.
        init_chunk_size (int): Initial learning window size. Defaults to 20.
        lag_qtrs (int): How many quarters before regional data are published. Defaults to 6.
        n_its (int): Iterations of ADVI for Bayesian inference. Defaults to 100000.
        n_posterior_samples (int): Samples of the posterior. Defaults to 3000.

    Returns:
        pd.DataFrame: Combined out-of-sample nowcasts and outturns across
            all rolling steps.
    """
    df = df.sort_values("datetime")
    stop: int = df.loc[df["measure"] == aggregate_measure, "datetime"].nunique()
    start = init_chunk_size
    num_steps = math.ceil((stop - start) / step_size)
    datetime_spine: pd.Series = df.loc[df["measure"] == aggregate_measure, "datetime"]
    datetime_spine = datetime_spine.sort_values()
    logger.info(f"Running out-of-sample analysis on {region_measure}:\n")
    logger.info(
        f"    burn-in: from {datetime_spine.iloc[0].strftime('%Y-%b')} to {datetime_spine.iloc[start].strftime('%Y-%b')}"
    )
    logger.info(f"    steps: {num_steps}")
    logger.info(f"    step size: {step_size}")
    df_results = pd.DataFrame()
    counter = 1
    for T_it in range(start, stop, step_size):
        # wedge details. NB start data point is always zeroth entry
        start_segment_oos = T_it - lag_qtrs
        logger.info(
            f"Nowcast step running up to {datetime_spine.iloc[T_it].strftime('%Y-%b')}"
        )
        logger.info(
            f"    Out-of-sample period: {datetime_spine.iloc[start_segment_oos].strftime('%Y-%b')} to {datetime_spine.iloc[T_it].strftime('%Y-%b')}"
        )
        # Prepare data for model run. Only take entries to the end of the segment.
        df_it = df.loc[df["datetime"] <= datetime_spine.iloc[T_it]].copy()
        # Create a mask for those values that will be masked because
        # we wish to attempt to predict them
        oos_mask = (
            (df_it["region"].isin(region_names))
            & (df_it["measure"] == region_measure)
            & (df_it["datetime"] >= datetime_spine.iloc[start_segment_oos])
        )
        # Block the annual regional data that has yet to be observed
        df_it_masked = df_it.copy()
        df_it_masked.loc[oos_mask, "value"] = np.nan
        amb = Ambric(
            df_it_masked,
            macro_names=macro_names,
            region_names=region_names,
            region_covariate_names=region_covariate_names,
            n_factors=n_factors,
            aggregate_measure=aggregate_measure,
            aggregation_region=aggregation_region,
            region_measure=region_measure,
        )

        logger.info("Ambric model created with ID: " + amb.model_id)

        logger.info(
            f"Fitting Ambric model with {n_its} iterations and {n_posterior_samples} posterior samples"
        )

        amb.fit(
            n_model_fit_iterations=n_its,
            n_posterior_samples=n_posterior_samples,
        )

        logger.info("AMBRIC model fit complete")

        # Extract estimates
        results_df_it = amb.populate_results()
        # | datetime | region | value | measure | type
        # We only want the nowcast from this
        results_df_it = results_df_it.loc[results_df_it["type"] == "nowcast"].copy()
        # Now we wish to combine it with only the relevant entries in the true data
        df_only_relevant = df.loc[
            df["datetime"].isin(results_df_it["datetime"].unique()), :
        ].copy()
        df_only_relevant["type"] = "outturn"
        # The earliest OOS value will be the one that is only 1 quarter away from publication, and so on. The max should be lag_qtrs.
        # Put this in as the difference in time to the most recent
        # data point for which there's a known value
        results_df_it["quarters_to_publication"] = quarter_differences(
            results_df_it["datetime"], datetime_spine.iloc[start_segment_oos - 1]
        )
        # For outturns, qtrs to pub doesn't make sense
        df_only_relevant["quarters_to_publication"] = np.nan

        est_and_orig_df = pd.concat([results_df_it, df_only_relevant], axis=0)
        # Now wish to filter down to just those entries that are out-of-sample
        oos_datetimes = pd.Series(
            [
                x
                for x in df_it["datetime"].unique()
                if x >= datetime_spine.iloc[start_segment_oos]
            ]
        )
        est_and_orig_df = est_and_orig_df.loc[
            est_and_orig_df["datetime"].isin(oos_datetimes), :
        ].copy()
        est_and_orig_df["nowcast_index"] = T_it
        logger.info(f"Time period {T_it}/{len(datetime_spine)} complete")
        logger.info(f"Step {counter} of {num_steps} complete")
        logger.info("----------------------------------------")
        counter = counter + 1
        est_and_orig_df = est_and_orig_df.loc[
            est_and_orig_df["measure"].isin([aggregate_measure, region_measure]), :
        ].copy()
        df_results = pd.concat([df_results, est_and_orig_df], ignore_index=True)
    return df_results
