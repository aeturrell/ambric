"""
BRAMBLE — Bayesian Regional Augmented Machine-Bridged Latent Estimation
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
import pymc as pm
import pytensor.tensor as pt
import xgboost as xgb
from great_tables import GT
from loguru import logger
from scipy.optimize import minimize
from sklearn.decomposition import FactorAnalysis
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from ambric.diagnostics import (
    live_recession_indicator,
    plot_current_nowcast,
    plot_estimated_regional_quarterly,
    plot_national_quarterly_vs_implied,
    plot_regional_annual_estimate,
    plot_single_region_annual_estimate,
    rmse_national_quarterly,
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
    Z_panel: list[npt.NDArray], n_factors: int, standardise=True
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
    if standardise:
        Z_stacked = StandardScaler().fit_transform(Z_stacked)

    fa = FactorAnalysis(n_components=n_factors, random_state=42)
    factors = fa.fit_transform(Z_stacked)

    reconstructed = fa.transform(Z_stacked) @ fa.components_
    explained_var = 1 - ((Z_stacked - reconstructed) ** 2).sum() / (Z_stacked**2).sum()

    logger.info(
        f"FA: Extracted {n_factors} factors. Approximate explained variance: {explained_var:.1%}"
    )
    logger.info(
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
) -> tuple[xgb.XGBRegressor, npt.NDArray[np.float64]]:
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

    model = xgb.XGBRegressor(**default_params)
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
    logger.info(f"Bridge: delta (XGBoost loading) = {delta:.4f}")
    logger.info(f"Bridge: intercept = {intercept:.4f}")

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

    logger.info(
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

    logger.info("Building ambric model:")
    logger.info(f"   {K} factors, {M} macro series, {T} quarters, {R} regions")
    logger.info(f"   Bridge signal: shape {bridge_signal.shape}")

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
        sigma_uk = pm.HalfNormal("sigma_uk", sigma=0.05)
        sigma_ann = pm.HalfNormal("sigma_ann", sigma=0.2, shape=R)

        # --- Factor AR(1) dynamics ---
        phi_f = pm.Normal("phi_f", 0.7, 0.1, shape=K)
        sigma_f = pm.HalfNormal("sigma_f", 0.1, shape=K)

        factors_latent = pm.Normal("factors_latent", mu=0, sigma=1, shape=(T, K))

        pm.Potential(
            "factor_ar_prior",
            -0.5
            * pt.sum(
                ((factors_latent[1:] - phi_f * factors_latent[:-1]) / sigma_f) ** 2
                + 2 * pt.log(sigma_f)
            ),
        )

        # --- Factor observation model ---
        sigma_exog = pm.HalfNormal("sigma_exog", sigma=0.2, shape=K)
        pm.Normal("obs_factors", mu=factors_latent, sigma=sigma_exog, observed=factors)

        # --- Exogenous mean: factors + macro + bridge ---
        mu_exog = (
            pt.dot(factors_latent, Lambda.T)
            + pt.dot(macro, Gamma.T)
            + bridge_data * delta_r
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
                    (y_reg[1:] - phi_r * y_reg[:-1] - (1 - phi_r) * mu_exog[1:])
                    / sigma_eps
                )
                ** 2
                + 2 * pt.log(sigma_eps)
            ),
        )

        pm.Potential(
            "regional_init",
            -0.5 * pt.sum(((y_reg[0] - mu_exog[0]) / sigma_eps) ** 2),
        )

        # --- Constraint: UK Quarterly Growth ---
        mu_uk = pt.sum(y_reg * w, axis=1)
        pm.StudentT("obs_uk", nu=nu_uk, mu=mu_uk, sigma=sigma_uk, observed=y_uk)

        # --- Constraint: Annual Regional Growth (temporal convolution) ---
        y_lags = [y_reg * OMEGA[0]]
        for j in range(1, 7):
            shifted = pt.concatenate([pt.zeros((j, R)), y_reg[:-j, :]], axis=0)
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
    """Constrained Bayesian Latent Trees model.

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
        self.xgb_model: xgb.XGBRegressor | None = None
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
                f"   Posterior samples: {self.trace.posterior.draw.shape[0]}\n"
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
            output_dir: Directory for output files (model diagram, etc.).
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
        logger.info("\n[4/5] Building ambric Bayesian model...")
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

        self.trace.to_netcdf(f"{path}/model_trace_{self.model_id}.nc")
        logger.info(f"Model trace saved to {path}/model_trace_{self.model_id}.nc")

    def populate_results(self) -> pd.DataFrame:
        if self.trace is None:
            raise ValueError(
                "Model trace is not available. Fit the model before saving the trace."
            )
        y_q_uk_est_point, _, y_a_r_est_point = trace_to_series(self.trace)

        annual_regional_ests = pd.DataFrame(
            index=self.datetime_ts, columns=self.region_names, data=y_a_r_est_point
        )
        annual_regional_long_est = pd.melt(
            annual_regional_ests.reset_index(), id_vars="datetime", var_name="region"
        )
        annual_regional_long_est["type"] = "nowcast"
        annual_regional_long_est["measure"] = self.region_measure
        annual_national_ests = pd.DataFrame(
            index=self.datetime_ts,
            data=y_q_uk_est_point,
            columns=[self.aggregation_region],
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
            self.y_uk, y_q_uk_est_point, self.datetime_ts, self.model_id, path=path
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
        """
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

    def live_recession_indicator(self) -> GT:
        """Produces a table giving nowcast indicating growth vs recession by region. Quarterly frequency but q on 4q estimates

        Returns:
            GT: great_table of recession nowcasts.
        """
        _, _, y_a_r_est_point = trace_to_series(self.trace)
        # Simple recession indicator based on sign of quarterly growth in the most recent quarter for each region
        gt_table = live_recession_indicator(
            y_a_r_est_point,
            region_names=self.region_names,
            datetime_ts=self.datetime_ts,
            lag_qtrs=self.lag_qtrs,
        )
        return gt_table

    def live_point_estimates(self) -> GT:
        """Produces a table giving nowcast point estimates by region. Quarterly frequency but q_on_4q estimates.

        Returns:
            GT: great_table of recession nowcasts.
        """
        _, _, y_a_r_est_point = trace_to_series(self.trace)
        # Simple recession indicator based on sign of quarterly growth in the most recent quarter for each region
        df = (
            pd.DataFrame(
                y_a_r_est_point[-2:, :],
                index=self.datetime_ts.iloc[-2:],
                columns=self.region_names,
            )
            * 100
        ).round(2)
        df.index.name = "Date"
        df = df.reset_index()
        gt_table = GT(df).tab_header(
            title=f"Nowcast for {self.datetime_ts.iloc[-1].strftime('%Y-%m')} (% annual growth)",
        )
        return gt_table


def run_out_of_sample_exercise(
    df: pd.DataFrame,
    macro_names: list[str],
    region_names: list[str],
    region_covariate_names: list[str],
    n_factors: int = 4,
    aggregate_measure: str = "gva_q_on_q",
    aggregation_region: str = "uk",
    region_measure: str = "gva_q_on_4q",
    no_steps: int = 4,
    init_chunk_size: int = 20,
    lag_qtrs: int = 6,
    n_its=100000,
    n_posterior_samples=3000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run out-of-sample exercise to evaluate model performance.

    Masks the most recent annual regional data by `lag_qtrs` quarters and fits the model in chunks. End scores for the out-of-sample period for annual regional estimates are recorded and returned.

    Args:
        df (pd.DataFrame): Dataframe containing relevant columns.
        macro_names (list[str]): Names of macro series.
        region_names (list[str]): Names of regions.
        region_covariate_names (list[str]): Names of by-region covariates.
        n_factors (int, optional): Factors. Defaults to 4.
        aggregate_measure (str, optional): Nation-wide measure. Defaults to "gva_q_on_q".
        aggregation_region (str, optional): Top level geography. Defaults to "uk".
        region_measure (str, optional): Growth measure regional. Defaults to "gva_q_on_4q".
        no_steps (int, optional): How many chunks to perform out of sample exercise in. Defaults to 4.
        init_chunk_size (int, optional): Initial learning window size. Defaults to 20.
        lag_qtrs (int, optional): How many quarters before the data are published. Defaults to 6.
        n_its (int, optional): Iterations of ADVI for Bayesian inference. Defaults to 100000.
        n_posterior_samples (int, optional): Samples of the posterior. Defaults to 3000.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Out-of-sample summary statistics, Annual-regional predictions and outturns, national quarterly predictions and outturns
    """
    logger.info("This will take time to run")
    T: int = df.loc[df["measure"] == aggregate_measure, "datetime"].nunique()
    t_chunk_size: int = (np.floor((T - init_chunk_size) / no_steps)).astype(int)
    df_results = pd.DataFrame()
    df_annual_regional = pd.DataFrame()
    df_quarterly_national = pd.DataFrame()
    for i, step in enumerate(range(no_steps + 1)):
        if step == no_steps:
            T_max = T
        else:
            T_max = int(init_chunk_size + step * t_chunk_size)

        # wedge details. NB start segment is always zeroth entry
        start_segment = 0
        start_segment_oos = T_max - lag_qtrs
        end_segment = T_max
        datetimes_this_wedge = df["datetime"].unique()[start_segment:end_segment]
        datetimes_this_wedge_oos = df["datetime"].unique()[
            start_segment_oos:end_segment
        ]
        # Prepare data for model run
        # Only take entries to the end of the segment.
        this_df = df[df["datetime"].isin(datetimes_this_wedge)].copy()
        # Create a filter for those values that will be masked
        filter = (
            (this_df["region"].isin(region_names))
            & (this_df["measure"] == region_measure)
            & (this_df["datetime"].isin(datetimes_this_wedge_oos))
        )
        # get the values that will be masked for oos evaluation
        series_y_actual_annual_oos = (
            df.loc[
                (df["measure"] == region_measure)
                & (df["datetime"].isin(datetimes_this_wedge))
            ]
            .pivot_table(
                index="datetime", columns="region", values="value", dropna=False
            )
            .copy()
        )
        y_actual_annual_oos = series_y_actual_annual_oos.values
        y_actual_annual_oos = y_actual_annual_oos[-lag_qtrs:]

        # Block the annual regional data that has yet to be observed
        this_df.loc[
            filter,
            "value",
        ] = np.nan
        amb = Ambric(
            this_df,
            macro_names,
            region_names,
            region_covariate_names,
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

        logger.info("Ambric model fit complete")

        # Extract estimates
        y_q_uk_est_point, y_q_r_est_point, y_a_r_est_point = trace_to_series(amb.trace)

        # Check that the model really didn't have access to the oos
        # annual numbers
        assert np.all(np.isnan(amb.y_annual[start_segment_oos:end_segment, :]))
        # known answer unseen by model
        rmse_national_q = rmse_national_quarterly(amb.y_uk, y_q_uk_est_point)
        # Out of sample annual rmse

        y_a_r_est_point_realtime = y_a_r_est_point[-lag_qtrs:, :].copy()
        rmse_out_of_sample = np.sqrt(
            np.nanmean(np.square(y_a_r_est_point_realtime - y_actual_annual_oos))
        )
        logger.info(
            f"Step {step+1}/{no_steps+1} | T_max={T_max} for this iteration | RMSE National Q: {rmse_national_q:.4f}  | RMSE OOS Annual: {rmse_out_of_sample:.4f}"
        )
        df_sim_here = pd.DataFrame(
            {
                "step": step + 1,
                "T_max": T_max,
                "rmse_national_quarterly": rmse_national_q,
                "rmse_out_of_sample_annual": rmse_out_of_sample,
            },
            index=pd.Index([0]),
        )
        df_sim_here["nowcast_index"] = i
        df_results = pd.concat([df_results, df_sim_here], ignore_index=True)
        df_ar_here_outturn = pd.DataFrame(y_actual_annual_oos)
        df_ar_here_outturn.columns = region_names
        df_ar_here_outturn.index = series_y_actual_annual_oos.index[-lag_qtrs:]
        df_ar_here_outturn["quarters_to_publication"] = range(1, lag_qtrs + 1, 1)
        df_ar_here_outturn["type"] = "outturn"
        df_ar_here_nowcast = pd.DataFrame(y_a_r_est_point_realtime)
        df_ar_here_nowcast.columns = region_names
        df_ar_here_nowcast.index = series_y_actual_annual_oos.index[-lag_qtrs:]
        df_ar_here_nowcast["type"] = "nowcast"
        df_ar_here_nowcast["quarters_to_publication"] = range(1, lag_qtrs + 1, 1)
        df_ar_here = pd.concat([df_ar_here_nowcast, df_ar_here_outturn], axis=0)
        df_ar_here["nowcast_index"] = i

        df_annual_regional = pd.concat([df_annual_regional, df_ar_here], axis=0)

        df_uk_q_here = pd.DataFrame(data=amb.y_uk, columns=pd.Index(["outturn"]))
        df_uk_q_here.index = datetimes_this_wedge
        df_uk_q_here["nowcast"] = y_q_uk_est_point
        # cut only to the regional data oos period
        df_uk_q_here = df_uk_q_here.iloc[-lag_qtrs:, :]
        df_uk_q_here["nowcast_index"] = i
        df_quarterly_national = pd.concat([df_quarterly_national, df_uk_q_here], axis=0)

    return df_results, df_annual_regional, df_quarterly_national
