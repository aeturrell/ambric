
# AMBRIC

[![Status](https://img.shields.io/pypi/status/ambric.svg)](https://pypi.org/project/ambric/)
[![Python Version](https://img.shields.io/pypi/pyversions/ambric)](https://pypi.org/project/ambric)
[![Read the documentation at https://aeturrell.github.io/ambric/](https://img.shields.io/badge/docs-passing-brightgreen)](https://aeturrell.github.io/ambric/)
[![Tests](https://github.com/aeturrell/ambric/workflows/Tests/badge.svg)](https://github.com/aeturrell/ambric/actions?workflow=Tests)
[![Codecov](https://codecov.io/gh/aeturrell/ambric/branch/main/graph/badge.svg)](https://codecov.io/gh/aeturrell/ambric)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Source](https://img.shields.io/badge/source%20code-github-lightgrey?style=for-the-badge)](https://github.com/aeturrell/ambric)


![macOS](https://img.shields.io/badge/mac%20os-000000?style=for-the-badge&logo=macos&logoColor=F0F0F0)


https://academic.oup.com/jrsssc

## **Augmented Mixed-frequency Bayesian Regional Inference with Constraints**

AMBRIC is a Bayesian state-space model for estimating latent regional growth in a given variable from sparse and temporally misaligned observations. It combines factor analysis, autoregressive dynamics at both the factor and regional level, and observation constraints from aggregate (UK-wide) and Annual regional data.

**When would I need this model?**

- When you have a highly lagged annual release of, say, economic growth at the regional level but quarterly growth at the national level with minimal lag.
- When you want to nowcast regional growth—both quarterly and annual—from published national quarterly growth.

It was originally designed to nowcast UK gross value-added (GVA) for regions following the publication of the equivalent quarterly national growth rate.

The core model features are:

- Hierarchical structure: Factor and macro loadings are partially pooled across regions, enabling information sharing while allowing regional heterogeneity.

- Dual AR(1) dynamics: Persistence is modelled at both the factor level (common shocks) and regional level (idiosyncratic dynamics).

- Robust observations: Student-t likelihoods for UK and annual constraints provide robustness to outliers.

- Mixed frequency: Only annual growth data are observed, but latent quarterly growth data are estimated.

- Cross-sectional constraint: National and regional growth rates are constrained to be consistent.

- Temporal constraints: Annual regional growth rates are constrained to be consistent with latent quarterly growth rates.

- Soft implementation of constraints: Regional growth rates are not always *exactly* consistent with the national growth rate, for measurement, extra-regional, and rounding reasons. Regional weights, $w$, are estimated rather than fixed, allowing the model to learn the effective contribution of each region to the national total.

- Dimensional reduction: With a rich set of data on each region, it would be extremely difficult to solve the model due to the increasing number of parameters. Factor analysis is applied to the panel of regional indicators first to reduce the dimensionality to a manageable level.

- XGBoost bridge signal: An XGBoost model is trained on annually-aggregated regional indicators and macro variables to predict annual regional growth. These annual predictions are then disaggregated to quarterly frequency via a MIDAS bridge equation, producing a quarterly signal $s_{t,r}$ that enters the state-space model with a hierarchical loading $\delta_r$. This allows the model to incorporate non-linear relationships captured by XGBoost while retaining the Bayesian uncertainty quantification of the state-space framework.

## Model Details

### Variable Definitions

Let $Y$ be the variable of interest in levels.

- $t = 1, \ldots, T$ at quarterly frequency.
- $r = 1, \ldots, R$ denotes the $R$ regions of the nation.
- $Y^\text{UK}_t$ is the level in quarter $t$ for the whole nation (observed).
- $y^\text{UK}_t = \log(Y^\text{UK}_t) - \log(Y^\text{UK}_{t-1})$ is the quarterly growth rate for the whole nation (observed).
- $Y_{t, r}$ is the level for region $r$ in quarter $t$ (never observed)
- $Y_{t, r}^A = Y_{t, r} + Y_{t-1, r} + Y_{t-2, r} + Y_{t-3, r}$ is the annual level for region $r$. Observed for Q4 only, and with a lag.
- $y_{t, r}^A = \log(Y_{t, r}^A) - \log(Y^{r,A}_{t-4,r})$ is the annual growth in region $r$; observed Q4 only. $y^A_t = (y^{A}_{t,1}, \ldots, y_{t, r}^A)'$ is the vector of these.
- $y_{t, r} = \log(Y_{t, r}) - \log(Y_{t-1, r})$ is the quarterly growth rate in region $r$ (never observed). $y^Q_t = (y_{t,1}, \ldots, y_{t, r})'$ is the vector of these.
- $\boldsymbol{Z}_t$ is a panel of regional indicators, with elements $Z_{j,r,t}$.
- $s_{t,r}$ is a quarterly bridge signal for region $r$, derived from XGBoost annual predictions disaggregated via a MIDAS bridge equation. $\mathbf{s}_t = (s_{t,1}, \ldots, s_{t,R})'$ is the vector of these.


The key dimensions of the problem are:

| Symbol | Description |
|--------|-------------|
| $T$ | Number of time periods (quarters) |
| $R$ | Number of regions |
| $J$ | Number of indicators per region in the regional data |
| $K$ | Number of latent factors drawn from the regional panel data |
| $M$ | Number of national macroeconomic covariates |


### Model Equations

The core equation of AMBRIC is:

$$
\begin{equation}
\mathbf{y}_t = \boldsymbol{\Phi}_r \mathbf{y}_{t-1} + (\mathbf{I} - \boldsymbol{\Phi}_r)(\boldsymbol{\Lambda} \boldsymbol{\Phi}_f \mathbf{F}_{t-1} + \boldsymbol{\Gamma} \mathbf{X}_t + \boldsymbol{\delta} \odot \mathbf{s}_t) + \boldsymbol{\epsilon}_t
\end{equation}
$$

where $\mathbf{y}_t$ is the vector of regional quarterly growth rates, $\mathbf{F}_t$ is a vector of factors based on a regional panel of indicators, $\mathbf{X}_t$ is a vector of national statistics, and $\mathbf{s}_t$ is a quarterly bridge signal derived from XGBoost predictions via a MIDAS bridge equation. The auto-regressive terms in equation (1) are diagonal matrices; $\boldsymbol{\Phi}_r = \text{diag}(\boldsymbol{\phi}_r)$, and $\boldsymbol{\Phi}_f = \text{diag}(\boldsymbol{\phi}_f)$. $\boldsymbol{\Lambda}$ are *factor loadings*, $\boldsymbol{\Gamma}$ are *macro loadings*, and $\boldsymbol{\delta} = (\delta_1, \ldots, \delta_R)'$ are *bridge signal loadings* with $\odot$ denoting element-wise multiplication.

#### Observed vs estimated data

We observe $\boldsymbol{Z}_t$, $y_t^\text{UK}$, and, with a significant lag, $y_{t, r}^A$ for $t\mod 4 \equiv 0$ (ie 4th quarter only.)

The model estimates many parameters, but those that are "outputs" are $y_{t,r}$ and $y_{t,r}^A$, the latter only for $t\mod 4 \neq 0$.

#### Cross-sectional and temporal constraints

$\mathbf{y}_t$ are latent variables; we only observe the left-hand sides of the following assumed relationships:

$$
y_t^{\text{UK}} = \mathbf{w}^\top \mathbf{y}_t \quad \text{ and }\quad \boldsymbol{y}_t^{A} = \boldsymbol{\Omega}(L) \mathbf{y}_t
$$

although the latter with a lag. In this, the first equation is the cross-sectional constraint and the latter is the temporal constraint, which uses a lag polynomial $\boldsymbol{\Omega}(L) = \sum_{j=0}^{6} \Omega_j L^j$. The cross-sectional constraint ensures that quarterly regional growth is consistent with quarterly national growth, while the temporal constraint ensures that the regional growth is consistent with UK annual growth. (NB: weights not shown here for brevity.)

Because these are soft constraints, they enter the model as:

$$
y_t^{\text{UK}} \sim \mathcal{T}(\nu_{\text{UK}}, \mathbf{w}^\top \mathbf{y}_t, \sigma_{\text{UK}}), \quad \mathbf{y}_t^{\text{A}} \sim \mathcal{T}\left(\nu_{\text{A}}, \sum_{j=0}^{6} \Omega_j \mathbf{y}_{t-j}, \boldsymbol{\sigma}_{\text{A}}\right)
$$

where $\mathcal{T}$ is the Student's T-distribution.

#### Auto-regressive behaviours

The (Bayesian) auto-regressive behaviour of the quarterly regional growth rates is given by

$$y_{t,r} \mid y_{t-1,r} \sim \mathcal{N}\left(\phi_r \, y_{t-1,r} + (1 - \phi_r) \, \mu_{t,r}^{\text{exog}}, \, \sigma_{\varepsilon,r}\right)$$

where

$$
\mu_{t,r}^{\text{exog}} = \boldsymbol{\Lambda}_r \mathbf{F}_t + \boldsymbol{\Gamma}_r \mathbf{X}_t + \delta_r \, s_{t,r}
$$

Note that the $(1 - \phi_r)$ scaling ensures that $\mathbb{E}[y_{t,r}] = \mu_{t,r}^{\text{exog}}$.

To reduce the dimensionality of $\boldsymbol{Z}_t$, the panel of regional indicators, we use *factor analysis*, which finds an $\boldsymbol{F}_t^{\text{obs}}$ with dimension $K$ such that

$$
\mathbf{F}_t = \mathbf{Z}_t \mathbf{W} + \boldsymbol{\mu}, \quad \mathbf{Z}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_K), \quad \text{Cov}(\mathbf{F}_t) = \mathbf{W}^\top \mathbf{W} + \boldsymbol{\Psi}
$$

where these factors are autoregressive such that

$$
\mathbf{F}_t = \text{diag}(\boldsymbol{\phi}_f) \mathbf{F}_{t-1} + \boldsymbol{\eta}_t, \quad \mathbf{F}_t^{\text{obs}} = \mathbf{F}_t + \boldsymbol{\epsilon}_t^f
$$

#### Bayesian priors

##### Regional panel and factors

$$\Lambda_\mu \sim \mathcal{N}(0, 0.5) \quad [K]$$

$$\Lambda_\sigma \sim \text{HalfNormal}(0.2) \quad [K]$$

$$\Lambda_{r,k} \sim \mathcal{N}(\Lambda_{\mu,k}, \Lambda_{\sigma,k}) \quad [R \times K]$$

$$\sigma_{\text{exog}} \sim \text{HalfNormal}(0.2) \quad [K]$$

$$F_{t,k}^{\text{obs}} \sim \mathcal{N}(F_{t,k}, \sigma_{\text{exog},k})$$

$$\phi_f \sim \mathcal{N}(0.7, 0.1) \quad [K]$$

$$\sigma_f \sim \text{HalfNormal}(0.1) \quad [K]$$

##### Macro indicators

$$\Gamma_\mu \sim \mathcal{N}(0, 0.5) \quad [M]$$

$$\Gamma_\sigma \sim \text{HalfNormal}(0.15) \quad [M]$$

$$\Gamma_{r,m} \sim \mathcal{N}(\Gamma_{\mu,m}, \Gamma_{\sigma,m}) \quad [R \times M]$$

##### Bridge signal loadings

$$\delta_\mu \sim \mathcal{N}(0, 0.3)$$

$$\delta_\sigma \sim \text{HalfNormal}(0.15)$$

$$\delta_r \sim \mathcal{N}(\delta_\mu, \delta_\sigma) \quad [R]$$

##### Growth

$$\sigma_\varepsilon \sim \text{HalfNormal}(0.03) \quad [R]$$

$$\sigma_{\text{A}} \sim \text{HalfNormal}(0.2) \quad [R]$$

$$\phi_r \sim \mathcal{N}(0.5, 0.15) \quad [R]$$

$$\sigma_{\text{UK}} \sim \text{HalfNormal}(0.05)$$

#### Parameters

| Parameter | Shape |
| ----------:|:---------|
| $\Lambda_\mu$ | $K$ |
| $\Lambda_\sigma$ | $K$ |
| $\Lambda$ | $R \times K$ |
| $\Gamma_\mu$ | $M$ |
| $\Gamma_\sigma$ | $M$ |
| $\Gamma$ | $R \times M$ |
| $\delta_\mu$ | $1$ |
| $\delta_\sigma$ | $1$ |
| $\delta_r$ | $R$ |
| $\phi_f$ | $K$ |
| $\sigma_f$ | $K$ |
| $F$ | $T \times K$ |
| $\sigma_{\text{exog}}$ | $K$ |
| $\phi_r$ | $R$ |
| $\sigma_\varepsilon$ | $R$ |
| $y_r$ | $T \times R$ |
| $w$ | $R$ |
| $\sigma_{\text{UK}}$ | $1$ |
| $\sigma_{\text{A}}$ | $R$ |
| $\nu_{\text{UK}}$ | $1$ |
| $\nu_{\text{A}}$ | $1$ |

Total: $6K + 2M + RK + RM + TK + TR + 5R + 5$

### Model solution

Prior to estimating the Bayesian model, an XGBoost model is trained on annually-aggregated regional indicators and macro variables to predict annual regional growth. These annual predictions are disaggregated to quarterly frequency using a MIDAS bridge equation, producing the bridge signal $\mathbf{s}_t$. The Bayesian state-space model is then estimated using [PyMC](https://www.pymc.io/welcome.html) and [pytensor](https://pytensor.readthedocs.io/en/latest/) via ADVI (Automatic Differentiation Variational Inference).
