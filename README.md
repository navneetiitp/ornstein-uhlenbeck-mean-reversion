# Ornstein-Uhlenbeck Mean-Reversion Research

A reproducible quantitative research project that models an equity-pair spread as an **Ornstein-Uhlenbeck (OU) mean-reverting process**, estimates mean-reversion dynamics, and evaluates a causal trading strategy against a static OLS benchmark.

This is the fourth project in a statistical-arbitrage research progression:

1. **Statistical Arbitrage Using Pairs Trading** — cointegration, static OLS hedge ratio, rolling Z-score.
2. **Multi-Pair Statistical Arbitrage Portfolio** — pair selection, portfolio construction and portfolio-level risk.
3. **Kalman Filter Dynamic Hedge Ratio** — recursive state-space estimation and time-varying hedge ratios.
4. **Ornstein-Uhlenbeck Mean Reversion** — stochastic-process modeling of mean-reversion dynamics.

> **Scope:** This project uses daily equity data. It is quantitative research rather than an HFT system. A true HFT implementation would require tick/quote data, order-book information, latency, queue position, fill probability and market-impact modeling.

---

## Research Objective

The project investigates whether the mean-reverting behavior of an equity-pair spread can be characterized using an Ornstein-Uhlenbeck process and converted into a systematic trading strategy without using future information.

The research focuses on:

- mean-reversion speed
- long-run equilibrium level
- OU volatility
- half-life
- dynamic spread behavior
- causal signal generation
- transaction costs and slippage
- out-of-sample performance
- comparison with a simpler static-hedge benchmark

The objective is not to maximize a historical performance metric, but to evaluate a clearly defined statistical hypothesis under a reproducible research framework.

---

## Mathematical Model

The continuous-time Ornstein-Uhlenbeck process is:

\[
dX_t = \theta(\mu-X_t)dt+\sigma dW_t
\]

where:

- \(X_t\) = spread
- \(\theta > 0\) = mean-reversion speed
- \(\mu\) = long-run mean
- \(\sigma\) = diffusion volatility
- \(W_t\) = Brownian motion

A larger \(\theta\) corresponds to faster mean reversion.

The corresponding half-life is:

\[
t_{1/2}=\frac{\ln 2}{\theta}
\]

The continuous OU stationary standard deviation is:

\[
\sigma_{\text{stationary}}
=
\frac{\sigma}{\sqrt{2\theta}}
\]

---

## Discrete-Time Estimation

For empirical estimation, the process is represented as:

\[
\Delta X_t = a+bX_{t-1}+\epsilon_t
\]

The discrete autoregressive coefficient is:

\[
\phi=1+b
\]

The implementation derives:

\[
\theta=-\ln(\phi)
\]

and:

\[
\mu=\frac{a}{1-\phi}
\]

The OU estimate is accepted only when:

\[
0<\phi<1
\]

which corresponds to a mean-reverting discrete process.

This formulation connects the statistical regression directly to the continuous-time OU parameters.

---

# Spread Construction

For the selected equity pair, the training data is used to estimate the log-price relationship:

\[
\log(P_A)
=
\alpha+\beta\log(P_B)+\epsilon
\]

The resulting spread is:

\[
S_t
=
\log(P_{A,t})
-
\alpha
-
\beta\log(P_{B,t})
\]

The hedge ratio is estimated using the permitted training information.

The resulting spread is then used as the input to the OU estimation and trading model.

---

# Research Pipeline

```text
Historical Price Data
        │
        ▼
Chronological Train / Validation / Test Split
        │
        ▼
Training-Only Pair Screening
        │
        ▼
Cointegration / ADF Analysis
        │
        ▼
OLS Hedge-Ratio Estimation
        │
        ▼
Spread Construction
        │
        ▼
OU Parameter Estimation
        │
        ├── θ: Mean-Reversion Speed
        ├── μ: Long-Run Mean
        ├── σ: Volatility
        └── Half-Life
        │
        ▼
Rolling OU Re-estimation
        │
        ▼
Z-Score Signal
        │
        ▼
Next-Observation Execution
        │
        ▼
Transaction Costs + Slippage
        │
        ▼
Performance / Risk Analysis
        │
        ▼
Static OLS Benchmark Comparison
