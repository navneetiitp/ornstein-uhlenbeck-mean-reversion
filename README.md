# Ornstein-Uhlenbeck Mean-Reversion Research

A reproducible quantitative research project that models an equity-pair spread as an **Ornstein-Uhlenbeck (OU) mean-reverting process**, estimates mean-reversion speed and half-life, and evaluates an execution-aware trading strategy against a static OLS benchmark.

This is the fourth project in a statistical-arbitrage research progression:

1. Single-pair statistical arbitrage with cointegration and static OLS.
2. Multi-pair statistical-arbitrage portfolio construction.
3. Dynamic hedge-ratio estimation with a Kalman filter.
4. **OU process modeling and mean-reversion dynamics.**

> **Scope:** the research uses daily equity data. It is not an HFT system. A true HFT implementation would require tick/quote data, order-book information, latency, queue position, fill probability and market-impact modeling.

## Research question

The project asks whether a spread that exhibits mean-reverting dynamics can be characterized through an OU process and converted into a systematic trading rule without using future information.

The key quantities are:

- mean-reversion speed `theta`
- long-run mean `mu`
- volatility `sigma`
- half-life
- OU-standardized spread
- turnover and implementation costs
- out-of-sample risk-adjusted performance

## Mathematical model

The continuous-time OU process is:

`dX_t = theta (mu - X_t) dt + sigma dW_t`

where:

- `X_t` is the spread
- `theta > 0` is mean-reversion speed
- `mu` is the long-run mean
- `sigma` is diffusion volatility
- `W_t` is Brownian motion

For estimation, the process is represented in discrete time as:

`Delta X_t = a + b X_{t-1} + epsilon_t`

with:

`phi = 1 + b`

`theta = -log(phi)`

`mu = a / (1 - phi)`

The implementation accepts an OU estimate only when the fitted discrete coefficient satisfies `0 < phi < 1`, which corresponds to a mean-reverting discrete process.

The OU half-life is:

`half_life = log(2) / theta`

The continuous OU stationary standard deviation is:

`sigma_stationary = sigma / sqrt(2 * theta)`

## Spread construction

For the selected pair, the training sample is used to estimate a log-price relationship:

`log(P_A) = alpha + beta * log(P_B) + epsilon`

The spread is:

`S_t = log(P_A,t) - alpha - beta * log(P_B,t)`

Pair selection, OLS estimation and cointegration screening are performed using training information only.

## Research design

The supplied dataset covers 2018–2024 and contains eight NIFTY equity series.

The chronological research split is:

| Period | Purpose |
|---|---|
| 2018–2021 | Pair screening and model development |
| 2022 | Validation / development boundary |
| 2023–2024 | Final out-of-sample evaluation |

All pair candidates are screened on the training period. Candidate relationships are evaluated using return correlation, Engle-Granger ADF evidence on the residual, and OU half-life. The final pair is selected without using the final test period.

## Trading model

The reference strategy uses:

- rolling OU estimation window: 252 observations
- parameter refit frequency: 21 observations
- entry: `|Z| >= 2.0`
- exit: `|Z| <= 0.5`
- risk stop: `|Z| >= 3.5`
- execution: next trading observation
- transaction cost assumption: 5 bps per unit turnover
- slippage assumption: 2 bps per unit turnover

The strategy uses normalized two-leg weights based on the training hedge ratio. Position changes are charged transaction costs and slippage.

## Why the next observation?

The signal at time `t` is generated from information available at `t`. The return is realized over the following observation interval. This avoids assuming that a signal can be generated and executed at the same unknown future price.

## Benchmark

A static OLS version of the same pair is evaluated using the same test period, execution convention and implementation-cost assumptions.

The comparison is intended to answer a research question rather than guarantee that one model will outperform the other:

> Does explicit OU modeling of mean-reversion dynamics provide useful information beyond a simpler static-hedge framework?

## Outputs

The research pipeline creates:

```text
outputs/
├── plots/
│   ├── equity_curve_comparison.png
│   ├── drawdown.png
│   ├── hedge_ratio.png
│   ├── spread.png
│   ├── zscore.png
│   └── theta.png
│
└── tables/
    ├── training_pair_screen.csv
    ├── selected_pair.csv
    ├── dynamic_backtest.csv
    ├── static_benchmark.csv
    ├── performance_comparison.csv
    └── research_config.csv
```

The repository intentionally reports the measured results produced by the code rather than hard-coding attractive performance numbers.

## Backtest integrity

The implementation is designed around causal research:

- pair selection uses training data
- test-period performance is not used for pair selection
- rolling OU fits use observations available before each signal
- signals are executed on the next observation
- transaction costs and slippage are charged on turnover
- the final test period is not used for parameter optimization

No backward-looking smoother is used to generate trading decisions.

## Important limitations

### Daily data

This project cannot establish HFT profitability. Daily observations do not contain enough information to model queue position, exchange latency, order-book dynamics or intraday market impact.

### Simplified costs

Fixed bps costs are a research assumption. Live execution would require bid/ask, impact, borrow and venue-specific fee models.

### Structural breaks

An OU fit can describe historical mean reversion while failing after a fundamental or regime change.

### Universe bias

The supplied universe is a manually selected set of NIFTY equities rather than a point-in-time constituent database, so survivorship bias may remain.

### Parameter risk

OU estimates can be unstable when the sample is short or the process is close to a random walk. Half-life should therefore be treated as an estimated model quantity, not a guaranteed forecast.

## Reproduce

```bash
python -m venv .venv
```

Windows:

```bash
.venv\\Scripts\\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the research pipeline:

```bash
python src/ou_research.py
```

Run tests:

```bash
pytest -q
```

## Interview topics

See [`docs/INTERVIEW.md`](docs/INTERVIEW.md) for questions covering:

- OU processes
- discretization
- parameter estimation
- half-life
- stationarity
- spread construction
- signal thresholds
- look-ahead bias
- transaction costs
- structural breaks
- model risk
- why daily OU research is different from HFT

## Quant/HFT relevance

The project demonstrates:

- stochastic-process modeling
- statistical estimation
- mean-reversion dynamics
- half-life analysis
- causal backtesting
- transaction-cost modeling
- benchmark construction
- risk measurement
- robustness-oriented research

It deliberately avoids calling a daily-bar strategy “HFT”. The natural HFT extension would replace daily data with tick/quote data and add microstructure-aware execution, latency and market-impact models.

## Repository structure

```text
ornstein-uhlenbeck-mean-reversion/
├── data/
│   └── raw/
│       └── stock_data.csv
├── src/
│   └── ou_research.py
├── outputs/
│   ├── plots/
│   └── tables/
├── docs/
│   └── INTERVIEW.md
├── tests/
│   └── test_ou_model.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Research philosophy

The objective is not to maximize a historical Sharpe ratio.

The objective is to formulate a statistical hypothesis, estimate it using information available at the time, evaluate it out of sample, account for implementation costs, and understand the failure modes.

A weaker out-of-sample result is more useful than an inflated result produced by leakage or test-period optimization.
