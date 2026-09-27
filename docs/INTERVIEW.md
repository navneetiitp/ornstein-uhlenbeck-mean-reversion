# Interview Notes — Ornstein-Uhlenbeck Mean Reversion

## Core model

### What is an OU process?
A continuous-time mean-reverting diffusion:

`dX_t = theta(mu - X_t)dt + sigma dW_t`

`theta` controls mean-reversion speed, `mu` is the long-run mean, and `sigma` is diffusion volatility.

### How is it estimated here?
The discrete representation is fit through:

`Delta X_t = a + b X_{t-1} + epsilon_t`

with `phi = 1 + b`, `theta = -log(phi)`, and `mu = a/(1-phi)` when `0 < phi < 1`.

### What is half-life?
`half_life = log(2) / theta`.
It measures the time required for a deviation to decay by roughly 50% under the continuous OU model.

## Trading

The spread is standardized with a trailing OU-based distribution. Entry occurs at an absolute Z-score of 2, exit at 0.5, and a 3.5-sigma stop is used. Signals are executed on the next observation.

## Look-ahead bias

Pair selection is performed on the training period. Test-period signals use only observations available before the signal timestamp. Rolling OU fits exclude the current observation. No future observations are used to revise earlier estimates.

## Why not call this HFT?
The data are daily bars. A true HFT claim would require tick/quote data, order-book state, latency, queue position, fill modeling, and market-impact assumptions. This project demonstrates statistical modeling and execution-aware research that can be extended to higher-frequency data.

## Main risks
Structural breaks, unstable mean reversion, parameter sensitivity, transaction costs, liquidity, shorting constraints, and model misspecification.

## What would you improve next?
Use point-in-time universes, walk-forward pair selection, realistic borrow/impact models, multiple independent test periods, and tick-level execution data.
