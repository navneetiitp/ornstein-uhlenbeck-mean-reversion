from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/stock_data.csv"
PLOTS = ROOT / "outputs/plots"
TABLES = ROOT / "outputs/tables"

TRAIN_END = "2021-12-31"
VALIDATION_END = "2022-12-31"
ENTRY = 2.0
EXIT = 0.5
STOP = 3.5
ROLLING_WINDOW = 252
REFIT_EVERY = 21
MAX_HALF_LIFE = 120.0
COST_BPS = 5.0
SLIPPAGE_BPS = 2.0


def load_prices() -> pd.DataFrame:
    df = pd.read_csv(DATA, parse_dates=[0]).rename(columns={"Unnamed: 0": "date"}).set_index("date")
    return df.sort_index().ffill().dropna(how="any")


def fit_ou(x: pd.Series) -> dict:
    x = pd.Series(x).dropna().astype(float)
    lag = x.shift(1)
    dx = x - lag
    reg = pd.concat([dx.rename("dx"), lag.rename("lag")], axis=1).dropna()
    if len(reg) < 60 or reg["lag"].std() == 0:
        raise ValueError("Insufficient variation for OU estimation")
    b, a = np.polyfit(reg["lag"].to_numpy(), reg["dx"].to_numpy(), 1)
    phi = 1.0 + b
    if not (0 < phi < 1):
        raise ValueError("Estimated discrete OU coefficient is not mean reverting")
    theta = -np.log(phi)
    mu = a / (1 - phi)
    resid = reg["dx"] - (a + b * reg["lag"])
    innovation_std = float(resid.std(ddof=1))
    sigma = innovation_std * np.sqrt(2 * theta / (1 - phi**2))
    half_life = np.log(2) / theta
    stationary_std = sigma / np.sqrt(2 * theta)
    return {"phi": float(phi), "theta": float(theta), "mu": float(mu), "sigma": float(sigma),
            "half_life": float(half_life), "stationary_std": float(stationary_std), "n_obs": len(reg)}


def pair_stats(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in itertools.combinations(train.columns, 2):
        y = np.log(train[a])
        x = np.log(train[b])
        beta, alpha = np.polyfit(x.to_numpy(), y.to_numpy(), 1)
        spread = y - (alpha + beta * x)
        try:
            adf_p = float(adfuller(spread, autolag="AIC")[1])
            ou = fit_ou(spread)
        except Exception:
            continue
        corr = float(train[[a, b]].pct_change().dropna().corr().iloc[0, 1])
        rows.append({"asset_a": a, "asset_b": b, "alpha": alpha, "beta": beta,
                     "adf_pvalue": adf_p, "return_correlation": corr,
                     **{k: ou[k] for k in ["phi", "theta", "mu", "sigma", "half_life", "stationary_std", "n_obs"]}})
    out = pd.DataFrame(rows)
    out = out.sort_values(["adf_pvalue", "half_life"]).reset_index(drop=True)
    return out


def choose_pair(stats: pd.DataFrame) -> pd.Series:
    eligible = stats[(stats.adf_pvalue < 0.05) & (stats.half_life.between(2, MAX_HALF_LIFE))]
    if eligible.empty:
        return stats.iloc[0]
    return eligible.iloc[0]


def dynamic_backtest(prices: pd.DataFrame, pair: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    a, b = pair.asset_a, pair.asset_b
    dates = prices.index[(prices.index >= start) & (prices.index <= end)]
    all_spread = pd.Series(index=prices.index, dtype=float)
    train_ref = prices.loc[:start - pd.Timedelta(days=1)]
    x = np.log(train_ref[b]); y = np.log(train_ref[a])
    beta, alpha = np.polyfit(x.to_numpy(), y.to_numpy(), 1)
    all_spread.loc[:] = np.log(prices[a]) - (alpha + beta * np.log(prices[b]))

    records = []
    position = 0.0
    prev_weights = np.array([0.0, 0.0])
    params = None
    for i, dt in enumerate(dates):
        hist_end = dt - pd.Timedelta(days=1)
        hist = all_spread.loc[:hist_end].dropna().tail(ROLLING_WINDOW)
        if len(hist) < 100:
            continue
        if params is None or i % REFIT_EVERY == 0:
            try:
                params = fit_ou(hist)
            except ValueError:
                continue
        spread_t = float(all_spread.loc[dt])
        z = (spread_t - params["mu"]) / params["stationary_std"] if params["stationary_std"] > 0 else np.nan
        if np.isnan(z):
            continue
        if position == 0:
            if z <= -ENTRY:
                position = 1.0
            elif z >= ENTRY:
                position = -1.0
        else:
            if abs(z) <= EXIT or abs(z) >= STOP:
                position = 0.0
        # Execute next session: weights are based on signal at t and applied to t+1 return.
        beta_t = float(beta)
        scale = 1.0 / (1.0 + abs(beta_t))
        target_weights = np.array([position * scale, -position * beta_t * scale])
        next_dt = dates[i + 1] if i + 1 < len(dates) else None
        if next_dt is None:
            ret_a = ret_b = 0.0
        else:
            ret_a = prices.loc[next_dt, a] / prices.loc[dt, a] - 1.0
            ret_b = prices.loc[next_dt, b] / prices.loc[dt, b] - 1.0
        gross = target_weights[0] * ret_a + target_weights[1] * ret_b
        turnover = float(np.abs(target_weights - prev_weights).sum())
        cost = turnover * (COST_BPS + SLIPPAGE_BPS) / 10000.0
        net = gross - cost
        records.append({"date": dt, "spread": spread_t, "zscore": z, "position": position,
                        "beta": beta_t, "mu": params["mu"], "theta": params["theta"],
                        "half_life": params["half_life"], "weight_a": target_weights[0],
                        "weight_b": target_weights[1], "gross_return": gross, "turnover": turnover,
                        "cost": cost, "net_return": net})
        prev_weights = target_weights
    return pd.DataFrame(records).set_index("date")


def static_backtest(prices: pd.DataFrame, pair: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    a, b = pair.asset_a, pair.asset_b
    train = prices.loc[:start - pd.Timedelta(days=1)]
    beta, alpha = np.polyfit(np.log(train[b]), np.log(train[a]), 1)
    spread = np.log(prices[a]) - (alpha + beta * np.log(prices[b]))
    dates = prices.index[(prices.index >= start) & (prices.index <= end)]
    hist = spread.loc[:start - pd.Timedelta(days=1)].tail(ROLLING_WINDOW)
    position = 0.0; prev=np.array([0.0,0.0]); rows=[]
    for i,dt in enumerate(dates):
        if len(hist) < 100: hist=spread.loc[:dt].tail(ROLLING_WINDOW)
        else: hist = pd.concat([hist, spread.loc[[dt]]]).tail(ROLLING_WINDOW)
        mu=hist.iloc[:-1].mean(); sd=hist.iloc[:-1].std()
        z=(spread.loc[dt]-mu)/sd if sd>0 else np.nan
        if position==0 and z <= -ENTRY: position=1
        elif position==0 and z >= ENTRY: position=-1
        elif position!=0 and (abs(z)<=EXIT or abs(z)>=STOP): position=0
        scale=1/(1+abs(beta)); w=np.array([position*scale,-position*beta*scale])
        if i+1<len(dates):
            nd=dates[i+1]; ra=prices.loc[nd,a]/prices.loc[dt,a]-1; rb=prices.loc[nd,b]/prices.loc[dt,b]-1
        else: ra=rb=0
        gross=w[0]*ra+w[1]*rb; turnover=np.abs(w-prev).sum(); cost=turnover*(COST_BPS+SLIPPAGE_BPS)/10000; net=gross-cost
        rows.append({"date":dt,"spread":spread.loc[dt],"zscore":z,"position":position,"beta":beta,"gross_return":gross,"turnover":turnover,"cost":cost,"net_return":net})
        prev=w
    return pd.DataFrame(rows).set_index('date')


def metrics(df: pd.DataFrame) -> dict:
    r=df.net_return.fillna(0); equity=(1+r).cumprod(); dd=equity/equity.cummax()-1
    vol=r.std(ddof=1)*np.sqrt(252); sharpe=(r.mean()/r.std(ddof=1)*np.sqrt(252)) if r.std(ddof=1)>0 else np.nan
    downside=r[r<0].std(ddof=1); sortino=(r.mean()/downside*np.sqrt(252)) if pd.notna(downside) and downside>0 else np.nan
    years=len(r)/252; cagr=equity.iloc[-1]**(1/years)-1 if years>0 else np.nan
    return {"total_return":equity.iloc[-1]-1,"cagr":cagr,"annualized_vol":vol,"sharpe":sharpe,"sortino":sortino,
            "max_drawdown":dd.min(),"calmar":cagr/abs(dd.min()) if dd.min()<0 else np.nan,
            "avg_daily_return":r.mean(),"turnover":df.turnover.sum(),"cost_paid":df.cost.sum(),
            "active_days":int((df.position!=0).sum()),"observations":len(df)}


def main():
    for d in [PLOTS,TABLES]: d.mkdir(parents=True,exist_ok=True)
    prices=load_prices(); train=prices.loc[:TRAIN_END]; val=prices.loc[pd.Timestamp(TRAIN_END)+pd.Timedelta(days=1):VALIDATION_END]
    stats=pair_stats(train); stats.to_csv(TABLES/'training_pair_screen.csv',index=False)
    pair=choose_pair(stats); pd.DataFrame([pair]).to_csv(TABLES/'selected_pair.csv',index=False)
    test_start=pd.Timestamp(VALIDATION_END)+pd.Timedelta(days=1); test_end=prices.index.max()
    dyn=dynamic_backtest(prices,pair,test_start,test_end); sta=static_backtest(prices,pair,test_start,test_end)
    dyn.to_csv(TABLES/'dynamic_backtest.csv'); sta.to_csv(TABLES/'static_benchmark.csv')
    rows=[]
    for name,df in [('dynamic_ou',dyn),('static_ols',sta)]:
        m=metrics(df); m['model']=name; rows.append(m)
    pd.DataFrame(rows).set_index('model').to_csv(TABLES/'performance_comparison.csv')
    dyn['equity']=(1+dyn.net_return).cumprod(); sta['equity']=(1+sta.net_return).cumprod()
    plt.figure(figsize=(12,6)); plt.plot(dyn.index,dyn.equity,label='Dynamic OU'); plt.plot(sta.index,sta.equity,label='Static OLS'); plt.title('Out-of-Sample Equity'); plt.legend(); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(PLOTS/'equity_curve_comparison.png',dpi=160); plt.close()
    dd=dyn.equity/dyn.equity.cummax()-1; plt.figure(figsize=(12,5)); plt.plot(dd); plt.title('Dynamic OU Drawdown'); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(PLOTS/'drawdown.png',dpi=160); plt.close()
    for col,title,file in [('beta','Static OLS Hedge Ratio / OU Baseline Beta','hedge_ratio.png'),('zscore','OU Standardized Spread','zscore.png'),('spread','Pair Spread','spread.png'),('theta','Rolling OU Mean-Reversion Speed','theta.png')]:
        plt.figure(figsize=(12,5)); plt.plot(dyn.index,dyn[col]); plt.title(title); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(PLOTS/file,dpi=160); plt.close()
    summary={"asset_a":pair.asset_a,"asset_b":pair.asset_b,"train_end":TRAIN_END,"validation_end":VALIDATION_END,"test_start":str(test_start.date()),"test_end":str(test_end.date()),"entry_z":ENTRY,"exit_z":EXIT,"stop_z":STOP,"rolling_window":ROLLING_WINDOW,"refit_every":REFIT_EVERY,"transaction_cost_bps_per_turnover":COST_BPS,"slippage_bps_per_turnover":SLIPPAGE_BPS}
    pd.DataFrame([summary]).to_csv(TABLES/'research_config.csv',index=False)
    print('Selected pair:',pair.asset_a,pair.asset_b); print(pd.DataFrame(rows).set_index('model').to_string())

if __name__=='__main__': main()
