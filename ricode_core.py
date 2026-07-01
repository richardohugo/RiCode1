"""
RiCode core: data access + risk/factor/simulation math.

Pure functions only (no Streamlit). Everything here is unit-testable with
synthetic return series; only `get_prices` touches the network.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _download_close(symbol: str, period: str) -> pd.Series | None:
    import yfinance as yf

    df = yf.download(symbol, period=period, auto_adjust=True,
                     progress=False, threads=False)
    if df is None or len(df) == 0:
        return None
    # df may be a DataFrame with a 'Close' column, possibly MultiIndexed.
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna()
    return close if len(close) >= 2 else None


def get_prices(ticker: str, period: str = "2y") -> pd.Series:
    """Adjusted daily closes as a clean float Series indexed by date.

    Robust to the various shapes yfinance returns across versions. A bare
    symbol with no suffix (e.g. BBRI) is retried with .JK appended, so IDX
    codes work without the Yahoo suffix.
    """
    t = ticker.strip().upper()
    if not t:
        raise ValueError("Empty ticker.")
    close = _download_close(t, period)
    if close is None and "." not in t and not t.startswith("^"):
        retry = t + ".JK"
        close = _download_close(retry, period)
        if close is not None:
            t = retry
    if close is None:
        raise ValueError(f"No data returned for '{ticker}'. Check the symbol.")
    close.name = t
    return close


def simple_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()


def log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1)).dropna()


def annualized_mu_sigma(prices: pd.Series) -> tuple[float, float]:
    """Annualized drift and volatility from daily log returns."""
    lr = log_returns(prices)
    mu = float(lr.mean() * TRADING_DAYS)
    sigma = float(lr.std(ddof=1) * np.sqrt(TRADING_DAYS))
    return mu, sigma


# --------------------------------------------------------------------------- #
# Value at Risk
# --------------------------------------------------------------------------- #
def ewma_vol(returns: pd.Series, lam: float = 0.94) -> float:
    """RiskMetrics EWMA daily volatility (latest estimate).

    sigma2_t = lam * sigma2_{t-1} + (1 - lam) * r2_{t-1}, zero-mean convention.
    Reacts to volatility clustering much faster than the equal-weight
    sample standard deviation.
    """
    r2 = pd.Series(returns, dtype=float).dropna() ** 2
    var = r2.ewm(alpha=1.0 - lam, adjust=True).mean().iloc[-1]
    return float(np.sqrt(var))


def bootstrap_horizon(returns: np.ndarray, horizon: int,
                      n_boot: int = 20_000, seed: int | None = 7) -> np.ndarray:
    """Compounded h-day returns built by iid resampling of daily returns.

    Avoids the sqrt-t shortcut for multi-day historical VaR: instead of
    scaling the 1-day quantile, it compounds actual daily returns drawn
    with replacement. Seeded by default so results are reproducible.
    """
    rng = np.random.default_rng(seed)
    draws = rng.choice(returns, size=(n_boot, horizon), replace=True)
    return np.prod(1.0 + draws, axis=1) - 1.0


def var_report(returns: pd.Series, conf: float = 0.95,
               horizon: int = 1, position: float = 0.0,
               lam: float = 0.94) -> dict:
    """VaR / CVaR on simple daily returns. Losses are POSITIVE fractions.

    Methods at the requested horizon:
      hist / cvar : empirical quantile / tail mean. For horizon > 1 the
                    h-day distribution is built by bootstrap compounding
                    of daily returns (no sqrt-t shortcut).
      param       : normal, mean and sd scaled by the normal model's own
                    time-aggregation rule.
      cf          : Cornish-Fisher with skew and excess kurtosis aggregated
                    to the horizon (skew/sqrt(h), kurtosis/h under iid).
      ewma        : RiskMetrics, EWMA volatility with zero mean.

    1-day values (hist1, cvar1) are always included for charting.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    mu, sd = r.mean(), r.std(ddof=1)
    skew = float(stats.skew(r))
    exkurt = float(stats.kurtosis(r, fisher=True))  # excess (normal -> 0)
    sd_ewma = ewma_vol(pd.Series(r), lam=lam)

    h = int(horizon)
    sqrt_h = np.sqrt(h)
    mu_h, sd_h = mu * h, sd * sqrt_h
    z = stats.norm.ppf(1 - conf)  # negative, e.g. -1.645 at 95%

    def cf_quantile(zq, s, k):
        return (zq
                + (zq**2 - 1) / 6 * s
                + (zq**3 - 3*zq) / 24 * k
                - (2*zq**3 - 5*zq) / 36 * s**2)

    # 1-day empirical
    q1 = np.quantile(r, 1 - conf)
    tail1 = r[r <= q1]
    hist1 = -q1
    cvar1 = -tail1.mean() if tail1.size else hist1

    # horizon empirical: bootstrap-compound daily returns when h > 1
    if h > 1:
        rh = bootstrap_horizon(r, h)
        qh = np.quantile(rh, 1 - conf)
        tailh = rh[rh <= qh]
        hist = -qh
        cvar = -tailh.mean() if tailh.size else hist
    else:
        hist, cvar = hist1, cvar1

    # moment aggregation for Cornish-Fisher at the horizon (iid scaling)
    skew_h = skew / sqrt_h
    exkurt_h = exkurt / h
    z_cf = cf_quantile(z, skew_h, exkurt_h)

    param = -(mu_h + sd_h * z)
    cf = -(mu_h + sd_h * z_cf)
    ewma = -(sd_ewma * sqrt_h * z)          # RiskMetrics zero-mean convention

    out = {
        "n": n, "conf": conf, "horizon": h,
        "mean_daily": mu, "vol_daily": sd,
        "vol_ann": sd * np.sqrt(TRADING_DAYS),
        "vol_ewma_ann": sd_ewma * np.sqrt(TRADING_DAYS),
        "skew": skew, "exkurt": exkurt,
        "hist": hist, "param": param, "cf": cf, "ewma": ewma, "cvar": cvar,
        "hist1": hist1, "cvar1": cvar1,
        "worst_day": -r.min(), "best_day": r.max(),
    }
    if position:
        for k in ("hist", "param", "cf", "ewma", "cvar"):
            out[k + "_cash"] = out[k] * position
    return out


def rolling_var(returns: pd.Series, window: int = 250, conf: float = 0.95) -> pd.Series:
    """Rolling 1-day historical VaR (positive losses)."""
    return -returns.rolling(window).quantile(1 - conf).dropna()


def var_backtest(returns: pd.Series, window: int = 250,
                 conf: float = 0.95) -> dict | None:
    """Out-of-sample check of rolling historical VaR + Kupiec POF test.

    Each day's return is compared against the VaR estimated from the
    window ending the PREVIOUS day (no look-ahead). Kupiec's test asks
    whether the observed breach count is consistent with the target rate;
    a small p-value means the model misses its coverage.
    """
    var_t = -returns.rolling(window).quantile(1 - conf)
    var_prev = var_t.shift(1)
    mask = var_prev.notna()
    if int(mask.sum()) < 30:
        return None
    r = returns[mask]
    v = var_prev[mask]
    breach = r < -v
    n, x = int(len(r)), int(breach.sum())
    p = 1 - conf

    def loglik(prob):
        # x*log(prob) with the 0*log(0) = 0 convention
        with np.errstate(divide="ignore", invalid="ignore"):
            a = x * np.log(prob) if x else 0.0
            b = (n - x) * np.log(1 - prob) if x < n else 0.0
        return a + b

    rate = x / n
    lr = -2.0 * (loglik(p) - loglik(rate)) if 0 < x < n else (
        -2.0 * loglik(p) if x == 0 else np.inf)
    pval = float(stats.chi2.sf(lr, df=1))

    return {"n": n, "breaches": x, "expected": p * n, "rate": rate,
            "kupiec_lr": float(lr), "kupiec_p": pval,
            "breach_dates": r.index[breach],
            "breach_returns": r[breach]}


# --------------------------------------------------------------------------- #
# Monte Carlo simulation
# --------------------------------------------------------------------------- #
def mc_paths_gbm(spot: float, mu: float, sigma: float, days: int,
                 n_paths: int, seed: int | None = None,
                 antithetic: bool = True) -> np.ndarray:
    """Geometric Brownian motion price paths, shape (n_paths, days + 1).

    `mu` and `sigma` are annualized. With `antithetic=True` each random
    draw Z is paired with -Z, which cancels odd-order sampling error and
    tightens the quantile estimates at no extra cost (variance reduction).
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / TRADING_DAYS
    if antithetic:
        half = (n_paths + 1) // 2
        z = rng.standard_normal((half, days))
        z = np.vstack([z, -z])[:n_paths]
    else:
        z = rng.standard_normal((n_paths, days))
    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    paths = np.empty((n_paths, days + 1))
    paths[:, 0] = spot
    paths[:, 1:] = spot * np.exp(np.cumsum(increments, axis=1))
    return paths


def mc_paths_bootstrap(spot: float, returns: pd.Series, days: int,
                       n_paths: int, seed: int | None = None) -> np.ndarray:
    """Price paths built by resampling ACTUAL daily returns with replacement.

    Keeps the fat tails and skew of the real return distribution (which a
    normal model smooths away), though not its volatility clustering.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 20:
        raise ValueError("Need at least 20 daily returns to bootstrap paths.")
    rng = np.random.default_rng(seed)
    draws = rng.choice(r, size=(n_paths, days), replace=True)
    paths = np.empty((n_paths, days + 1))
    paths[:, 0] = spot
    paths[:, 1:] = spot * np.cumprod(1.0 + draws, axis=1)
    return paths


def mc_summary(paths: np.ndarray, conf: float = 0.95) -> dict:
    """Percentile bands over time + terminal-distribution statistics."""
    spot = float(paths[0, 0])
    qs = [5, 25, 50, 75, 95]
    bands = np.percentile(paths, qs, axis=0)          # (5, days+1)
    terminal = paths[:, -1]
    term_ret = terminal / spot - 1.0
    q = np.quantile(term_ret, 1 - conf)
    tail = term_ret[term_ret <= q]
    return {
        "spot": spot, "qs": qs, "bands": bands, "terminal": terminal,
        "median": float(np.median(terminal)),
        "p5": float(bands[0, -1]), "p95": float(bands[4, -1]),
        "prob_up": float((terminal > spot).mean()),
        "var": float(-q),
        "cvar": float(-tail.mean()) if tail.size else float(-q),
    }


# --------------------------------------------------------------------------- #
# Single-factor (market) model:  r_i - rf = alpha + beta (r_m - rf) + eps
# --------------------------------------------------------------------------- #
def factor_model(stock_ret: pd.Series, mkt_ret: pd.Series,
                 rf_annual: float = 0.0) -> dict:
    rf_daily = rf_annual / TRADING_DAYS
    df = pd.concat([stock_ret, mkt_ret], axis=1, join="inner").dropna()
    df.columns = ["stock", "mkt"]
    y = df["stock"].to_numpy() - rf_daily
    x = df["mkt"].to_numpy() - rf_daily
    n = y.size
    if n < 30:
        raise ValueError("Not enough overlapping history to fit the model.")

    xbar, ybar = x.mean(), y.mean()
    Sxx = np.sum((x - xbar) ** 2)
    if Sxx == 0:
        raise ValueError("Benchmark returns have zero variance.")
    Sxy = np.sum((x - xbar) * (y - ybar))
    beta = Sxy / Sxx
    alpha = ybar - beta * xbar

    resid = y - (alpha + beta * x)
    sse = np.sum(resid ** 2)
    sst = np.sum((y - ybar) ** 2)
    r2 = 1 - sse / sst
    s2 = sse / (n - 2)                       # residual variance
    se_beta = np.sqrt(s2 / Sxx)
    se_alpha = np.sqrt(s2 * (1 / n + xbar**2 / Sxx))
    t_beta = beta / se_beta
    t_alpha = alpha / se_alpha
    p_alpha = 2 * stats.t.sf(abs(t_alpha), n - 2)

    resid_vol_ann = np.sqrt(s2) * np.sqrt(TRADING_DAYS)
    total_vol_ann = y.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sys_vol_ann = abs(beta) * x.std(ddof=1) * np.sqrt(TRADING_DAYS)

    return {
        "n": n, "beta": beta,
        "alpha_daily": alpha, "alpha_ann": alpha * TRADING_DAYS,
        "r2": r2, "t_alpha": t_alpha, "t_beta": t_beta, "p_alpha": p_alpha,
        "resid_vol_ann": resid_vol_ann, "total_vol_ann": total_vol_ann,
        "sys_vol_ann": sys_vol_ann,
        "sys_share": r2, "spec_share": 1 - r2,
        "x": x, "y": y,               # excess returns, for the scatter
    }


def rolling_beta(stock_ret: pd.Series, mkt_ret: pd.Series, window: int = 63) -> pd.Series:
    df = pd.concat([stock_ret, mkt_ret], axis=1, join="inner").dropna()
    df.columns = ["stock", "mkt"]
    cov = df["stock"].rolling(window).cov(df["mkt"])
    var = df["mkt"].rolling(window).var()
    return (cov / var).dropna()
