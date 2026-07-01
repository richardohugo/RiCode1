# RiCode — live IDX risk tools

Three Streamlit tools over live Yahoo Finance data, in the RiCode house style.

- **Value at Risk** — historical, EWMA (RiskMetrics), parametric (normal) and
  Cornish-Fisher VaR, plus CVaR / expected shortfall, all shown on one scale;
  a return-distribution chart, and a rolling-VaR line with an out-of-sample
  breach backtest (Kupiec test).
- **Monte Carlo** — thousands of simulated price paths with a probability fan
  (P5–P95), terminal distribution, probability of profit, and simulated
  VaR/CVaR. Two engines: a smooth model with antithetic variance reduction,
  or a bootstrap that resamples the stock's actual trading days.
- **Single-Factor Model** — market-model regression giving beta, alpha, R², the
  systematic-vs-specific variance split, a fitted scatter, and rolling beta.
- **Reader's Guide** — every metric explained in plain language for readers with
  no finance background.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)

1. Push this folder to a GitHub repo.
2. On [share.streamlit.io](https://share.streamlit.io), create an app pointing at `app.py`.
3. It installs `requirements.txt` automatically. No secrets or API keys needed.

## Notes on the data

- Symbols follow Yahoo Finance. IDX tickers use the `.JK` suffix (`BBCA.JK`,
  `BBRI.JK`, `TLKM.JK`), and prices are in **IDR** despite Yahoo's `$` labels — set
  the currency label in the sidebar accordingly.
- The default factor benchmark is `^JKSE`, the Jakarta Composite. Swap in any
  benchmark you like (`^GSPC`, a sector ETF, another stock).
- Prices are auto-adjusted (splits/dividends). Returns are simple daily returns.

## What it is not

Educational tooling, not financial advice. VaR is a threshold, not a worst case —
read the CVaR alongside it, and watch the breach backtest. The Monte Carlo fan is
a range, not a forecast: it knows nothing the price history doesn't. The factor
model is a single-factor OLS: beta estimates tightly, alpha rarely does. All of it
is backward-looking.

## Files

```
app.py                   Streamlit UI + theming
ricode_core.py           data access + risk/simulation/factor math (pure, testable)
requirements.txt
.streamlit/config.toml   theme
```
