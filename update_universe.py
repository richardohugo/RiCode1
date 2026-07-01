"""Refresh idx_stocks.csv from Yahoo's screener (all Indonesian equities).

Run occasionally to pick up new listings:  python update_universe.py
The app also accepts any symbol typed manually, so an out-of-date list
only affects the dropdown, not what the tools can analyze.
"""
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

OUT = Path(__file__).parent / "idx_stocks.csv"


def main():
    q = yf.EquityQuery("eq", ["region", "id"])
    rows, offset, total = [], 0, None
    while True:
        r = yf.screen(q, size=250, offset=offset,
                      sortField="intradaymarketcap", sortAsc=False)
        quotes = r.get("quotes", [])
        total = r.get("total", 0)
        if not quotes:
            break
        for it in quotes:
            sym = it.get("symbol", "")
            name = it.get("longName") or it.get("shortName") or ""
            if sym.endswith(".JK") and name:
                rows.append((sym, name.strip()))
        offset += len(quotes)
        print(f"fetched {offset}/{total}")
        if offset >= total:
            break
        time.sleep(0.5)

    df = (pd.DataFrame(rows, columns=["symbol", "name"])
          .drop_duplicates("symbol").sort_values("symbol"))
    df.to_csv(OUT, index=False)
    print(f"saved {len(df)} tickers to {OUT}")


if __name__ == "__main__":
    main()
