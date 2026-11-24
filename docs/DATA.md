# Data

## Inputs (and nothing else)

Per stock and session: **adjusted close**, **close**, **low**, from the first
session of 2023 (2023-01-03). The signal never reads open, high, volume,
fundamentals, options or any other series. VXN (CBOE Nasdaq-100 volatility
index, FRED `VXNCLS`) is used **only to validate** the finished signal and
never enters it.

| field | vendor basis | use |
|---|---|---|
| `adj_close` | split + dividend adjusted | returns, previous close |
| `close` | split adjusted | adjustment factor `adj_close / close` |
| `low` | split adjusted (same basis as `close`) | adjusted low = `low * adj_close / close` |

Adjusted prices carry no look-ahead: each new ex-date rescales all earlier
prices by one constant, and every quantity used is a ratio of two prices dated
on or before the current session, so the rescaling cancels.

## Hard rules enforced in code

* **Era rule** (`config.assert_era`): rows before 2023-01-03 are dropped before
  any computation; returns, scales and filters warm up inside the era.
* **Holdout lock** (`config.assert_dev`, `data.load_dev_panel`): development
  code reads `data/raw/dev.parquet` truncated at 2025-12-31. The 2026 holdout
  lives in a separate file that is downloaded only after the model spec is
  frozen and committed, and is readable only through
  `data.load_holdout_panel(<sha256 of config/frozen.json>)`.

## Vendor and reproducibility

Bars come from Yahoo Finance via `yfinance` (`auto_adjust=False`). Raw files
are git-ignored; `data/manifest_<name>.json` commits row counts, tickers,
date span and a SHA-256 of the canonical CSV (6 decimals), and
`fearsignal.data.verify(name)` checks a local copy against it. Vendor
adjusted closes are revised retroactively after later dividends; such
revisions change the hash but, by the ratio argument above, not the signal
beyond floating-point noise.

## Universe

134 US-listed AI value-chain stocks in 11 segments (`fearsignal/universe.py`).
Two names were dropped because the vendor no longer serves their history
under the ticker (CFLT, PSTG). Listing overrides, each set by inspecting the
vendor series:

| ticker | trade_start | reason |
|---|---|---|
| OKLO | 2024-05-10 | de-SPAC; earlier rows are the AltC shell (flat ~$10-18, -77% "return" on the merger date) |
| TLN | 2024-07-10 | Nasdaq listing; earlier rows are thin OTC prints (many sessions with low = close) |
| SERV | 2024-04-18 | Nasdaq uplisting; earlier rows are OTC (first row prints a -150% log "return") |
| GEV | 2024-04-02 | regular-way trading after the GE spin-off |
| SNDK | 2025-02-24 | regular-way trading after the WDC spin-off |

Later listings (ARM, ALAB, TEM, NBIS, WRD, PONY, CRWV, ...) join automatically
once they have 126 in-era returns. Large dividend ex-dates (ASX, HIMX) are the
reason the adjusted low is needed: on those dates the raw low sits far below
the raw previous close with no economic loss.
