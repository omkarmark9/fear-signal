"""The declared method for the read-time earnings calendar (research/trading/PROSPECTIVE.md, R-v2).

v2 built data/raw/earnings_calendar.json the same way, for 2023-11..2026-11.

1. **Fetch.** Only an agent with the Robinhood connector can fetch. For every month from
   2026-08 through 2028-07, it calls ``get_earnings_calendar`` with:
   - ``symbols``: the 134 universe tickers (src/fearsignal/universe.py), in one or more
     batches;
   - ``start_date``: the month's first day;
   - ``days``: the month's length.

   Each call's input and full result are saved as one JSON file,
   {"input": {...}, "payload": {...}}, in data/raw/earnings_prospective_windows/.
2. **Merge** with ``merge()`` below:
   - every month must be covered by successful calls whose symbol sets together hold all
     134 tickers, otherwise nothing is written;
   - rows are taken in month order, keeping the first of each (symbol, report date);
   - conflicting duplicates are listed;
   - the output is data/raw/earnings_calendar_prospective.json, with
     {fetched_utc, source, windows, rows}.
3. **The read** checks the file's data as well as its windows
   (prospective.calendar_covers) and records its SHA-256.

    .venv/bin/python research/trading/earnings_calendar.py merge
"""

from __future__ import annotations

import calendar as cal
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "earnings_prospective_windows"
OUT = ROOT / "data" / "raw" / "earnings_calendar_prospective.json"
FIRST, LAST = (2026, 8), (2028, 7)


def expected_windows(first=FIRST, last=LAST) -> list[tuple[str, int]]:
    out, (y, m) = [], first
    while (y, m) <= last:
        out.append((f"{y}-{m:02d}-01", cal.monthrange(y, m)[1]))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def merge(raw_dir: Path = RAW_DIR, out: Path = OUT, symbols: list[str] | None = None, write: bool = True) -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    from fearsignal import universe
    symbols = symbols or universe.tickers()
    calls = []
    for f in sorted(Path(raw_dir).glob("*.json")):
        doc = json.loads(f.read_text())
        inp, pay = doc.get("input", {}), doc.get("payload") or {}
        if "data" not in pay:
            continue
        calls.append({"start": inp.get("start_date"), "days": inp.get("days"), "symbols": inp.get("symbols") or [],
                      "results": pay["data"].get("results", []), "file": f.name,
                      "mtime": dt.datetime.fromtimestamp(f.stat().st_mtime, dt.timezone.utc)})
    exp = expected_windows()
    missing = []
    for st, days in exp:
        covered = set()
        for c in calls:
            if c["start"] == st and c["days"] == days:
                covered |= set(c["symbols"])
        if not set(symbols) <= covered:
            missing.append((st, sorted(set(symbols) - covered)[:10]))
    rows, seen, conflicts = [], {}, []
    for st, days in exp:
        lo = dt.date.fromisoformat(st)
        hi = lo + dt.timedelta(days=days - 1)
        for c in (c for c in calls if c["start"] == st and c["days"] == days):
            for r in c["results"]:
                if not lo <= dt.date.fromisoformat(r["report"]["date"]) <= hi or r["symbol"] not in c["symbols"]:
                    continue
                key = (r["symbol"], r["report"]["date"])
                if key in seen:
                    if r != seen[key]:
                        conflicts.append({"key": key, "kept": seen[key], "dropped": r})
                    continue
                seen[key] = r
                rows.append(r)
    report = {"calls": len(calls), "windows_expected": len(exp), "missing": missing, "rows": len(rows),
              "conflicting_duplicates": conflicts}
    if missing:
        report["written"] = False
        return report
    if write:
        doc = {"fetched_utc": max(c["mtime"] for c in calls).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "source": "robinhood get_earnings_calendar, monthly windows",
               "windows": [st for st, _ in exp], "rows": rows}
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=1))
        tmp.replace(out)
    report["written"] = bool(write)
    return report


if __name__ == "__main__":
    print(json.dumps(merge(write=len(sys.argv) > 1 and sys.argv[1] == "merge"), indent=1, default=str))
