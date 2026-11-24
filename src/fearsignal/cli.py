"""Command line: the research protocol, one step per command.

    fearsignal download-dev                 bars 2023-01-03 .. 2025-12-31 (+ VXN for validation)
    fearsignal evaluate-dev                 walk-forward model comparison on 2023-2025
    fearsignal freeze --stock-spec NAME     fit on 2023-2025, write config/frozen.json
    fearsignal dev-reference                pre-registered metrics, in-sample on dev
    fearsignal download-holdout             bars and VXN through the last full session
    fearsignal holdout --sha256 HASH        the single frozen run on 2026
    fearsignal latest [--top N]             latest market and stock readings
"""

from __future__ import annotations

import argparse
import json
import urllib.request

from . import data, evaluate, frozen, pipeline

FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VXNCLS&cosd=2023-01-01&coed={end}"


def _fred(end: str, name: str) -> None:
    with urllib.request.urlopen(FRED.format(end=end), timeout=60) as r:
        (data.RAW / name).write_bytes(r.read())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="fearsignal", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download-dev")
    sub.add_parser("evaluate-dev")
    f = sub.add_parser("freeze")
    f.add_argument("--stock-spec", required=True)
    f.add_argument("--n-boot", type=int, default=200)
    sub.add_parser("dev-reference")
    h = sub.add_parser("download-holdout")
    h.add_argument("--end-exclusive", required=True, help="first session NOT to include, e.g. 2026-11-24")
    hr = sub.add_parser("holdout")
    hr.add_argument("--sha256", required=True)
    lt = sub.add_parser("latest")
    lt.add_argument("--top", type=int, default=15)
    a = ap.parse_args(argv)

    if a.cmd == "download-dev":
        data.download("dev", "2023-01-03", "2026-01-01")
        _fred("2025-12-31", "vxn_dev.csv")
    elif a.cmd == "evaluate-dev":
        out = pipeline.OUT / "dev"
        out.mkdir(parents=True, exist_ok=True)
        evaluate.run(out_path=out / "walkforward.json")
    elif a.cmd == "freeze":
        pipeline.freeze(a.stock_spec, n_boot=a.n_boot)
        print("frozen spec sha256:", frozen.spec_hash())
    elif a.cmd == "dev-reference":
        res = pipeline.dev_reference(frozen.load())
        out = pipeline.OUT / "dev" / "dev_reference.json"
        out.write_text(json.dumps(res, indent=2, default=float) + "\n")
    elif a.cmd == "download-holdout":
        # overlap December 2025 so the loader can confirm the two pulls agree
        data.download("holdout", "2025-12-01", a.end_exclusive)
        _fred(a.end_exclusive, "vxn_full.csv")
    elif a.cmd == "holdout":
        pipeline.holdout(a.sha256)
    elif a.cmd == "latest":
        import pandas as pd
        mk = pd.read_csv(pipeline.OUT / "holdout" / "market_fear.csv", index_col=0)
        st = pd.read_csv(pipeline.OUT / "holdout" / "stock_fear.csv.gz")
        last = st["date"].max()
        print(mk.tail(10).to_string())
        print(st[st.date == last].sort_values("Z", ascending=False).head(a.top).to_string(index=False))


if __name__ == "__main__":
    main()
