"""Command line: the research protocol, one step per command.

    fearsignal download-dev                 bars 2023-01-03 .. 2025-12-31 (+ VXN for validation)
    fearsignal evaluate-dev                 walk-forward model comparison on 2023-2025
    fearsignal freeze --stock-spec NAME     fit on 2023-2025, write config/frozen.json
    fearsignal dev-reference                pre-registered metrics, in-sample on dev
    fearsignal download-holdout             bars and VXN through the last full session
    fearsignal holdout --sha256 HASH        the single frozen run on 2026 (refuses a second run)
    fearsignal figures                      docs/figures from the holdout outputs
    fearsignal posthoc-2026                 labelled post-hoc diagnostic of the v3 market scale
    fearsignal freeze-v3                    re-estimate v3 on 2023-01-03..2026-11-23
    fearsignal v3-reference                 in-sample v3 metrics, signal history and F5 diagnostic
    fearsignal download-prospective --end-exclusive D   bars + VXN after 2026-11-23
    fearsignal prospective --sha256 HASH    the single pre-registered v3 read (400 sessions)
    fearsignal live [--days N]              frozen v3 on the latest bars (display only)
    fearsignal forecast --target D          next-session forecasts, saved before the session
    fearsignal score-forecast --target D    score a saved forecast; append to the scoreboard
    fearsignal latest [--top N]             latest market and stock readings

    trading model (model 2; docs/TRADING.md)
    fearsignal trading-data --end-exclusive D   refresh LIVE OHLCV (backtest inputs are hash-locked)
    fearsignal trading-placebo [--reps N]   placebo calibration of the selection gate
    fearsignal trading-dev                  dev selection 2024-2025 (once; writes config/trading_selected.json)
    fearsignal trading-holdout --sha256 H   the single 2026 read of the selection
    fearsignal trading-live [--days N]      both models on recent sessions, and the next session's tickets
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
    hr.add_argument("--rerun-reason", default=None,
                    help="required to run the holdout again; recorded in outputs/holdout/RUNLOG.md")
    sub.add_parser("figures")
    sub.add_parser("posthoc-2026")
    fv = sub.add_parser("freeze-v3")
    fv.add_argument("--n-boot", type=int, default=200)
    sub.add_parser("v3-reference")
    dp = sub.add_parser("download-prospective")
    dp.add_argument("--end-exclusive", required=True)
    pr = sub.add_parser("prospective")
    pr.add_argument("--sha256", required=True)
    pr.add_argument("--rerun-reason", default=None)
    lv = sub.add_parser("live")
    lv.add_argument("--days", type=int, default=7)
    lv.add_argument("--top", type=int, default=10)
    lv.add_argument("--end-exclusive", default=None,
                    help="first session NOT to include (default: today's UTC date)")
    lv.add_argument("--source", choices=["yahoo", "robinhood"], default="yahoo",
                    help="robinhood: completed sessions from connector bars saved in "
                         "data/raw/robinhood/*.json (no download here)")
    fc = sub.add_parser("forecast")
    fc.add_argument("--target", required=True)
    fc.add_argument("--source", choices=["yahoo", "robinhood"], default="robinhood")
    sf = sub.add_parser("score-forecast")
    sf.add_argument("--target", required=True)
    sf.add_argument("--source", choices=["yahoo", "robinhood"], default="robinhood")
    lt = sub.add_parser("latest")
    lt.add_argument("--top", type=int, default=15)
    td = sub.add_parser("trading-data")
    td.add_argument("--end-exclusive", required=True)
    tp = sub.add_parser("trading-placebo")
    tp.add_argument("--reps", type=int, default=300)
    tp.add_argument("--workers", type=int, default=4)
    tv = sub.add_parser("trading-dev")
    tv.add_argument("--rerun-reason", default=None)
    th = sub.add_parser("trading-holdout")
    th.add_argument("--sha256", required=True)
    th.add_argument("--rerun-reason", default=None)
    tl = sub.add_parser("trading-live")
    tl.add_argument("--days", type=int, default=6)
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
        frozen.committed_spec_hash()   # no 2026 bar is fetched before the spec is committed
        # overlap December 2025 so the loader can confirm the two pulls agree
        data.download("holdout", "2025-12-01", a.end_exclusive)
        _fred(a.end_exclusive, "vxn_full.csv")
    elif a.cmd == "holdout":
        pipeline.holdout(a.sha256, rerun_reason=a.rerun_reason)
    elif a.cmd == "figures":
        from . import figures
        fig_dir = pipeline.ROOT / "docs" / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        spec = frozen.load()
        vxn = evaluate.load_vxn(data.RAW / "vxn_full.csv", dev_only=False)
        figures.market_figure(pipeline.OUT / "holdout" / "market_fear.csv", vxn,
                              spec["holdout_start"], fig_dir / "market_fear.png")
        figures.reliability_figure(pipeline.OUT / "holdout" / "holdout_results.json",
                                   fig_dir / "tail_reliability.png")
    elif a.cmd == "posthoc-2026":
        (pipeline.OUT / "v3").mkdir(parents=True, exist_ok=True)
        res = pipeline.posthoc_2026_market_scale()
        (pipeline.OUT / "v3" / "posthoc_2026_market_scale.json").write_text(
            json.dumps(res, indent=2, default=float) + "\n")
    elif a.cmd == "freeze-v3":
        pipeline.freeze_v3(n_boot=a.n_boot)
        print("frozen v3 sha256:", frozen.spec_hash(frozen.FROZEN_V3_PATH))
    elif a.cmd == "v3-reference":
        res = pipeline.v3_reference(json.loads(frozen.FROZEN_V3_PATH.read_text()))
        (pipeline.OUT / "v3" / "estimation_reference.json").write_text(
            json.dumps(res, indent=2, default=float) + "\n")
        (pipeline.OUT / "v3" / "posthoc_2026_F5.json").write_text(
            json.dumps(pipeline.posthoc_2026_f5(), indent=2, default=float) + "\n")
        mk, st = pipeline.v3_insample_outputs()
        mk.round(4).to_csv(pipeline.OUT / "v3" / "market_fear_v3_insample.csv")
        num = st.select_dtypes("number").columns
        st.assign(**st[num].round(4)).to_csv(pipeline.OUT / "v3" / "stock_fear_v3_insample.csv.gz",
                                              index=False, compression="gzip")
    elif a.cmd == "download-prospective":
        frozen.committed_spec_hash(frozen.FROZEN_V3_PATH)   # nothing after 2026-11-23 before v3 is committed
        # overlap November 2026 so the loader can align the adjustment basis
        data.download("prospective", "2026-11-02", a.end_exclusive)
        _fred(a.end_exclusive, "vxn_prospective.csv")
    elif a.cmd == "prospective":
        pipeline.prospective(a.sha256, rerun_reason=a.rerun_reason)
    elif a.cmd == "live":
        import pandas as pd
        if a.source == "yahoo" or a.end_exclusive:
            end = a.end_exclusive or pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
            data.download("live", "2026-11-02", end, manifest=False)
        mk, st, rep = pipeline.live(a.days, source=a.source, with_report=True)
        print("data:", {k: v for k, v in rep.items() if k != "history_bars_filled"},
              "| history bars filled:", rep.get("history_bars_filled", []))
        cols = ["M", "M_lo", "M_hi", "M_pct", "F_M", "band", "high_conviction", "breadth", "n_stocks"]
        print(mk[cols].round(2).to_string())
        for d in mk.index[-1:]:
            day = st[st["date"] == d].sort_values("Z", ascending=False).head(a.top)
            print(f"\nmost afraid stocks on {d.date()}")
            print(day[["ticker", "x", "Z", "Z_lo", "Z_hi", "band", "q_next", "I"]].round(2).to_string(index=False))
    elif a.cmd == "forecast":
        out = pipeline.OUT / "live"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"forecast_{a.target}.json"
        if path.exists():
            raise SystemExit(f"{path} exists; a forecast is written once, before its session")
        path.write_text(json.dumps(pipeline.live_forecast(a.target, a.source), indent=2, default=float) + "\n")
        print("wrote", path)
    elif a.cmd == "score-forecast":
        out = pipeline.OUT / "live"
        fcd = json.loads((out / f"forecast_{a.target}.json").read_text())
        res = pipeline.live_score(fcd, a.source)
        res["source"] = a.source
        (out / f"score_{a.target}.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
        board = out / "SCOREBOARD.md"
        new = not board.exists()
        with open(board, "a") as fh:
            if new:
                fh.write("# Live forecast scoreboard\n\n| session | count (90% range) | M_pct (90% range) | Brier model / clim | flagged |\n|---|---|---|---|---|\n")
            fh.write(f"| {res['target_session']} | {res['realized_count_ge_3sigma']} ({res['count_range_90'][0]:.0f}-{res['count_range_90'][1]:.0f}) | "
                     f"{res['realized_m_pct']:.2f} ({res['m_pct_range_90'][0]:.2f}-{res['m_pct_range_90'][1]:.2f}) | "
                     f"{res['brier_model']:.4f} / {res['brier_climatology']:.4f} | {'yes' if res['flagged'] else 'no'} |\n")
        print(json.dumps(res, indent=2, default=float))
    elif a.cmd == "latest":
        import pandas as pd
        mk = pd.read_csv(pipeline.OUT / "holdout" / "market_fear.csv", index_col=0)
        st = pd.read_csv(pipeline.OUT / "holdout" / "stock_fear.csv.gz")
        last = st["date"].max()
        print(mk.tail(10).to_string())
        print(st[st.date == last].sort_values("Z", ascending=False).head(a.top).to_string(index=False))
    elif a.cmd.startswith("trading-"):
        _trading(a)


def _trading(a) -> None:
    import pandas as pd
    from .trading import backtest, market_data, placebo, protocol
    out = protocol.OUT
    if a.cmd == "trading-data":
        # live refresh only: the declared backtest inputs are hash-locked and never rewritten here
        market_data.download(end_exclusive=a.end_exclusive, live=True)
        print("wrote", market_data.LIVE_PATH)
    elif a.cmd == "trading-placebo":
        commit = protocol.declaration_commit()
        protocol.check_inputs()
        rf, prices = backtest.load_inputs()
        res = placebo.calibrate(backtest.placebo_spec(rf, prices), reps=a.reps, workers=a.workers)
        res.update(declaration_commit=commit, inputs=protocol.input_manifest())
        protocol.write_json(out / "placebo.json", res)
        print(json.dumps(res, indent=1))
    elif a.cmd == "trading-dev":
        commit = protocol.declaration_commit()
        protocol.check_inputs()
        protocol.check_dev_once(a.rerun_reason)
        dev = backtest.run_dev()
        d = out / "dev"
        d.mkdir(parents=True, exist_ok=True)
        for k, f in dev["families"].items():
            f["daily"].to_csv(d / f"daily_{k}.csv", float_format="%.8g")
        rfe = dev["RF"]
        pd.DataFrame({n: rfe["variants"][n]["base"]["d"] for n in rfe["variants"]}).to_csv(
            d / "daily_RF_d.csv", float_format="%.8g")
        pd.DataFrame({n: rfe["variants"][n]["E"] for n in rfe["variants"]}).to_csv(d / "RF_exposure.csv")
        res = backtest.strip(dev)
        res["RF"]["variants"] = {n: {k: backtest.strip(v) for k, v in rv.items() if k != "E"}
                                 for n, rv in rfe["variants"].items()}
        res["RF"].pop("sigma_hat", None)
        res["declaration_commit"] = commit
        protocol.write_json(d / "dev_results.json", res)
        protocol.write_json(protocol.SELECTED, backtest.selection_from_dev(dev, commit))
        protocol.log_dev(commit, a.rerun_reason)
        print(json.dumps(json.loads(protocol.SELECTED.read_text()), indent=1))
    elif a.cmd == "trading-holdout":
        protocol.check_inputs()
        protocol.check_holdout_unlock(a.sha256, a.rerun_reason)
        sel = json.loads(protocol.SELECTED.read_text())
        res = backtest.run_holdout(sel)
        h = out / "holdout"
        h.mkdir(parents=True, exist_ok=True)
        for k, v in res.items():
            if "daily" in v:
                v["daily"].to_csv(h / f"daily_{k}.csv", float_format="%.8g")
                v["trades"].to_csv(h / f"trades_{k}.csv", index=False, float_format="%.6g")
        clean = {k: backtest.strip(v) for k, v in res.items()}
        clean["_meta"] = {"selection_sha256": a.sha256, "inputs": protocol.input_manifest()}
        protocol.write_json(h / "holdout_results.json", clean)
        protocol.log_holdout(a.sha256, "2026-11-27", a.rerun_reason)
        print(json.dumps(clean, indent=1))
    elif a.cmd == "trading-live":
        from .trading import live
        live.main(a.days)


if __name__ == "__main__":
    main()
