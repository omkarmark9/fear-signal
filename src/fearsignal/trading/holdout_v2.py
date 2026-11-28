"""The single descriptive 2026 read of v2's StepM-survivors (research/trading/v2/LOG.md).

Declared before the read: no v2 trial passed the full selection rule, so the
StepM-rejected trials are read on 2026 once, descriptively. They stay at
Tier 0 and become the prospective test's pre-registered hypotheses. The read
runs once. Its start is logged before any 2026 outcome is computed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from . import diagnostics as D, explore as X, gates, protocol

OUT = X.REG / "holdout_2026.json"
LOG = X.REG / "HOLDOUT_RUNLOG.md"


def main(rerun_reason: str | None = None) -> dict:
    if (OUT.exists() or LOG.exists()) and not rerun_reason:
        raise RuntimeError("the v2 2026 read has already run; pass a rerun reason")
    sel = json.loads((X.REG / "selection.json").read_text())
    with open(LOG, "a") as fh:
        when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        fh.write(f"| {when} | read started | trials: {len(sel['stepm_rejected'])} | {rerun_reason or 'first read'} |\n")
    recs, _ = X.load_registry()
    ctx = X.Ctx()
    out = {}
    for cid in sel["stepm_rejected"]:
        r = recs[recs.id == cid].iloc[0]
        tr = X.Trial(r["family"], r["kind"], r["params"], int(r["round"]))
        out[cid] = D.event_diagnostics(tr, ctx, *gates.HOLDOUT, n_placebo=200, seed=1)
    out["_meta"] = {"window": [str(d.date()) for d in gates.HOLDOUT], "status": "descriptive (Tier 0)",
                    "inputs": protocol.input_manifest()}
    protocol.write_json(OUT, out)
    with open(LOG, "a") as fh:
        when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        fh.write(f"| {when} | read completed | 2026-11-27 | {rerun_reason or 'first read'} |\n")
    return out
