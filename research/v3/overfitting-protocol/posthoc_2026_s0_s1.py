"""POST-HOC DIAGNOSTIC ONLY (2026 is a spent holdout; nothing here is used for
selection). On 2026 both parameter sets are genuinely out of sample: S0 (v1 as
published, fitted by its author on 2023-2025) and S1 as frozen in
config/frozen.json (fitted on 2023-2025). Compare their one-step predictive log
scores under the common censored, zero-inflated likelihood (dev zero rate from
the frozen spec for both), exactly as the dev walk-forward did.
Output: posthoc_2026_s0_s1.json"""
import json, os, sys
os.environ["GIT_OPTIONAL_LOCKS"] = "0"
import numpy as np
sys.path.insert(0, "/tmp/claude-0/v3/overfitting-protocol")
sys.path.insert(0, "/home/user/fear-signal/src")
import spa
from fearsignal import data, features, frozen, hmm, model as mdl

spec = json.load(open("/home/user/fear-signal/config/frozen.json"))
panel = data.load_holdout_panel(frozen.committed_spec_hash())
feat = features.compute(panel)
obs = feat.obs
z0 = spec["dev_zero_rate"]
sm = spec["stock_model"]
s1 = hmm.HMM(hmm.HalfNormalEmission(np.array(sm["s"]), None, "none", False), np.array(sm["A"]))
s0 = mdl.fit_stock_model(mdl.CANDIDATES[0], obs)
ls = {}
for name, m in (("S0", s0), ("S1", s1)):
    fr = mdl.as_scoring_model(m, z0).filter(obs)
    ls[name] = fr.logc
dates = obs.dates
mask = np.asarray(dates >= np.datetime64("2026-01-02"))
nob = obs.eligible[mask].sum(axis=1)
day0 = ls["S0"][mask].sum(axis=1)
day1 = ls["S1"][mask].sum(axis=1)
keep = nob > 0
d = (day0 - day1)[keep] / nob[keep].mean()
per_obs = float((day0 - day1)[keep].sum() / nob[keep].sum())
se = float(np.sqrt(spa.lrv(d, "nw", nw_lags=10)[0] / len(d)))
r_fixed = spa.spa(d, block=10, B=10_000, studentize="fixed")
r_bt = spa.spa(d, block=10, B=10_000, studentize="replicate")
r_bt_rev = spa.spa(-d, block=10, B=10_000, studentize="replicate")
rob = spa.episode_robustness(d)
dd = np.asarray(dates[mask])[keep]
for e in rob["drop_one"]:
    e["date"] = str(dd[e["day_index"]])[:10]
out = {"label": "POST-HOC 2026 DIAGNOSTIC (spent holdout); not evidence for selection",
       "days": int(keep.sum()), "stock_days": int(nob[keep].sum()),
       "S0_minus_S1_per_obs": per_obs, "nw10_se_per_obs": se, "t": per_obs / se,
       "S0_beats_S1_p_fixed": r_fixed["p_c"], "S0_beats_S1_p_bootstrap_t": r_bt["p_c"],
       "S1_beats_S0_p_bootstrap_t": r_bt_rev["p_c"], "episode_robustness": rob,
       "logscore_per_obs": {"S0": float(day0[keep].sum() / nob[keep].sum()),
                            "S1": float(day1[keep].sum() / nob[keep].sum())}}
print(json.dumps(out, indent=1, default=float))
json.dump(out, open("/tmp/claude-0/v3/overfitting-protocol/posthoc_2026_s0_s1.json", "w"), indent=1, default=float)
