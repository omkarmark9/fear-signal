"""Part (b), second pass after the size checks: the same T1/T2 questions with the
bootstrap-t SPA_c and the prepivoted (double-bootstrap) p-value, plus the
leave-one-episode-out check. Output: apply_walkforward_bt.json"""
import json, sys
import numpy as np
sys.path.insert(0, "/tmp/claude-0/v3/overfitting-protocol")
import spa
from apply_walkforward import clean, NAMES

z = np.load("/home/user/fear-signal/outputs/dev/walkforward_daily.npz")
sc = z["scored"]; scale = z["nobs"][sc].mean(); dates = z["dates"][sc]
ls = {k: z[f"logscore__{k}"][sc] / scale for k in NAMES}
fam = ["S2_shared_K3", "S3_state_K3", "S4_state_K2", "S5_state_K4"]
d = np.column_stack([ls[k] - ls["S1_v1_refit"] for k in fam])
out = {"T1_bootstrap_t": spa.spa(d, block=10, B=10_000, studentize="replicate", names=fam)}
out["T1_bootstrap_t_block20"] = spa.spa(d, block=20, B=10_000, studentize="replicate", names=fam)
d01 = ls["S0_v1_published"] - ls["S1_v1_refit"]
out["T2_S0_beats_S1_bootstrap_t"] = spa.spa(d01, block=10, B=10_000, studentize="replicate")
out["T2_S0_beats_S1_calibrated"] = spa.spa_calibrated(d01, block=10, B=999, R=999, seed=3)
out["T2_S0_beats_S1_calibrated_bt"] = spa.spa_calibrated(d01, block=10, B=499, R=499, studentize="replicate", seed=3)
out["T1_calibrated"] = spa.spa_calibrated(d, block=10, B=999, R=999, seed=4, names=fam)
out["T2_S1_beats_S0_bootstrap_t"] = spa.spa(-d01, block=10, B=10_000, studentize="replicate")
rob = {}
for k in ["S0_v1_published"] + fam:
    r = spa.episode_robustness(ls[k] - ls["S1_v1_refit"])
    for e in r["drop_one"]:
        e["date"] = str(dates[e["day_index"]])
    rob[k] = r
out["episode_robustness_vs_S1"] = rob
print(json.dumps(clean({k: ({kk: vv for kk, vv in v.items() if kk in ("T", "p_c", "p_l", "p_u", "t", "p_c_raw",
                                                                        "p_c_calibrated", "null_p_quantiles")}
                            if k.startswith("T") else v) for k, v in out.items()}), indent=1))
json.dump(clean(out), open("/tmp/claude-0/v3/overfitting-protocol/apply_walkforward_bt.json", "w"), indent=1)
