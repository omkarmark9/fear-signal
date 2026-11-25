"""Size of spa_functional_calibrated on the mc_functional null (does prepivoting
remove the ~2x oversize of spa_functional?). R_mc = 150, outer R = 99, inner B = 199."""
import json, sys
import numpy as np
from scipy import stats as sst
sys.path.insert(0, "/tmp/claude-0/v3/overfitting-protocol")
import spa
from mc_functional import make, corr_rows

rng = np.random.default_rng(78)
raw, cal = [], []
for i in range(150):
    v, ser = make(rng, 363, [0.62] * 4)
    rk = [sst.rankdata(s) for s in ser]; rv = sst.rankdata(v)
    def stat(ix):
        b = corr_rows(rk[0][ix], rv[ix])
        return np.array([corr_rows(r[ix], rv[ix]) - b for r in rk[1:]])
    r = spa.spa_functional_calibrated(stat, 363, block=20.0, B=199, R=99, seed=int(rng.integers(2**31)))
    raw.append(r["p_c_raw"]); cal.append(r["p_c_calibrated"])
    if i % 25 == 24:
        print(i + 1, "raw<.05", np.mean(np.array(raw) < .05), "cal<.05", np.mean(np.array(cal) < .05), flush=True)
out = {"R_mc": len(raw), "raw_rej05": float(np.mean(np.array(raw) < .05)), "cal_rej05": float(np.mean(np.array(cal) < .05)),
       "raw_rej10": float(np.mean(np.array(raw) < .10)), "cal_rej10": float(np.mean(np.array(cal) < .10))}
print(out)
json.dump(out, open("/tmp/claude-0/v3/overfitting-protocol/mc_functional_cal.json", "w"), indent=1)
