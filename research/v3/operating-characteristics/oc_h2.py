"""H2' operating characteristic: P(logistic calibration slope estimate in [0.8, 1.25]).

Normal approximation: slope_hat ~ N(b, se_N^2), se_N = se_ref * sqrt(n_ref / N),
with the day-clustered s.e. from
  * 2026 holdout: se_ref = 0.118 at n_ref = 225 days
  * dev walk-forward: se_ref = 0.188 at n_ref = 378 days.
Exact under the approximation (no Monte Carlo error). The full H2' rule also
needs Brier skill > 0, so P(pass) <= the numbers here.
Output: oc_h2.json
"""
import json

import numpy as np
from scipy import optimize
from scipy.stats import norm

LO, HI = 0.8, 1.25
REFS = {"2026": (0.118, 225), "dev": (0.188, 378)}
NS = [300, 400]
SLOPES = [1.0, 0.9, 0.8, 0.7]


def p_band(b, se):
    return float(norm.cdf((HI - b) / se) - norm.cdf((LO - b) / se))


out = {"band": [LO, HI], "refs": {k: {"se": v[0], "n_days": v[1]} for k, v in REFS.items()}, "table": {}}
for lab, (se0, n0) in REFS.items():
    for n in NS:
        se = se0 * np.sqrt(n0 / n)
        row = {"se": float(se)}
        for b in SLOPES:
            row[f"P_band|b={b}"] = p_band(b, se)
        bmax = optimize.minimize_scalar(lambda b: -p_band(b, se), bounds=(0.8, 1.25), method="bounded").x
        row["max_P_band"] = p_band(bmax, se)
        # true slope (below the band centre) at which P(band) = 0.5 / 0.8; None if never reached
        for tgt in (0.5, 0.8):
            row[f"b_low_where_P={tgt}"] = (float(optimize.brentq(lambda b: p_band(b, se) - tgt, 0.3, bmax))
                                           if row["max_P_band"] > tgt else None)
        out["table"][f"{lab}:N={n}"] = row
        print(lab, n, {k: (None if v is None else round(v, 4)) for k, v in row.items()})
with open("/tmp/claude-0/v3/oc/oc_h2.json", "w") as fh:
    json.dump(out, fh, indent=1)
