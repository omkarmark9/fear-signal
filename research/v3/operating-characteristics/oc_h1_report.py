"""Tables from oc_h1_cells/*.npz (written by oc_h1.py). Output: oc_h1_report.json + stdout.

Per cell: P(pass) of the pre-registered rule (2.5% bootstrap percentile > 0.30),
CI coverage and tails, and two size-repairing alternatives evaluated on the
same replications:
  * R97.5: lower 1.25% percentile > 0.30 (i.e. a 97.5% two-sided percentile CI)
  * Rc:    2.5% percentile > c*, c* = the 97.5% quantile across boundary
           replications of the lower bound (pooled worst case over DGPs at N)
Calibration at the boundary: alpha* (lower percentile level giving 2.5%,
linear interpolation over the stored levels) and c* with a bootstrap s.e.
"""
import glob
import json
import os

import numpy as np

D = "/tmp/claude-0/v3/oc"
cfg = json.load(open(f"{D}/oc_h1.json"))
pops = {(c["dgp"], c["target_rho"], c["N"]): c["pop_rho"] for c in cfg["cells"]}
cells = {}
for f in sorted(glob.glob(f"{D}/oc_h1_cells/*.npz")):
    if f.endswith("_quick.npz"):
        continue
    b = os.path.basename(f)[:-4]
    dgp, r, n = b.split("_")
    cells[(dgp, float(r[3:]), int(n[1:]))] = dict(np.load(f))
QLEV = cfg["settings"]["qlev"]
iq = {a: i for i, a in enumerate(QLEV)}


def p_se(x):
    p = float(np.mean(x))
    return p, float(np.sqrt(p * (1 - p) / len(x)))


rng = np.random.default_rng(1)
calib = {}
for (dgp, r, n), z in cells.items():
    if r != 0.30:
        continue
    lo = z["q"][:, iq[0.025]]
    cstar = float(np.quantile(lo, 0.975))
    bs = [np.quantile(rng.choice(lo, len(lo)), 0.975) for _ in range(2000)]
    rates = np.array([np.mean(z["q"][:, iq[a]] > 0.30) for a in QLEV if a < 0.5])
    levs = np.array([a for a in QLEV if a < 0.5])
    astar = float(np.interp(0.025, rates, levs)) if rates.min() <= 0.025 <= rates.max() else None
    calib[f"{dgp}:N={n}"] = {"c_star": cstar, "c_star_se": float(np.std(bs)), "alpha_star": astar,
                             "rates_by_alpha": dict(zip([f"{a:g}" for a in levs], rates.tolist()))}
cpool = {n: max(v["c_star"] for k, v in calib.items() if k.endswith(f"N={n}"))
         for n in (300, 400) if any(k.endswith(f"N={n}") for k in calib)}

rows = []
for (dgp, r, n), z in sorted(cells.items(), key=lambda kv: (kv[0][2], kv[0][1], kv[0][0])):
    q, est = z["q"], z["est"]
    pop = pops[(dgp, r, n)]
    lo, hi = q[:, iq[0.025]], q[:, iq[0.975]]
    row = {"dgp": dgp, "rho": r, "N": n, "R": len(est), "pop_rho": pop}
    row["P_pass"], row["P_pass_se"] = p_se(lo > 0.30)
    row["P_pass_R97.5"], row["P_pass_R97.5_se"] = p_se(q[:, iq[0.0125]] > 0.30)
    row["P_pass_Rc"], row["P_pass_Rc_se"] = p_se(lo > cpool[n]) if n in cpool else (None, None)
    row["coverage95"], row["coverage95_se"] = p_se((lo <= pop) & (pop <= hi))
    row["miss_low_side(lo>true)"], _ = p_se(lo > pop)
    row["miss_high_side(hi<true)"], _ = p_se(hi < pop)
    row["mean_rho_hat"], row["sd_rho_hat"] = float(est.mean()), float(est.std(ddof=1))
    row["boot_sd_ratio"] = float(z["bsd"].mean() / est.std(ddof=1))
    row["boot_sd_ratio_atanh"] = float(z["bzsd"].mean() / np.arctanh(est).std(ddof=1))
    rows.append(row)
out = {"calibration_at_boundary": calib, "c_star_pooled_worst": cpool, "rows": rows}
json.dump(out, open(f"{D}/oc_h1_report.json", "w"), indent=1)

print("calibration", json.dumps(calib, indent=1), "\nc* pooled", cpool)
hdr = ("dgp", "rho", "N", "R", "pop", "P_pass", "se", "P97.5", "se", "Pc*", "se", "cov95", "se",
       "lo>true", "hi<true", "mean", "sd", "bsd/sd", "bsdz/sdz")
print(" ".join(f"{h:>8}" for h in hdr))
for w in rows:
    vals = (w["dgp"], w["rho"], w["N"], w["R"], w["pop_rho"], w["P_pass"], w["P_pass_se"], w["P_pass_R97.5"],
            w["P_pass_R97.5_se"], w["P_pass_Rc"], w["P_pass_Rc_se"], w["coverage95"], w["coverage95_se"],
            w["miss_low_side(lo>true)"], w["miss_high_side(hi<true)"], w["mean_rho_hat"], w["sd_rho_hat"],
            w["boot_sd_ratio"], w["boot_sd_ratio_atanh"])
    print(" ".join(f"{v:>8.4f}" if isinstance(v, float) else f"{str(v):>8}" for v in vals))
