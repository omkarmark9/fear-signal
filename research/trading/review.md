# Results review of the trading model

This review ran after the single holdout read, as an independent workflow
with three reviewers:
- an independent recomputation that does not use the engine;
- an adversarial overfitting and robustness critic;
- a claims checker for the docs and commit messages.

It could run anything except the one-shot commands. Its findings changed
the live ND action and the description of every result.

## Overfitting and robustness critic

**Reproduced exactly from the hash-locked inputs:**
- ND1: dev +17.49 bp/day (t 2.543); 2026 +6.38 bp/day (t 0.609).
- SR1: dev −3.41 bp/day (t −0.859); 2026 +19.82 bp/day (t 2.089).
- RF1: dev variance ratio 0.8418; 2026 0.9922.
- ND's calibrated SPA_c p: 0.020 at B = R = 999.

| id | finding | disposition |
|---|---|---|
| ROB-1 (blocker) | The ND payoff that was tested is a rotation: sell the name and hold β × the AI basket. The live action said to sell to cash, which earned nothing in either sample (dev −0.52% per event, t −0.32; 2026 +0.42%). The live output also said to sell for windows already open, a mid-window start that was never tested. | Fixed. Live ND now says: rotate at the signal's next open, and "do not add" for windows already open. |
| ROB-2 | ND's 2026 "agreement" is uninformative. 40% of random-name draws agree; one lone position (INOD) supplies 4.1 of the 6.4 bp/day; a one-day delay gives −8.5 bp/day; one earnings cluster supplies 88% of the per-event sum; N2 fails in 2026. | Reported. Tier 1 is the declared rule mechanically applied, and it is described as a pilot on dev evidence only. |
| ROB-3 | ND's dev adoption was marginal. The per-event t is 1.6–1.8 clustered. Removing 2 of 145 events gives a calibrated SPA p of 0.136. The SPA on the 1/H series gives 0.057. | Reported next to the gate figures. |
| ROB-4 | RF's dev pass rests on April 2025: without it the variance ratio is 0.954, and the bootstrap gives P(ratio > 0.85) = 0.53. The 2026 failure matches dev without April. | Reported. |
| ROB-5 | SR's 2026 gain is two news jumps (APP +70%, SIMO +54%). Without them the per-event mean is −0.06%; the median is −0.06%; the clustered t is 0.5–0.8. | Reported. There is no basis for a new SR version. |
| ROB-6 | The exploratory event study had already seen outcomes overlapping 65% of ND's dev event windows, and ND's edge sits in those events. | Added to the provenance in `candidates.md` and to the caveats. |

**Robustness that held:**
- ND on dev against random names on the same days: 99.3rd percentile.
- ND on dev at 25 and 50 bps: t 2.31 and 1.93.
- ND on dev with β = 1 or unshrunk OLS β: t 2.74 and 2.41.
- ND on dev with entry at the t+2 open or the t+1 close: t 2.10 and 2.05.
- ND leave-one-cluster-out: minimum t 1.98.
- ND's same-stock-adjusted excess: +1.64% (dev), +1.27% (2026).
- SR's dev failure, under every variation.
- RF's 2026 failure, at γ = 2, 5, 10 and 20, at 25 bps, with a delay, and against the causal static mix.

## Independent recomputation

Run with trading/engine.py, gates.py and backtest.py replaced by stub modules
that raise on use, so none of their code executed. It reproduced exactly:
- every event list, for SR1–SR4 and ND1–ND3;
- every dev and 2026 mean (differences around 1e-17), every daily CSV
  (within rounding), the robustness runs, the signatures and the betas
  (6e-15);
- RF's exposure path and statistics;
- ND's calibrated SPA_c p (0.0200).

| id | finding | disposition |
|---|---|---|
| RC-1 | The docs put the calendar-time t next to per-event means. Per event, ND's t is 1.80 by day and 1.57 by episode on dev, against a calendar-time 2.54. SR's 2026 per-event t is 0.81 / 0.51, against 2.09. | Fixed. Both statistics are now labelled. |
| RC-2 | ND's pass on gate (b) hinges on single names. Dropping ALAB (2025-09-24) gives calibrated p 0.107; AEHR 0.071; MU 0.067; BIDU 0.060. Gate (a) holds for every removal. | Added to TRADING.md and the README. |
