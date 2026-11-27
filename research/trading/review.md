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
| ROB-5 | SR's 2026 per-event gain is two news jumps (APP +70%, SIMO +54%). Without them the per-event mean is −0.06%; the median is −0.06%; the clustered t is 0.5–0.8. | Reported. The calendar-time figure is stronger (see CL-1). Either way 2026 cannot select SR; a new version needs its own declaration and a prospective test. |
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

## Claims checker

It checked every number and claim in the docs and commit messages against
the outputs, and reproduced them.

| id | finding | disposition |
|---|---|---|
| CL-1 (major) | "Two news jumps" explains only SR's per-event mean. Its 2026 calendar-time gain survives removing them (+14.2 bp/day, t 1.45) and sits at about the 98th percentile against random names. 90% of it comes from about 33 thin sessions. | Fixed. Both statistics are reported; 2026 still cannot select SR. |
| CL-2 (major) | "The later big winners hurt ND" rests on one cut with t −0.47, and other cuts disagree. | Removed. The a-priori survivorship argument stays, labelled untestable within the universe. |
| CL-3 (major) | ND's dev events that the earlier event study never touched earn +0.51% (t 0.24); the overlapping ones +3.31% (t 1.97). | Fixed. The section is titled "adopted on dev, marginally", and the untouched-subset figures are shown. |
| CL-4 | The protocol section said "ten minor issues"; the audit found 15, 4 of them major. | Fixed. |
| CL-5 | A failed declared prediction was omitted: the second-half share was 30% against at least 40%, and −24% in 2026. | Added. |
| CL-6 | "It needs the news gap" overstated N3: the difference has t 0.57. | Reworded. |
| CL-7 | Some calendar-time statistics were unlabelled, and the 1/H p of 0.057 was missing from the README. | Fixed. |
| CL-8 | Sizing and option choice were described as live, but trading-live does not call them. | Stated plainly. ND's covered-call alternative is not offered. |
| CL-9 | Two provenance passages contradicted the event-study caveat. | Qualified. |
| CL-10 | The Tier-3 row showed SR's ticket conditions for every family. | Split by family. |
| CL-11 | The live log could be overwritten, and no prospective-read rule exists for model 2. | The log is append-only; the missing rule is stated. |

**Skeptic votes on ROB-1.** One skeptic confirmed it against the pre-fix
code. The other refuted it as "already fixed at HEAD", which is
consistent with the fix.
