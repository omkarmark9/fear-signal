# Trading model v2: research log (checkpoints)

Newest entries at the bottom. Each entry gives the time (UTC), what ran,
the result, the commit, and the next step. To resume after a reset, read
this file and `PLAN.md`, then continue from the last "next" line.

## 2026-11-28, start

**State.** v1 is complete:
- dev selection 3065e17;
- holdout b3be90a;
- review fixes through 6ad09dc.

ND1 is at Tier 1 (pilot, rotation); SR is at Tier 0; RF is off.

**Done.** PLAN.md declares:
- the permutation space: 108 Round-1 trials plus 10 v1 trials;
- the statistics: StepM, SPA, PBO/CSCV, DSR;
- the mechanical selection rule.

It was written before any v2 trial ran.

**Next.**
1. Fetch the earnings calendar for 2023-12..2026-11 (background agent).
2. Build `trading/multitest.py` and `trading/explore.py`, with tests.
3. Register the v1 trials.
4. Run Round 1.
