# v3 course-correction round: raw provenance

Everything the five-agent round produced is kept here as it was produced, so
every claim in `docs/V3.md` can be traced back. The round was four
diagnosis agents, one per v2 weak spot, plus one adversarial critic.

| folder | what |
|---|---|
| `market-aggregation/` | why v2's market index lost to v1's on 2026 changes; candidates C1–C4 |
| `tail-calibration/` | why the tail forecast's slope fell to 0.80 in 2026; candidates C0–C3 |
| `volatility-increment/` | why the frozen HAR failed in 2026; candidates C1–C3 |
| `overfitting-protocol/` | the selection protocol (SPA gate, Holm, prospective holdout), its simulations and the power analysis |
| `critic/` | the critic's independent re-computations |
| `round_findings_and_critique.json` | every agent's structured report and the critic's verdicts |

Each `candidates.md` was written before its candidates were scored. The
critic checked this from file times for the market and volatility issues.
For the tail issue the check could not be done, because an addendum was
appended later; that addendum's candidate (C3) was quarantined as
non-adoptable. SHA-256 of the declarations as committed:

```
159e7b874cad70050a138b2e41f7b3b2266314b8302e7f8d2e97d7dca15ccfe5  market-aggregation/candidates.md
5aefb329ab135666d03c30ea349fd4b99c8b950b04273f438644f7202b6e1a60  tail-calibration/candidates.md
d2c6cdf8dfdc1af7e02354702a81ff758effedc0cb129e7fc7a98ce429b83511  volatility-increment/candidates.md
462ce7455fe6957d5a892b022c8dec67a9f56b6750353259cbfcc460f54bc653  overfitting-protocol/candidates.md
```

Scripts reference the scratch layout they ran in (`/tmp/claude-0/v3/...`,
including the shared fold-model cache). They are records, not package code.
The SPA implementation was promoted to `src/fearsignal/spa.py`, and its tests
to `tests/test_spa.py`.
