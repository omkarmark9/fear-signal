# Trading model research: provenance

The declared candidate set for model 2 (`candidates.md`) came from a design
panel, not from searching backtests.

* **Three designers.** Each worked independently, through its own lens:
  - empirical asset pricing;
  - risk management and portfolio construction;
  - options practice with level-2 approval.
* **What they were told.**
  - They had the same facts about the fear model.
  - They had the evidence against easy profits: the failed return test,
    the sign-flipping event study and the hindsight-chosen universe.
  - They had the constraints: long only, no shorting, level-2 options,
    data from 2024 onward.
  - They knew how selection would work.
* **What they were forbidden.**
  - To compute or look at any return that followed a signal in
    2024–2026.
  - To write anything in the repository.

  They were, however, given the aggregate result of an earlier
  exploratory event study, and two of them cite it. Its outcome windows
  overlap 65% of ND's dev events, and ND's dev edge sits in exactly those
  events. So the declaration was not fitted to outcomes by the
  designers, but it was not made in ignorance of them either (see
  `candidates.md`).
* **What they did.** They read the code and docs and computed event counts
  from signals only. Their scripts compute same-day and trailing
  quantities, never forward returns. These scripts were audited.
* **A judge.** It chose the declared set from the three proposals: at most 5
  families and 16 variants, every rule exactly computable, rejections
  explained.

| file | content |
|---|---|
| `proposals/*.json` | each designer's structured proposal, verbatim |
| `declaration.json` | the judge's output, verbatim |
| `candidates.md` | the declaration as implemented, with the input hashes and the gate rules |

`candidates.md` and the code that implements it (`src/fearsignal/trading/families.py`)
were committed before any backtest of a declared rule was run.
