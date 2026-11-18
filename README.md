# Team-05 Red Team

A red-team analysis of the ElasticML MS3 specification and implementation: what the specification guarantees, what the implementation does, where the two differ, and which guarantees, including every performance guarantee, the specification never makes.

| Folder | Contents |
|---|---|
| [`jepsen-report/`](jepsen-report/) | **Jepsen-style report**, structured after Jepsen's published analyses: background, claimed guarantees, test design, safety, liveness, performance (including the performance guarantees the specification doesn't make), claims against evidence, and discussion. Figures are in `jepsen-report/figures/`. |
| [`gap-analysis/`](gap-analysis/) | Full specification gap analysis: per-property verdicts, every finding with code references, claim-by-claim checks, unstated values, observations, and the round-3 performance tables |
| [`tests/`](tests/) | In-process red-team tests, Paxos schedule fuzzer, and performance calibration (pytest) |
| [`hokea_tests/`](hokea_tests/) | Red-team, nemesis, churn and straggler tests that run on a local HOKEA Docker cluster |
| [`logs/`](logs/) | Raw logs from every run |
| [`scripts/`](scripts/) | `plot_perf.py`, which regenerates the report's figures from `logs/14_hokea_perf.log` |

## Headline results

**Safety and liveness**
- **28 confirmed findings** (RT-01 to RT-28), each backed by a strict-xfail test that asserts the specification's requirement. **14 reproduce on the live HOKEA cluster.**
- Violated: S1 (at-most-once), S2 (one value per slot, under a single-voter disk rollback that the specification permits), S3 (default mode), S4 (at reducers), S5, S7 (dedup state), L1, L4 (the published profile has no backup reducer).
- Held: S6, S8, and S2/S1 under the stated assumptions. That covers 12 fuzzed in-process schedules, plus 1,081 updates on HOKEA under random partitions and 25% packet loss.

**Performance.** The specification makes no performance guarantees, so we tested against stated, reasonable assumptions:

| Scenario (6 workers, published accuracy profile, equal seed) | Commit rate vs healthy | Longest stall | Time to 80% accuracy |
|---|---|---|---|
| healthy | 100% (0.728/s) | 4.5 s | 120 s |
| workers leave and rejoin | **61%** | **34.7 s**, after every departure | 240 s |
| churn + voter isolation | **56%** | 34.2 s | 240 s |
| one worker throttled to 0.25 CPU | **41%** | — | — |

- If the class cluster's cores are about as fast as ours, the published 20-worker run spent ~73% of worker time waiting rather than computing.
- One 519 KB request killed a voter.

## Running the tests

The tests import the `elasticml` package and are meant to be dropped into the ElasticML MS3 tree. That source code is not included here. See [the report's reproduction section](jepsen-report/README.md#reproducing-these-results).

No ElasticML code was modified as part of this analysis.
