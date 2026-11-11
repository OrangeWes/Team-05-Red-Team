# Team-05 Red Team

A red-team analysis of the ElasticML MS3 specification and implementation: what the specification guarantees, what the implementation does, and where the two differ.

| Folder | Contents |
|---|---|
| [`jepsen-report/`](jepsen-report/) | **Jepsen-style report** covering all 27 confirmed issues (RT-01 to RT-27): architecture, workloads, nemeses, histories, verdicts and recommendations |
| [`gap-analysis/`](gap-analysis/) | Full specification gap analysis: per-property verdicts, claim-by-claim checks, the 15 guarantees the specification does not make, and 30+ unstated values and observations |
| [`tests/`](tests/) | In-process red-team tests and the Paxos schedule fuzzer (pytest) |
| [`hokea_tests/`](hokea_tests/) | Red-team tests that run on a local HOKEA Docker cluster |
| [`logs/`](logs/) | Raw test logs from every run, including two independent HOKEA runs |

## Headline results

- **27 confirmed issues**, each backed by a strict-xfail test that asserts the specification's requirement. **6 reproduce on HOKEA.**
- Violated: S1 (at-most-once), S2 (one value per slot, under a single-voter disk rollback that the specification permits), S3 (default mode), S4 (at reducers), S5, S7 (dedup state), L1.
- Held: S6, S8, L4, and S2 under the stated assumptions (12-seed fuzzing).

The tests import the `elasticml` package and are meant to be dropped into the ElasticML MS3 tree. That source code is not included here. See [jepsen-report §7](jepsen-report/README.md#7-reproducing).

No product code was modified as part of this analysis.
