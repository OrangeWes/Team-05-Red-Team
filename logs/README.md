# Log index

Every log begins with the command that produced it. All commands run from the ElasticML MS3 root as
`PYTHONPATH=src python -m pytest -p no:cacheprovider ...`. The table below shows only the arguments after that. HOKEA runs use the local Docker backend (HOKEA `427b946`, `unset HOKEA_NAMESPACE KUBECONFIG`). Section numbers refer to the ElasticML MS3 `PROJECT_SPEC.pdf`.

`xfail` = a strict expected failure, meaning the test asserts a specification requirement the system does not meet. Runs with `--runxfail` show those as `FAILED`.

## Rounds 1–2: in-process tests

| Log | Arguments | Result | Purpose |
|---|---|---|---|
| `01_full_suite.log` | `-rfEX -q` (all of `tests/`, including ElasticML's own suite) | 601 passed, 29 xfailed | Baseline. Their own tests pass, and ours show the expected failures. |
| `02_redteam_verbose.log` | `-v -rxX tests/test_redteam_spec_gaps.py tests/test_redteam_spec_gaps_round2.py tests/test_redteam_paxos_fuzz.py` | 17 passed, 29 xfailed | Every red-team test, one per line |
| `03_redteam_runxfail_tracebacks.log` | the same files, with `--runxfail --tb=short -v` | 29 failed, 17 passed | Evidence: traceback for each unmet requirement |
| `04_redteam_failure_lines.log` | the same files, with `--runxfail --tb=line -q` | 29 failed, 17 passed | Same as log 03, one line per failure |

### Specification property checked by each test in logs 02–04

| Test | Specification | Result |
|---|---|---|
| `test_s2_randomized_schedules_choose_one_value_per_slot[0-11]` | S2 | passed (12 fuzzed schedules) |
| `test_demo_voter_restart_without_durable_files_breaks_s2` | S2, §5 (excluded fault; demo) | passed |
| `test_s6_read_at_replica_that_missed_every_acknowledged_write` | S6 | passed |
| `test_s8_isolated_replica_cannot_acknowledge_commit_or_authoritative_read` | S8 | passed |
| `test_l4_backup_reducer_commits_after_primary_dies_before_forwarding` | L4 | passed |
| `test_l4_s1_backup_rebuilds_same_batch_after_primary_committed` | L4, S1 | passed |
| rt01 | S5, L1 | xfail |
| rt02, rt04, rt08 | S5 | xfail |
| rt03 (both tests) | S5, S7, §7 | xfail |
| rt05 | L1 (worker crash) | xfail |
| rt06 | S4, L1, L4 | xfail |
| rt07 | S5, L1 | xfail |
| rt09 | S1 | xfail |
| rt10 | L1, L4 | xfail |
| rt11, rt24 | §7 metrics | xfail |
| rt12 | §5 authentication | xfail |
| rt13 | §6 dataset hashes | xfail |
| rt14a, rt14b, rt14c | §9 | xfail |
| rt14d, rt25 | §7 run evidence | xfail |
| rt17 | S2 under a §5 single-voter rollback | xfail |
| rt18 | §5 heterogeneous speed, L1, L2 | xfail |
| rt19 | §5 bounded jobs | xfail |
| rt20 | §7 "one deterministic run" | xfail |
| rt21 | S3 (default mode) | xfail |
| rt22 | §2 cells | xfail |
| rt23 | §8, §12 | xfail |
| rt26 | §5 compaction | xfail |
| rt27 | §10 local HOKEA suite | xfail |

## Rounds 1–2: HOKEA

| Log | Arguments | Result | Specification |
|---|---|---|---|
| `10_hokea_baseline_happy_path.log` | `-v -s hokea_tests/test_elasticml.py::test_happy_path_trains_and_converges` (ElasticML's own test) | **ERROR**: voters not healthy after 600 s ("must provide exactly 6 pods") | §10. Evidence for RT-27. |
| `11_hokea_redteam.log` | `-v -s -rxXfE hokea_tests/test_redteam_hokea.py` | 2 passed, 5 xfailed | See below |
| `12_hokea_redteam_events.log` | Not a separate run. The 33 `ELASTICML_REDTEAM` event lines extracted from log 11. | — | Same as log 11 |
| `13_hokea_redteam_rerun.log` | `-v -s hokea_tests/test_redteam_hokea.py` (second independent run) | 2 passed, 5 xfailed | Same as log 11 (reproducibility check) |

Tests in logs 11 and 13:

| Test | Specification | Result |
|---|---|---|
| `test_hk_rt01_overflowing_member_cannot_halt_aggregation` | S5, L1 | xfail |
| `test_hk_rt03_reserved_prefix_keeps_replica_dedup_state_identical` | S5, S7 | xfail |
| `test_hk_rt05_survivors_keep_committing_after_worker_pod_crash` | L1 (worker pod killed and never restarted) | xfail |
| `test_hk_rt06_future_base_request_cannot_halt_reducer_tree` | S4, L1, L4 | xfail |
| `test_hk_rt17_single_voter_disk_rollback_preserves_slot_1` | S2 under a §5 single-voter rollback | xfail |
| `test_hk_rt15_demo_amnesiac_voter_restart_lets_slot_1_change` | §5 excluded fault (demo) | passed |
| `test_hk_s6_read_at_voter_that_missed_each_acknowledged_write` | S6 | passed |

## Round 3: performance and HOKEA

| Log | Arguments | Result |
|---|---|---|
| `14_hokea_perf.log` (final) | `-v -s hokea_tests/test_redteam_perf_hokea.py` | 1 passed, 2 failed |
| `14a_hokea_perf_voter1_sampler.log` | the same, first run | **Superseded.** The sampler read voter 1 only. Kept as evidence of learner lag (O-19, S6). |
| `15_hokea_round3.log` | `-v -s --runxfail hokea_tests/test_redteam_hokea_round3.py` | 9 failed, 2 passed |
| `16_hokea_round3_rerun.log` | the same, with `-k 'rt07 or rt08 or rt09 or rt19'` | 3 failed, 1 passed |
| `17_inprocess_perf.log` | `-v -s tests/test_redteam_perf.py` | 3 passed, 2 xfailed |

| Log | Test | Specification | Result |
|---|---|---|---|
| 14 | `test_ph_healthy` | baseline for P1–P15 | passed |
| 14 | `test_ph_churn` (PH-02) | L1, §5 churn | failed against assumptions A3/A4 |
| 14 | `test_ph_churn_partition` (PH-03) | L1, L3, §5 | failed against assumptions A3/A4 |
| 15 | rt02, rt04 | S5 | failed |
| 15 | rt12 | §5 authentication | failed |
| 15 | rt20 | §7 determinism | failed |
| 15 | `test_hk_straggler_slows_cluster_by_at_most_its_share` (PH-04) | §5 heterogeneous speed | failed |
| 15 | rt22 | §2 cells | passed (not reproduced) |
| 15 | `test_hk_s1_s2_randomized_partitions_and_loss_keep_one_history` | S1, S2 | passed (both held) |
| 16 | rt07 | S5, L1 | failed |
| 16 | rt08/rt10 | S5, L1, L4 | failed |
| 16 | rt19 | §5 bounded jobs | failed |
| 16 | rt09 | S1 | passed (not reproduced on HOKEA; the in-process reproduction stands) |
| 17 | PF-01, PF-02, PF-05 | calibration, commit path, codec | passed |
| 17 | PF-03 | L1, §5 churn | xfail |
| 17 | PF-04 | §2, L4 | xfail |

The specification makes no performance guarantees. §7 calls results "measurements, not guarantees". The log 14 failures are therefore judged against the stated assumptions A1–A4 in the [report](../jepsen-report/README.md), not against a specification limit.

**For rt07, rt08, rt09 and rt19, use log 16, not log 15.** In log 15 those tests hit two bugs in our own test harness: the reducer on port 8101 was not ready yet, and rt19's predicate never fired. Log 16 reran them after we fixed our test code. No ElasticML code was changed.

The full in-process suite run after round 3 (604 passed, 31 xfailed) has no separate log file.
