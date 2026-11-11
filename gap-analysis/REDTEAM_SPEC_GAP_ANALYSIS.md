# Red-team gap analysis against the MS3 specification

**Date:** 2026-11-10
**Specification:** `docs/submission/PROJECT_SPEC.pdf` ("ElasticML — Revised Project Specification (MS3)", 7 pages)
**Code under test:** the Team-05 ElasticML MS3 working tree (not included in this repository). It has no `.git` metadata, so no commit can be named. `file:line` references point into that tree.
**Environment:** Linux 6.11.2 x86_64, Python 3.14.6, torch 2.15.0, pytest 8.4.2, Docker 29.8.0, HOKEA `../hokea` at pinned commit `427b946`
**Scope of this round:** analysis and tests only. **No product code was changed.**

---

## 0. How to read this report

- **RT-xx** is a confirmed gap. Each one has at least one test that asserts what the specification requires. The test is marked `@pytest.mark.xfail(strict=True)`, so CI stays green but the gap stays visible. A fix turns the test into an XPASS failure, which tells you to remove the marker.
- **O-xx** is an observation that has no test, usually because the specification states no measurable requirement to test against.
- **"HOKEA"** means the test was also run on a real local HOKEA Docker cluster: separate voter, worker and reducer containers, real networking, and HOKEA `kill`/`start`/`isolate`/`heal` faults.

Commands:

```bash
# in-process red-team tests (expected failures shown as XFAIL)
PYTHONPATH=src python -m pytest -q tests/test_redteam_spec_gaps.py tests/test_redteam_spec_gaps_round2.py tests/test_redteam_paxos_fuzz.py
```

```bash
# same tests, expected failures shown as hard failures
PYTHONPATH=src python -m pytest -q --runxfail tests/test_redteam_spec_gaps.py tests/test_redteam_spec_gaps_round2.py tests/test_redteam_paxos_fuzz.py
```

```bash
# HOKEA cluster tests (Docker; never Kubernetes)
unset HOKEA_NAMESPACE KUBECONFIG; PYTHONPATH=src python -m pytest -v -s hokea_tests/test_redteam_hokea.py
```

## 1. Results at a glance

| Run | Result | Log |
|---|---|---|
| Existing suite before this work | 584 passed | — |
| Full default suite, including new tests | **601 passed, 29 xfailed** (27 s) | [01_full_suite.log](../logs/01_full_suite.log) |
| Red-team in-process tests | **17 passed, 29 xfailed** | [02_redteam_verbose.log](../logs/02_redteam_verbose.log) |
| Same tests with `--runxfail` | **29 failed, 17 passed**. Every failure is at the requirement assertion. | [03_…tracebacks.log](../logs/03_redteam_runxfail_tracebacks.log), [04_…failure_lines.log](../logs/04_redteam_failure_lines.log) |
| Repository's own local HOKEA suite (`hokea_tests/test_elasticml.py`) | **ERROR**: all 3 voter containers exit, `HOKEA must provide exactly 6 pods` (RT-27) | [10_hokea_baseline_happy_path.log](../logs/10_hokea_baseline_happy_path.log) |
| New HOKEA red-team tests | **2 passed, 5 xfailed** (6 min 27 s) | [11_hokea_redteam.log](../logs/11_hokea_redteam.log), [12_hokea_redteam_events.log](../logs/12_hokea_redteam_events.log) |

**27 gaps (RT-01 to RT-27) are confirmed**, and 6 of them were reproduced on HOKEA.

## 2. Status of every Section 4 guarantee

The specification says: *"The correctness properties in Section 4 are guarantees, and each one has automated tests."*

| Property | Verdict | Why (finding IDs) |
|---|---|---|
| **S1** at-most-once | **NOT MET** in a supported profile | RT-09: a retried update is applied twice with `allow_stale`. RT-03: a contribution is silently swallowed. |
| **S2** one value per slot | **NOT MET under a fault the specification does not exclude** | RT-17: rolling back **one** voter's disk breaks S2, in-process and on HOKEA. Under the stated assumptions, randomized fuzzing found no violation. RT-15 shows the amnesia exclusion is load-bearing and is not detected. |
| **S3** durable ack | Met in peer mode. **NOT MET in the default single-coordinator mode** for valid configurations | RT-21: `storage_write_quorum + storage_read_quorum ≤ N` is accepted, and an acknowledged update disappears after restart. |
| **S4** version bounds | Met at voters. **NOT MET at reducers.** | RT-06 (also on HOKEA). |
| **S5** input validity | **NOT MET** | RT-01 (also HOKEA), RT-02, RT-03 (also HOKEA), RT-04, RT-07, RT-08. |
| **S6** read freshness | Met | New passing tests in-process and on HOKEA. The first read after failover is often 503 (RT-16). |
| **S7** deterministic application | Met for model weights. **NOT MET for dedup state and metrics.** | RT-03 on HOKEA: the voters hold the same log but different processed-update sets. |
| **S8** fail closed | Met for commits and `/v1/model` | New passing test. Which endpoints count as "authoritative" is undefined (§6 item 2). |
| **L1** worker loss | **NOT MET** | RT-05 (also HOKEA), RT-06 (also HOKEA), RT-01 (also HOKEA), RT-07, RT-10, RT-18, RT-19. |
| **L2** join / rejoin | Partially met | RT-06 (a returning worker from an old job poisons a reducer), RT-18 (a slow returning worker never contributes). |
| **L3** one-voter failover | Met with client retries | RT-16. The repository's own HOKEA L3 evidence restarts a voter, which falls in the excluded amnesia case (RT-15). |
| **L4** reducer failover | Met at component level | New passing tests. A backup inherits RT-06, RT-07 and RT-10. |

## 3. Complete list of confirmed gaps

Severity assumes the §5 fault model. "In-proc" names the test in `tests/`. "HOKEA" gives the cluster result.

| ID | Sev. | Spec | Finding | Code | In-proc test | HOKEA |
|---|---|---|---|---|---|---|
| RT-01 | **High** | S5, L1 | A **finite** gradient (1e36 × `sample_count` 1000) is admitted. The cohort's weighted sum then overflows, every later commit raises `CodecError` (**HTTP 500**), and the bucket is never cleared. Worker failover carries the same payload to the next voter. | `state.py:344-348`, `state.py:380`, `api.py:459-476` | `test_rt01_…` | **Reproduced**: codes `[200,500,500,200,500,500]`, all voters stuck at v0 |
| RT-02 | Med-High | S5 | No gradient-magnitude bound and no check that the model stays finite. Two accepted gradients of 3e38 with momentum 0.9 overflow the model to `inf`, and every `/v1/model` read then fails. Checkpoint errors are swallowed. AdamW was not tested. | `state.py:107-126` | `test_rt02_…` | — |
| RT-03 | Medium | S5, S7, §7 | The reserved aggregate-ID prefix is rejected only when aggregation is enabled. In direct mode a crafted ID is committed. Voters that learn by recovery add the fake manifest members to their dedup set and add +1,000,000 examples to the metric, while the proposer does not. The named victim ID is later answered `duplicate` without ever being applied. | `state.py:517-518`, `state.py:222-228` | `test_rt03_…` ×2 | **Reproduced**: same `learned_prefix_sha256`, but `processed_update_count` is 1 on node 1 and 2 on node 2 |
| RT-04 | Medium | S5 "oversized" | Only the encoded size is limited (4,000,000 base64 characters per field, 9 fields). A 20 KB NPZ allocated **32 MB** before being rejected, and an 83 KB one allocated 128 MB. | `codec.py:31-43` | `test_rt04_…` | — |
| RT-05 | **High** | L1 | The voter-side partial flush runs only when a new gradient POST arrives. Workers waiting on `pending` poll `/v1/updates/status`, which never flushes. After one worker crashes, the survivors stall forever. This is the HOKEA default for 4 or fewer workers. | `state.py:427-440`, `state.py:446-467` | `test_rt05_…` | **Reproduced**: 3-worker cluster, worker pod killed at v12. Two in-flight commits reached v14, then **no progress for 120 s**, with `pending_updates: 2`. |
| RT-06 | **High** | S4, L1, L4 | Reducers trust `base_version`, and `_max_observed_base_version` only grows. One future-base POST makes every honest submission `stale` from then on. | `gradient_reducer.py:192-206` | `test_rt06_…` | **Reproduced**: 5-worker reducer tree. One POST with base 10000 to each reducer replica froze training at **v23 for 120 s**. |
| RT-07 | Med-High | S5, L1 | Reducers don't check tensor names or shapes. One mismatched member stays in the bucket, so batch formation fails every time, and `status()` raises (HTTP 500). The cell is blocked. | `gradient_reducer.py:134-146`, `gradient_reducer.py:336-346` | `test_rt07_…` | — |
| RT-08 | Low-Med | S5 | The reducer decoder checks finiteness **before** int8 dequantisation, so scale × 127 can overflow to `inf`. The voter's decoder checks afterwards, so the two disagree. | `array_codec.py:17-23` | `test_rt08_…` | — |
| RT-09 | High (stale profile) | S1 | The primary and backup reducers share one ID, and the voter sees only aggregate IDs. With `allow_stale`, a retried `w1:u1` is applied twice (v1 and then v2). The selected profile is not affected. This is the earlier T-01, now reproduced end to end. | `gradient_reducer.py:148-156` | `test_rt09_…` | — |
| RT-10 | Medium | L1, L4 | A reducer ignores 422 and 429 replies from upstream, so members stay `pending` forever. | `gradient_reducer.py:308-321` | `test_rt10_…` | — |
| RT-11 | Low | §7 | The `duplicates`, `dropped` and `dropped_examples` counters live in memory and reset on restart. | `state.py:92-94` | `test_rt11_…` | — |
| RT-12 | Medium | §5 | Peer mode validates with `auth_token=None`, which leaves Paxos `prepare`/`accept` unauthenticated. Public default tokens `dev-only-change-me` and `elasticml-hokea-local` are committed. | `config.py:35`, `hokea_node.py:133` | `test_rt12_…` | — |
| RT-13 | Low-Med | §6 | "The loader verifies published hashes" holds only with `download=True`. Existing or tampered archives and the `.npy` arrays used in training are never hashed. The validation size is `min(5000, len//10)`. | `data.py:67-96`, `data.py:126` | `test_rt13_…` | — |
| RT-14a | Low | §9 | The specification claims Python 3.11–3.14, but CI tests only 3.12 and 3.14. | `ci.yml` | `test_rt14a_…` | — |
| RT-14b | Medium | §9 | The specification names commit `17f8bd4`, but the retained clean-artifact evidence names `ff56901`, and this tree has no git metadata. | `clean-artifact-verification.json` | `test_rt14b_…` | — |
| RT-14c | Low | §9 | `HOKEA_NET_BACKEND` is not read anywhere in ElasticML or HOKEA `427b946`. HOKEA chooses its backend in code (`Net(backend="auto")`). | — | `test_rt14c_…` | — |
| RT-14d | Medium | §7 | The record for the 94.68% run is missing: commit, script, per-role seeds, p50/p95, commit latency, CPU, memory, bytes, churn rates, network profile, batch size and fan-in. Its raw log is absent. | `lr006_20.json` | `test_rt14d_…` | — |
| RT-15 | Info / evidence validity | §5 excluded, L3 | Demo: a voter restarted with empty storage rejoins under the same identity and slot 1 is chosen again with a different value. Nothing detects it. **On HOKEA, a restart provably empties `/data`** (`learned_prefix_length: 0`), so the repository's own failover evidence (`test_hokea_tree_failover.py:124-127`) runs inside the excluded fault. | `consensus.py:86-96` | `test_demo_…` (pass) | **Reproduced (pass)**: slot-1 model sha changed from `d00108c3…` to `0772b07a…` |
| RT-16 | Low | L3 | After failover, the first authoritative read at a lagging replica usually returns 503. Its recovery ballot is below the writer's promise and there is no internal retry. | `consensus.py:196-217` | inside `test_s6_…` | **Observed**: first read 503 in 2 of 3 rotations |
| RT-17 | **High** | §5, S2 | §5 excludes rolled-back state only **"for a majority"**. Restoring **one** voter from an older snapshot (same identity, same process) lets a second value be chosen for an acknowledged slot. | `consensus.py:129-154` | `test_rt17_…` | **Reproduced**: node 1 slot-1 sha `d00108c3…`, node 3 sha `0772b07a…` |
| RT-18 | Medium | §5 heterogeneous speed, L1/L2 | A live worker that finishes 1.0 s into a round, against a 0.75 s flush, is answered `stale` **every** round (6 of 6) and never contributes. There is no fairness guarantee. | `gradient_reducer.py:336-346` | `test_rt18_…` | — |
| RT-19 | Medium | §5 bounded jobs | The voter's 409 `job_complete` is mapped to `stale` with a made-up `model_version` (base+1 = 2). Tree-mode workers loop instead of stopping. | `gradient_reducer.py:313-315` | `test_rt19_…` | — |
| RT-20 | Medium | §7 "one deterministic run" | With identical inputs, the committed model depends on when updates arrive relative to the flush deadline. The "deterministic" claim is unsupported. | `gradient_reducer.py:336-346` | `test_rt20_…` | — |
| RT-21 | Medium | S3 (default mode) | `validate()` accepts W=1, R=2. An `applied` update written to one mirror is gone after restart. Single-coordinator mode is the default `election_backend`. | `config.py:102-105`, `storage.py:170-188` | `test_rt21_…` | — |
| RT-22 | Medium | §2 cells | Membership is per replica and not replicated. Two voters with different registration views put worker `w3` in cell 0 and cell 2. | `api.py:216-225`, `membership.py:20-74` | `test_rt22_…` | — |
| RT-23 | Low | §8, §12 | The specification says the profiles are "healthy, bad, terrible, mixed" and that the simulated-day framing was replaced. The code has a `good` profile, "Good day" labels and a `simulated_hours` field. | `lab_dashboard.py:66-90` | `test_rt23_…` | — |
| RT-24 | Low-Med | §7 | "Unique committed examples" = 2,444,800, but the training split is 55,000. The metric counts exposures (≈44 passes over the data), not unique examples. | evidence file | `test_rt24_…` | — |
| RT-25 | Medium | §7 equal-seed fault comparisons | No retained `fashion_cnn_v5` evidence has a churn, request-loss or replica-failure run. Every MS3-model record is healthy only (`faults_injected: 0` or absent). | `evidence/model` | `test_rt25_…` | — |
| RT-26 | Medium | §5 "compaction … not part of the selected runtime" | Every HOKEA voter, the runtime behind the reported results, hard-codes `peer_log_compaction_enabled: True`. | `hokea_node.py:235` | `test_rt26_…` | **Confirmed** in a live voter's `/data/coordinator.json` |
| RT-27 | Medium | §10 "the local HOKEA suite runs in Docker" | `hokea_tests/conftest.py` starts 3 nodes, but the image requires 3 + workers (6 for the default 3). Every container exits, so the documented local suite cannot run. | `hokea_tests/conftest.py`, `hokea_node.py:316-318` | `test_rt27_…` | **Reproduced**: baseline run ERROR after 717 s |

## 4. Test logs (verbatim excerpts)

### 4.1 Full suite, [01_full_suite.log](../logs/01_full_suite.log)

```
601 passed, 29 xfailed, 2 warnings in 27.19s
```

### 4.2 Red-team tests, normal mode, [02_redteam_verbose.log](../logs/02_redteam_verbose.log)

```
tests/test_redteam_spec_gaps.py::test_rt01_overflowing_cohort_member_cannot_stall_honest_workers XFAIL
tests/test_redteam_spec_gaps.py::test_rt02_accepted_updates_keep_model_finite_and_readable XFAIL
tests/test_redteam_spec_gaps.py::test_rt03_reserved_aggregate_prefix_rejected_before_proposal XFAIL
tests/test_redteam_spec_gaps.py::test_rt03_live_and_recovered_replicas_deduplicate_identically XFAIL
tests/test_redteam_spec_gaps.py::test_rt04_oversized_decoded_payload_rejected_before_large_allocation XFAIL
tests/test_redteam_spec_gaps.py::test_rt05_surviving_workers_commit_after_cohort_member_crashes XFAIL
tests/test_redteam_spec_gaps.py::test_rt06_future_base_submission_cannot_poison_reducer XFAIL
tests/test_redteam_spec_gaps.py::test_rt07_wrong_shape_member_rejected_without_blocking_cell XFAIL
tests/test_redteam_spec_gaps.py::test_rt08_reducer_decoder_rejects_int8_scale_overflow XFAIL
tests/test_redteam_spec_gaps.py::test_rt09_worker_contribution_applied_at_most_once_across_reducer_failover XFAIL
tests/test_redteam_spec_gaps.py::test_rt10_upstream_rejection_gives_workers_a_terminal_status XFAIL
tests/test_redteam_spec_gaps.py::test_rt11_reported_duplicate_and_drop_counts_survive_restart XFAIL
tests/test_redteam_spec_gaps.py::test_rt12_peer_mode_requires_authentication XFAIL
tests/test_redteam_spec_gaps.py::test_rt13_loader_rejects_files_that_do_not_match_published_hashes XFAIL
tests/test_redteam_spec_gaps.py::test_rt14a_ci_matrix_covers_every_supported_python XFAIL
tests/test_redteam_spec_gaps.py::test_rt14b_retained_evidence_matches_the_declared_commit XFAIL
tests/test_redteam_spec_gaps.py::test_rt14c_documented_hokea_net_backend_is_implemented XFAIL
tests/test_redteam_spec_gaps.py::test_rt14d_selected_run_evidence_records_every_section7_field XFAIL
tests/test_redteam_spec_gaps.py::test_s6_read_at_replica_that_missed_every_acknowledged_write PASSED
tests/test_redteam_spec_gaps.py::test_s8_isolated_replica_cannot_acknowledge_commit_or_authoritative_read PASSED
tests/test_redteam_spec_gaps.py::test_l4_backup_reducer_commits_after_primary_dies_before_forwarding PASSED
tests/test_redteam_spec_gaps.py::test_l4_s1_backup_rebuilds_same_batch_after_primary_committed PASSED
tests/test_redteam_spec_gaps_round2.py::test_rt17_single_voter_rollback_preserves_single_value_per_slot XFAIL
tests/test_redteam_spec_gaps_round2.py::test_rt18_slow_but_live_worker_eventually_contributes XFAIL
tests/test_redteam_spec_gaps_round2.py::test_rt19_job_completion_reaches_workers_through_reducers XFAIL
tests/test_redteam_spec_gaps_round2.py::test_rt20_equal_inputs_give_equal_model_regardless_of_arrival_timing XFAIL
tests/test_redteam_spec_gaps_round2.py::test_rt21_acknowledged_update_survives_restart_for_every_valid_quorum_config XFAIL
tests/test_redteam_spec_gaps_round2.py::test_rt22_every_voter_assigns_a_worker_to_the_same_cell XFAIL
tests/test_redteam_spec_gaps_round2.py::test_rt23_dashboard_matches_section8_profiles_and_timing_model XFAIL
tests/test_redteam_spec_gaps_round2.py::test_rt24_unique_committed_examples_cannot_exceed_training_split XFAIL
tests/test_redteam_spec_gaps_round2.py::test_rt25_ms3_model_has_equal_seed_fault_scenario_evidence XFAIL
tests/test_redteam_spec_gaps_round2.py::test_rt26_hokea_runtime_does_not_enable_experimental_compaction XFAIL
tests/test_redteam_spec_gaps_round2.py::test_rt27_local_hokea_suite_cluster_size_matches_image_topology XFAIL
tests/test_redteam_paxos_fuzz.py::test_s2_randomized_schedules_choose_one_value_per_slot[0..11] PASSED (12 seeds)
tests/test_redteam_paxos_fuzz.py::test_demo_voter_restart_without_durable_files_breaks_s2 PASSED
================== 17 passed, 29 xfailed, 2 warnings in 5.33s ==================
```

### 4.3 Why each expected failure fails, from `--runxfail` ([04_redteam_failure_lines.log](../logs/04_redteam_failure_lines.log))

```
tests/test_redteam_spec_gaps.py:146: assert (200 < 500 and 500 < 500)                                   # RT-01 honest submit got HTTP 500
tests/test_redteam_spec_gaps.py:164: assert False                                                       # RT-02 parameters not finite
tests/test_redteam_spec_gaps.py:177: Failed: DID NOT RAISE <class 'elasticml.state.InvalidGradient'>    # RT-03 reserved prefix accepted
tests/test_redteam_spec_gaps.py:192: AssertionError: assert {'__elasticml...', 'victim-1'} == {'__elasticml...2NhaAMV3Cn0='}  # RT-03 dedup divergence
tests/test_redteam_spec_gaps.py:222: AssertionError: rejection allocated 32.0 MB                        # RT-04
tests/test_redteam_spec_gaps.py:246: AssertionError: assert 'pending' == 'applied'                      # RT-05
tests/test_redteam_spec_gaps.py:265: AssertionError: assert 'stale' == 'applied'                        # RT-06
src/elasticml/gradient_reducer.py:143: ValueError: gradient tensor sets and shapes must match within a reducer batch  # RT-07 honest w2 blocked
tests/test_redteam_spec_gaps.py:299: Failed: DID NOT RAISE <class 'ValueError'>                         # RT-08
tests/test_redteam_spec_gaps.py:325: assert 2 == 1                                                      # RT-09 version 2: u1 applied twice
tests/test_redteam_spec_gaps.py:344: AssertionError: assert 'pending' != 'pending'                      # RT-10
tests/test_redteam_spec_gaps.py:366: AssertionError: duplicates                                         # RT-11
tests/test_redteam_spec_gaps.py:379: Failed: DID NOT RAISE <class 'ValueError'>                         # RT-12
tests/test_redteam_spec_gaps.py:400: Failed: DID NOT RAISE <class 'ValueError'>                         # RT-13
tests/test_redteam_spec_gaps.py:413: AssertionError: {'3.12', '3.14'}                                   # RT-14a
tests/test_redteam_spec_gaps.py:421: AssertionError: assert False                                       # RT-14b
tests/test_redteam_spec_gaps.py:429: assert False                                                       # RT-14c
tests/test_redteam_spec_gaps.py:446: AssertionError: {'missing': ['source_commit', 'script', 'data_selection_seed', 'fault_schedule_seed', 'p50', 'p95', ...], 'raw_log_present': False}  # RT-14d
tests/test_redteam_spec_gaps_round2.py:77: AssertionError: slot 1 chose v1 and then v2                  # RT-17
tests/test_redteam_spec_gaps_round2.py:102: AssertionError: ['stale', 'stale', 'stale', 'stale', 'stale', 'stale']  # RT-18
tests/test_redteam_spec_gaps_round2.py:118: AssertionError: {'aggregation_enabled': True, 'model_version': 2, 'status': 'stale'}  # RT-19
tests/test_redteam_spec_gaps_round2.py:145: assert False                                                # RT-20 models differ
tests/test_redteam_spec_gaps_round2.py:175: assert (0 == 1)                                             # RT-21 acked update lost
tests/test_redteam_spec_gaps_round2.py:195: assert 0 == 2                                               # RT-22 cell 0 vs cell 2
tests/test_redteam_spec_gaps_round2.py:206: AssertionError: assert 'healthy' in {'bad': ..., 'good': ..., 'terrible': ...}  # RT-23
tests/test_redteam_spec_gaps_round2.py:219: assert 2444800 <= (60000 - 5000)                            # RT-24
tests/test_redteam_spec_gaps_round2.py:234: AssertionError: no MS3-model run with injected faults       # RT-25
tests/test_redteam_spec_gaps_round2.py:248: assert True is False                                        # RT-26
tests/test_redteam_spec_gaps_round2.py:265: AssertionError: (3, 6)                                      # RT-27
29 failed, 17 passed, 2 warnings in 4.88s
```

Comments after `#` were added for this report. Full tracebacks are in [03_redteam_runxfail_tracebacks.log](../logs/03_redteam_runxfail_tracebacks.log).

### 4.4 Repository's own HOKEA suite, [10_hokea_baseline_happy_path.log](../logs/10_hokea_baseline_happy_path.log)

```
hokea_tests/test_elasticml.py::test_happy_path_trains_and_converges ERROR
E           --- hokea-elasticml-local-node1 last logs ---
E               raise ValueError(f"HOKEA must provide exactly {expected_pods} pods with a valid NODE_ID")
E           ValueError: HOKEA must provide exactly 6 pods with a valid NODE_ID
   (same for node2 and node3)
========================= 1 error in 717.60s (0:11:57) =========================
```

### 4.5 HOKEA red-team run, [11_hokea_redteam.log](../logs/11_hokea_redteam.log) / [12_hokea_redteam_events.log](../logs/12_hokea_redteam_events.log)

```
XFAIL test_hk_rt01_overflowing_member_cannot_halt_aggregation
  {"event": "rt01_result", "codes": [200, 500, 500, 200, 500, 500], "versions": {"1": 0, "2": 0, "3": 0}}
XFAIL test_hk_rt03_reserved_prefix_keeps_replica_dedup_state_identical
  {"event": "rt03_voter_config", "peer_log_compaction_enabled": true, "gradient_aggregation_enabled": false, "max_staleness": 8, "max_job_updates": 1000}
  {"event": "rt03_states", "1": {"model_version": 1, "processed_update_count": 1, "learned_prefix_sha256": "0548e8d4…"},
                           "2": {"model_version": 1, "processed_update_count": 2, "learned_prefix_sha256": "0548e8d4…"}, …}
XFAIL test_hk_rt05_survivors_keep_committing_after_worker_pod_crash
  {"event": "rt05_before_kill", "versions": {"1": 12, "2": 12, "3": 12}}
  {"event": "rt05_after_kill", "versions": {"1": 14, "2": 14, "3": 14}, "status": {…, "aggregate_commits": 14, "pending_updates": 2, …}}
XFAIL test_hk_rt06_future_base_request_cannot_halt_reducer_tree
  {"event": "rt06_before_ghost", "versions": {"1": 11, "2": 11, "3": 11}}
  {"event": "rt06_ghost", "node": 4, "out": "200 {\"status\":\"pending\",\"model_version\":10000,…}"}
  {"event": "rt06_ghost", "node": 5, "out": "200 {\"status\":\"pending\",\"model_version\":10000,…}"}
  {"event": "rt06_after_ghost", "versions_at_ghost": 23, "versions": {"1": 23, "2": 23, "3": 23}}
PASSED test_hk_rt15_demo_amnesiac_voter_restart_lets_slot_1_change
  {"event": "rt15_first_commit", "model_version": 1, "model_sha256": "d00108c3…", "learned_prefix_length": 1}
  {"event": "rt15_restarted_voter", "model_version": 0, "processed_update_count": 0, "learned_prefix_length": 0}
  {"event": "rt15_after_second_proposer", "model_version": 1, "model_sha256": "0772b07a…", "learned_prefix_length": 1}
XFAIL test_hk_rt17_single_voter_disk_rollback_preserves_slot_1
  {"event": "rt17_first_commit", "model_version": 1, "model_sha256": "d00108c3…"}
  {"event": "rt17_rolled_back_voter_2"}
  {"event": "rt17_after_second_proposer", "status": 200, "model_version": 1, "model_sha256": "0772b07a…"}
PASSED test_hk_s6_read_at_voter_that_missed_each_acknowledged_write
  {"event": "s6_round", "writer": 1, "reader": 2, "acknowledged": 1, "first_status": 200, "observed": 1}
  {"event": "s6_round", "writer": 2, "reader": 3, "acknowledged": 2, "first_status": 503, "observed": 2}
  {"event": "s6_round", "writer": 3, "reader": 1, "acknowledged": 3, "first_status": 503, "observed": 3}
=================== 2 passed, 5 xfailed in 386.95s (0:06:26) ===================
```

Hashes are shortened. Full values are in the log. HOKEA cluster sizes: RT-01 = 3 voters + 2 worker pods; RT-03, RT-15, RT-17 and S6 = 3 + 1 (workers idle); RT-05 = 3 + 3 (training, voter-side aggregation); RT-06 = 3 + 5 (training, reducer tree, 1 logical reducer × 2 replicas on pods 4 and 5).

## 5. Specification claims checked against the artifact

| § | Claim | Holds? | Evidence |
|---|---|---|---|
| 2 | `gradient_reducer_count` defaults to 0 | Yes | `config.py:51` |
| 2 | Tree distribution above four workers | Yes | `tree_default = total_workers > 4` (`hokea_node.py:411`) |
| 2 | "Each primary reducer has a deterministic backup" | Partly | Placement is deterministic. The backup shares the primary's ID (enables RT-09) and inherits RT-06, RT-07 and RT-10. |
| 2 | int8 transport with error feedback | First hop only | Reducer→parent is float16 (`hokea_node.py:418`) |
| 3 | No etcd, ZooKeeper or consensus library | Yes | `election_backend` limited to `single`/`peer` (`config.py:106`) |
| 3 | "Reducers add aggregation capacity, not authority" | Authority yes; liveness no | RT-06: one reducer input can halt all commits |
| 5 | Excluded: majority corruption or rollback | **Incomplete exclusion** | RT-17: minority rollback breaks S2 |
| 5 | A HOKEA restart comes back with no memory | **Yes, confirmed** | RT-15 on HOKEA: `learned_prefix_length: 0` after `start` |
| 5 | Compaction is not part of the selected runtime | **No** | RT-26 |
| 5 | `max_job_updates` default 1,000 | Yes | `config.py:42`. The selected run ended at v962. |
| 6 | Loader verifies published hashes | **No** (download only) | RT-13 |
| 6 | 5,000-example validation split | Conditional | `min(5000, len//10)` |
| 7 | Each run records its load parameters and names script, commit and log | **No** | RT-14d |
| 7 | Equal-seed comparisons across fault scenarios | **No MS3-model evidence** | RT-25 |
| 7 | "Unique committed examples" | Mislabelled | RT-24 |
| 7 | "One deterministic run" | **Unsupported** | RT-20 |
| 7 | 94.68%, 94% at 945 s, 95% not reached | Matches evidence | `lr006_20.json` |
| 8 | Profiles "healthy, bad, terrible, mixed"; real-second timing | **No** | RT-23 |
| 9 | Commit `17f8bd4` | **No** | RT-14b |
| 9 | Python 3.11–3.14 | Declared, not tested | RT-14a |
| 9 | `HOKEA_NET_BACKEND` | **No** | RT-14c |
| 9 | CPU wheel index | Yes | `Dockerfile:28`, `ci.yml` |
| 10 | GitHub Actions runs on each commit | Configured | `on: push / pull_request` |
| 10 | Local HOKEA suite runs in Docker | **No** | RT-27 |

## 6. Guarantees the specification does not make (gaps in the specification itself)

1. **What a "worker contribution" is (S1).** It is never defined. In tree mode the voters never see worker update IDs.
2. **Which reads are authoritative (S6, S8).** Only `/v1/model` goes through quorum. The models workers actually train on come attached to acknowledgements or reducer caches and are local reads. `/v1/updates/status`, `/v1/status`, `/health` and `/v1/workers/register` answer from a minority partition.
3. **No numbers for "bounded".** Cell size, partial-flush timeout and minimum, buffers, staleness limit and request-body size are all unstated (see §7).
4. **No liveness time bounds.** No recovery-time or failover-to-commit target is set, and no starvation or fairness guarantee is given for slow workers (RT-18).
5. **No numerical-validity guarantee.** Inputs must be finite, but the model is not required to stay finite and gradients are not bounded (RT-01, RT-02).
6. **Minority disk faults.** The only stated exclusions are majority corruption, rollback and loss, plus amnesiac restart. Minority rollback (RT-17) and minority bit corruption of `slot-*.json` or `proposer.json`, which carry no checksums, are left in scope and are unsafe.
7. **No detection guarantees for excluded faults** (RT-15, O-4).
8. **No authentication requirement** (RT-12).
9. **Durability scope.** Process crash or power loss is not stated. `fsync` plus a directory `fsync` is implemented.
10. **Which deployment mode the guarantees cover.** The default `election_backend` is `single`: one process with three mirror directories, usually on one disk. S3, S8 and L3 as written do not apply to it, and RT-21 breaks S3 there.
11. **The selected profile is not specified.** Cohort, fan-in, flush, staleness, `max_job_updates`, wire precision and learning-rate schedule appear only in code and evidence.
12. **Job lifecycle.** What workers and reducers must do at `max_job_updates`, and how a new job prevents old workers from interfering (RT-06, RT-19), is not stated.
13. **Reads as proposers.** The liveness assumption of "eventually limited proposer contention" counts only the root reducer. Every authoritative read at a lagging replica also runs Paxos prepares (RT-16, O-12).
14. **Membership consistency.** Cell assignment and rotating-cohort admission depend on each voter's private, unreplicated registry (RT-22).
15. **Measurement definitions.** "Unique examples", "dropped", "duplicate" and "recovery time" are not defined, and the counters are volatile (RT-11, RT-24).

## 7. Values the implementation uses that the specification does not state

| Area | Value in code | Where |
|---|---|---|
| Staleness limit | 8 in direct mode. **0** in aggregation mode unless `allow_stale` is set (HOKEA env `ELASTICML_HOKEA_MAX_STALENESS`, 0–100). | `config.py:29`, `state.py:511`, `hokea_node.py:191` |
| Stale learning-rate discount | `lr / (1 + staleness)` | `state.py:376` |
| Voter partial flush | 2.0 s, minimum `ceil(cohort/2)`, only on POST | `config.py:45`, `state.py:431` |
| Reducer partial flush | 0.75 s (0.5 s at 16+ workers), minimum 1, hard bound 2× timeout | `hokea_node.py:40-44`, `gradient_reducer.py:342` |
| Cell size / fan-in | 8 for `fashion_cnn_v5`/int8/16 MiB. 20 workers → leaves of 7/7/6 plus a root of 3. | `reducer_tree.py:76` |
| Reducer limits | Groups 1–32, count 0–32, replicas 1–2, ID ≤ 64 characters, 64 pending updates, 16 MiB, 4,096 cached outcomes | `gradient_reducer.py:54-66` |
| Backup placement | Backup ordinal = primary ordinal + number of logical reducers. Same reducer ID. | `reducer_tree.py:56-73` |
| HOKEA cohort | ≤ 4 workers: `ceil(0.75 × n)` (max 8). Otherwise all workers. | `hokea_node.py:26-32` |
| Preferred voter | The first URL in the upstream list, sticky after success | `client.py:68` |
| Wire and commit precision | Worker int8 or float16 (HOKEA default **float16**). Reducer float16. Commit float32, or float16 by default for Fashion with aggregation in HOKEA. | `hokea_node.py:155,175` |
| int8 scheme | Symmetric, per tensor, `max|x|/127` | `quantization.py:37-39` |
| Input limits | `update_id` ≤ 256, `worker_id` ≤ 128, 4 M base64 characters per gradient field, 1–8 microsteps, `sample_count` 1–1 M, batch × bundle ≤ 1 M. **No total body or decoded-size limit.** | `api.py:21-49` |
| Voter buffers | 256 updates / 16 MiB | `config.py:46-47` |
| Timeouts | Peer RPC 2.0 s (HOKEA 1.5 s), hedge to the next peer after 25 ms, worker request 5 s (HOKEA 1.5 s), heartbeat 3 s, worker TTL 15 s, startup barrier 120 s | `config.py`, `consensus.py:264` |
| Admission control | Training queue `min(128, max(20, 2×cohort))`, model queue 128, 4 s wait, **one active request**, `Retry-After: 0.1` | `api.py:60-187` |
| Recovery | Learned import ≤ 8 records per RPC | `peer_store.py:151` |
| Checkpoints | Every 25 versions, errors ignored. MirroredStore keeps 2 generations. | `config.py:31`, `storage.py:207-212` |
| Compaction (HOKEA) | Enabled, 64 pending slots | `hokea_node.py:235-236` |
| Mirror quorums | W = 2, R = 2 of 3 by default. Any 1 ≤ W, R ≤ N is accepted. | `config.py:33-34,102-105` |
| Peer membership | "At least three" peers accepted. 4 or more, including even counts, pass validation. | `config.py:151` |
| Learning rate | Default 0.03 (0.2 for `fashion_mlp*`). HOKEA Fashion: 0.2/0.3/0.4 depending on cohort. Selected run: peak 0.006, one-cycle starting at peak/10, 860 planned updates, effective batch 2,560. | `config.py:55`, `hokea_node.py:206-210` |
| Model init | `torch.manual_seed(7)` (reseeds the **global** RNG) | `models.py:210-211` |
| Accuracy targets | 0.70, 0.80, 0.8533, 0.94, 0.95 (from evidence, not the specification) | evidence file |
| Dataset hash | MD5 (download only) | `data.py:33-38` |
| Auth | Optional. HOKEA fallback token `elasticml-hokea-local`. | `hokea_node.py:133` |
| Dashboard | 1–25 trials, 10–900 s each, `simulated_hours` 0.05–168, up to 12 workers | `lab_dashboard.py:66-90` |

## 8. Other observations (no test)

- **O-1:** The two partial-flush rules disagree: the voter needs `ceil(n/2)` members and the reducer needs 1.
- **O-2:** `attach_next_model` takes two separate registry snapshots. A lease that expires between them causes a 500 **after** the commit is applied (`api.py:201-225`).
- **O-3:** Checkpoint failures are swallowed (`state.py:420-424`, `state.py:568-572`).
- **O-4:** Nothing stops two processes from opening one replica directory: there is no lock, only an identity and fingerprint check (`coordinator.py:51-58`).
- **O-5:** A reducer can mark an in-flight batch `stale` that the voter later applies. Workers are told `stale` for applied work, and dropped counts are overstated (`gradient_reducer.py:119-121`).
- **O-6:** `/v1/status` needs no authentication and lists worker IDs and hostnames (`api.py:526`).
- **O-7:** Only one `/v1/model` request runs at a time per replica (`api.py:60-92`).
- **O-8:** The selected run ended at v962, 38 versions short of the default 1,000 job cap. The ~11,000-version continuous run must have used an unstated non-default cap.
- **O-9:** `slot-*.json`, `proposer.json` and `identity.json` have no checksums. Value files do (`value_store.py:51-57`). A rolled-back `proposer.json` re-issues old ballot numbers.
- **O-10:** Peer ballots are not type-checked. Float or bool rounds pass `check_peer` (`api.py:314-326`, `consensus.py:26-28`).
- **O-11:** Worker-ID uniqueness is not enforced. Two live workers with one ID collide in the pending checks.
- **O-12:** An authoritative read at a lagging replica runs `recover()` (a Paxos prepare) on slots up to the accepted frontier, including in-flight ones. Reads can therefore pre-empt a writer's accept and turn its success into a 503 "unknown outcome". Analysis only; no deterministic reproduction yet.
- **O-13:** A worker restart resets `sequence` to 0, so it retrains the same data positions. With sharding, a dead worker's shard is never trained, and this is not counted as dropped (`worker.py:91`, `data.py:150-170`).
- **O-14:** The int8 error-feedback residual is kept even when its gradient is discarded as stale.
- **O-15:** Undocumented components that affect behaviour: gossip data distribution (`elasticml-gossip-worker`), the adaptive network policy (`ELASTICML_ADAPTIVE_NETWORK`), and "request loss" injected inside the client rather than on the network (`client.py:32-39`).
- **O-16:** The evaluation split is chosen by an environment variable (`ELASTICML_EVALUATION_SPLIT`), so nothing in code prevents test-split use during a run (`api.py:418`).
- **O-17:** Dashboard campaigns drive the native lab without the reducer tier, so the MS3 path is never exercised there. The specification acknowledges the campaign results are from MS2.
- **O-18:** `/v1/checkpoint` lets any authenticated client force checkpoint writes.

## 9. Test inventory

| File | Tests | Normal result |
|---|---|---|
| [tests/test_redteam_spec_gaps.py](../tests/test_redteam_spec_gaps.py) | RT-01 to RT-14d (18 xfail), S6, S8, L4 ×2 (4 pass) | 4 passed, 18 xfailed |
| [tests/test_redteam_spec_gaps_round2.py](../tests/test_redteam_spec_gaps_round2.py) | RT-17 to RT-27 except RT-16 (11 xfail) | 11 xfailed |
| [tests/test_redteam_paxos_fuzz.py](../tests/test_redteam_paxos_fuzz.py) | S2 fuzz × 12 seeds, RT-15 demo | 13 passed |
| [hokea_tests/test_redteam_hokea.py](../hokea_tests/test_redteam_hokea.py) | RT-01, RT-03, RT-05, RT-06, RT-17 (xfail), RT-15 demo, S6 | 2 passed, 5 xfailed |

## 10. Log index

All logs are in [`docs/redteam_logs/`](../logs/), and each starts with a header giving the date, host, Python and torch versions, and the exact command.

| File | Contents |
|---|---|
| [01_full_suite.log](../logs/01_full_suite.log) | Whole default suite |
| [02_redteam_verbose.log](../logs/02_redteam_verbose.log) | Every red-team test with its XFAIL reason |
| [03_redteam_runxfail_tracebacks.log](../logs/03_redteam_runxfail_tracebacks.log) | Tracebacks for all 29 expected failures |
| [04_redteam_failure_lines.log](../logs/04_redteam_failure_lines.log) | One line per failure |
| [10_hokea_baseline_happy_path.log](../logs/10_hokea_baseline_happy_path.log) | Repository's HOKEA suite failing to start (RT-27) |
| [11_hokea_redteam.log](../logs/11_hokea_redteam.log) | Full HOKEA red-team run |
| [12_hokea_redteam_events.log](../logs/12_hokea_redteam_events.log) | Structured events from the HOKEA run, with model payloads elided |

## 11. Relationship to earlier reviews and limits

- RT-09 is T-01 from `SPEC_TEST_GAP_ANALYSIS.md`, now reproduced end to end. This review supersedes that document's view of S2, S5 and L1.
- **Not done:** no class-cluster (Kubernetes) run, no `--runxfail` rerun on HOKEA, and no AdamW check for RT-02.
- The HOKEA xfails are backed by the event lines in §4.5 rather than by a `--runxfail` assertion line.
- Deterministic in-process tests cannot exhaust asynchronous schedules. The S2 fuzz is a randomized search, not a proof.
- Severities assume trusted clients, as §5 requires. RT-03, RT-06 and RT-12 are more serious if clients are untrusted.
