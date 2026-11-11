# ElasticML MS3: a Jepsen-style analysis

**Team-05 Red Team** · 2026-11-11
System under test: ElasticML, "Revised Project Specification (MS3)", 7 pages
Test harness: pytest 8.4.2 in-process, plus HOKEA (local Docker fault-injection cluster) at commit `427b946`
Environment: Linux 6.11.2 x86_64, Python 3.14.6, torch 2.15.0, Docker 29.8.0

---

## Summary

ElasticML is an elastic, fault-tolerant system for data-parallel neural-network training. Three voter processes run a hand-written multi-slot Paxos log. Each chosen slot is one optimizer step. Workers compute gradients. Optionally, a tree of gradient reducers aggregates them before a root reducer proposes the aggregate to the voters. The MS3 specification promises eight safety properties (S1–S8) and four liveness properties (L1–L4) under a stated fault model (§5). It also says *"each one has automated tests."*

We tested those claims with 46 new in-process tests, a randomized Paxos schedule fuzzer and 7 tests on a real HOKEA cluster. HOKEA runs separate voter, worker and reducer containers, and injects process kills, restarts, network isolation and disk rollback.

**We found 27 confirmed issues.** Each has a test that asserts what the specification requires and fails today. Six of them reproduce on HOKEA. We ran the HOKEA suite twice, on 2026-11-10 and 2026-11-11, with the same verdicts both times.

The most important results:

- **S2 (one value per slot) is violated by a fault the specification permits.** §5 excludes rollback of a *majority* of voters' disks. Rolling back **one** voter's disk is enough to make a second value be chosen for an acknowledged slot. This reproduces on HOKEA (RT-17).
- **Training halts permanently after one worker crash.** This happens in the HOKEA default configuration (four or fewer workers, voter-side aggregation). The partial flush that should release the surviving workers runs only when a new gradient POST arrives, and survivors only poll (RT-05).
- **One request halts the reducer tree.** A single gradient whose `base_version` is in the future makes every honest submission `stale` forever (RT-06).
- **One finite but very large gradient makes every voter return HTTP 500** and leaves the model at version 0. Client failover then spreads the bad payload to the next voter (RT-01).
- **S1 (at-most-once) is violated** in the supported `allow_stale` profile: a retried worker update is applied twice through the primary and backup reducers (RT-09).
- **S3 (durable acknowledgement) is violated** in the default single-coordinator mode, for configurations its validator accepts (RT-21).
- **Replica state diverges** on the HOKEA runtime. Voters with the same learned log keep different deduplication sets after a crafted update ID (RT-03).
- The specification's reproducibility and evidence claims do not match the artifact. The repository's own HOKEA suite cannot start (RT-27). HOKEA enables log compaction that the specification calls "not part of the selected runtime" (RT-26). The "one deterministic run" is timing-dependent (RT-20). The named commit does not match the evidence (RT-14b).

No product code was modified. All tests are marked `xfail(strict=True)`, so they turn into failures the moment a fix lands, as a signal to remove the marker.

---

## 1. Background

### 1.1 Architecture as specified

| Role | Count | Responsibility |
|---|---|---|
| Voter | 3 | Durable Paxos acceptor per slot, plus learner and state machine. Applies the chosen gradient to the model and answers reads. |
| Worker | N (elastic) | Registers, pulls the model, computes gradients, and submits them with `update_id` and `base_version`. |
| Leaf reducer | ⌈N / cell⌉ | Batches the gradients of one cell and forwards one aggregate upward. It has a primary and a deterministic backup that share one reducer ID. |
| Root reducer | 1 (+ backup) | Combines leaf aggregates and proposes them to the voters. |

When the tree is disabled (the HOKEA default at four or fewer workers), the voters aggregate a cohort themselves and commit it with a partial flush.

### 1.2 Claimed guarantees

| ID | Property (paraphrased from §4) |
|---|---|
| S1 | Each worker contribution is applied at most once. |
| S2 | At most one value is chosen per log slot. |
| S3 | An acknowledged (`applied`) update survives crashes. |
| S4 | Committed updates respect version and staleness bounds. |
| S5 | Invalid input (non-finite, malformed, oversized) is rejected and does not affect the model. |
| S6 | An authoritative read reflects every acknowledged write. |
| S7 | Replicas apply the chosen log deterministically to identical state. |
| S8 | A replica without a quorum fails closed for commits and authoritative reads. |
| L1 | Training continues after worker loss. |
| L2 | Workers can join and rejoin. |
| L3 | Training continues after one voter fails. |
| L4 | Training continues after a reducer fails. |

### 1.3 Fault model (§5)

The fault model includes crash-stop and crash-recover processes, message loss, delay, reordering and duplication, partitions, heterogeneous worker speed, and bounded jobs (`max_job_updates`, 1,000 by default).

It **excludes** the following:

- Byzantine behaviour;
- corruption, rollback or loss of state on **a majority** of voters;
- a voter restart that comes back without its storage ("amnesia").

Clients are trusted.

---

## 2. Test design

### 2.1 Harnesses

| Harness | What it exercises | Files |
|---|---|---|
| In-process | The real FastAPI apps, `ConsensusRuntime`, `GradientReducer`, `MirroredStore` and codecs, wired together with `TestClient` and fake transports. Fast and deterministic. | [`tests/test_redteam_spec_gaps.py`](../tests/test_redteam_spec_gaps.py), [`tests/test_redteam_spec_gaps_round2.py`](../tests/test_redteam_spec_gaps_round2.py) |
| Paxos fuzzer | `FlakyCluster` transport: drops 15% of requests, loses 15% of replies, restarts 5% of acceptors from durable files, with concurrent proposers. Checks S2 over 12 seeds. | [`tests/test_redteam_paxos_fuzz.py`](../tests/test_redteam_paxos_fuzz.py) |
| HOKEA | Real containers built from the project Dockerfile: 3 voters plus 1–5 worker or reducer pods on a Docker network. HOKEA supplies `kill`, `start`, `isolate` and `heal` (iptables). `/data` is a per-node tmpfs. | [`hokea_tests/test_redteam_hokea.py`](../hokea_tests/test_redteam_hokea.py) |

### 2.2 Workloads

| Workload | Description |
|---|---|
| **single-update** | One gradient per voter, then read every voter's `model_sha256`, `processed_update_ids_sha256` and `learned_prefix_sha256`. Used for S2, S7 and amnesia checks. |
| **rotation** | Three rounds of "write at voter *i* while isolating voter *i+1*, heal, read at *i+1*". Used for S6 and S8. |
| **training** | Real workers training `fashion_cnn_v5` for up to 120 s. Progress is the committed `model_version` sampled across voters. Used for L1 and L4. |
| **adversarial input** | Crafted gradients: very large but finite, wrong tensor shape, int8 scale overflow, a compressed bomb, a reserved ID prefix, and a future `base_version`. Used for S4 and S5. |
| **retry** | The same `update_id` sent through primary and backup reducers, or after a voter restart. Used for S1 and S3. |

### 2.3 Nemeses

| Nemesis | Mechanism |
|---|---|
| Worker crash | `cluster.kill(worker)` (HOKEA). An in-process member never returns. |
| Voter isolate and heal | HOKEA `net.isolate` and `net.heal`. In-process transport drop. |
| Voter amnesia | HOKEA `kill` then `start` (tmpfs is wiped). In-process: delete the acceptor files. Excluded by §5; used as a control. |
| Single-voter disk rollback | Snapshot one voter's `/data/peer`, then restore it later under the same identity. **Not excluded by §5.** |
| Reducer failover | Kill the primary before or after it forwards. The backup takes over with the same ID. |
| Timing skew | A worker that arrives after the 0.75 s reducer flush deadline. Equal inputs with different arrival orders. |

### 2.4 Checkers

- **S2:** for each slot, every voter that learned it holds the same value hash.
- **S1:** for each `update_id`, how many times it was applied (model version deltas).
- **S7:** after any history, equal `learned_prefix_sha256` implies equal model, dedup set and metrics.
- **Liveness:** `model_version` advances within a window (30–120 s) after the nemesis ends.
- **S5:** an invalid input returns a 4xx response, the model stays finite, and later honest requests succeed.

---

## 3. Results

| Property | Verdict |
|---|---|
| S1 at-most-once | **Violated** (RT-09, RT-03) |
| S2 one value per slot | **Violated under a permitted fault** (RT-17). No violation in 12-seed fuzzing under the stated assumptions. |
| S3 durable ack | Holds in peer mode. **Violated** in default single-coordinator mode (RT-21). |
| S4 version bounds | Holds at voters. **Violated at reducers** (RT-06). |
| S5 input validity | **Violated** (RT-01, RT-02, RT-03, RT-04, RT-07, RT-08) |
| S6 read freshness | Holds. The first read after failover usually returns 503 (RT-16). |
| S7 deterministic application | Holds for weights. **Violated** for dedup state and metrics (RT-03). |
| S8 fail closed | Holds for commits and `/v1/model`. |
| L1 worker loss | **Violated** (RT-01, RT-05, RT-06, RT-07, RT-10, RT-18, RT-19) |
| L2 join and rejoin | Partial (RT-06, RT-18) |
| L3 voter failover | Holds with client retries. The repository's evidence for it relies on an excluded fault (RT-15). |
| L4 reducer failover | Holds at component level. The backup inherits RT-06, RT-07 and RT-10. |

Severity assumes trusted clients, as §5 does. RT-03, RT-06 and RT-12 become more serious with untrusted clients.

### 3.1 Safety violations

#### RT-17: Rolling back one voter's disk lets a slot be chosen twice (S2) · **High** · HOKEA ✔

§5 excludes rolled-back state only "for a majority" of voters, so restoring one voter from an older snapshot is in scope. Paxos safety depends on each acceptor never forgetting a promise or an acceptance. If one acceptor of three forgets, a new proposer can form a majority with that acceptor and a voter that never saw the first value.

**HOKEA history** ([log 11](../logs/11_hokea_redteam.log), [log 13](../logs/13_hokea_redteam_rerun.log)):

```
t0  node1 ← POST first        → 200 applied, model_version 1
    {"event":"rt17_first_commit","node_id":"node-1","model_version":1,"model_sha256":"d00108c3…"}
t1  snapshot taken before t0 restored onto node2 /data/peer   (same identity, same process)
t2  node1 isolated
t3  node3 ← POST second       → 200 applied, model_version 1
    {"event":"rt17_after_second_proposer","node_id":"node-3","model_version":1,"model_sha256":"0772b07a…"}
```

Slot 1 has two chosen values: `d00108c3…` and `0772b07a…`. Both were acknowledged to clients as `applied`.

Root cause: `slot-*.json` and `proposer.json` are written atomically but carry no epoch, checksum or monotonic counter that a peer could compare (`consensus.py:129-154`, O-9). The specification should either exclude rollback of *any* voter or require detection.

#### RT-09: A retried worker update is applied twice across reducer failover (S1) · **High** (in the `allow_stale` profile)

The primary and backup reducers share one reducer ID. Each builds its own batch ID from its local membership, and the voter deduplicates only aggregate IDs. When the primary forwards `w1:u1` and dies before replying, the worker retries at the backup. The backup builds a different aggregate containing the same `w1:u1`. With `allow_stale`, the voter applies it again.

```
primary  ← w1:u1  → forwards agg A{w1:u1} → voter applies, v0→v1 ; primary dies before replying
backup   ← w1:u1 (retry) → forwards agg B{w1:u1}, base 0 → voter (allow_stale) applies, v1→v2
assert model_version == 1   →   2 == 1  FAILED
```

The selected profile (staleness 0) is protected only by accident: B is stale. The specification never defines "worker contribution" (§6 of the [gap analysis](../gap-analysis/REDTEAM_SPEC_GAP_ANALYSIS.md)).

#### RT-21: An acknowledged update is lost after restart in default mode (S3) · Medium

The default `election_backend` is `single`: one process that writes to three mirror directories with write quorum W and read quorum R. `validate()` accepts any 1 ≤ W, R ≤ N, including W = 1, R = 2, where W + R ≤ N.

```
config W=1 R=2 N=3 ; POST u1 → 200 applied (model_version 1) ; written to mirror 0 only
restart ; read quorum picks mirrors 1,2 → model_version 0
assert (0 == 1) FAILED
```

#### RT-03: A crafted aggregate ID makes replicas' dedup state diverge (S5, S7, S1) · Medium · HOKEA ✔

The reserved prefix `__elasticml_aggregate_v1__:` is rejected only when aggregation is enabled. In direct mode (the HOKEA config for this test), a worker-supplied ID with that prefix is committed. The proposer treats it as an ordinary ID. Voters that learn the slot by recovery decode it as an aggregate manifest, add the named members (here `victim-1`) to their processed set, and add 1,000,000 examples to their metrics. Later, the genuine `victim-1` is answered `duplicate` and never applied.

```
{"event":"rt03_voter_config","peer_log_compaction_enabled":true,"gradient_aggregation_enabled":false,...}
node1 ← POST "__elasticml_aggregate_v1__:eNqrVspNzU1KLSpWsoqOV…" → 200 applied, v1
{"event":"rt03_states",
  "1":{"processed_update_count":1,"learned_prefix_sha256":"3779325…"},
  "2":{"processed_update_count":2,"learned_prefix_sha256":"3779325…"},
  "3":{"processed_update_count":2,"learned_prefix_sha256":"3779325…"}}
```

The voters hold the same learned prefix but different state, which violates S7 for dedup state.

#### RT-06: One future-base request permanently freezes the reducer tree (S4, L1, L4) · **High** · HOKEA ✔

Reducers trust the worker's `base_version`, and `_max_observed_base_version` only ever increases (`gradient_reducer.py:192-206`). After one POST with `base_version = 10000`, every honest worker's gradient is older than the observed maximum and is answered `stale`.

```
5-worker reducer tree (1 logical reducer × 2 replicas on pods 4,5), training
{"event":"rt06_before_ghost","versions":{"1":11,"2":11,"3":11}}
pod4 ← POST base_version 10000 → 200 {"status":"pending","model_version":10000}
pod5 ← POST base_version 10000 → 200 {"status":"pending","model_version":10000}
{"event":"rt06_after_ghost","versions_at_ghost":23,"versions":{"1":23,"2":23,"3":23}}
… no further commits for 120 s
```

The rerun froze at v5 instead of v23. The behaviour is identical. A worker from an old job that rejoins after a reset produces exactly this input, so the trigger needs no malice.

### 3.2 Liveness violations

#### RT-05: Survivors stall forever after one worker crashes (L1) · **High** · HOKEA ✔

With voter-side aggregation (the HOKEA default for four or fewer workers), a cohort commits when it is full or when the partial flush (2.0 s, at least ⌈cohort/2⌉ members) fires. The flush is checked **only inside the gradient POST handler** (`state.py:427-440`). Surviving workers whose gradients are `pending` poll `/v1/updates/status`, which never flushes. After one worker dies, nobody POSTs again.

```
3 workers, cohort 3, training
{"event":"rt05_before_kill","versions":{"1":12,"2":12,"3":12}}
kill worker pod
{"event":"rt05_after_kill","versions":{"1":14,"2":14,"3":14},
 "status":{"aggregate_commits":14,"pending_updates":2,"dropped":0,...}}
… versions unchanged for 120 s
```

The rerun showed 13/14 → 15, then stopped with `pending_updates: 2`.

#### RT-01: One very large but finite gradient blocks all voters (S5, L1) · **High** · HOKEA ✔

Validation checks `isfinite` on each gradient, but the weighted cohort sum (`grad × sample_count`) is not checked. A 1e36 gradient with `sample_count` 1000 overflows to `inf` at commit time. `CodecError` is raised inside the commit path, so every later POST returns HTTP 500, and the bucket is never cleared. The worker's client then fails over to the next voter, carrying the same payload.

```
node1 ← big (1e36 × 1000) → 200 pending
node1 ← u2 → 500 Internal Server Error
node1 ← u3 → 500 Internal Server Error
node2 ← big (failover)   → 200 pending
node2 ← u2 → 500 ; node2 ← u3 → 500
{"event":"rt01_result","codes":[200,500,500,200,500,500],"versions":{"1":0,"2":0,"3":0}}
```

Identical in both HOKEA runs.

#### RT-07: A wrong-shape gradient blocks a reducer cell (S5, L1) · Med-High

Reducers do not validate tensor names or shapes on admission. The bad member stays in the bucket, every batch formation raises `ValueError: gradient tensor sets and shapes must match`, and `status()` returns 500. Honest workers in the cell are blocked.

#### RT-10: Upstream 422 or 429 leaves workers pending forever (L1, L4) · Medium

The reducer ignores 422 and 429 replies from the voter (`gradient_reducer.py:308-321`). Members stay `pending` with no terminal status and no retry.

#### RT-18: A slow but live worker never contributes (§5 heterogeneous speed, L1, L2) · Medium

A worker that finishes 1.0 s into each round, against a 0.75 s reducer flush, misses every batch. It is answered `stale` 6 times out of 6. The specification includes heterogeneous speed in the fault model but makes no fairness or starvation guarantee.

#### RT-19: Job completion is reported as `stale` through reducers (§5 bounded jobs) · Medium

The voter returns 409 `job_complete` at `max_job_updates`. The reducer maps that reply to `{"status":"stale","model_version":2}`, with a made-up version of base+1. Workers in tree mode loop forever instead of stopping.

#### RT-16: The first read after failover usually returns 503 (L3) · Low · HOKEA observed

A lagging replica serving an authoritative read runs recovery prepares with a ballot lower than the writer's promise, and does not retry internally. The read is correct when retried.

```
{"event":"s6_round","writer":2,"reader":3,"acknowledged":2,"first_status":503,"observed":2}
{"event":"s6_round","writer":3,"reader":1,"acknowledged":3,"first_status":503,"observed":3}
```

This happened in 2 of 3 rotations, in both HOKEA runs.

### 3.3 Input-validation violations (S5)

| ID | Sev. | Finding |
|---|---|---|
| RT-02 | Med-High | Gradient magnitude is unbounded and model finiteness is not enforced. Two accepted 3e38 gradients with momentum 0.9 drive the weights to `inf`, after which every `/v1/model` read fails. Checkpoint errors are swallowed. |
| RT-04 | Medium | Only the *encoded* size is limited (4,000,000 base64 characters per field, 9 fields). A 20 KB NPZ allocated 32 MB before rejection, and an 83 KB one allocated 128 MB, so a decompression bomb is possible. |
| RT-08 | Low-Med | The reducer's decoder checks finiteness *before* int8 dequantization, so a scale × 127 that overflows passes. The voter's decoder checks after dequantization, so the two decoders disagree. |

### 3.4 Deployment, security and observability

| ID | Sev. | Finding |
|---|---|---|
| RT-12 | Medium | Peer mode validates with `auth_token=None`, leaving Paxos `prepare` and `accept` unauthenticated. Default tokens (`dev-only-change-me`, `elasticml-hokea-local`) are committed to the repository. |
| RT-11 | Low | The `duplicates`, `dropped` and `dropped_examples` counters are in memory only and reset on restart. |
| RT-22 | Medium | Worker membership is per voter and not replicated. Two voters with different registration views put worker `w3` in cell 0 and in cell 2. |
| RT-26 | Medium | §5 says compaction is "not part of the selected runtime". Every HOKEA voter hard-codes `peer_log_compaction_enabled: True` (confirmed in a live voter's `/data/coordinator.json`, and printed in the `rt03_voter_config` event). |
| RT-13 | Low-Med | "The loader verifies published hashes" holds only with `download=True`. Existing or tampered archives, and the `.npy` arrays actually used for training, are never hashed. |

### 3.5 Evidence and reproducibility claims

| ID | Sev. | Finding |
|---|---|---|
| RT-27 | Medium | §10 says the local HOKEA suite runs in Docker. `hokea_tests/conftest.py` starts 3 nodes, but the image requires 3 + workers (6 by default). Every container exits with `ValueError: HOKEA must provide exactly 6 pods`, after 717 s ([log 10](../logs/10_hokea_baseline_happy_path.log)). |
| RT-15 | Info | Control experiment for an excluded fault. A HOKEA `start` wipes `/data` (`learned_prefix_length: 0`), and slot 1 is then re-chosen (`d00108c3…` became `0772b07a…`). Nothing detects this. The repository's own L3 failover evidence restarts a voter, so it runs inside this excluded fault. |
| RT-20 | Medium | §7 says "one deterministic run". With identical inputs, the committed model depends on arrival time relative to the flush deadline. |
| RT-24 | Low-Med | "Unique committed examples" is 2,444,800 against a 55,000-example training split, so it counts exposures, not unique examples. |
| RT-25 | Medium | §7 promises equal-seed fault comparisons. No retained `fashion_cnn_v5` evidence has a churn, loss or failure run (`faults_injected: 0`). |
| RT-14a | Low | Python 3.11–3.14 is claimed. CI tests only 3.12 and 3.14. |
| RT-14b | Medium | The specification names commit `17f8bd4`. The clean-artifact evidence names `ff56901`. |
| RT-14c | Low | `HOKEA_NET_BACKEND` is documented but read nowhere in ElasticML or HOKEA. |
| RT-14d | Medium | The selected 94.68% run record lacks commit, script, per-role seeds, p50/p95, commit latency, CPU, memory, bytes, churn, network profile, batch size and fan-in, and its raw log is absent. |
| RT-23 | Low | The dashboard still uses the `good` profile, "Good day" labels and `simulated_hours`, which §8 and §12 say were replaced. |

---

## 4. What held up

- **S2 under the stated assumptions.** 12 randomized schedules with 15% request loss, 15% reply loss, 5% durable restarts and concurrent proposers found no slot with two values.
- **S6.** In-process and on HOKEA, a read at a voter that missed every acknowledged write always reflected all of them, once it returned 200.
- **S8.** An isolated voter refused to acknowledge commits or serve authoritative `/v1/model` reads.
- **L4.** A backup reducer committed after its primary died, either before forwarding or after committing, without double application (in the default staleness-0 profile).
- **Durability in peer mode.** Acceptor writes use `fsync` plus a directory `fsync`.

---

## 5. Gaps in the specification itself

The specification is silent on the following points, so they cannot be tested as written:

1. What a "worker contribution" is (S1). In tree mode, voters never see worker IDs.
2. Which reads are authoritative. Only `/v1/model` uses quorum. The models workers train on come attached to acknowledgements or reducer caches.
3. Any number behind "bounded": cell size, flush timeouts, buffers, staleness limit, request body size.
4. Liveness time bounds, and fairness for slow workers.
5. Numerical validity: the model is not required to stay finite.
6. Minority disk faults: rollback and bit corruption of unchecksummed `slot-*.json` and `proposer.json`.
7. Detection of excluded faults.
8. Any authentication requirement.
9. Durability scope: process crash versus power loss.
10. Which deployment mode the guarantees cover. The default `single` mode is not peer Paxos.
11. The selected run profile, which exists only in code and evidence.
12. Job lifecycle: what happens at `max_job_updates`, and how old workers are fenced out.
13. Reads acting as proposers, which the proposer-contention assumption does not account for.
14. Membership consistency across voters.
15. Definitions of the metrics ("unique", "dropped", "duplicate", "recovery time").

More than 30 implementation constants that the specification does not state are listed in §7 of the [full gap analysis](../gap-analysis/REDTEAM_SPEC_GAP_ANALYSIS.md#7-values-the-implementation-uses-that-the-specification-does-not-state). Eighteen untested observations (O-1 to O-18) are in its §8.

---

## 6. Recommendations

1. **Fix the fault model or the protocol.** Either exclude rollback of *any* voter in §5, or add an epoch or monotonic counter to acceptor state so that a peer can detect a rolled-back acceptor and stop it from voting (RT-17, RT-15).
2. **Flush on a timer, not on POST.** Run the voter partial flush from a background task, or from the status-poll path as well (RT-05).
3. **Bound inputs after decoding.** Check finiteness of the weighted sum and of the post-step model, limit gradient norm, cap decoded size before allocation, dequantize before checking finiteness, and validate shapes on reducer admission (RT-01, RT-02, RT-04, RT-07, RT-08).
4. **Never trust client versions.** Clamp `base_version` to the reducer's last observed upstream version, and reset it per job (RT-06).
5. **Make deduplication end to end.** Carry worker `update_id`s through aggregates and deduplicate at the voter state machine. Reject the reserved prefix in every mode (RT-09, RT-03).
6. **Validate W + R > N** in `MirroredStore` configuration (RT-21).
7. **Give reducers terminal states.** Propagate 409, 422 and 429 replies to workers verbatim (RT-10, RT-19).
8. **Require authentication in peer mode**, and remove the committed default tokens (RT-12).
9. **Make the evidence reproducible.** Fix the HOKEA conftest node count, disable compaction or document it, record every §7 field, and name the actual commit (RT-14, RT-24–RT-27).

---

## 7. Reproducing

The tests import the `elasticml` package. Copy `tests/` and `hokea_tests/` into the ElasticML MS3 tree. HOKEA must be installed from `../hokea` at `427b946`.

```bash
PYTHONPATH=src python -m pytest -q tests/test_redteam_spec_gaps.py tests/test_redteam_spec_gaps_round2.py tests/test_redteam_paxos_fuzz.py
```

```bash
PYTHONPATH=src python -m pytest -q --runxfail tests/test_redteam_spec_gaps.py tests/test_redteam_spec_gaps_round2.py tests/test_redteam_paxos_fuzz.py
```

```bash
unset HOKEA_NAMESPACE KUBECONFIG; PYTHONPATH=src python -m pytest -v -s hokea_tests/test_redteam_hokea.py
```

| Run | Result | Log |
|---|---|---|
| Full suite, including red-team tests | 601 passed, 29 xfailed | [01](../logs/01_full_suite.log) |
| Red-team in-process tests | 17 passed, 29 xfailed | [02](../logs/02_redteam_verbose.log) |
| Same tests with `--runxfail` | 29 failed (each at its requirement assertion), 17 passed | [03](../logs/03_redteam_runxfail_tracebacks.log), [04](../logs/04_redteam_failure_lines.log) |
| Repository's own HOKEA suite | ERROR, cannot start (RT-27) | [10](../logs/10_hokea_baseline_happy_path.log) |
| HOKEA red-team run 1 | 2 passed, 5 xfailed, 386.95 s | [11](../logs/11_hokea_redteam.log), [12](../logs/12_hokea_redteam_events.log) |
| HOKEA red-team run 2 | 2 passed, 5 xfailed, 387.99 s | [13](../logs/13_hokea_redteam_rerun.log) |

## 8. Limits

- No run on the class Kubernetes cluster.
- The HOKEA expected failures are supported by structured event lines, not by a `--runxfail` rerun on HOKEA.
- The fuzzer is a randomized search, not a proof. Deterministic in-process tests cannot exhaust asynchronous schedules.
- AdamW was not tested for RT-02. O-12 (reads pre-empting writers) is an analysis without a deterministic reproduction.
- The ElasticML source code and the specification PDF are not included in this repository.
