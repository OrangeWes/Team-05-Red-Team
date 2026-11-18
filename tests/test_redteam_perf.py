"""Red-team performance tests (in-process), Team-05 round 3.

The MS3 specification (§7) says performance results are "logged measurements
from specific runs, not guarantees", so it gives nothing to assert against.
These tests therefore do two things:

1. **Calibrate and normalise.**  They measure what one CPU core of the test
   host can do with the real ``fashion_cnn_v5`` model and compare it with the
   rates implied by the selected run in
   ``docs/evidence/model/fashion_cnn_v5_hokea_lr006_20.json`` (20 one-core
   workers, 1355.89 examples/s, 0.53 commits/s, 94% at 945.01 s).  Local runs
   of the "full run" are much slower because this host has 8 cores, not the
   27 the selected topology needs.  Instead of comparing raw seconds we state
   an explicit assumption: **the class cluster's cores are about as fast as
   one core of the test host** (``ASSUMED_REFERENCE_CORE`` below) and derive
   every expectation from that.

2. **Assert assumed bounds.**  Where the spec promises a behaviour (L1:
   training continues after worker loss; §5: churn and heterogeneous speed
   are in the fault model) but no number, we state a bound that a reasonable
   reader would expect and test it.  Each bound is labelled ``ASSUMPTION``.
   Failing bounds are ``xfail(strict=True)`` with a PF-xx id, as in the other
   red-team files.

Every test prints one ``ELASTICML_PERF {json}`` line so the log carries the
raw numbers.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from elasticml.array_codec import arrays_to_b64, b64_to_arrays
from elasticml.config import CoordinatorConfig
from elasticml.gradient_reducer import GradientReducer
from elasticml.models import build_model
from elasticml.quantization import dequantize_int8_arrays, quantize_int8_error_feedback
from test_peer_runtime import PeerCluster
from test_redteam_spec_gaps import (REDUCER_ID, VoterUpstream, gradient_request,
                                    reducer_payload, tree_voter)

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "docs/evidence/model/fashion_cnn_v5_hokea_lr006_20.json"

# Selected-run constants (spec §7 and the evidence file).
REF_WORKERS = 20
REF_EXAMPLES_PER_SECOND = 1355.89
REF_COMMITS_PER_SECOND = 0.53
REF_TIME_TO_94 = 945.01
REF_BATCH_PER_WORKER = 128          # effective batch 2560 / 20 workers
REF_COMMIT_BUDGET = 1 / REF_COMMITS_PER_SECOND   # 1.887 s between commits

# Accuracy-profile reducer settings used by the repository's full-run harness
# (conftest.py: reducer flush 60 s for --elasticml-accuracy-profile, minimum
# partial size 1, one reducer replica).
ACCURACY_FLUSH_SECONDS = 60.0

# ASSUMPTION A1: a class-cluster core is within 2x of one test-host core.
ASSUMED_REFERENCE_CORE_RANGE = (0.5, 2.0)
# ASSUMPTION A2: after a worker leaves, the survivors' commit rate should
# recover to within a few healthy commit intervals (we allow 5 x 1.887 s).
ASSUMED_DEPARTURE_ROUND_BOUND = 5 * REF_COMMIT_BUDGET


def perf(event: str, **fields) -> None:
    print("ELASTICML_PERF " + json.dumps({"event": event, **fields}, default=float), flush=True)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _local_core_rate(model_name: str = "fashion_cnn_v5", batch: int = REF_BATCH_PER_WORKER,
                     repeats: int = 5) -> float:
    """Examples/s for one forward+backward pass on one thread."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = build_model(model_name)
        loss_fn = torch.nn.CrossEntropyLoss()
        x = torch.randn(batch, 1, 28, 28)
        y = torch.randint(0, 10, (batch,))
        for _ in range(2):
            model.zero_grad()
            loss_fn(model(x), y).backward()
        times = []
        for _ in range(repeats):
            started = time.perf_counter()
            model.zero_grad()
            loss_fn(model(x), y).backward()
            times.append(time.perf_counter() - started)
        return batch / statistics.median(times)
    finally:
        torch.set_num_threads(previous)


# --------------------------------------------------------------------------
# PF-01  Calibration and the "reasonable hardware" assumption
# --------------------------------------------------------------------------

def test_pf01_local_core_calibration_against_selected_run():
    """Normalise the selected run to this host.

    If the class cluster's cores are as fast as ours (A1), 20 one-core workers
    could compute ``20 x local_rate`` examples/s.  The selected run reports
    1355.89 examples/s, so the fraction of worker CPU time spent computing
    gradients is ``1355.89 / (20 x local_rate)``.  The rest is coordination
    and waiting.
    """
    evidence = json.loads(EVIDENCE.read_text())
    assert evidence["performance"]["unique_committed_examples_per_second"] == pytest.approx(
        REF_EXAMPLES_PER_SECOND)
    local_rate = _local_core_rate()
    per_worker_ref = REF_EXAMPLES_PER_SECOND / REF_WORKERS
    ideal_ref = REF_WORKERS * local_rate
    compute_fraction = REF_EXAMPLES_PER_SECOND / ideal_ref
    compute_seconds_per_commit = REF_BATCH_PER_WORKER / local_rate
    examples_to_94 = REF_TIME_TO_94 * REF_EXAMPLES_PER_SECOND
    perf("pf01_calibration",
         local_core_examples_per_second=round(local_rate, 1),
         reference_examples_per_second_per_worker=round(per_worker_ref, 1),
         reference_compute_fraction=round(compute_fraction, 3),
         reference_commit_interval_seconds=round(REF_COMMIT_BUDGET, 3),
         local_compute_seconds_per_commit=round(compute_seconds_per_commit, 3),
         examples_to_94_percent=round(examples_to_94),
         projected_time_to_94_if_compute_bound_seconds=round(examples_to_94 / ideal_ref, 1))
    # A1 sanity: one local core must at least sustain the per-worker rate the
    # reference achieved, otherwise "reasonable hardware" would be faster than
    # ours and every projection below would be optimistic.
    assert local_rate >= per_worker_ref * ASSUMED_REFERENCE_CORE_RANGE[0]


# --------------------------------------------------------------------------
# PF-02  Where the 1.887 s commit interval goes
# --------------------------------------------------------------------------

class FashionPeerCluster(PeerCluster):
    """Three in-process peer voters running the MS3 model and aggregation."""

    def __init__(self, root, monkeypatch):
        super().__init__(root, monkeypatch)
        for name in "abc":
            self.configs[name] = CoordinatorConfig(
                node_id=name, public_url=f"http://{name}", election_backend="peer",
                peer_urls={peer: f"http://{peer}" for peer in "abc"},
                peer_state_root=str(root / f"fashion-{name}"), checkpoint_every=25,
                max_staleness=0, max_job_updates=1000, model="fashion_cnn_v5",
                optimizer="adamw", learning_rate=0.006,
                gradient_aggregation_enabled=True, gradient_aggregation_cohort_size=1,
                gradient_reducer_count=1, gradient_aggregation_commit_dtype="float16")
            self.restart(name)


def test_pf02_coordination_path_fits_the_reference_commit_interval(tmp_path, monkeypatch):
    """Time every non-compute step of one synchronous round at fan-in 7.

    7 is the largest leaf cell of the selected 20-worker tree (7/7/6).  The
    measured path is: each worker int8-encodes its gradient, the reducer
    decodes and averages them, then the reducer forwards a float16 aggregate
    that three in-process voters commit through Paxos.  The network is not
    included, so this is a lower bound on coordination cost.
    """
    peers = FashionPeerCluster(tmp_path, monkeypatch)
    try:
        state = peers.runtimes["a"].state
        rng = np.random.default_rng(7)
        gradients = {name: rng.standard_normal(tuple(p.shape)).astype(np.float32) * 1e-3
                     for name, p in state.model.named_parameters()}
        fan_in = 7
        encode_times, round_times = [], []
        for round_index in range(3):
            reducer = GradientReducer(REDUCER_ID, VoterUpstream(peers.clients["a"]),
                                      group_size=fan_in, wire_dtype="float16")
            base = state.version
            payloads = []
            for _ in range(fan_in):
                started = time.perf_counter()
                encoded, _ = quantize_int8_error_feedback(gradients)
                payloads.append(arrays_to_b64(encoded))
                encode_times.append(time.perf_counter() - started)
            started = time.perf_counter()
            result = None
            for worker in range(fan_in):
                result = reducer.submit(gradient_request(
                    f"w{worker}:{round_index}", f"w{worker}", base, payloads[worker],
                    samples=REF_BATCH_PER_WORKER))
            round_times.append(time.perf_counter() - started)
            assert result["status"] == "applied", result
        local_rate = _local_core_rate()
        compute = REF_BATCH_PER_WORKER / local_rate
        coordination = statistics.median(encode_times) + statistics.median(round_times)
        perf("pf02_commit_budget", fan_in=fan_in,
             worker_int8_encode_seconds=round(statistics.median(encode_times), 4),
             reducer_plus_paxos_commit_seconds=round(statistics.median(round_times), 4),
             worker_compute_seconds=round(compute, 3),
             reference_commit_interval_seconds=round(REF_COMMIT_BUDGET, 3),
             unexplained_seconds_per_commit=round(REF_COMMIT_BUDGET - compute - coordination, 3))
        assert compute + coordination < REF_COMMIT_BUDGET
    finally:
        peers.close()


# --------------------------------------------------------------------------
# PF-03  A departed worker costs every later round the full flush timeout
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=(
    "PF-03: reducer group_size is fixed at start-up, so after one worker leaves every round in "
    "its cell waits the full partial-flush timeout (60 s in the accuracy profile) instead of "
    "recovering; survivors' commit interval grows from ~1.9 s to ~60 s until the worker returns"))
def test_pf03_departed_worker_does_not_stall_every_later_round(tmp_path):
    runtime, client = tree_voter(tmp_path)
    clock = Clock()
    group = 6
    reducer = GradientReducer(REDUCER_ID, VoterUpstream(client), group_size=group,
                              wire_dtype="float32", minimum_partial_size=1,
                              flush_timeout_seconds=ACCURACY_FLUSH_SECONDS, clock=clock)
    payload = reducer_payload(runtime.state)
    round_seconds = []
    for round_index in range(3):     # worker w5 has left; w0..w4 keep training
        base = runtime.state.version
        started = clock.now
        for worker in range(group - 1):
            reducer.submit(gradient_request(f"w{worker}:{round_index}", f"w{worker}", base, payload))
        while reducer.status(f"w0:{round_index}", base)["status"] != "applied":
            clock.now += 0.5             # survivors poll every 0.5 s
            assert clock.now - started < 10 * ACCURACY_FLUSH_SECONDS
        round_seconds.append(clock.now - started)
    survivors_rate = (group - 1) * REF_BATCH_PER_WORKER / statistics.mean(round_seconds)
    perf("pf03_departed_worker", group_size=group, flush_timeout_seconds=ACCURACY_FLUSH_SECONDS,
         round_seconds=round_seconds,
         survivor_examples_per_second=round(survivors_rate, 1),
         assumed_bound_seconds=round(ASSUMED_DEPARTURE_ROUND_BOUND, 2))
    assert max(round_seconds[1:]) <= ASSUMED_DEPARTURE_ROUND_BOUND, round_seconds


# --------------------------------------------------------------------------
# PF-04  The selected profile runs without backup reducers
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=(
    "PF-04 / RT-28: §2 says each primary reducer has a deterministic backup and L4 relies on it, "
    "but the accuracy profile used for the selected run sets ELASTICML_HOKEA_REDUCER_REPLICAS=1"))
def test_pf04_accuracy_profile_keeps_a_backup_reducer():
    conftest = (REPO / "conftest.py").read_text()
    start = conftest.index("if accuracy_profile:")
    block = conftest[start:conftest.index("if learning_rate is not None:", start)]
    replicas = None
    for line in block.splitlines():
        if "ELASTICML_HOKEA_REDUCER_REPLICAS" in line:
            replicas = line.split(":")[1].strip().strip(",").strip("'\"")
    perf("pf04_reducer_replicas", accuracy_profile_reducer_replicas=replicas)
    assert replicas == "2"


# --------------------------------------------------------------------------
# PF-05  Codec cost per update (no assertion; recorded for the report)
# --------------------------------------------------------------------------

def test_pf05_int8_codec_cost_and_payload_size():
    model = build_model("fashion_cnn_v5")
    rng = np.random.default_rng(3)
    arrays = {name: rng.standard_normal(tuple(p.shape)).astype(np.float32)
              for name, p in model.named_parameters()}
    started = time.perf_counter()
    encoded, _ = quantize_int8_error_feedback(arrays)
    payload = arrays_to_b64(encoded)
    encoded_at = time.perf_counter()
    decoded = dequantize_int8_arrays(b64_to_arrays(payload))
    decoded_at = time.perf_counter()
    float32_bytes = sum(a.nbytes for a in arrays.values())
    perf("pf05_codec", parameters=sum(a.size for a in arrays.values()),
         float32_bytes=float32_bytes, int8_base64_bytes=len(payload),
         compression_ratio=round(float32_bytes / len(payload), 2),
         encode_seconds=round(encoded_at - started, 4),
         decode_seconds=round(decoded_at - encoded_at, 4))
    assert set(decoded) == set(arrays)
    # Base64 adds 4/3 over raw int8, so the wire saving is about 3x, not 4x.
    assert float32_bytes / len(payload) < 4
