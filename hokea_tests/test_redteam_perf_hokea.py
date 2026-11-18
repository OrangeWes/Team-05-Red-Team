"""Red-team performance runs on a local HOKEA Docker cluster (round 3).

Three equal-seed training runs with the repository's accuracy profile
(``fashion_cnn_v5``, AdamW one-cycle, int8 worker transport, float16 commits,
synchronous reducer tree, 60 s reducer flush, one reducer replica), scaled
from 20 workers to 6 so the cluster fits on an 8-core host:

* ``healthy``          no faults.
* ``churn``            workers leave and rejoin: every 60 s one worker pod is
                       killed and started again 20 s later.  The victim order
                       is fixed and includes pod 4, which also hosts the only
                       reducer, because §5 says workers leave "somewhat
                       arbitrarily" and a reducer host is still a worker.
* ``churn_partition``  the same churn plus two 30 s isolations of voter 2.

§7 lists "equal-seed comparisons across healthy, churn, request-loss, and
replica-failure scenarios" as part of the evaluation but no retained MS3 run
has one (RT-25).  These runs provide it.  The spec gives no numeric
performance targets, so the bounds below are labelled ASSUMPTION; see
tests/test_redteam_perf.py for the hardware normalisation.

Run (Docker, never Kubernetes; about 35 minutes)::

    unset HOKEA_NAMESPACE KUBECONFIG
    PYTHONPATH=src python -m pytest -v -s hokea_tests/test_redteam_perf_hokea.py
"""
from __future__ import annotations

import json
import math
import statistics
import threading
import time

import pytest
import requests

from conftest import ROOT, TOKEN

HEADERS = {"Authorization": f"Bearer {TOKEN}"}
WORKERS = 6
SEED = 50929546                 # the selected run's seed
MEASURE_SECONDS = 420
CHURN_PERIOD = 60
CHURN_DOWN = 20
CHURN_VICTIMS = [7, 5, 9, 4, 6, 8]  # node indices; 4 hosts the only reducer
PARTITIONS = [(150, 30), (330, 30)]  # (start offset, duration) for voter 2
EVAL_PERIOD = 60
LOCAL_CORE_EXAMPLES_PER_SECOND = 255.3  # PF-01 on this host (Xeon Gold 6238R)

# ASSUMPTION A3: with workers leaving and rejoining, survivors should never
# go longer than worker TTL (15 s) + one 15 s margin without a commit.
ASSUMED_MAX_COMMIT_GAP = 30.0
# ASSUMPTION A4: losing one of six workers for a third of the time removes
# about 6% of compute; we accept up to a 30% throughput loss.
ASSUMED_MIN_THROUGHPUT_RATIO = 0.70

RESULTS: dict[str, dict] = {}


def accuracy_env(workers: int) -> dict[str, str]:
    """The repository's --elasticml-accuracy-profile environment, continuous."""
    effective_batch = 128 * workers
    updates = math.ceil(40 * 55_000 / effective_batch)
    warmup = max(1, min(100, round(updates * 0.15)))
    return {
        "ELASTICML_AUTH_TOKEN": TOKEN,
        "ELASTICML_ENABLE_DEBUG_ENDPOINTS": "1",
        "ELASTICML_ENABLE_EVALUATION_ENDPOINT": "1",
        "ELASTICML_EVALUATION_DATASET": "fashion_mnist",
        "ELASTICML_EVALUATION_CACHE": "/opt/elasticml/fashion-mnist",
        "ELASTICML_EVALUATION_SPLIT": "validation",
        "ELASTICML_HOKEA_WORKER_STEPS": "",
        "ELASTICML_HOKEA_MAX_UPDATES": "2000000000",
        "ELASTICML_HOKEA_TOTAL_WORKERS": str(workers),
        "ELASTICML_HOKEA_MAX_PODS": "40",
        "ELASTICML_HOKEA_SEED": str(SEED),
        "ELASTICML_HOKEA_DURATION_SECONDS": "0",
        "ELASTICML_HOKEA_DATASET": "fashion_mnist",
        "ELASTICML_HOKEA_MODEL": "fashion_cnn_v5",
        "ELASTICML_HOKEA_BATCH_SIZE": "128",
        "ELASTICML_HOKEA_DATA_PREPROCESSING": "unit",
        "ELASTICML_HOKEA_GRADIENT_BUNDLE_SIZE": "1",
        "ELASTICML_HOKEA_GRADIENT_WIRE_DTYPE": "int8",
        "ELASTICML_GRADIENT_AGGREGATION_COMMIT_DTYPE": "float16",
        "ELASTICML_GRADIENT_UPSTREAM_COHORT_SIZE": "1",
        "ELASTICML_GRADIENT_ALLOW_STALE_AGGREGATES": "0",
        "ELASTICML_HOKEA_MAX_STALENESS": "8",
        "ELASTICML_HOKEA_REDUCER_FLUSH_TIMEOUT_SECONDS": "60.0",
        "ELASTICML_HOKEA_REDUCER_MINIMUM_PARTIAL_SIZE": "1",
        "ELASTICML_HOKEA_REDUCER_REPLICAS": "1",
        "ELASTICML_GRADIENT_AGGREGATION_ENABLED": "1",
        "ELASTICML_HOKEA_REDUCER_TREE_ENABLED": "1",
        "ELASTICML_HOKEA_BATCHED_MICROSTEPS": "0",
        "ELASTICML_GRADIENT_AGGREGATION_COHORT_SIZE": str(workers),
        "ELASTICML_GRADIENT_AGGREGATION_FLUSH_TIMEOUT_SECONDS": "2.0",
        "ELASTICML_GRADIENT_AGGREGATION_MAX_PENDING_UPDATES": "256",
        "ELASTICML_GRADIENT_AGGREGATION_MAX_PENDING_BYTES": str(16 * 1024 * 1024),
        "ELASTICML_HOKEA_LEARNING_RATE": "0.006",
        "ELASTICML_HOKEA_LEARNING_RATE_SCHEDULE": "one_cycle_cosine",
        "ELASTICML_HOKEA_LEARNING_RATE_MIN": "0.000003",
        "ELASTICML_HOKEA_LEARNING_RATE_WARMUP_UPDATES": str(warmup),
        "ELASTICML_HOKEA_LEARNING_RATE_DECAY_UPDATES": str(updates - warmup),
        "ELASTICML_HOKEA_EFFECTIVE_BATCH_SIZE": str(effective_batch),
        "ELASTICML_HOKEA_PLANNED_OPTIMIZER_UPDATES": str(updates),
        "ELASTICML_HOKEA_PLANNED_EPOCHS": "40",
        "ELASTICML_HOKEA_MOMENTUM": "0",
        "ELASTICML_HOKEA_OPTIMIZER": "adamw",
        "ELASTICML_HOKEA_OPTIMIZER_WEIGHT_DECAY": "0.001",
        "ELASTICML_HOKEA_ADAM_BETA1": "0.9",
        "ELASTICML_HOKEA_ADAM_BETA2": "0.999",
        "ELASTICML_HOKEA_EVALUATION_EMA_DECAY": str(0.995 ** (effective_batch / 256)),
        "ELASTICML_HOKEA_LABEL_SMOOTHING": "0.05",
        "ELASTICML_HOKEA_WEIGHT_DECAY": "0",
    }


@pytest.fixture
def hokea_cluster(tmp_path, request):
    from hokea import Cluster
    scenario = request.node.name.split("_", 2)[2]
    return Cluster(dockerfile=str(ROOT), memory="1g", cpus=1.0, nodes=3 + WORKERS,
                   name=f"elasticml-perf-{scenario.replace('_', '-')[:20]}",
                   port=8100, peer_port=8100, env=accuracy_env(WORKERS),
                   workdir=tmp_path / "cluster", up_timeout=900)


def log(event, **fields):
    print("ELASTICML_PERF " + json.dumps({"event": event, **fields}, default=str), flush=True)


def get(node, path, timeout=4):
    return requests.get(f"{node.url}{path}", headers=HEADERS, timeout=timeout)


class Sampler(threading.Thread):
    """Polls every voter's learned version once a second.

    A single voter is not a reliable clock: a voter that is not proposing can
    lag and then learn many slots at once (seen on 2026-11-17 as jumps of ~25
    versions in 2 s).  The commit clock is therefore the most advanced voter;
    per-voter versions are kept to measure learner lag.  Only the cheap debug
    endpoint is used, because /v1/status runs a catch-up on the voter it asks.
    """

    def __init__(self, voters, started):
        super().__init__(daemon=True)
        self.voters, self.started, self.samples = voters, started, []
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            elapsed = time.monotonic() - self.started
            versions = []
            for voter in self.voters:
                try:
                    versions.append(int(get(voter, "/v1/debug/local-state", timeout=2)
                                        .json()["model_version"]))
                except (requests.RequestException, ValueError, KeyError):
                    versions.append(None)
            known = [v for v in versions if v is not None]
            if known:
                self.samples.append((elapsed, max(known), versions))
            self.stop.wait(1.0)


def reported_examples(voters):
    best = None
    for voter in voters:
        try:
            status = get(voter, "/v1/status", timeout=10).json()
            value = (status["model_version"], status["update_metrics"].get("included_examples", 0))
            best = value if best is None or value[0] > best[0] else best
        except (requests.RequestException, ValueError, KeyError):
            pass
    return best


def wait_for_training(voter, timeout=600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if int(get(voter, "/v1/debug/local-state").json()["model_version"]) >= 2:
                return
        except (requests.RequestException, ValueError, KeyError):
            pass
        time.sleep(2)
    raise AssertionError("cluster never started committing")


def evaluate(voter):
    for _ in range(3):
        try:
            response = get(voter, "/v1/evaluation", timeout=180)
            if response.status_code == 200:
                return float(response.json()["heldout_accuracy"])
        except requests.RequestException:
            pass
        time.sleep(2)
    return None


def run_scenario(name, cluster, net=None, churn=False, partitions=()):
    nodes = {node.index: node for node in cluster.nodes()}
    voter1, voter3 = nodes[1], nodes[3]
    wait_for_training(voter1)
    started = time.monotonic()
    voters = [nodes[1], nodes[2], nodes[3]]
    reported_start = reported_examples(voters)
    sampler = Sampler(voters, started)
    sampler.start()
    events, accuracy = [], [(0.0, evaluate(voter3))]
    schedule = []
    if churn:
        for i, victim in enumerate(CHURN_VICTIMS):
            schedule.append((30 + i * CHURN_PERIOD, "kill", victim))
            schedule.append((30 + i * CHURN_PERIOD + CHURN_DOWN, "start", victim))
    for offset, duration in partitions:
        schedule.append((offset, "isolate", 2))
        schedule.append((offset + duration, "heal", 2))
    for offset in range(EVAL_PERIOD, MEASURE_SECONDS + 1, EVAL_PERIOD):
        schedule.append((offset, "evaluate", 3))
    schedule.sort()
    for offset, action, index in schedule:
        delay = started + offset - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        at = round(time.monotonic() - started, 1)
        if action == "kill":
            cluster.kill(index)
        elif action == "start":
            cluster.start(index)
        elif action == "isolate":
            net.isolate(index)
        elif action == "heal":
            net.heal()
        elif action == "evaluate":
            accuracy.append((at, evaluate(voter3)))
            log("accuracy", scenario=name, at=at, accuracy=accuracy[-1][1])
            continue
        events.append((at, action, index))
        log("fault", scenario=name, at=at, action=action, node=index)
    remaining = started + MEASURE_SECONDS - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)
    sampler.stop.set()
    sampler.join(timeout=10)
    reported_end = reported_examples(voters)
    return summarise(name, sampler.samples, events, accuracy, reported_start, reported_end)


def summarise(name, samples, events, accuracy, reported_start=None, reported_end=None):
    assert len(samples) > 10, "sampler collected too little data"
    first, last = samples[0], samples[-1]
    span = last[0] - first[0]
    commits = last[1] - first[1]
    # Throughput is measured from commits: each commit is one synchronous
    # step of WORKERS x 128 examples.  The voter's own included_examples
    # counter is logged for comparison; on 2026-11-17 it rose ~3.7x faster
    # than commits x 768, more than six one-core workers can compute.
    examples = commits * 128 * WORKERS
    reported = (reported_end[1] - reported_start[1]) if reported_start and reported_end else None
    lags = [max(v) - min(v) for _, _, v in samples if None not in v]
    log("samples", scenario=name,
        samples=[(round(t, 1), v, per) for t, v, per in samples])
    commit_times = [cur[0] for prev, cur in zip(samples, samples[1:]) if cur[1] > prev[1]]
    gaps = [b - a for a, b in zip([first[0]] + commit_times, commit_times + [last[0]])]
    # Stall caused by each fault = the longest commit gap in the 70 s after it.
    # (The first commit after a kill is often one already in flight, so "time
    # to next commit" understates the stall; see log 14, 2026-11-17.)
    recovery = []
    for at, action, _ in events:
        if action in {"kill", "isolate"}:
            window = [t for t in commit_times if at - 2 <= t <= at + 70]
            stalls = [b - a for a, b in zip(window, window[1:])]
            recovery.append(round(max(stalls), 1) if stalls else None)
    examples_per_second = examples / span if span else 0.0
    targets = {}
    for target in (0.70, 0.80, 0.85):
        hit = next((t for t, a in accuracy if a is not None and a >= target), None)
        targets[f"time_to_{target:.2f}"] = hit
    result = dict(
        scenario=name, measured_seconds=round(span, 1), commits=commits,
        commits_per_second=round(commits / span, 3) if span else 0.0,
        examples=examples, examples_per_second=round(examples_per_second, 1),
        voter_reported_examples=reported,
        voter_reported_examples_per_second=round(reported / span, 1) if reported and span else None,
        max_learner_lag_versions=max(lags) if lags else None,
        compute_fraction=round(examples_per_second / (WORKERS * LOCAL_CORE_EXAMPLES_PER_SECOND), 3),
        max_commit_gap_seconds=round(max(gaps), 1) if gaps else None,
        median_commit_gap_seconds=round(statistics.median(gaps), 2) if gaps else None,
        stall_seconds_after_each_fault=recovery,
        accuracy=[(t, None if a is None else round(a, 4)) for t, a in accuracy],
        best_accuracy=max((a for _, a in accuracy if a is not None), default=None),
        **targets)
    log("summary", **result)
    RESULTS[name] = result
    return result


# --------------------------------------------------------------------------
# PH-01  healthy baseline
# --------------------------------------------------------------------------

def test_ph_healthy(cluster):
    result = run_scenario("healthy", cluster)
    assert result["commits"] > 0


# --------------------------------------------------------------------------
# PH-02  workers leave and rejoin
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=(
    "PH-02: confirmed 2026-11-17 (log 14): with one worker leaving and rejoining every 60 s, the "
    "commit rate fell to 61% of healthy (assumed bound 70%) and the longest stall was 34.7 s "
    "(assumed bound 30 s); 3 of 6 departures stalled every worker for 29-35 s"))
def test_ph_churn(cluster):
    result = run_scenario("churn", cluster, churn=True)
    baseline = RESULTS.get("healthy")
    if baseline is None:
        pytest.skip("healthy baseline did not run in this session")
    ratio = result["commits_per_second"] / max(baseline["commits_per_second"], 1e-9)
    log("churn_vs_healthy", throughput_ratio=round(ratio, 3),
        max_commit_gap=result["max_commit_gap_seconds"],
        assumed_max_commit_gap=ASSUMED_MAX_COMMIT_GAP,
        assumed_min_throughput_ratio=ASSUMED_MIN_THROUGHPUT_RATIO)
    assert result["max_commit_gap_seconds"] <= ASSUMED_MAX_COMMIT_GAP
    assert ratio >= ASSUMED_MIN_THROUGHPUT_RATIO


# --------------------------------------------------------------------------
# PH-03  churn plus a temporarily isolated voter
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=(
    "PH-03: confirmed 2026-11-17 (log 14): churn plus two 30 s isolations of voter 2 gave 56% of "
    "the healthy commit rate and a 34.2 s stall"))
def test_ph_churn_partition(cluster, net):
    result = run_scenario("churn_partition", cluster, net=net, churn=True, partitions=PARTITIONS)
    baseline = RESULTS.get("healthy")
    if baseline is None:
        pytest.skip("healthy baseline did not run in this session")
    ratio = result["commits_per_second"] / max(baseline["commits_per_second"], 1e-9)
    log("churn_partition_vs_healthy", throughput_ratio=round(ratio, 3),
        max_commit_gap=result["max_commit_gap_seconds"])
    assert result["max_commit_gap_seconds"] <= ASSUMED_MAX_COMMIT_GAP
    assert ratio >= ASSUMED_MIN_THROUGHPUT_RATIO
