"""Round 3: the remaining red-team tests, run on a local HOKEA Docker cluster.

The in-process tests in tests/ exercise ElasticML's code directly.  This file
drives the same findings against the running system instead: real voter,
worker and reducer containers, real HTTP, and HOKEA faults (kill, start,
isolate, heal, netem loss and delay, CPU throttling).  Requests that no honest
worker would send are injected from inside a worker pod with ``node.exec``,
exactly as RT-06 does in test_redteam_hokea.py.

IDs match the gap analysis.  ``xfail(strict=True)`` = the requirement was
confirmed NOT met on the running cluster.

Not portable to HOKEA (they stay in-process only):
  RT-11  counter persistence: a HOKEA restart wipes /data (excluded amnesia).
  RT-13  dataset hashes: the dataset is baked into the image at build time.
  RT-14a-d, RT-23-RT-25  documents and evidence files, not runtime behaviour.
  RT-21  single-coordinator MirroredStore: the HOKEA image always runs peer mode.
  PF-01/02/05  single-core micro-benchmarks.

Run (Docker, never Kubernetes; about 40 minutes)::

    unset HOKEA_NAMESPACE KUBECONFIG
    PYTHONPATH=src python -m pytest -v -s hokea_tests/test_redteam_hokea_round3.py
"""
from __future__ import annotations

import json
import random
import threading
import time

import pytest
import requests

from conftest import ROOT, TOKEN

HEADERS = {"Authorization": f"Bearer {TOKEN}"}
IDLE = {"ELASTICML_HOKEA_WORKER_STEPS": "0"}
TREE5_IDLE = {"ELASTICML_GRADIENT_AGGREGATION_ENABLED": "1", **IDLE}

SCENARIOS = {
    "rt02": dict(workers=1, env={**IDLE, "ELASTICML_HOKEA_MOMENTUM": "0.9"}),
    "rt04": dict(workers=1, env=IDLE),
    "rt07": dict(workers=5, env=TREE5_IDLE),
    "rt08": dict(workers=5, env=TREE5_IDLE),
    "rt09": dict(workers=5, env={**TREE5_IDLE, "ELASTICML_GRADIENT_ALLOW_STALE_AGGREGATES": "1",
                                 "ELASTICML_HOKEA_MAX_STALENESS": "8"}),
    "rt12": dict(workers=1, env=IDLE, token=None),
    "rt19": dict(workers=5, env={"ELASTICML_GRADIENT_AGGREGATION_ENABLED": "1",
                                 "ELASTICML_HOKEA_WORKER_STEPS": "",
                                 "ELASTICML_HOKEA_MAX_UPDATES": "6"}),
    "rt22": dict(workers=5, env={"ELASTICML_GRADIENT_AGGREGATION_ENABLED": "1",
                                 "ELASTICML_HOKEA_WORKER_STEPS": ""}),
    "s1": dict(workers=1, env={**IDLE, "ELASTICML_HOKEA_MAX_UPDATES": "100000"}),
    "straggler": dict(workers=5, env={"ELASTICML_GRADIENT_AGGREGATION_ENABLED": "1",
                                      "ELASTICML_HOKEA_WORKER_STEPS": "",
                                      "ELASTICML_HOKEA_DATASET": "fashion_mnist",
                                      "ELASTICML_HOKEA_MODEL": "fashion_cnn_v5",
                                      "ELASTICML_HOKEA_BATCH_SIZE": "128",
                                      "ELASTICML_HOKEA_REDUCER_FLUSH_TIMEOUT_SECONDS": "60.0",
                                      "ELASTICML_HOKEA_REDUCER_REPLICAS": "1"}),
}
DIRECT_RUN = {"ELASTICML_HOKEA_WORKER_STEPS": "", "ELASTICML_HOKEA_MAX_UPDATES": "20",
              "ELASTICML_HOKEA_SEED": "7"}


def cluster_for(name, workers, env, tmp_path, token=TOKEN):
    from hokea import Cluster
    full = {"ELASTICML_ENABLE_DEBUG_ENDPOINTS": "1", "ELASTICML_HOKEA_TOTAL_WORKERS": str(workers),
            "ELASTICML_HOKEA_SEED": "7", **env}
    if token is not None:
        full["ELASTICML_AUTH_TOKEN"] = token
    return Cluster(dockerfile=str(ROOT), memory="768m", cpus=1.0, nodes=3 + workers,
                   name=f"elasticml-r3-{name}", port=8100, peer_port=8100, env=full,
                   workdir=tmp_path / f"cluster-{name}", up_timeout=900)


@pytest.fixture
def hokea_cluster(tmp_path, request):
    key = request.node.name.split("_")[2]
    scenario = SCENARIOS[key]
    return cluster_for(key, scenario["workers"], scenario["env"], tmp_path,
                       token=scenario.get("token", TOKEN))


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def log(event, **fields):
    print("ELASTICML_REDTEAM " + json.dumps({"event": event, **fields}, default=str), flush=True)


def get(node, path, timeout=5, headers=HEADERS):
    return requests.get(f"{node.url}{path}", headers=headers, timeout=timeout)


def local_state(node):
    return get(node, "/v1/debug/local-state").json()


def version(node):
    try:
        return int(local_state(node)["model_version"])
    except (requests.RequestException, ValueError, KeyError):
        return None


def wait_until(predicate, timeout, interval=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    return None


# Runs inside a pod: builds gradient payloads that match the live model and
# sends them to the pod-local reducer (port 8101) or to a voter.
INJECT = r"""
import base64, io, json, sys, time
import numpy as np, requests
from elasticml.array_codec import arrays_to_b64
from elasticml.codec import b64_to_tensors
spec = json.loads(sys.argv[1]); token = sys.argv[2]
h = {"Authorization": "Bearer " + token}
model = requests.get("http://node-1:8100/v1/model", headers=h, timeout=10).json()
shapes = {k: tuple(v.shape) for k, v in b64_to_tensors(model["model_b64"]).items()}
def payload(kind, dtype):
    if kind == "overflow_int8":
        arrays = {}
        for k, s in shapes.items():
            arrays[k] = np.full(s, 127, np.int8)
            arrays["__elasticml_int8_scale__::" + k] = np.asarray(3e38, np.float32)
        buf = io.BytesIO(); np.savez_compressed(buf, **arrays)
        return base64.b64encode(buf.getvalue()).decode()
    arrays = {k: np.full(s, 0.01, dtype) for k, s in shapes.items()}
    if kind == "wrong_shape":
        arrays[next(iter(arrays))] = np.zeros((3, 3), dtype)
    return arrays_to_b64(arrays)
out = []
for m in spec["members"]:
    body = dict(update_id=m["id"], worker_id=m["worker"], base_version=m["base"],
                gradient_b64=payload(m.get("kind", "valid"), np.dtype(spec.get("dtype", "float16"))),
                loss=1.0, sample_count=m.get("samples", 1))
    try:
        r = requests.post(spec["url"] + "/v1/gradients", json=body, headers=h, timeout=15)
        out.append([m["id"], r.status_code, r.text[:160]])
    except Exception as exc:
        out.append([m["id"], "error", str(exc)[:160]])
print(json.dumps({"model_version": model["model_version"], "results": out}))
"""

STATUS = r"""
import json, sys, requests
spec = json.loads(sys.argv[1]); h = {"Authorization": "Bearer " + sys.argv[2]}
out = []
for uid in spec["ids"]:
    try:
        r = requests.get(spec["url"] + "/v1/updates/status", headers=h, timeout=10,
                         params={"update_id": uid, "base_version": spec["base"]})
        out.append([uid, r.status_code, r.text[:160]])
    except Exception as exc:
        out.append([uid, "error", str(exc)[:160]])
print(json.dumps(out))
"""


def pod_run(node, script, spec, token=TOKEN):
    result = node.exec("python", "-c", script, json.dumps(spec), token or "", check=False)
    text = (result.stdout or "").strip().splitlines()
    try:
        return json.loads(text[-1])
    except (IndexError, ValueError):
        return {"error": (result.stdout + result.stderr)[-400:]}


READY = r"""
import sys, time, requests
h = {"Authorization": "Bearer " + sys.argv[2]}
deadline = time.time() + 180
while time.time() < deadline:
    try:
        if requests.get("http://127.0.0.1:8101/v1/status", headers=h, timeout=2).status_code == 200:
            print('{"ready": true}'); break
    except Exception:
        pass
    time.sleep(1)
else:
    print('{"ready": false}')
"""


def reducer_ready(node):
    """The reducer process starts after the pod's /health is green; wait for it."""
    out = pod_run(node, READY, {})
    log("reducer_ready", node=node.index, out=out)
    assert out.get("ready"), f"reducer on node {node.index} never listened on 8101"


def members(prefix, count, base, kinds=()):
    kinds = list(kinds) + ["valid"] * (count - len(kinds))
    return [dict(id=f"{prefix}{i}:1", worker=f"{prefix}{i}", base=base, kind=k)
            for i, k in enumerate(kinds)]


# --------------------------------------------------------------------------
# RT-02  S5: accepted gradients drive the model to inf; reads then fail
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="RT-02 on HOKEA: two finite 3e38 gradients with momentum 0.9 "
                   "make the voters' model non-finite; authoritative reads fail afterwards")
def test_hk_rt02_accepted_gradients_keep_model_readable(cluster):
    nodes = cluster.nodes()
    big = r"""
import json, sys, requests, torch
from elasticml.codec import b64_to_tensors, tensors_to_b64
h = {"Authorization": "Bearer " + sys.argv[2]}
out = []
for i in range(2):
    m = requests.get("http://node-1:8100/v1/model", headers=h, timeout=10).json()
    g = {k: torch.full_like(v, 3e38) for k, v in b64_to_tensors(m["model_b64"]).items()}
    r = requests.post("http://node-1:8100/v1/gradients", headers=h, timeout=15, json=dict(
        update_id=f"overflow:{i}", worker_id="huge", base_version=m["model_version"],
        gradient_b64=tensors_to_b64(g), loss=1.0, sample_count=1))
    out.append([r.status_code, r.text[:120]])
print(json.dumps(out))
"""
    log("rt02_overflow_submits", out=pod_run(nodes[3], big, {}))
    reads = {}
    for node in nodes[:3]:
        try:
            response = get(node, "/v1/model", timeout=10)
            reads[node.index] = response.status_code
        except requests.RequestException as exc:
            reads[node.index] = str(exc)[:80]
    log("rt02_reads_after", reads=reads)
    assert all(code == 200 for code in reads.values()), reads


# --------------------------------------------------------------------------
# RT-04  S5 "oversized": a small request decodes to hundreds of MB
# --------------------------------------------------------------------------

BOMB = r"""
import base64, io, json, sys, numpy as np, requests
h = {"Authorization": "Bearer " + sys.argv[2]}
buf = io.BytesIO(); np.savez_compressed(buf, **{"net.0.weight": np.zeros((10000, 10000), np.float32)})
payload = base64.b64encode(buf.getvalue()).decode()
v = requests.get("http://node-1:8100/v1/model", headers=h, timeout=10).json()["model_version"]
try:
    r = requests.post("http://node-1:8100/v1/gradients", headers=h, timeout=60, json=dict(
        update_id="bomb:1", worker_id="bomb", base_version=v, gradient_b64=payload, loss=1.0))
    print(json.dumps({"request_chars": len(payload), "status": r.status_code, "body": r.text[:120]}))
except Exception as exc:
    print(json.dumps({"request_chars": len(payload), "status": "error", "body": str(exc)[:160]}))
"""


def memory_peak(node):
    result = node.exec("sh", "-c", "cat /sys/fs/cgroup/memory.peak 2>/dev/null || "
                       "cat /sys/fs/cgroup/memory/memory.max_usage_in_bytes", check=False)
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


@pytest.mark.xfail(strict=True, reason="RT-04 on HOKEA: one 519 KB request decodes to a 400 MB array "
                   "inside the voter before shape validation; on 2026-11-17 it OOM-killed voter 1")
def test_hk_rt04_small_request_cannot_exhaust_voter_memory(cluster):
    nodes = cluster.nodes()
    before = memory_peak(nodes[0])
    out = pod_run(nodes[3], BOMB, {})
    time.sleep(3)
    alive = nodes[0].is_running()
    after = memory_peak(nodes[0]) if alive else None
    growth = (after - before) / 2**20 if before and after else None
    log("rt04_bomb", request=out, voter_alive=alive, memory_peak_before_mb=before and before / 2**20,
        memory_peak_after_mb=after and after / 2**20, growth_mb=growth)
    assert alive, "voter 1 died after one 540 KB request"
    assert isinstance(out.get("status"), int) and 400 <= out["status"] < 500, out
    assert growth is not None and growth < 32, f"voter peak memory grew {growth:.0f} MB"


# --------------------------------------------------------------------------
# RT-07  S5, L1: one wrong-shape member blocks a reducer cell
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="RT-07 on HOKEA: the reducer admits a wrong-shape member; "
                   "the cell then never forms a batch and honest members stay pending")
def test_hk_rt07_wrong_shape_member_cannot_block_a_reducer_cell(cluster):
    nodes = cluster.nodes()
    reducer_ready(nodes[3])
    base = version(nodes[0]) or 0
    spec = dict(url="http://127.0.0.1:8101", members=members("rt7w", 5, base, ["wrong_shape"]))
    log("rt07_submit", out=pod_run(nodes[3], INJECT, spec))
    honest = [m["id"] for m in spec["members"][1:]]
    statuses = None
    for _ in range(10):
        time.sleep(2)
        statuses = pod_run(nodes[3], STATUS, dict(url="http://127.0.0.1:8101", ids=honest, base=base))
    log("rt07_honest_status_after_20s", statuses=statuses, voter_version=version(nodes[0]))
    assert version(nodes[0]) > base, "honest members never committed"


# --------------------------------------------------------------------------
# RT-08 + RT-10: reducer accepts an int8 payload the voter must reject, then
# ignores the voter's rejection, so members never get a terminal answer
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="RT-08/RT-10 on HOKEA: the reducer admits an int8 scale "
                   "overflow (finiteness checked before dequantisation); every member of the cell then "
                   "gets HTTP 500 instead of a terminal rejection")
def test_hk_rt08_rt10_int8_overflow_rejected_with_terminal_status(cluster):
    nodes = cluster.nodes()
    reducer_ready(nodes[3])
    base = version(nodes[0]) or 0
    spec = dict(url="http://127.0.0.1:8101", members=members("rt8w", 5, base, ["overflow_int8"]))
    submitted = pod_run(nodes[3], INJECT, spec)
    log("rt08_submit", out=submitted)
    time.sleep(5)
    statuses = pod_run(nodes[3], STATUS, dict(url="http://127.0.0.1:8101",
                                              ids=[m["id"] for m in spec["members"]], base=base))
    log("rt08_status_after_5s", statuses=statuses, voter_version=version(nodes[0]))
    first = submitted["results"][0]
    assert first[1] in (400, 413, 422), f"reducer admitted the overflow member: {first}"
    terminal = {"applied", "duplicate", "stale", "rejected"}
    assert all(json.loads(s[2]).get("status") in terminal for s in statuses if s[1] == 200), statuses


# --------------------------------------------------------------------------
# RT-09  S1: allow_stale + primary/backup reducers apply one contribution twice
# --------------------------------------------------------------------------

def test_hk_rt09_retry_at_backup_reducer_applies_once(cluster):
    """RT-09 NOT reproduced on HOKEA (2026-11-17, log 16): after the primary
    committed the five members, the backup kept the identical retry 'pending'
    and the model stayed at version 1.  The in-process test, where the backup
    flushes its batch, still applies the contribution twice."""
    nodes = cluster.nodes()
    reducer_ready(nodes[3])
    reducer_ready(nodes[4])
    base = version(nodes[0]) or 0
    batch = members("rt9w", 5, base)
    log("rt09_primary", out=pod_run(nodes[3], INJECT, dict(url="http://127.0.0.1:8101", members=batch)))
    after_primary = wait_until(lambda: (version(nodes[0]) or 0) > base and version(nodes[0]), 30)
    log("rt09_after_primary", version=after_primary)
    log("rt09_backup_retry",
        out=pod_run(nodes[4], INJECT, dict(url="http://127.0.0.1:8101", members=batch)))
    time.sleep(10)
    final = version(nodes[0])
    log("rt09_final", base=base, after_primary=after_primary, final=final)
    assert after_primary == base + 1
    assert final == base + 1, f"one set of five contributions advanced the model {final - base} times"


# --------------------------------------------------------------------------
# RT-12  §5: the cluster runs with a token published in the repository
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="RT-12 on HOKEA: started without ELASTICML_AUTH_TOKEN, every "
                   "voter falls back to the token committed in the repository and accepts it")
def test_hk_rt12_cluster_refuses_to_run_with_the_published_default_token(cluster):
    nodes = cluster.nodes()
    public = {"Authorization": "Bearer elasticml-hokea-local"}
    codes = {node.index: get(node, "/v1/model", headers=public).status_code for node in nodes[:3]}
    unauthenticated = {node.index: requests.post(f"{node.url}/v1/peers/prepare", json={},
                                                 timeout=5).status_code for node in nodes[:3]}
    log("rt12_public_token_reads", codes=codes, unauthenticated_prepare=unauthenticated)
    assert all(code in (401, 403) for code in codes.values()), codes


# --------------------------------------------------------------------------
# RT-19  §5 bounded jobs: tree-mode workers never learn that the job ended
# --------------------------------------------------------------------------

PROCS = r"""
import json, os
found = []
me = str(os.getpid())
for pid in os.listdir("/proc"):
    if pid.isdigit() and pid != me:
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode()
        except OSError:
            continue
        if "elasticml.worker" in cmd or "elasticml-worker" in cmd:
            found.append(cmd[:120])
print(json.dumps(found))
"""


@pytest.mark.xfail(strict=True, reason="RT-19 on HOKEA: after max_job_updates the reducer reports "
                   "'stale' instead of job_complete, so tree-mode worker processes keep running")
def test_hk_rt19_workers_stop_after_the_job_completes(cluster):
    nodes = cluster.nodes()
    # Non-proposing voters learn lazily (seen in this run: 6 / 0 / 0), so the
    # job is complete when the most advanced voter reaches max_job_updates.
    done = wait_until(lambda: max((version(n) or 0) for n in nodes[:3]) >= 6, 300)
    log("rt19_job_complete", reached=bool(done), versions={n.index: version(n) for n in nodes[:3]})
    assert done, "job never reached max_job_updates"
    time.sleep(60)
    running = {node.index: pod_run(node, PROCS, {}) for node in nodes[3:]}
    log("rt19_worker_processes_60s_after_completion", running=running)
    assert not any(running.values()), running


# --------------------------------------------------------------------------
# RT-20  §7 "one deterministic run": equal seed, different committed model
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="RT-20 on HOKEA: two runs with identical seed and config "
                   "commit different models at the same version")
def test_hk_rt20_equal_seed_runs_commit_identical_models(tmp_path):
    hashes = []
    for run in range(2):
        with cluster_for(f"rt20-{run}", 3, DIRECT_RUN, tmp_path) as cluster:
            nodes = cluster.nodes()
            done = wait_until(lambda: all((version(n) or 0) >= 20 for n in nodes[:3]), 300)
            state = local_state(nodes[0])
            log("rt20_run", run=run, reached=bool(done), model_version=state["model_version"],
                model_sha256=state["model_sha256"])
            hashes.append((state["model_version"], state["model_sha256"]))
    assert hashes[0][0] == hashes[1][0] == 20
    assert hashes[0][1] == hashes[1][1], hashes


# --------------------------------------------------------------------------
# RT-22  §2 cells: voters disagree about which workers are active
# --------------------------------------------------------------------------

def test_hk_rt22_voters_agree_on_worker_membership(cluster, net):
    """RT-22 NOT reproduced on HOKEA (2026-11-17, log 15): HOKEA workers
    heartbeat every voter, so all three active sets matched before and after a
    30 s isolation of voter 1.  The in-process test still shows that two voters
    with different registration views assign a worker to different cells; the
    registry is not replicated, it is only refreshed by heartbeats."""
    nodes = cluster.nodes()
    wait_until(lambda: (version(nodes[0]) or 0) >= 3, 240)
    def membership():
        out = {}
        for node in nodes[:3]:
            try:
                workers = get(node, "/v1/status").json()["workers"]
                out[node.index] = sorted(w["worker_id"] for w in workers if w["state"] == "active")
            except (requests.RequestException, ValueError, KeyError):
                out[node.index] = None
        return out
    before = membership()
    net.isolate(1)
    time.sleep(30)
    net.heal()
    time.sleep(10)
    after = membership()
    log("rt22_membership", before_failover=before, after_failover=after)
    assert len({json.dumps(v) for v in after.values()}) == 1, after


# --------------------------------------------------------------------------
# S1/S2 nemesis: concurrent clients, random isolate/heal and packet loss
# --------------------------------------------------------------------------

def test_hk_s1_s2_randomized_partitions_and_loss_keep_one_history(cluster, net):
    """Jepsen-style: 3 clients submit unique updates to random voters for 90 s
    while a nemesis isolates a random voter or adds 25% loss every 6 s.
    Checker: after healing, every voter holds the same learned prefix and
    model, every update was applied at most once (version == processed count),
    and every acknowledged update was applied (acked <= processed)."""
    from elasticml.codec import b64_to_tensors, tensors_to_b64
    import torch
    nodes = cluster.nodes()
    voters = nodes[:3]
    model = get(voters[0], "/v1/model").json()
    template = {k: torch.full_like(v, 1e-4) for k, v in b64_to_tensors(model["model_b64"]).items()}
    gradient = tensors_to_b64(template)
    acked, attempted, lock = set(), set(), threading.Lock()
    stop = threading.Event()
    history = []

    def client(cid):
        rng = random.Random(cid)
        n = 0
        while not stop.is_set():
            uid = f"c{cid}-{n}"
            n += 1
            with lock:
                attempted.add(uid)
            target = rng.choice(voters)
            body = dict(update_id=uid, worker_id=f"client{cid}", base_version=0,
                        gradient_b64=gradient, loss=1.0, sample_count=1)
            for _ in range(6):
                if stop.is_set():
                    break
                try:
                    body["base_version"] = int(local_state(target)["model_version"])
                    r = requests.post(f"{target.url}/v1/gradients", json=body, headers=HEADERS,
                                      timeout=4)
                    status = r.json().get("status") if r.status_code == 200 else r.status_code
                except (requests.RequestException, ValueError, KeyError):
                    status = "info"
                history.append((time.monotonic(), cid, uid, target.index, status))
                if status in ("applied", "duplicate"):
                    with lock:
                        acked.add(uid)
                    break
                if status == "stale":
                    continue
                time.sleep(0.2)

    threads = [threading.Thread(target=client, args=(c,), daemon=True) for c in range(3)]
    for t in threads:
        t.start()
    rng = random.Random(11)
    faults = []
    started = time.monotonic()
    while time.monotonic() - started < 90:
        victim = rng.choice([1, 2, 3])
        kind = rng.choice(["isolate", "loss"])
        if kind == "isolate":
            net.isolate(victim)
        else:
            net.slow(victim, delay_ms=50, jitter_ms=50, loss_pct=25)
        faults.append((round(time.monotonic() - started, 1), kind, victim))
        time.sleep(6)
        net.heal()
        time.sleep(2)
    stop.set()
    for t in threads:
        t.join(timeout=30)
    net.heal()
    for node in voters:          # authoritative reads make lagging voters learn
        wait_until(lambda: get(node, "/v1/model").status_code == 200, 60, 0.5)
    states = {node.index: local_state(node) for node in voters}
    summary = {i: {k: s[k] for k in ("model_version", "processed_update_count",
                                     "learned_prefix_sha256", "model_sha256")}
               for i, s in states.items()}
    log("nemesis_faults", faults=faults)
    log("nemesis_result", attempted=len(attempted), acked=len(acked),
        outcomes={str(k): sum(1 for h in history if h[4] == k)
                  for k in {h[4] for h in history}}, states=summary)
    assert len({s["learned_prefix_sha256"] for s in summary.values()}) == 1, summary
    assert len({s["model_sha256"] for s in summary.values()}) == 1, summary
    for s in summary.values():
        assert s["model_version"] == s["processed_update_count"]       # at most once
        assert len(acked) <= s["processed_update_count"] <= len(attempted)


# --------------------------------------------------------------------------
# PH-04  §5 heterogeneous speed: one throttled worker slows every worker
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="PH-04 on HOKEA: with the accuracy profile's synchronous "
                   "60 s reducer cell, one worker throttled to 0.25 CPU cuts the whole cluster's "
                   "commit rate by more than half")
def test_hk_straggler_slows_cluster_by_at_most_its_share(cluster):
    """ASSUMPTION A5: one straggler among five workers should cost at most
    about half the commit rate (it is 20% of the compute)."""
    nodes = cluster.nodes()
    wait_until(lambda: (version(nodes[0]) or 0) >= 2, 600)

    def rate(seconds):
        start_v, start_t = version(nodes[0]), time.monotonic()
        time.sleep(seconds)
        return ((version(nodes[0]) or 0) - (start_v or 0)) / (time.monotonic() - start_t)

    healthy = rate(120)
    cluster.throttle(nodes[-1].index, cpus=0.25)
    throttled = rate(120)
    cluster.unthrottle(nodes[-1].index)
    log("straggler", healthy_commits_per_second=round(healthy, 3),
        throttled_commits_per_second=round(throttled, 3),
        ratio=round(throttled / healthy, 3) if healthy else None, throttled_node=nodes[-1].index)
    assert healthy > 0
    assert throttled >= 0.5 * healthy
