"""Red-team findings reproduced on a local HOKEA Docker cluster.

Each test starts the real ElasticML image (three voting replicas plus worker
pods) and drives it over real HTTP with HOKEA-injected faults.  IDs match
docs/REDTEAM_SPEC_GAP_ANALYSIS.md.  ``xfail(strict=True)`` = the spec
requirement is confirmed NOT met on the running cluster.

Run (Docker required, never against Kubernetes):

    unset HOKEA_NAMESPACE KUBECONFIG
    PYTHONPATH=src python -m pytest -v -s hokea_tests/test_redteam_hokea.py
    # add --runxfail to see expected failures as hard failures
"""
from __future__ import annotations

import base64
import json
import time
import zlib

import pytest
import requests
import torch

from conftest import ROOT, TOKEN

HEADERS = {"Authorization": f"Bearer {TOKEN}"}
AGGREGATE_PREFIX = "__elasticml_aggregate_v1__:"

# Per-test cluster environment.  NODES = 3 voters + one pod per worker.
SCENARIOS = {
    "rt01": dict(workers=2, env={"ELASTICML_GRADIENT_AGGREGATION_ENABLED": "1",
                                 "ELASTICML_HOKEA_WORKER_STEPS": "0"}),
    "rt03": dict(workers=1, env={"ELASTICML_HOKEA_WORKER_STEPS": "0"}),
    "rt05": dict(workers=3, env={"ELASTICML_GRADIENT_AGGREGATION_ENABLED": "1",
                                 "ELASTICML_HOKEA_WORKER_STEPS": "100000"}),
    "rt06": dict(workers=5, env={"ELASTICML_GRADIENT_AGGREGATION_ENABLED": "1",
                                 "ELASTICML_HOKEA_WORKER_STEPS": "100000"}),
    "rt15": dict(workers=1, env={"ELASTICML_HOKEA_WORKER_STEPS": "0"}),
    "rt17": dict(workers=1, env={"ELASTICML_HOKEA_WORKER_STEPS": "0"}),
    "s6": dict(workers=1, env={"ELASTICML_HOKEA_WORKER_STEPS": "0"}),
}


@pytest.fixture
def hokea_cluster(tmp_path, request):
    from hokea import Cluster
    key = request.node.name.split("_")[2]
    scenario = SCENARIOS[key]
    env = {"ELASTICML_AUTH_TOKEN": TOKEN, "ELASTICML_ENABLE_DEBUG_ENDPOINTS": "1",
           "ELASTICML_HOKEA_TOTAL_WORKERS": str(scenario["workers"]),
           "ELASTICML_HOKEA_SEED": "7", **scenario["env"]}
    return Cluster(dockerfile=str(ROOT), memory="768m", cpus=1.0,
                   nodes=3 + scenario["workers"], name=f"elasticml-rt-{key}",
                   port=8100, peer_port=8100, env=env,
                   workdir=tmp_path / "cluster", up_timeout=600)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def log(event, **fields):
    print("ELASTICML_REDTEAM " + json.dumps({"event": event, **fields}, default=str),
          flush=True)


def get(node, path, timeout=4, **params):
    return requests.get(f"{node.url}{path}", headers=HEADERS, params=params, timeout=timeout)


def post(node, path, body, timeout=8):
    return requests.post(f"{node.url}{path}", headers=HEADERS, json=body, timeout=timeout)


def model(node, retries=40):
    for _ in range(retries):
        try:
            response = get(node, "/v1/model")
            if response.status_code == 200:
                return response.json()
        except requests.RequestException:
            pass
        time.sleep(0.25)
    raise AssertionError(f"node {node.index} never served an authoritative model")


def local_state(node):
    response = get(node, "/v1/debug/local-state")
    response.raise_for_status()
    return response.json()


def gradient_b64(node, value):
    from elasticml.codec import b64_to_tensors, tensors_to_b64
    weights = b64_to_tensors(model(node)["model_b64"])
    return tensors_to_b64({name: torch.full_like(tensor, value)
                           for name, tensor in weights.items()})


def submit(node, update_id, value, *, base=None, samples=1, worker="redteam", retries=20):
    body = dict(update_id=update_id, worker_id=worker,
                base_version=model(node)["model_version"] if base is None else base,
                gradient_b64=gradient_b64(node, value), loss=1.0, sample_count=samples)
    response = None
    for _ in range(retries):
        try:
            response = post(node, "/v1/gradients", body)
        except requests.RequestException as exc:
            log("submit_exception", node=node.index, update_id=update_id, error=str(exc))
            time.sleep(0.25)
            continue
        if response.status_code != 503:
            break
        time.sleep(0.25)
    log("submit", node=node.index, update_id=update_id[:48], status_code=response.status_code,
        body=response.text[:200])
    return response


def versions(nodes):
    out = {}
    for node in nodes:
        try:
            out[node.index] = int(get(node, "/v1/model").json()["model_version"])
        except Exception:
            out[node.index] = None
    return out


def wait_for_progress(nodes, minimum, timeout):
    deadline = time.monotonic() + timeout
    seen = {}
    while time.monotonic() < deadline:
        seen = versions(nodes)
        if max((v for v in seen.values() if v is not None), default=-1) >= minimum:
            return seen
        time.sleep(1.0)
    return seen


# --------------------------------------------------------------------------
# RT-01 on HOKEA: one finite overflowing gradient halts voter aggregation
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="RT-01 on HOKEA: finite overflowing member poisons the "
                   "voter cohort (HTTP 500); failover carries the poison to other voters")
def test_hk_rt01_overflowing_member_cannot_halt_aggregation(cluster):
    voters = cluster.nodes()[:3]
    codes = []
    for node in voters[:2]:  # worker fails over from voter 1 to voter 2 with the same payloads
        codes.append(submit(node, "big", 1e36, base=0, samples=1000, worker="w-big").status_code)
        codes.append(submit(node, "u2", 0.01, base=0, worker="w2").status_code)
        codes.append(submit(node, "u3", 0.01, base=0, worker="w3").status_code)
    seen = versions(voters)
    log("rt01_result", codes=codes, versions=seen)
    assert all(code < 500 for code in codes), codes
    assert max(v or 0 for v in seen.values()) >= 1


# --------------------------------------------------------------------------
# RT-03 on HOKEA: reserved prefix makes replica dedup state diverge
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="RT-03 on HOKEA: the proposing voter and the voters that "
                   "learn by recovery hold different processed-update sets for one log")
def test_hk_rt03_reserved_prefix_keeps_replica_dedup_state_identical(cluster):
    voters = cluster.nodes()[:3]
    config = json.loads(voters[0].exec("cat", "/data/coordinator.json").stdout)
    log("rt03_voter_config", peer_log_compaction_enabled=config["peer_log_compaction_enabled"],
        gradient_aggregation_enabled=config["gradient_aggregation_enabled"],
        max_staleness=config["max_staleness"], max_job_updates=config["max_job_updates"])
    manifest = json.dumps({"members": [["victim-1", 1_000_000]]}, separators=(",", ":")).encode()
    crafted = AGGREGATE_PREFIX + base64.urlsafe_b64encode(zlib.compress(manifest, 9)).decode()
    assert submit(voters[0], crafted, 0.01, worker="w1").status_code == 200
    for node in voters:
        model(node)  # authoritative read makes every voter catch up
    states = {node.index: local_state(node) for node in voters}
    log("rt03_states", **{str(k): {f: v[f] for f in ("model_version", "processed_update_count",
                                                     "processed_update_ids_sha256",
                                                     "learned_prefix_sha256")}
                          for k, v in states.items()})
    assert len({s["learned_prefix_sha256"] for s in states.values()}) == 1  # same log
    assert len({s["processed_update_ids_sha256"] for s in states.values()}) == 1


# --------------------------------------------------------------------------
# RT-05 on HOKEA: one worker pod crashes, survivors stall forever
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="RT-05 on HOKEA: with voter-side aggregation (<=4 workers, "
                   "the HOKEA default) losing one worker stops all later commits")
def test_hk_rt05_survivors_keep_committing_after_worker_pod_crash(cluster):
    voters = cluster.nodes()[:3]
    before = wait_for_progress(voters, 3, timeout=180)
    start = max(v or 0 for v in before.values())
    log("rt05_before_kill", versions=before)
    assert start >= 3, "cluster never trained"
    cluster.kill(cluster.nodes()[-1].index)  # an ordinary worker pod
    after = wait_for_progress(voters, start + 3, timeout=120)
    log("rt05_after_kill", versions=after,
        status=get(voters[0], "/v1/status").json().get("update_metrics"))
    assert max(v or 0 for v in after.values()) >= start + 3


# --------------------------------------------------------------------------
# RT-06 on HOKEA: one future-base request poisons the real reducer tree
# --------------------------------------------------------------------------

GHOST = r"""
import json, sys, numpy as np, requests
from elasticml.array_codec import arrays_to_b64
body = dict(update_id="ghost:1", worker_id="ghost", base_version=10000,
            gradient_b64=arrays_to_b64({"w": np.zeros(4, np.float16)}), loss=1.0)
r = requests.post("http://127.0.0.1:8101/v1/gradients", json=body, timeout=5,
                  headers={"Authorization": "Bearer " + sys.argv[1]})
print(r.status_code, r.text[:200])
"""


@pytest.mark.xfail(strict=True, reason="RT-06 on HOKEA: reducers trust worker base_version; one "
                   "future-base POST to the leaf reducers halts tree training")
def test_hk_rt06_future_base_request_cannot_halt_reducer_tree(cluster):
    voters = cluster.nodes()[:3]
    before = wait_for_progress(voters, 3, timeout=240)
    start = max(v or 0 for v in before.values())
    log("rt06_before_ghost", versions=before)
    assert start >= 3, "cluster never trained"
    # 5 workers -> one logical reducer with replicas on worker pods 4 and 5.
    for node in cluster.nodes()[3:5]:
        result = node.exec("python", "-c", GHOST, TOKEN, check=False)
        log("rt06_ghost", node=node.index, out=(result.stdout + result.stderr).strip()[-200:])
    time.sleep(5)
    mid = max(v or 0 for v in versions(voters).values())
    after = wait_for_progress(voters, mid + 3, timeout=120)
    log("rt06_after_ghost", versions_at_ghost=mid, versions=after)
    assert max(v or 0 for v in after.values()) >= mid + 3


# --------------------------------------------------------------------------
# RT-15 on HOKEA: container restart = amnesia; S2 breaks (excluded fault)
# --------------------------------------------------------------------------

def test_hk_rt15_demo_amnesiac_voter_restart_lets_slot_1_change(cluster, net):
    """Passing demonstration: proves §5's claim that a HOKEA restart loses
    /data and shows that nothing prevents the amnesiac voter from rejoining."""
    n1, n2, n3 = cluster.nodes()[:3]
    net.isolate(3)
    assert submit(n1, "first", 0.01).json()["status"] == "applied"
    original = local_state(n1)
    log("rt15_first_commit", **original)
    cluster.kill(2)
    cluster.start(2)
    cluster.wait_healthy(timeout=120)
    restarted = local_state(n2)
    log("rt15_restarted_voter", **restarted)
    assert restarted["learned_prefix_length"] == 0 and restarted["model_version"] == 0
    cluster.kill(1)
    net.heal()
    assert submit(n3, "second", 0.05).json()["status"] == "applied"
    model(n3)
    rewritten = local_state(n3)
    log("rt15_after_second_proposer", **rewritten)
    assert rewritten["model_version"] == original["model_version"] == 1
    assert rewritten["learned_prefix_sha256"] != original["learned_prefix_sha256"], \
        "expected two different records chosen for slot 1"


# --------------------------------------------------------------------------
# RT-17 on HOKEA: rolling back ONE voter's disk (not excluded) breaks S2
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="RT-17 on HOKEA: §5 only excludes majority rollback, yet "
                   "restoring one voter's /data/peer snapshot lets slot 1 be chosen twice")
def test_hk_rt17_single_voter_disk_rollback_preserves_slot_1(cluster, net):
    n1, n2, n3 = cluster.nodes()[:3]
    model(n1)
    n2.exec("sh", "-c", "cp -a /data/peer /data/peer.snapshot")
    net.isolate(3)
    assert submit(n1, "first", 0.01).json()["status"] == "applied"
    original = local_state(n1)
    log("rt17_first_commit", **original)
    n2.exec("sh", "-c", "rm -rf /data/peer && cp -a /data/peer.snapshot /data/peer")
    log("rt17_rolled_back_voter_2")
    cluster.kill(1)
    net.heal()
    response = submit(n3, "second", 0.05)
    model(n3)
    after = local_state(n3)
    log("rt17_after_second_proposer", status=response.status_code, **after)
    assert after["learned_prefix_sha256"] == original["learned_prefix_sha256"]


# --------------------------------------------------------------------------
# S6 on HOKEA (expected to pass): read at a voter that missed the write
# --------------------------------------------------------------------------

def test_hk_s6_read_at_voter_that_missed_each_acknowledged_write(cluster, net):
    voters = cluster.nodes()[:3]
    first_attempt_503 = 0
    for index in range(3):
        writer, reader = voters[index % 3], voters[(index + 1) % 3]
        net.isolate(reader.index)
        response = submit(writer, f"rot-{index}", 0.001 * (index + 1))
        assert response.status_code == 200 and response.json()["status"] == "applied"
        acknowledged = response.json()["model_version"]
        net.heal()
        net.isolate(writer.index)
        first = get(reader, "/v1/model")
        first_attempt_503 += first.status_code == 503
        observed = model(reader)["model_version"]
        log("s6_round", writer=writer.index, reader=reader.index, acknowledged=acknowledged,
            first_status=first.status_code, observed=observed)
        assert observed >= acknowledged
        net.heal()
    log("s6_summary", first_attempt_503=first_attempt_503)
