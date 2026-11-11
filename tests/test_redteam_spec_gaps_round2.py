"""Red-team round 2 against docs/submission/PROJECT_SPEC.pdf (MS3).

Same convention as tests/test_redteam_spec_gaps.py: ``xfail(strict=True)``
marks a requirement that is confirmed NOT met (RT-17 onward in
docs/REDTEAM_SPEC_GAP_ANALYSIS.md).  Run as hard failures with ``--runxfail``.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import make_gradient
from elasticml.config import CoordinatorConfig
from elasticml.consensus import Ballot, DurableAcceptor, PaxosProposer
from elasticml.gradient_reducer import GradientReducer
from elasticml.array_codec import arrays_to_b64
from elasticml.state import TrainingState
from elasticml.storage import MirroredStore
from test_redteam_spec_gaps import (REDUCER_ID, VoterUpstream, gradient_request,
                                    reducer_payload, tree_voter, voter)

REPO = Path(__file__).resolve().parents[1]
NODES = ("a", "b", "c")


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class Transport:
    def __init__(self, acceptors, down):
        self.acceptors, self.down = acceptors, down

    def _peer(self, peer):
        if peer in self.down:
            raise ConnectionError(peer)
        return self.acceptors[peer]

    def prepare(self, peer, slot, ballot):
        return self._peer(peer).prepare(slot, ballot)

    def accept(self, peer, slot, ballot, value):
        return self._peer(peer).accept(slot, ballot, value)


# --------------------------------------------------------------------------
# RT-17  §5 / S2: minority ROLLBACK is not excluded, yet breaks S2
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="RT-17: §5 excludes rolled-back state only 'for a majority'; "
                   "restoring ONE voter from an older snapshot lets a second value be chosen")
def test_rt17_single_voter_rollback_preserves_single_value_per_slot(tmp_path):
    acceptors = {n: DurableAcceptor(tmp_path / n / "acceptor") for n in NODES}
    first_ballot = Ballot(1, "a")
    acceptors["b"].prepare(1, first_ballot)
    shutil.copytree(tmp_path / "b", tmp_path / "b-snapshot")  # e.g. VM/volume snapshot
    acceptors["a"].prepare(1, first_ballot)
    for node in ("a", "b"):
        assert acceptors[node].accept(1, first_ballot, {"value": "v1"}).accepted
    # v1 is now chosen (a, b).  b is restored from its earlier snapshot:
    # it keeps its identity and its promise, but forgets the acceptance.
    shutil.rmtree(tmp_path / "b")
    shutil.copytree(tmp_path / "b-snapshot", tmp_path / "b")
    acceptors["b"] = DurableAcceptor(tmp_path / "b" / "acceptor")
    proposer = PaxosProposer("c", tmp_path / "c" / "proposer", list(NODES),
                             Transport(acceptors, down={"a"}))
    second = proposer.propose(1, {"value": "v2"})
    assert second.value == {"value": "v1"}, "slot 1 chose v1 and then v2"


# --------------------------------------------------------------------------
# RT-18  §5 heterogeneous speed (included fault) + L1/L2: slow worker starves
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="RT-18: with synchronous aggregation a worker slower than the "
                   "partial-flush deadline is answered 'stale' every round and never contributes")
def test_rt18_slow_but_live_worker_eventually_contributes(tmp_path):
    runtime, client = tree_voter(tmp_path)
    clock = Clock()
    reducer = GradientReducer(REDUCER_ID, VoterUpstream(client), group_size=3,
                              wire_dtype="float32", flush_timeout_seconds=0.75, clock=clock)
    payload = reducer_payload(runtime.state)
    slow_outcomes = []
    for round_index in range(6):
        base = runtime.state.version
        reducer.submit(gradient_request(f"w1:{round_index}", "w1", base, payload))
        reducer.submit(gradient_request(f"w2:{round_index}", "w2", base, payload))
        clock.now += 0.8  # bounded partial flush fires
        assert reducer.status(f"w1:{round_index}", base)["status"] == "applied"
        clock.now += 0.2  # slow-but-finite worker finishes 1.0 s after round start
        slow_outcomes.append(
            reducer.submit(gradient_request(f"w3:{round_index}", "w3", base, payload))["status"])
    assert "applied" in slow_outcomes or "pending" in slow_outcomes, slow_outcomes


# --------------------------------------------------------------------------
# RT-19  §5 bounded jobs: job end is reported as 'stale' through reducers
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="RT-19: the voter's 409 job_complete is mapped to 'stale' with a "
                   "fabricated model_version (base+1), so tree-mode workers loop instead of stopping")
def test_rt19_job_completion_reaches_workers_through_reducers(tmp_path):
    runtime, client = tree_voter(tmp_path, max_job_updates=1)
    reducer = GradientReducer(REDUCER_ID, VoterUpstream(client), group_size=1,
                              wire_dtype="float32")
    payload = reducer_payload(runtime.state)
    assert reducer.submit(gradient_request("w1:a", "w1", 0, payload))["status"] == "applied"
    result = reducer.submit(gradient_request("w1:b", "w1", 1, payload))
    assert result["status"] not in {"stale", "pending"}, result
    assert int(result.get("model_version", 1)) <= runtime.state.version


# --------------------------------------------------------------------------
# RT-20  §7 "The 94.68% result is one deterministic run"
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="RT-20: committed aggregates depend on wall-clock arrival "
                   "relative to the partial-flush deadline, so equal seeds/inputs do not give "
                   "an equal model; the 'deterministic run' claim is unsupported")
def test_rt20_equal_inputs_give_equal_model_regardless_of_arrival_timing(tmp_path):
    def run(root, late_seconds):
        runtime, client = tree_voter(root)
        clock = Clock()
        reducer = GradientReducer(REDUCER_ID, VoterUpstream(client), group_size=3,
                                  wire_dtype="float32", flush_timeout_seconds=0.75, clock=clock)
        for worker, value in (("w1", 0.01), ("w2", 0.02)):
            reducer.submit(gradient_request(f"{worker}:0", worker, 0,
                                            reducer_payload(runtime.state, value)))
        clock.now += late_seconds
        reducer.status("w1:0", 0)
        reducer.submit(gradient_request("w3:0", "w3", 0, reducer_payload(runtime.state, 0.09)))
        return {k: v.clone() for k, v in runtime.state.model.state_dict().items()}

    on_time = run(tmp_path / "on-time", 0.1)
    late = run(tmp_path / "late", 0.8)
    assert all(torch.equal(on_time[name], late[name]) for name in on_time)


# --------------------------------------------------------------------------
# RT-21  S3 in the DEFAULT (single-coordinator) mode: W + R <= N is accepted
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="RT-21: validate() accepts storage_write_quorum + "
                   "storage_read_quorum <= N; an 'applied' update written to one mirror vanishes "
                   "after restart")
def test_rt21_acknowledged_update_survives_restart_for_every_valid_quorum_config(tmp_path, monkeypatch):
    roots = [tmp_path / x for x in "abc"]
    CoordinatorConfig(storage_roots=[str(r) for r in roots], storage_write_quorum=1,
                      storage_read_quorum=2).validate()  # accepted today
    original = MirroredStore._create_immutable

    def only_first_mirror(path, text):
        if roots[0] not in path.parents:
            raise OSError("mirror unavailable")
        return original(path, text)

    monkeypatch.setattr(MirroredStore, "_create_immutable", staticmethod(only_first_mirror))
    state = TrainingState(model_name="tiny_mlp", learning_rate=0.02, max_staleness=8,
                          checkpoint_every=100, store=MirroredStore(roots, 1, 2))
    base, gradient, loss = make_gradient(state)
    assert state.apply_update(update_id="acked", worker_id="w1", base_version=base,
                              gradient_b64=gradient, loss=loss).status == "applied"
    monkeypatch.setattr(MirroredStore, "_create_immutable", staticmethod(original))
    restarted = TrainingState(model_name="tiny_mlp", learning_rate=0.02, max_staleness=8,
                              checkpoint_every=100, store=MirroredStore(roots, 1, 2))
    assert restarted.version == 1 and "acked" in restarted.processed_update_ids


# --------------------------------------------------------------------------
# RT-22  cell assignment depends on each voter's private membership view
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="RT-22: membership is per-replica and unreplicated; two voters "
                   "with different registration views place the same worker in different cells")
def test_rt22_every_voter_assigns_a_worker_to_the_same_cell(tmp_path):
    options = dict(gradient_aggregation_enabled=True, gradient_aggregation_cohort_size=8,
                   gradient_reducer_count=3)
    runtime_a, client_a = voter(tmp_path / "A", **options)
    runtime_b, client_b = voter(tmp_path / "B", **options)
    for worker in [f"w{i}" for i in range(6)]:
        client_a.post("/v1/workers/register", json={"worker_id": worker})
        if worker != "w0":  # replica B was briefly unreachable during w0's registration
            client_b.post("/v1/workers/register", json={"worker_id": worker})
    cell_a = client_a.get("/v1/model", params={"worker_id": "w3"}).json()["gradient_reducer_index"]
    cell_b = client_b.get("/v1/model", params={"worker_id": "w3"}).json()["gradient_reducer_index"]
    assert cell_a == cell_b


# --------------------------------------------------------------------------
# RT-23  §8 dashboard wording vs implementation
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="RT-23: §8/§12 say 'healthy' profile and timed runs replacing "
                   "the simulated-day framing; the dashboard still uses 'good' days and simulated_hours")
def test_rt23_dashboard_matches_section8_profiles_and_timing_model():
    from elasticml.lab_dashboard import DAY_PROFILES, CampaignRequest
    assert "healthy" in DAY_PROFILES
    assert "simulated_hours" not in CampaignRequest.model_fields


# --------------------------------------------------------------------------
# RT-24  §7 "unique committed examples" is really example exposures
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="RT-24: the reported 'unique_committed_examples' (2,444,800) "
                   "exceeds the 55,000-example training split; it counts exposures, not unique examples")
def test_rt24_unique_committed_examples_cannot_exceed_training_split():
    record = json.loads((REPO / "docs/evidence/model/fashion_cnn_v5_hokea_lr006_20.json")
                        .read_text(encoding="utf-8"))
    assert record["performance"]["unique_committed_examples"] <= 60_000 - 5_000


# --------------------------------------------------------------------------
# RT-25  §7 equal-seed fault comparisons for the MS3 model
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="RT-25: no retained fashion_cnn_v5 evidence has a churn, "
                   "request-loss or replica-failure run; every MS3 model record is healthy-only")
def test_rt25_ms3_model_has_equal_seed_fault_scenario_evidence():
    faulted = []
    for path in (REPO / "docs/evidence/model").glob("fashion_cnn_v5*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if (record.get("correctness", {}).get("faults_injected") or 0) > 0:
            faulted.append(path.name)
    assert faulted, "no MS3-model run with injected faults"


# --------------------------------------------------------------------------
# RT-26  §5 "Checkpoint and log compaction ... are not part of the selected runtime"
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="RT-26: every HOKEA voter (the runtime behind the reported "
                   "cluster results) hard-codes peer_log_compaction_enabled=True")
def test_rt26_hokea_runtime_does_not_enable_experimental_compaction():
    from elasticml.hokea_node import hokea_configuration
    env = {"NODE_ID": "1", "N_NODES": "4", "ELASTICML_HOKEA_TOTAL_WORKERS": "1",
           "PEERS": "10.0.0.1:8100,10.0.0.2:8100,10.0.0.3:8100,10.0.0.4:8100"}
    config, _ = hokea_configuration(env)
    assert config["peer_log_compaction_enabled"] is False


# --------------------------------------------------------------------------
# RT-27  §10 "the local HOKEA suite runs in Docker"
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="RT-27: hokea_tests/conftest.py starts 3 nodes, but the image "
                   "requires 3 voters + one pod per worker (6 for the default 3 workers); every "
                   "container exits with 'HOKEA must provide exactly 6 pods'")
def test_rt27_local_hokea_suite_cluster_size_matches_image_topology():
    import re
    from elasticml.hokea_node import hokea_pod_count
    source = (REPO / "hokea_tests" / "conftest.py").read_text(encoding="utf-8")
    nodes = int(re.search(r"nodes=(\d+)", source).group(1))
    workers = int(re.search(r"ELASTICML_HOKEA_TOTAL_WORKERS\"?:\s*\"?(\d+)", source).group(1)) \
        if "ELASTICML_HOKEA_TOTAL_WORKERS" in source else 3  # image default
    assert nodes == hokea_pod_count(workers), (nodes, hokea_pod_count(workers))
