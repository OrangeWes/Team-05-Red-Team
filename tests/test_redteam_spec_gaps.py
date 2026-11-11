"""Red-team tests derived from docs/submission/PROJECT_SPEC.pdf (MS3).

Each test names the specification clause it exercises.  Tests marked
``xfail(strict=True)`` encode a requirement the current implementation does
NOT meet; they were confirmed failing on 2026-11-10 and are tracked in
docs/REDTEAM_SPEC_GAP_ANALYSIS.md under the same RT-xx identifier.  Because
they are strict, fixing a defect turns the test into an XPASS failure, which
is the signal to delete the marker.

Run the expected failures as hard failures with::

    PYTHONPATH=src python -m pytest -q --runxfail tests/test_redteam_spec_gaps.py
"""
from __future__ import annotations

import base64
import gzip
import io
import json
import re
import struct
import time
import tracemalloc
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from conftest import make_gradient
from elasticml.api import GradientRequest, create_app
from elasticml.array_codec import arrays_to_b64, b64_to_arrays
from elasticml.client import ClientResponse
from elasticml.codec import tensors_to_b64
from elasticml.config import CoordinatorConfig
from elasticml.coordinator import build_runtime
from elasticml.gradient_reducer import GradientReducer
from elasticml.state import InvalidGradient, TrainingState, UpdateRejected, _AGGREGATE_PREFIX
from elasticml.storage import MirroredStore
from test_peer_runtime import PeerCluster

REPO = Path(__file__).resolve().parents[1]
REDUCER_ID = "hokea-reducer-l0-0"  # primary and backup share the logical id


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def make_state(tmp_path: Path, **kwargs) -> TrainingState:
    options = dict(model_name="tiny_mlp", learning_rate=0.02, max_staleness=8,
                   checkpoint_every=100,
                   store=MirroredStore([tmp_path / x for x in "abc"], 2, 2))
    options.update(kwargs)
    return TrainingState(**options)


def voter(tmp_path: Path, **kwargs):
    config = CoordinatorConfig(storage_roots=[str(tmp_path / x) for x in "abc"], **kwargs)
    runtime = build_runtime(config)
    return runtime, TestClient(create_app(runtime), raise_server_exceptions=False)


def full_gradient(state: TrainingState, value: float) -> str:
    return tensors_to_b64({name: torch.full_like(param, value)
                           for name, param in state.model.named_parameters()})


def reducer_payload(state: TrainingState, value: float = 0.01, *, wrong_shape=False,
                    extra_tensor=False) -> str:
    arrays = {name: np.full(tuple(param.shape), value, np.float32)
              for name, param in state.model.named_parameters()}
    if wrong_shape:
        arrays[next(iter(arrays))] = np.zeros((3, 3), np.float32)
    if extra_tensor:
        arrays["unknown.layer"] = np.zeros((2,), np.float32)
    return arrays_to_b64(arrays)


def gradient_request(update_id: str, worker_id: str, base: int, payload: str,
                     samples: int = 1) -> GradientRequest:
    return GradientRequest(update_id=update_id, worker_id=worker_id, base_version=base,
                           gradient_b64=payload, loss=1.0, sample_count=samples)


class VoterUpstream:
    """Adapts the real voter FastAPI app to the reducer's upstream client API."""

    def __init__(self, client: TestClient):
        self.client = client
        self.calls: list[tuple[str, str, int]] = []

    def request(self, method, path, *, json=None):
        response = self.client.request(method, path, json=json)
        self.calls.append((method, path.split("?")[0], response.status_code))
        try:
            data = response.json()
        except ValueError:
            data = {"detail": response.text}
        return ClientResponse(response.status_code, data)


def tree_voter(tmp_path: Path, **kwargs):
    options = dict(gradient_aggregation_enabled=True, gradient_aggregation_cohort_size=1,
                   gradient_reducer_count=1)
    options.update(kwargs)
    return voter(tmp_path, **options)


def crafted_aggregate_id(members) -> str:
    manifest = json.dumps({"members": members}, separators=(",", ":")).encode()
    return _AGGREGATE_PREFIX + base64.urlsafe_b64encode(zlib.compress(manifest, 9)).decode()


def poll(fn, predicate, *, seconds=2.0, interval=0.02):
    deadline = time.monotonic() + seconds
    value = fn()
    while not predicate(value) and time.monotonic() < deadline:
        time.sleep(interval)
        value = fn()
    return value


# --------------------------------------------------------------------------
# RT-01  S5 / L1: a finite gradient whose weighted cohort sum overflows
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="RT-01: finite-but-huge member poisons the voter cohort; "
                   "commit raises CodecError (HTTP 500) forever for that base version")
def test_rt01_overflowing_cohort_member_cannot_stall_honest_workers(tmp_path):
    runtime, client = voter(tmp_path, gradient_aggregation_enabled=True,
                            gradient_aggregation_cohort_size=2)
    state = runtime.state
    big = client.post("/v1/gradients", json=dict(
        update_id="big", worker_id="w-big", base_version=0,
        gradient_b64=full_gradient(state, 1e36), loss=1.0, sample_count=1000))
    honest = client.post("/v1/gradients", json=dict(
        update_id="u2", worker_id="w2", base_version=0,
        gradient_b64=full_gradient(state, 0.01), loss=1.0))
    later = client.post("/v1/gradients", json=dict(
        update_id="u3", worker_id="w3", base_version=0,
        gradient_b64=full_gradient(state, 0.01), loss=1.0))
    # S5: invalid input is rejected before proposal (4xx), never a server error.
    assert big.status_code < 500 and honest.status_code < 500 and later.status_code < 500
    # L1: honest surviving workers still produce a later commit.
    assert state.version >= 1


# --------------------------------------------------------------------------
# RT-02  S5: accepted finite gradients can make the model non-finite
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="RT-02: no gradient-magnitude bound; momentum overflows to inf "
                   "and every later authoritative model read fails")
def test_rt02_accepted_updates_keep_model_finite_and_readable(tmp_path):
    runtime, client = voter(tmp_path, momentum=0.9)
    state = runtime.state
    for index in range(2):
        client.post("/v1/gradients", json=dict(
            update_id=f"u{index}", worker_id="w1", base_version=state.version,
            gradient_b64=full_gradient(state, 3e38), loss=1.0))
    assert all(torch.isfinite(p).all() for p in state.model.parameters())
    assert client.get("/v1/model").status_code == 200


# --------------------------------------------------------------------------
# RT-03  S5 / S7: reserved aggregate prefix accepted in direct mode
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="RT-03: direct mode accepts the reserved aggregate update_id "
                   "prefix; only aggregation mode rejects it")
def test_rt03_reserved_aggregate_prefix_rejected_before_proposal(tmp_path):
    state = make_state(tmp_path)
    base, gradient, loss = make_gradient(state, "w1")
    with pytest.raises(InvalidGradient):
        state.apply_update(update_id=crafted_aggregate_id([["victim-1", 1_000_000]]),
                           worker_id="w1", base_version=base, gradient_b64=gradient, loss=loss)
    assert state.version == 0


@pytest.mark.xfail(strict=True, reason="RT-03: live and WAL-recovered replicas derive different "
                   "dedup sets and example metrics from the same chosen log")
def test_rt03_live_and_recovered_replicas_deduplicate_identically(tmp_path):
    live = make_state(tmp_path)
    base, gradient, loss = make_gradient(live, "w1")
    live.apply_update(update_id=crafted_aggregate_id([["victim-1", 1_000_000]]),
                      worker_id="w1", base_version=base, gradient_b64=gradient, loss=loss)
    recovered = make_state(tmp_path)  # replays the same durable WAL
    assert recovered.version == live.version
    assert recovered.processed_update_ids == live.processed_update_ids
    assert (recovered.update_metrics["included_examples"]
            == live.update_metrics["included_examples"])
    # The victim's real contribution must not be silently swallowed as a duplicate.
    base, gradient, loss = make_gradient(recovered, "victim")
    result = recovered.apply_update(update_id="victim-1", worker_id="victim",
                                    base_version=base, gradient_b64=gradient, loss=loss)
    assert result.status == "applied"


# --------------------------------------------------------------------------
# RT-04  S5 "oversized": decoded size is unbounded (compressed NPZ)
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="RT-04: only encoded characters are bounded; a ~20 KB "
                   "payload allocates tens of MB before shape validation rejects it")
def test_rt04_oversized_decoded_payload_rejected_before_large_allocation(tmp_path):
    state = make_state(tmp_path)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{"net.0.weight": np.zeros((2000, 2000), np.float32)})
    payload = base64.b64encode(buffer.getvalue()).decode()
    assert len(payload) < 64_000  # far below MAX_GRADIENT_B64_CHARS (4,000,000)
    tracemalloc.start()
    try:
        with pytest.raises(UpdateRejected):
            state.apply_update(update_id="bomb", worker_id="w1", base_version=0,
                               gradient_b64=payload, loss=1.0)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 4 * 1024 * 1024, f"rejection allocated {peak / 1e6:.1f} MB"


# --------------------------------------------------------------------------
# RT-05  L1: voter partial flush only happens on a new POST
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="RT-05: /v1/updates/status never triggers the bounded partial "
                   "flush; survivors of a lost cohort member poll 'pending' forever")
def test_rt05_surviving_workers_commit_after_cohort_member_crashes(tmp_path):
    runtime, client = voter(tmp_path, gradient_aggregation_enabled=True,
                            gradient_aggregation_cohort_size=3,
                            gradient_aggregation_flush_timeout_seconds=0.05)
    for worker in ("w1", "w2", "w3"):
        assert client.post("/v1/workers/register", json={"worker_id": worker}).status_code == 200
    for worker in ("w1", "w2"):  # w3 crashes before submitting
        response = client.post("/v1/gradients", json=dict(
            update_id=f"{worker}:u", worker_id=worker, base_version=0,
            gradient_b64=full_gradient(runtime.state, 0.01), loss=1.0))
        assert response.json()["status"] == "pending"
    time.sleep(0.1)  # exceed the bounded flush timeout
    status = poll(lambda: client.get("/v1/updates/status", params=dict(
        update_id="w1:u", base_version=0, worker_id="w1")).json()["status"],
        lambda value: value == "applied")
    assert status == "applied"
    assert runtime.state.version == 1


# --------------------------------------------------------------------------
# RT-06  S4 / L1 at the reducer tier: one future-base request poisons a cell
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="RT-06: reducers trust worker base_version; one future base "
                   "evicts honest work and makes every later submission 'stale'")
def test_rt06_future_base_submission_cannot_poison_reducer(tmp_path):
    runtime, client = tree_voter(tmp_path)
    reducer = GradientReducer(REDUCER_ID, VoterUpstream(client), group_size=2,
                              wire_dtype="float32")
    payload = reducer_payload(runtime.state)
    assert reducer.submit(gradient_request("w1:a", "w1", 0, payload))["status"] == "pending"
    # e.g. a worker left over from an earlier job whose log reached version 500
    reducer.submit(gradient_request("ghost:a", "ghost", 500, payload))
    reducer.submit(gradient_request("w2:a", "w2", 0, payload))
    assert reducer.status("w1:a", 0)["status"] == "applied"
    assert runtime.state.version == 1


# --------------------------------------------------------------------------
# RT-07  S5 / L1 at the reducer tier: one wrong-shape member blocks a cell
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="RT-07: reducer admits gradients without shape validation; a "
                   "mismatched member stays in the bucket and batch formation fails forever")
def test_rt07_wrong_shape_member_rejected_without_blocking_cell(tmp_path):
    runtime, client = tree_voter(tmp_path)
    reducer = GradientReducer(REDUCER_ID, VoterUpstream(client), group_size=2,
                              wire_dtype="float32", flush_timeout_seconds=0.05)
    reducer.submit(gradient_request("w1:a", "w1", 0, reducer_payload(runtime.state)))
    with pytest.raises(ValueError):
        reducer.submit(gradient_request("bad:a", "bad", 0,
                                        reducer_payload(runtime.state, wrong_shape=True)))
    reducer.submit(gradient_request("w2:a", "w2", 0, reducer_payload(runtime.state)))
    assert reducer.status("w1:a", 0)["status"] == "applied"
    assert runtime.state.version == 1


# --------------------------------------------------------------------------
# RT-08  S5: int8 finiteness is checked before dequantisation in the reducer
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="RT-08: b64_to_arrays checks finiteness before int8 "
                   "dequantisation, so scale*127 overflow yields inf inside the reducer")
def test_rt08_reducer_decoder_rejects_int8_scale_overflow():
    arrays = {"w": np.array([127, -127], np.int8),
              "__elasticml_int8_scale__::w": np.asarray(3e38, np.float32)}
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    with pytest.raises(ValueError):
        b64_to_arrays(base64.b64encode(buffer.getvalue()).decode())


# --------------------------------------------------------------------------
# RT-09  S1: overlapping primary/backup reducer batches (stale profile)
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="RT-09: with gradient_aggregation_allow_stale the voter sees "
                   "only reducer aggregate ids, so a retried worker update is applied twice")
def test_rt09_worker_contribution_applied_at_most_once_across_reducer_failover(tmp_path):
    runtime, client = tree_voter(tmp_path, gradient_aggregation_allow_stale=True,
                                 max_staleness=8)
    upstream = VoterUpstream(client)
    payload = reducer_payload(runtime.state)
    primary = GradientReducer(REDUCER_ID, upstream, group_size=2, wire_dtype="float32")
    primary.submit(gradient_request("w1:u1", "w1", 0, payload))
    primary.submit(gradient_request("w2:u2", "w2", 0, payload))
    assert runtime.state.version == 1
    # Primary dies before w1 sees its ack; w1 retries the SAME update at the
    # deterministic backup, where a slow worker w3 also lands on base 0.
    backup = GradientReducer(REDUCER_ID, upstream, group_size=2, wire_dtype="float32")
    backup.submit(gradient_request("w1:u1", "w1", 0, payload))
    backup.submit(gradient_request("w3:u3", "w3", 0, payload))
    # S1: w1:u1 already changed the model; a second aggregate containing it
    # must not be applied.
    assert runtime.state.version == 1


# --------------------------------------------------------------------------
# RT-10  error propagation: upstream 4xx leaves reducer members pending forever
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="RT-10: reducer _advance ignores 422/429 replies, so workers "
                   "are never told their aggregate was rejected and poll 'pending' forever")
def test_rt10_upstream_rejection_gives_workers_a_terminal_status(tmp_path):
    runtime, client = tree_voter(tmp_path)
    upstream = VoterUpstream(client)
    reducer = GradientReducer(REDUCER_ID, upstream, group_size=2, wire_dtype="float32")
    payload = reducer_payload(runtime.state, extra_tensor=True)  # wrong model at both workers
    reducer.submit(gradient_request("w1:a", "w1", 0, payload))
    reducer.submit(gradient_request("w2:a", "w2", 0, payload))
    assert any(code == 422 for _, _, code in upstream.calls)
    for _ in range(5):
        status = reducer.status("w1:a", 0)["status"]
    assert status != "pending"


# --------------------------------------------------------------------------
# RT-11  §7 measurement integrity: stale/duplicate/dropped counters reset on restart
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="RT-11: §7 duplicate/dropped metrics are volatile and reset "
                   "when a replica restarts, so reported counts undercount")
def test_rt11_reported_duplicate_and_drop_counts_survive_restart(tmp_path):
    state = make_state(tmp_path, max_staleness=0)
    base, gradient, loss = make_gradient(state, "w1")
    state.apply_update(update_id="a", worker_id="w1", base_version=base,
                       gradient_b64=gradient, loss=loss)
    state.apply_update(update_id="a", worker_id="w1", base_version=base,
                       gradient_b64=gradient, loss=loss)  # duplicate
    with pytest.raises(UpdateRejected):
        state.apply_update(update_id="b", worker_id="w2", base_version=0,
                           gradient_b64=gradient, loss=loss)  # dropped as stale
    before = dict(state.update_metrics)
    restarted = make_state(tmp_path, max_staleness=0)
    for key in ("duplicates", "dropped", "dropped_examples"):
        assert restarted.update_metrics[key] == before[key], key


# --------------------------------------------------------------------------
# RT-12  §5 fault model: peer mode runs with no authentication at all
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="RT-12: peer mode validates with auth_token=None, leaving "
                   "prepare/accept open to any network client; the spec only excludes stolen tokens")
def test_rt12_peer_mode_requires_authentication(tmp_path):
    config = CoordinatorConfig(node_id="a", election_backend="peer",
                               peer_urls={n: f"http://{n}:8100" for n in "abc"},
                               peer_state_root=str(tmp_path), auth_token=None)
    with pytest.raises(ValueError):
        config.validate()


# --------------------------------------------------------------------------
# RT-13  §6 dataset: "the loader verifies published hashes"
# --------------------------------------------------------------------------

def _idx_gz(path: Path, images: bool, count: int) -> None:
    header = (struct.pack(">IIII", 2051, count, 28, 28) if images
              else struct.pack(">II", 2049, count))
    body = bytes(count * (784 if images else 1))
    path.write_bytes(gzip.compress(header + body))


@pytest.mark.xfail(strict=True, reason="RT-13: checksums are verified only when download=True; "
                   "existing/tampered archives and prepared .npy arrays are never hashed")
def test_rt13_loader_rejects_files_that_do_not_match_published_hashes(tmp_path):
    from elasticml.data import FASHION_FILES, prepare_fashion_mnist
    for name in FASHION_FILES:
        _idx_gz(tmp_path / name, "images" in name, 20)
    with pytest.raises(ValueError):
        prepare_fashion_mnist(tmp_path, download=False)


# --------------------------------------------------------------------------
# RT-14  §9 / §10 deployment and reproducibility claims
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="RT-14a: spec claims Python 3.11-3.14 but CI tests only 3.12 "
                   "and 3.14")
def test_rt14a_ci_matrix_covers_every_supported_python():
    text = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    versions = set(re.findall(r"'(3\.\d+)'", text))
    assert {"3.11", "3.12", "3.13", "3.14"} <= versions, versions


@pytest.mark.xfail(strict=True, reason="RT-14b: spec names commit 17f8bd4 but the retained clean-"
                   "artifact evidence was produced from ff56901 and this tree has no git metadata")
def test_rt14b_retained_evidence_matches_the_declared_commit():
    evidence = json.loads((REPO / "docs" / "evidence" / "ms3" /
                           "clean-artifact-verification.json").read_text(encoding="utf-8"))
    assert str(evidence.get("source_commit", "")).startswith("17f8bd4")


@pytest.mark.xfail(strict=True, reason="RT-14c: HOKEA_NET_BACKEND is documented in §9 but is not "
                   "read anywhere in this repository")
def test_rt14c_documented_hokea_net_backend_is_implemented():
    sources = [*REPO.glob("*.py"), *REPO.glob("scripts/*"), *REPO.glob("src/elasticml/*.py"),
               *REPO.glob("hokea_tests/*.py")]
    assert any("HOKEA_NET_BACKEND" in path.read_text(encoding="utf-8", errors="ignore")
               for path in sources if path.is_file())


@pytest.mark.xfail(strict=True, reason="RT-14d: the selected 94.68% run record omits §7 fields: "
                   "commit, script, per-role seeds, latency p50/p95, CPU/memory/bytes, churn "
                   "rates, network profile; its raw log is not in the checkout")
def test_rt14d_selected_run_evidence_records_every_section7_field():
    path = REPO / "docs" / "evidence" / "model" / "fashion_cnn_v5_hokea_lr006_20.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    flat = json.dumps(record).lower()
    required = ["source_commit", "script", "data_selection_seed", "fault_schedule_seed",
                "p50", "p95", "commit_latency", "cpu", "memory", "durable_bytes",
                "peer_payload_bytes", "join_rate", "leave_rate", "network_profile",
                "batch_size", "fan_in"]
    missing = [field for field in required if field not in flat]
    raw = (path.parent / record["raw_result"]).resolve()
    assert not missing and raw.exists(), {"missing": missing, "raw_log_present": raw.exists()}


# ==========================================================================
# New coverage that PASSES today (fills gaps in the compliance matrix)
# ==========================================================================

@pytest.fixture
def peers(tmp_path, monkeypatch):
    cluster = PeerCluster(tmp_path, monkeypatch)
    yield cluster
    cluster.close()


def test_s6_read_at_replica_that_missed_every_acknowledged_write(peers):
    """S6 with rotating quorums: the reader never accepted the write it must see."""
    nodes = "abc"
    for index in range(4):
        writer, reader, third = (nodes[(index + k) % 3] for k in range(3))
        peers.down = {reader}
        result = peers.successful_submit(writer, peers.payload(f"rot-{index}", writer))
        assert result.json()["status"] == "applied"
        acknowledged = result.json()["model_version"]
        peers.down = {writer}  # read quorum = reader + third, reader missed the accept
        # The first read usually returns 503: the lagging reader's recovery
        # ballot is below the writer's promise (RT-16).  L3 only promises
        # eventual progress, so retry like PeerCluster.converge().
        response = poll(lambda: peers.clients[reader].get("/v1/model"),
                        lambda r: r.status_code != 503, seconds=2.0, interval=0.01)
        assert response.status_code == 200, response.text
        assert response.json()["model_version"] >= acknowledged
    peers.down = set()
    peers.converge()


def test_s8_isolated_replica_cannot_acknowledge_commit_or_authoritative_read(peers):
    peers.down = {"b", "c"}
    before = peers.runtimes["a"].state.local_debug_snapshot()
    write = peers.submit("a", peers.payload("isolated"))
    read = peers.clients["a"].get("/v1/model")
    assert write.status_code == 503 and read.status_code == 503
    assert peers.runtimes["a"].state.local_debug_snapshot() == before


def test_l4_backup_reducer_commits_after_primary_dies_before_forwarding(tmp_path):
    runtime, client = tree_voter(tmp_path)
    upstream = VoterUpstream(client)
    payload = reducer_payload(runtime.state)
    primary = GradientReducer(REDUCER_ID, upstream, group_size=2, wire_dtype="float32")
    primary.submit(gradient_request("w1:a", "w1", 0, payload))  # pending, then primary dies
    backup = GradientReducer(REDUCER_ID, upstream, group_size=2, wire_dtype="float32")
    backup.submit(gradient_request("w1:a", "w1", 0, payload))  # worker retry
    result = backup.submit(gradient_request("w2:a", "w2", 0, payload))
    assert result["status"] == "applied"
    assert runtime.state.version == 1
    assert [call for call in upstream.calls if call[0] == "POST"] == [
        ("POST", "/v1/gradients", 200)]


def test_l4_s1_backup_rebuilds_same_batch_after_primary_committed(tmp_path):
    runtime, client = tree_voter(tmp_path)
    upstream = VoterUpstream(client)
    payload = reducer_payload(runtime.state)
    primary = GradientReducer(REDUCER_ID, upstream, group_size=2, wire_dtype="float32")
    primary.submit(gradient_request("w1:a", "w1", 0, payload))
    primary.submit(gradient_request("w2:a", "w2", 0, payload))
    assert runtime.state.version == 1
    weights = {k: v.clone() for k, v in runtime.state.model.state_dict().items()}
    # Primary died before relaying; both workers retry unchanged IDs at backup.
    backup = GradientReducer(REDUCER_ID, upstream, group_size=2, wire_dtype="float32")
    backup.submit(gradient_request("w1:a", "w1", 0, payload))
    result = backup.submit(gradient_request("w2:a", "w2", 0, payload))
    assert result["status"] in {"applied", "duplicate"}
    assert runtime.state.version == 1
    for name, value in runtime.state.model.state_dict().items():
        assert torch.equal(value, weights[name])
