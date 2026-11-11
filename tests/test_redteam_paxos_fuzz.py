"""Red-team S2 checks for the project-written Paxos (docs/submission/PROJECT_SPEC.pdf §4).

``test_s2_randomized_schedules_choose_one_value_per_slot`` explores seeded
interleavings of three concurrent proposers with dropped requests, lost
replies, crash/restart of acceptors with intact disks, and the
recover-then-propose (``_prepared_empty``) fast path.  It is a randomized
search, not a proof; every seed is reproducible.

``test_demo_voter_restart_without_durable_files_breaks_s2`` is a deliberate
counterexample (like tests/test_weakened_quorum_demo.py).  It shows that the
excluded fault "a voter restarts without its durable files" is load-bearing:
nothing in the runtime detects it, and HOKEA container restarts (used by
test_hokea_tree_failover.py) fall in that class.  See RT-15 in
docs/REDTEAM_SPEC_GAP_ANALYSIS.md.
"""
from __future__ import annotations

import random
import shutil
import threading
from collections import defaultdict

import pytest

from elasticml.consensus import DurableAcceptor, NoQuorum, PaxosProposer

NODES = ("a", "b", "c")


class FlakyCluster:
    """In-memory transport over real DurableAcceptor directories."""

    def __init__(self, root, seed, drop=0.15, lose_reply=0.15, restart=0.05):
        self.root = root
        self.rng = random.Random(seed)
        self.rng_lock = threading.Lock()
        self.drop, self.lose_reply, self.restart_p = drop, lose_reply, restart
        self.locks = {node: threading.Lock() for node in NODES}
        self.acceptors = {node: DurableAcceptor(root / node / "acceptor") for node in NODES}
        self.restarts = 0

    def _roll(self):
        with self.rng_lock:
            return self.rng.random()

    def _call(self, peer, operation, *args):
        if self._roll() < self.drop:
            raise ConnectionError("request dropped")
        with self.locks[peer]:  # one process owns a directory at a time
            if self._roll() < self.restart_p:
                self.acceptors[peer] = DurableAcceptor(self.root / peer / "acceptor")
                self.restarts += 1
            result = getattr(self.acceptors[peer], operation)(*args)
        if self._roll() < self.lose_reply:
            raise TimeoutError("reply lost after durable state change")
        return result

    def prepare(self, peer, slot, ballot):
        return self._call(peer, "prepare", slot, ballot)

    def accept(self, peer, slot, ballot, value):
        return self._call(peer, "accept", slot, ballot, value)


class ReliableTransport:
    def __init__(self, acceptors):
        self.acceptors = acceptors

    def prepare(self, peer, slot, ballot):
        return self.acceptors[peer].prepare(slot, ballot)

    def accept(self, peer, slot, ballot, value):
        return self.acceptors[peer].accept(slot, ballot, value)


@pytest.mark.parametrize("seed", range(12))
def test_s2_randomized_schedules_choose_one_value_per_slot(tmp_path, seed):
    cluster = FlakyCluster(tmp_path, seed)
    proposers = {node: PaxosProposer(node, tmp_path / node / "proposer", list(NODES), cluster)
                 for node in NODES}
    chosen = defaultdict(set)
    chosen_lock = threading.Lock()
    slots = (1, 2, 3)

    def drive(node):
        rng = random.Random(f"{seed}:{node}")
        for slot in slots:
            for attempt in range(25):
                try:
                    if rng.random() < 0.3:
                        recovered = proposers[node].recover(slot)
                        if recovered is not None:
                            with chosen_lock:
                                chosen[slot].add(repr(recovered.value))
                    result = proposers[node].propose(
                        slot, {"slot": slot, "by": node, "attempt": attempt})
                except NoQuorum:
                    continue
                with chosen_lock:
                    chosen[slot].add(repr(result.value))
                break

    threads = [threading.Thread(target=drive, args=(node,)) for node in NODES]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
        assert not thread.is_alive(), "proposer thread hung"

    # Re-read every slot from intact disks with a reliable network.
    verifier = PaxosProposer("a", tmp_path / "a" / "proposer", list(NODES),
                             ReliableTransport({n: DurableAcceptor(tmp_path / n / "acceptor")
                                                for n in NODES}))
    for slot in slots:
        for _ in range(10):  # a fresh proposer must climb past promised ballots
            try:
                final = verifier.recover(slot)
                break
            except NoQuorum:
                continue
        else:
            pytest.fail(f"reliable verifier could not read slot {slot}")
        if final is not None:
            chosen[slot].add(repr(final.value))
        assert len(chosen[slot]) <= 1, (seed, slot, chosen[slot])


def test_demo_voter_restart_without_durable_files_breaks_s2(tmp_path):
    acceptors = {n: DurableAcceptor(tmp_path / n / "acceptor") for n in NODES}
    down = {"c"}

    class Transport(ReliableTransport):
        def _check(self, peer):
            if peer in down:
                raise ConnectionError(peer)

        def prepare(self, peer, slot, ballot):
            self._check(peer)
            return super().prepare(peer, slot, ballot)

        def accept(self, peer, slot, ballot, value):
            self._check(peer)
            return super().accept(peer, slot, ballot, value)

    transport = Transport(acceptors)
    first = PaxosProposer("a", tmp_path / "a" / "proposer", list(NODES), transport)
    v1 = first.propose(1, {"value": "v1"})
    assert set(v1.witnesses) == {"a", "b"}

    # b's container restarts with an empty /data under the same identity,
    # and a becomes unreachable.  Nothing refuses b's rejoin.
    shutil.rmtree(tmp_path / "b")
    acceptors["b"] = DurableAcceptor(tmp_path / "b" / "acceptor")
    down.clear()
    down.add("a")
    second = PaxosProposer("c", tmp_path / "c" / "proposer", list(NODES), transport)
    v2 = second.propose(1, {"value": "v2"})

    assert v1.value != v2.value, "two different values chosen for slot 1"
