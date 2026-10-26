"""Tests for memweave: decay scoring, contradiction policies, baselines,
persistence (cross-session sync), compression, and the benchmark stream.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memweave.agent import MemoryAgent
from memweave.baselines import BufferOnlyMemory, VectorNoDecayMemory
from memweave.benchgen import generate
from memweave.embedder import Embedder
from memweave.store import MemweaveStore


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder()


# ---------------------------------------------------------------------- #
# decay scoring
# ---------------------------------------------------------------------- #
def test_decay_decreases_with_age(embedder: Embedder) -> None:
    store = MemweaveStore(embedder, half_life_sessions=2.0)
    fact = store.add_fact("maria", "city", "Lisbon", session=0)
    assert store.decay_factor(fact, now_session=0) == 1.0
    assert store.decay_factor(fact, now_session=2) == pytest.approx(0.5)
    assert store.decay_factor(fact, now_session=4) == pytest.approx(0.25)


def test_reaffirmation_refreshes_decay(embedder: Embedder) -> None:
    store = MemweaveStore(embedder, half_life_sessions=2.0)
    fact = store.add_fact("maria", "city", "Lisbon", session=0)
    store.add_fact("maria", "city", "Lisbon", session=6)
    assert len(store.facts) == 1  # restatement, not a duplicate
    assert fact.access_count == 1
    assert store.decay_factor(fact, now_session=6) == 1.0


def test_newer_fact_outranks_stale_under_decay(embedder: Embedder) -> None:
    store = MemweaveStore(embedder, policy="keep_both", half_life_sessions=2.0)
    store.add_fact("maria", "city", "Lisbon", session=0)
    store.add_fact("maria", "city", "Porto", session=5)
    ranked = store.retrieve("Which city does maria live in?", "maria", now_session=6)
    assert ranked[0][0].value == "Porto"


# ---------------------------------------------------------------------- #
# contradiction policies
# ---------------------------------------------------------------------- #
def test_supersede_policy(embedder: Embedder) -> None:
    store = MemweaveStore(embedder, policy="supersede")
    old = store.add_fact("maria", "city", "Lisbon", session=0)
    store.add_fact("maria", "city", "Porto", session=3)
    assert old.superseded is True
    assert store.current_fact("maria", "city").value == "Porto"
    assert len(store.contradictions) == 1
    ranked = store.retrieve("Which city does maria live in?", "maria", now_session=3)
    assert all(f.value != "Lisbon" for f, _ in ranked)


def test_keep_both_policy(embedder: Embedder) -> None:
    store = MemweaveStore(embedder, policy="keep_both")
    store.add_fact("maria", "city", "Lisbon", session=0)
    store.add_fact("maria", "city", "Porto", session=3)
    current = [f for f in store.facts if not f.superseded]
    assert len(current) == 2
    assert len(store.contradictions) == 1


def test_flag_policy(embedder: Embedder) -> None:
    store = MemweaveStore(embedder, policy="flag")
    old = store.add_fact("maria", "city", "Lisbon", session=0)
    new = store.add_fact("maria", "city", "Porto", session=3)
    assert old.conflicted and new.conflicted
    assert not old.superseded


def test_no_contradiction_for_distinct_slots(embedder: Embedder) -> None:
    store = MemweaveStore(embedder)
    store.add_fact("maria", "city", "Lisbon", session=0)
    store.add_fact("maria", "visited_city", "Lisbon", session=1,
                   text="maria visited Lisbon last spring")
    assert store.contradictions == []


def test_invalid_policy_rejected(embedder: Embedder) -> None:
    with pytest.raises(ValueError):
        MemweaveStore(embedder, policy="shred")


# ---------------------------------------------------------------------- #
# compression
# ---------------------------------------------------------------------- #
def test_context_budget_and_one_per_slot(embedder: Embedder) -> None:
    store = MemweaveStore(embedder, policy="keep_both")
    for slot, value in [("city", "Lisbon"), ("city", "Porto"),
                        ("hobby", "chess"), ("diet", "vegan")]:
        store.add_fact("maria", slot, value, session=0)
    context = store.assemble_context("maria", "maria", now_session=1, char_budget=60)
    assert sum(len(f.text) + 1 for f in context) <= 60
    slots = [f.slot for f in context]
    assert len(slots) == len(set(slots))


# ---------------------------------------------------------------------- #
# persistence / cross-session sync
# ---------------------------------------------------------------------- #
def test_save_load_roundtrip(embedder: Embedder, tmp_path: Path) -> None:
    store = MemweaveStore(embedder, policy="supersede")
    store.add_fact("maria", "city", "Lisbon", session=0)
    store.add_fact("maria", "city", "Porto", session=2)
    store.add_fact("maria", "hobby", "chess", session=1)
    path = tmp_path / "memory.json"
    store.save(path)
    restored = MemweaveStore.load(path, embedder)
    assert restored.current_fact("maria", "city").value == "Porto"
    assert len(restored.contradictions) == 1
    before = store.retrieve("Which city does maria live in?", "maria", 3)
    after = restored.retrieve("Which city does maria live in?", "maria", 3)
    assert [f.value for f, _ in before] == [f.value for f, _ in after]


def test_sync_continuation_matches_uninterrupted(embedder: Embedder, tmp_path: Path) -> None:
    stream = generate()
    statements = [s for s in stream.statements if s.user == "maria"]
    probes = [p for p in stream.probes if p.user == "maria"]

    whole = MemweaveStore(embedder)
    agent = MemoryAgent(whole)
    for s in statements:
        agent.observe(s)
    expected = [agent.answer(p, now_session=8).value for p in probes]

    split = MemweaveStore(embedder)
    agent2 = MemoryAgent(split)
    path = tmp_path / "sync.json"
    for s in statements:
        if s.session == 4:
            split.save(path)
            split = MemweaveStore.load(path, embedder)
            agent2 = MemoryAgent(split)
        agent2.observe(s)
    got = [agent2.answer(p, now_session=8).value for p in probes]
    assert got == expected


# ---------------------------------------------------------------------- #
# baselines
# ---------------------------------------------------------------------- #
def test_buffer_evicts_oldest() -> None:
    buf = BufferOnlyMemory(capacity=3)
    for i in range(5):
        buf.add_fact("maria", "city", f"City{i}", session=i)
    values = [f.value for f in buf.buffer]
    assert values == ["City2", "City3", "City4"]


def test_vector_baseline_keeps_stale_fact(embedder: Embedder) -> None:
    mem = VectorNoDecayMemory(embedder)
    mem.add_fact("maria", "city", "Lisbon", session=0)
    mem.add_fact("maria", "city", "Porto", session=3)
    assert len(mem.facts) == 2  # nothing is ever resolved


# ---------------------------------------------------------------------- #
# benchmark stream sanity + end-to-end behaviour
# ---------------------------------------------------------------------- #
def test_stream_is_deterministic_and_labelled() -> None:
    a, b = generate(), generate()
    assert [(s.user, s.slot, s.value, s.session) for s in a.statements] == [
        (s.user, s.slot, s.value, s.session) for s in b.statements
    ]
    assert len(a.probes) == a.n_users * 10
    assert sum(1 for p in a.probes if p.updated) == a.n_updates == a.n_users * 4
    for probe in a.probes:
        if probe.updated:
            assert probe.stale_value is not None
            assert probe.stale_value != probe.gold_value


def test_memweave_answers_current_value_after_update(embedder: Embedder) -> None:
    store = MemweaveStore(embedder, policy="supersede")
    agent = MemoryAgent(store)
    from memweave.benchgen import Probe, Statement

    agent.observe(Statement("maria", "city", "Lisbon", 0, "plant"))
    agent.observe(Statement("maria", "city", "Porto", 4, "update"))
    probe = Probe(
        user="maria",
        slot="city",
        query="Which city does maria live in?",
        gold_value="Porto",
        stale_value="Lisbon",
        updated=True,
    )
    assert agent.answer(probe, now_session=8).value == "Porto"
