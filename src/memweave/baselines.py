"""Baselines measured against memweave.

(a) ``BufferOnlyMemory`` — a short-term buffer holding only the most
    recent ``capacity`` facts, ranked by recency. No embeddings, no
    contradiction handling: whatever scrolled out is forgotten, and when
    both versions of a fact are in the buffer the most recent wins by
    position alone.

(b) ``VectorNoDecayMemory`` — the full episodic store ranked by pure
    embedding cosine. Nothing is ever superseded or decayed, so a stale
    fact whose text is nearly identical to its replacement competes on
    embedding noise alone.
"""

from __future__ import annotations

import numpy as np

from .embedder import Embedder
from .facts import Fact


class BufferOnlyMemory:
    def __init__(self, capacity: int = 20) -> None:
        self.capacity = capacity
        self.buffer: list[Fact] = []
        self._next_id = 0

    def add_fact(
        self, user: str, slot: str, value: str, session: int, text: str = ""
    ) -> Fact:
        fact = Fact(
            fact_id=self._next_id,
            user=user,
            slot=slot,
            value=value,
            session=session,
            text=text,
        )
        self._next_id += 1
        self.buffer.append(fact)
        overflow = len(self.buffer) - self.capacity
        if overflow > 0:
            del self.buffer[:overflow]
        return fact

    def retrieve(
        self, query: str, user: str, now_session: int, k: int = 8, **_: object
    ) -> list[tuple[Fact, float]]:
        mine = [f for f in self.buffer if f.user == user]
        mine.sort(key=lambda f: f.session, reverse=True)
        return [(f, 1.0) for f in mine[:k]]

    def assemble_context(
        self, query: str, user: str, now_session: int, char_budget: int = 1200
    ) -> list[Fact]:
        chosen: list[Fact] = []
        used = 0
        for fact, _ in self.retrieve(query, user, now_session, k=10_000):
            cost = len(fact.text) + 1
            if used + cost > char_budget:
                continue
            chosen.append(fact)
            used += cost
        return chosen


class VectorNoDecayMemory:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.facts: list[Fact] = []
        self._embeddings: list[np.ndarray] = []
        self._next_id = 0

    def add_fact(
        self, user: str, slot: str, value: str, session: int, text: str = ""
    ) -> Fact:
        fact = Fact(
            fact_id=self._next_id,
            user=user,
            slot=slot,
            value=value,
            session=session,
            text=text,
        )
        self._next_id += 1
        self.facts.append(fact)
        self._embeddings.append(self.embedder.encode([fact.text])[0])
        return fact

    def retrieve(
        self, query: str, user: str, now_session: int, k: int = 8, **_: object
    ) -> list[tuple[Fact, float]]:
        candidates = [
            (i, f) for i, f in enumerate(self.facts) if f.user == user
        ]
        if not candidates:
            return []
        q = self.embedder.encode([query])[0]
        scored = [
            (fact, float(np.dot(q, self._embeddings[i])))
            for i, fact in candidates
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    def assemble_context(
        self, query: str, user: str, now_session: int, char_budget: int = 1200
    ) -> list[Fact]:
        chosen: list[Fact] = []
        used = 0
        for fact, _ in self.retrieve(query, user, now_session, k=10_000):
            cost = len(fact.text) + 1
            if used + cost > char_budget:
                continue
            chosen.append(fact)
            used += cost
        return chosen
