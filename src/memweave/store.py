"""The memweave episodic memory store.

Two mechanisms distinguish it from a plain vector store:

1. **Time-decay relevance scoring.** Retrieval score is
   ``cosine(query, fact) * 0.5 ** (age / half_life) * reinforcement`` where
   age is measured in sessions since the fact was last seen and
   reinforcement grows (logarithmically) with how often the fact has been
   recalled or reaffirmed. Recent and re-used facts outrank stale ones
   with near-identical text.

2. **Explicit contradiction policy.** Slots are single-valued. When a new
   fact contradicts a current fact for the same (user, slot), the store
   resolves it by a named policy instead of silently keeping both:
   - ``supersede``: the old fact is marked superseded (kept for audit,
     excluded from default retrieval); the new fact becomes current.
   - ``keep_both``: both stay current, distinguished only by timestamps;
     retrieval ranks them by decay like any other facts.
   - ``flag``: both stay current and are marked conflicted; retrieval
     annotates results so a caller can surface the conflict.

The store persists to JSON (embeddings are recomputed on load) so a new
process/session can sync from disk and continue — the cross-session path
exercised in the tests and benchmark.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .embedder import Embedder
from .facts import Fact

POLICIES = ("supersede", "keep_both", "flag")


@dataclass
class ContradictionEvent:
    user: str
    slot: str
    old_value: str
    new_value: str
    session: int
    policy: str


@dataclass
class MemweaveStore:
    embedder: Embedder
    policy: str = "supersede"
    half_life_sessions: float = 4.0
    reinforcement_alpha: float = 0.15
    facts: list[Fact] = field(default_factory=list)
    contradictions: list[ContradictionEvent] = field(default_factory=list)
    _embeddings: list[np.ndarray] = field(default_factory=list)
    _next_id: int = 0

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(f"unknown policy {self.policy!r}; choose {POLICIES}")

    # ------------------------------------------------------------------ #
    # writes
    # ------------------------------------------------------------------ #
    def add_fact(
        self, user: str, slot: str, value: str, session: int, text: str = ""
    ) -> Fact:
        """Add a fact, resolving any contradiction per the store policy.

        Restating the *same* value is a reaffirmation, not a contradiction:
        it refreshes last_seen and counts as a recall (reinforcement).
        """
        for existing in self.facts:
            if (
                existing.user == user
                and existing.slot == slot
                and not existing.superseded
            ):
                if existing.value == value:
                    existing.last_seen = session
                    existing.access_count += 1
                    return existing
                self.contradictions.append(
                    ContradictionEvent(
                        user=user,
                        slot=slot,
                        old_value=existing.value,
                        new_value=value,
                        session=session,
                        policy=self.policy,
                    )
                )
                if self.policy == "supersede":
                    existing.superseded = True
                elif self.policy == "flag":
                    existing.conflicted = True
                # keep_both: nothing to mark; timestamps separate them.
        fact = Fact(
            fact_id=self._next_id,
            user=user,
            slot=slot,
            value=value,
            session=session,
            text=text,
            conflicted=(self.policy == "flag" and self._has_conflict(user, slot)),
        )
        self._next_id += 1
        self.facts.append(fact)
        self._embeddings.append(self.embedder.encode([fact.text])[0])
        return fact

    def _has_conflict(self, user: str, slot: str) -> bool:
        return any(
            f.user == user and f.slot == slot and not f.superseded
            for f in self.facts
        )

    # ------------------------------------------------------------------ #
    # reads
    # ------------------------------------------------------------------ #
    def decay_factor(self, fact: Fact, now_session: int) -> float:
        age = max(0, now_session - fact.last_seen)
        return 0.5 ** (age / self.half_life_sessions)

    def score(self, fact: Fact, cosine: float, now_session: int) -> float:
        reinforcement = 1.0 + self.reinforcement_alpha * math.log1p(
            fact.access_count
        )
        return cosine * self.decay_factor(fact, now_session) * reinforcement

    def retrieve(
        self,
        query: str,
        user: str,
        now_session: int,
        k: int = 8,
        include_superseded: bool = False,
        record_access: bool = False,
    ) -> list[tuple[Fact, float]]:
        """Rank the user's current facts for a query by decayed relevance."""
        candidates = [
            (i, f)
            for i, f in enumerate(self.facts)
            if f.user == user and (include_superseded or not f.superseded)
        ]
        if not candidates:
            return []
        q = self.embedder.encode([query])[0]
        scored: list[tuple[Fact, float]] = []
        for i, fact in candidates:
            cosine = float(np.dot(q, self._embeddings[i]))
            scored.append((fact, self.score(fact, cosine, now_session)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        top = scored[:k]
        if record_access:
            for fact, _ in top:
                fact.access_count += 1
        return top

    def current_fact(self, user: str, slot: str) -> Fact | None:
        current = [
            f
            for f in self.facts
            if f.user == user and f.slot == slot and not f.superseded
        ]
        if not current:
            return None
        return max(current, key=lambda f: f.session)

    # ------------------------------------------------------------------ #
    # context compression
    # ------------------------------------------------------------------ #
    def assemble_context(
        self,
        query: str,
        user: str,
        now_session: int,
        char_budget: int = 1200,
        k: int = 32,
    ) -> list[Fact]:
        """Greedily pack the highest-scored facts under a character budget.

        One fact per slot (the best-scoring one) — compression drops the
        superseded/duplicate versions that a raw history dump would carry.
        """
        ranked = self.retrieve(query, user, now_session, k=k)
        chosen: list[Fact] = []
        seen_slots: set[str] = set()
        used = 0
        for fact, _ in ranked:
            if fact.slot in seen_slots:
                continue
            cost = len(fact.text) + 1
            if used + cost > char_budget:
                continue
            chosen.append(fact)
            seen_slots.add(fact.slot)
            used += cost
        return chosen

    # ------------------------------------------------------------------ #
    # cross-session persistence ("sync")
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        payload = {
            "policy": self.policy,
            "half_life_sessions": self.half_life_sessions,
            "reinforcement_alpha": self.reinforcement_alpha,
            "next_id": self._next_id,
            "facts": [
                {
                    "fact_id": f.fact_id,
                    "user": f.user,
                    "slot": f.slot,
                    "value": f.value,
                    "session": f.session,
                    "text": f.text,
                    "superseded": f.superseded,
                    "conflicted": f.conflicted,
                    "access_count": f.access_count,
                    "last_seen": f.last_seen,
                }
                for f in self.facts
            ],
            "contradictions": [
                {
                    "user": c.user,
                    "slot": c.slot,
                    "old_value": c.old_value,
                    "new_value": c.new_value,
                    "session": c.session,
                    "policy": c.policy,
                }
                for c in self.contradictions
            ],
        }
        Path(path).write_text(json.dumps(payload, indent=1))

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder) -> "MemweaveStore":
        payload = json.loads(Path(path).read_text())
        store = cls(
            embedder=embedder,
            policy=payload["policy"],
            half_life_sessions=payload["half_life_sessions"],
            reinforcement_alpha=payload["reinforcement_alpha"],
        )
        store._next_id = payload["next_id"]
        for raw in payload["facts"]:
            store.facts.append(Fact(**raw))
        store.contradictions = [
            ContradictionEvent(**c) for c in payload["contradictions"]
        ]
        if store.facts:
            store._embeddings = list(
                embedder.encode([f.text for f in store.facts])
            )
        return store
