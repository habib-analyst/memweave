"""A minimal conversational-agent harness over a memory backend.

The agent observes structured facts as sessions proceed and answers
slot questions from retrieved memory. Answering is deliberately
mechanical — the value of the highest-ranked retrieved fact whose slot
matches the question — so the benchmark measures the *memory*, not an
LLM's reading comprehension. Any of the three shipped backends
(memweave, buffer-only, vector-no-decay) can sit under the same harness.
"""

from __future__ import annotations

from dataclasses import dataclass

from .benchgen import Probe, Statement


@dataclass
class Answer:
    value: str | None
    retrieved: list[tuple[str, str]]  # (slot, value) pairs, ranked
    context_chars: int


class MemoryAgent:
    def __init__(self, memory: object) -> None:
        self.memory = memory

    def observe(self, statement: Statement) -> None:
        self.memory.add_fact(  # type: ignore[attr-defined]
            user=statement.user,
            slot=statement.slot,
            value=statement.value,
            session=statement.session,
            text=statement.text,
        )

    def answer(self, probe: Probe, now_session: int, k: int = 8) -> Answer:
        ranked = self.memory.retrieve(  # type: ignore[attr-defined]
            probe.query, probe.user, now_session, k=k
        )
        value: str | None = None
        for fact, _score in ranked:
            if fact.slot == probe.slot:
                value = fact.value
                break
        context = self.memory.assemble_context(  # type: ignore[attr-defined]
            f"everything known about {probe.user}", probe.user, now_session
        )
        return Answer(
            value=value,
            retrieved=[(f.slot, f.value) for f, _ in ranked],
            context_chars=sum(len(f.text) + 1 for f in context),
        )
