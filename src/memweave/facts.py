"""Structured fact records.

A fact is a (user, slot, value) triple rendered as natural text for
embedding, e.g. Fact(user="maria", slot="city", value="Lisbon") renders as
"maria's city is Lisbon". Two facts with the same (user, slot) but
different values are, by definition, a contradiction: the slot is
single-valued, so at most one value can be current.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Fact:
    fact_id: int
    user: str
    slot: str
    value: str
    session: int  # session index at which the fact was stated
    text: str = ""
    superseded: bool = False  # replaced by a newer value under 'supersede'
    conflicted: bool = False  # part of an unresolved contradiction ('flag')
    access_count: int = 0  # times this fact was recalled (reinforcement)
    last_seen: int = 0  # last session the fact was stated or reaffirmed

    def __post_init__(self) -> None:
        if not self.text:
            self.text = f"{self.user}'s {self.slot} is {self.value}"
        if self.last_seen == 0:
            self.last_seen = self.session

    def render(self) -> str:
        return self.text
