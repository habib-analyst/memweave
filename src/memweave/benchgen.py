"""Deterministic multi-session user-fact streams for the benchmark.

Each synthetic user gets 8 sessions of statements against 10 single-valued
slots. The generator plants, per user:

- **facts**: an initial value for every slot (sessions 0-1);
- **contradictions**: 4 of the 10 slots receive a *new* value in sessions
  4-5, superseding the planted one;
- **reaffirmations**: 2 never-updated slots are restated verbatim in
  sessions 2 and 6 (a memory that is used should survive decay);
- **distractors**: near-miss statements on look-alike slots
  (``visited_city`` vs ``city``, ``dream_job`` vs ``occupation``,
  ``considering_diet`` vs ``diet``) that embed close to real facts but
  must never be answered as the slot's value.

Probes are issued after the final session (t=8): one per (user, slot),
each labelled with the current gold value and, for updated slots, the
superseded (stale) value.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

SLOTS: dict[str, list[str]] = {
    "city": ["Lisbon", "Porto", "Braga", "Coimbra", "Faro", "Aveiro"],
    "occupation": ["teacher", "nurse", "welder", "accountant", "chef", "plumber"],
    "employer": ["Northwind", "Helios Health", "City Council", "Freelance", "Orbital Labs", "Kite Foods"],
    "diet": ["vegetarian", "vegan", "omnivore", "pescatarian", "halal", "kosher"],
    "hobby": ["chess", "trail running", "ceramics", "photography", "salsa dancing", "woodworking"],
    "pet": ["a cat named Milo", "a dog named Rex", "a parrot named Kiwi", "no pet", "a rabbit named Clover", "two goldfish"],
    "language": ["Python", "Rust", "Go", "TypeScript", "Julia", "Kotlin"],
    "timezone": ["UTC+0", "UTC+1", "UTC+2", "UTC-5", "UTC+5", "UTC+9"],
    "instrument": ["piano", "violin", "guitar", "flute", "drums", "cello"],
    "sport": ["swimming", "cycling", "climbing", "rowing", "tennis", "badminton"],
}

DISTRACTORS: dict[str, tuple[str, list[str]]] = {
    # look-alike slot -> (render template, values)
    "visited_city": ("{u} visited {v} last spring", SLOTS["city"]),
    "dream_job": ("{u} once dreamed of working as a {v}", ["pilot", "architect", "vet", "journalist"]),
    "considering_diet": ("{u} is considering trying a {v} diet", ["keto", "paleo", "mediterranean", "raw"]),
}

QUESTIONS: dict[str, str] = {
    "city": "Which city does {u} live in?",
    "occupation": "What is {u}'s occupation?",
    "employer": "Who does {u} work for?",
    "diet": "What diet does {u} follow?",
    "hobby": "What is {u}'s hobby?",
    "pet": "What pet does {u} have?",
    "language": "What programming language does {u} use?",
    "timezone": "What timezone is {u} in?",
    "instrument": "What instrument does {u} play?",
    "sport": "What sport does {u} play?",
}

USERS = [
    "maria", "jonas", "priya", "tomas", "aisha", "lukas",
    "nina", "omar", "elena", "david", "sana", "pavel",
    "ingrid", "karim", "lucia", "henrik",
]

N_SESSIONS = 8
PROBE_TIME = 8


@dataclass
class Statement:
    user: str
    slot: str
    value: str
    session: int
    kind: str  # plant | update | reaffirm | distractor
    text: str = ""


@dataclass
class Probe:
    user: str
    slot: str
    query: str
    gold_value: str
    stale_value: str | None  # superseded value, if the slot was updated
    updated: bool


@dataclass
class Stream:
    statements: list[Statement] = field(default_factory=list)
    probes: list[Probe] = field(default_factory=list)
    n_updates: int = 0
    n_users: int = 0


def generate(seed: int = 20261026, users: list[str] | None = None) -> Stream:
    rng = random.Random(seed)
    users = users or USERS
    stream = Stream(n_users=len(users))
    slot_names = list(SLOTS.keys())

    for user in users:
        per_user: list[Statement] = []
        initial: dict[str, str] = {}
        for slot in slot_names:
            initial[slot] = rng.choice(SLOTS[slot])

        updated_slots = rng.sample(slot_names, 4)
        stable_slots = [s for s in slot_names if s not in updated_slots]
        reaffirmed = rng.sample(stable_slots, 2)
        new_values: dict[str, str] = {}
        for slot in updated_slots:
            choices = [v for v in SLOTS[slot] if v != initial[slot]]
            new_values[slot] = rng.choice(choices)

        def plant(slot: str, session: int, kind: str, value: str) -> None:
            per_user.append(
                Statement(user=user, slot=slot, value=value, session=session, kind=kind)
            )

        for slot in slot_names[:5]:
            plant(slot, 0, "plant", initial[slot])
        for slot in slot_names[5:]:
            plant(slot, 1, "plant", initial[slot])

        distractor_slots = list(DISTRACTORS.keys())
        for session, count in ((2, 3), (3, 3), (6, 3), (7, 2)):
            for _ in range(count):
                dslot = rng.choice(distractor_slots)
                template, values = DISTRACTORS[dslot]
                value = rng.choice(values)
                per_user.append(
                    Statement(
                        user=user,
                        slot=dslot,
                        value=value,
                        session=session,
                        kind="distractor",
                        text=template.format(u=user, v=value),
                    )
                )
            if session == 2:
                plant(reaffirmed[0], 2, "reaffirm", initial[reaffirmed[0]])
            if session == 6:
                plant(reaffirmed[1], 6, "reaffirm", initial[reaffirmed[1]])

        for slot in updated_slots[:2]:
            plant(slot, 4, "update", new_values[slot])
        for slot in updated_slots[2:]:
            plant(slot, 5, "update", new_values[slot])

        per_user.sort(key=lambda s: (s.session, s.kind))
        stream.statements.extend(per_user)
        stream.n_updates += len(updated_slots)

        for slot in slot_names:
            was_updated = slot in updated_slots
            stream.probes.append(
                Probe(
                    user=user,
                    slot=slot,
                    query=QUESTIONS[slot].format(u=user),
                    gold_value=new_values[slot] if was_updated else initial[slot],
                    stale_value=initial[slot] if was_updated else None,
                    updated=was_updated,
                )
            )
    return stream
