"""memweave: episodic agent memory with time-decay relevance scoring and
explicit contradiction policies, plus a scripted benchmark that measures
stale-fact error rate.

No LLM is required for the memory layer or the benchmark: facts are
structured (user, slot, value) records rendered as text and embedded with
all-MiniLM-L6-v2.
"""

from .facts import Fact
from .store import MemweaveStore, ContradictionEvent
from .baselines import BufferOnlyMemory, VectorNoDecayMemory

__all__ = [
    "Fact",
    "MemweaveStore",
    "ContradictionEvent",
    "BufferOnlyMemory",
    "VectorNoDecayMemory",
]

__version__ = "0.1.0"
