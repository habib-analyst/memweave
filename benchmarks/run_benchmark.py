"""Run the memweave stale-fact benchmark.

Feeds one deterministic stream of multi-session user facts (planted facts,
planted contradictions, reaffirmations, distractors) into three memory
systems behind the same agent harness and measures, per system:

- answer accuracy on post-stream slot probes;
- **stale-fact error rate**: fraction of probes on *updated* slots answered
  with the superseded value;
- retrieval precision/recall@k against the current fact for each probe;
- context coverage: fraction of a user's current facts that fit in a
  1,200-char compressed context.

Usage:  python benchmarks/run_benchmark.py [--seed N] [--out PATH]
Writes results/results.json and prints a summary table. CPU only, no LLM.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memweave.agent import MemoryAgent
from memweave.baselines import BufferOnlyMemory, VectorNoDecayMemory
from memweave.benchgen import PROBE_TIME, generate
from memweave.embedder import Embedder
from memweave.store import MemweaveStore

K = 8
CHAR_BUDGET = 1200


def evaluate(name: str, make_memory, seed: int = 20261026) -> dict:
    stream = generate(seed=seed)
    per_user_memory = {}
    agents = {}
    for user in {s.user for s in stream.statements}:
        per_user_memory[user] = make_memory()
        agents[user] = MemoryAgent(per_user_memory[user])
    for statement in stream.statements:
        agents[statement.user].observe(statement)

    n = correct = stale = missing = wrong = 0
    upd_n = upd_correct = upd_stale = 0
    stable_n = stable_correct = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for probe in stream.probes:
        answer = agents[probe.user].answer(probe, PROBE_TIME, k=K)
        relevant = {(probe.slot, probe.gold_value)}
        retrieved = set(answer.retrieved)
        precisions.append(len(retrieved & relevant) / max(1, len(answer.retrieved)))
        recalls.append(1.0 if relevant & retrieved else 0.0)
        n += 1
        if probe.updated:
            upd_n += 1
        else:
            stable_n += 1
        if answer.value is None:
            missing += 1
        elif answer.value == probe.gold_value:
            correct += 1
            if probe.updated:
                upd_correct += 1
            else:
                stable_correct += 1
        elif probe.stale_value is not None and answer.value == probe.stale_value:
            stale += 1
            if probe.updated:
                upd_stale += 1
        else:
            wrong += 1

    # context coverage: how many of each user's current facts survive
    # compression into a fixed char budget
    gold_by_user: dict[str, set[tuple[str, str]]] = {}
    for probe in stream.probes:
        gold_by_user.setdefault(probe.user, set()).add((probe.slot, probe.gold_value))
    coverages: list[float] = []
    chars: list[float] = []
    for user, gold in gold_by_user.items():
        context = per_user_memory[user].assemble_context(
            f"everything known about {user}", user, PROBE_TIME, char_budget=CHAR_BUDGET
        )
        have = {(f.slot, f.value) for f in context}
        coverages.append(len(have & gold) / len(gold))
        chars.append(sum(len(f.text) + 1 for f in context))

    return {
        "system": name,
        "probes": n,
        "accuracy": correct / n,
        "stale_errors": stale,
        "stale_error_rate_all_probes": stale / n,
        "missing_rate": missing / n,
        "wrong_value_rate": wrong / n,
        "updated_probes": upd_n,
        "updated_accuracy": upd_correct / upd_n,
        "stale_error_rate_updated": upd_stale / upd_n,
        "stable_probes": stable_n,
        "stable_accuracy": stable_correct / stable_n,
        "retrieval_precision_at_k": sum(precisions) / len(precisions),
        "retrieval_recall_at_k": sum(recalls) / len(recalls),
        "context_coverage": sum(coverages) / len(coverages),
        "context_chars_mean": sum(chars) / len(chars),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "results.json",
    )
    args = parser.parse_args()

    embedder = Embedder()
    systems = [
        ("buffer_only", lambda: BufferOnlyMemory(capacity=20)),
        ("vector_no_decay", lambda: VectorNoDecayMemory(embedder)),
        (
            "memweave_supersede",
            lambda: MemweaveStore(embedder, policy="supersede", half_life_sessions=4.0),
        ),
        (
            "memweave_decay_keep_both",
            lambda: MemweaveStore(embedder, policy="keep_both", half_life_sessions=4.0),
        ),
        (
            "memweave_supersede_hl2",
            lambda: MemweaveStore(embedder, policy="supersede", half_life_sessions=2.0),
        ),
        (
            "memweave_supersede_hl8",
            lambda: MemweaveStore(embedder, policy="supersede", half_life_sessions=8.0),
        ),
    ]
    results = [evaluate(name, make, seed=args.seed) for name, make in systems]
    stream = generate(seed=args.seed)
    payload = {
        "seed": args.seed,
        "config": {
            "users": stream.n_users,
            "sessions": 8,
            "slots_per_user": 10,
            "updates_per_user": 4,
            "probes": len(stream.probes),
            "k": K,
            "char_budget": CHAR_BUDGET,
            "buffer_capacity": 20,
            "half_life_sessions": 4.0,
            "embedder": "all-MiniLM-L6-v2",
        },
        "systems": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    for r in results:
        print(
            f"{r['system']:<20} acc={r['accuracy']:.3f} "
            f"stale(updated)={r['stale_error_rate_updated']:.3f} "
            f"missing={r['missing_rate']:.3f} "
            f"P@{K}={r['retrieval_precision_at_k']:.3f} "
            f"R@{K}={r['retrieval_recall_at_k']:.3f} "
            f"ctx_cov={r['context_coverage']:.3f}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
