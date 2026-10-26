# memweave

**Episodic agent memory that knows when its facts expire.** A memory-enabled
conversational agent whose long-term store does two things a plain vector
store does not: it *decays* facts with time and *resolves contradictions by
an explicit policy* — and it ships with a benchmark that measures the
failure mode this prevents: answering with a fact that was later
contradicted.

## The measured problem

On a deterministic stream of 16 users × 8 sessions (10 facts per user, 4 of
them later contradicted, plus near-miss distractor statements), a standard
vector memory with no decay and no contradiction handling answered **53.1%
of updated-fact probes with the superseded, stale value** — the old and new
facts have nearly identical text, so cosine ranking between the two
versions is close to a coin flip — and the stale fact keeps winning it.

## Results (seed 20261026, 160 probes, 64 on updated slots)

| System | Accuracy | Stale-fact error (updated probes) | Updated acc. | Stable acc. | Missing | R@8 | Context chars @ full coverage |
|---|---|---|---|---|---|---|---|
| Buffer only (last 20) | 0.300 | 0.000 | 0.500 | 0.167 | 0.700 | 0.300 | 683 (cov 0.71) |
| Vector, no decay | 0.788 | **0.531** | 0.469 | 1.000 | 0.000 | 1.000 | 878 |
| memweave supersede, half-life 2 | 0.569 | 0.000 | 1.000 | 0.281 | 0.431 | 0.569 | 389 |
| memweave supersede, half-life 4 | 0.762 | 0.000 | 1.000 | 0.604 | 0.238 | 0.762 | 389 |
| memweave decay only (keep-both, hl 4) | 0.625 | 0.000 | 0.984 | 0.385 | 0.375 | 0.625 | 389 |
| **memweave supersede, half-life 8** | **0.944** | **0.000** | **1.000** | 0.906 | 0.056 | 0.944 | 389 |

Full numbers: [`results/results.json`](results/results.json). Method and
honest limits: [`PAPER.md`](PAPER.md).

Three findings:

1. **Stale answers are the dominant failure of plain vector memory** —
   not forgetting. It never missed (missing rate 0.000); it confidently
   retrieved the wrong *version*.
2. **Decay alone fixes currency, supersede makes it structural.**
   With keep-both + decay, the newer value already outranks the stale one
   (0 stale answers). The supersede policy additionally removes the stale
   fact from retrieval entirely, keeps an audit trail of every
   contradiction, and compresses assembled context to 389 chars at the
   same 100% coverage the vector store needs 878 chars for.
3. **Half-life is a real knob, not a detail.** Too short (2 sessions) and
   stable facts decay out of retrieval (stable accuracy 0.281); the sweep
   is reported in full rather than cherry-picked.

## Design

- **Short-term buffer + long-term store.** The agent harness carries a
  session buffer; durable facts live in the episodic store.
- **Facts are structured** `(user, slot, value)` triples rendered as text
  and embedded with all-MiniLM-L6-v2. Slots are single-valued, which makes
  contradictions *detectable* instead of a matter of interpretation.
- **Decay scoring:** `score = cosine × 0.5^(age / half_life) × (1 + 0.15·ln(1 + recalls))`.
  Restating a fact reaffirms it (refreshes its clock, adds reinforcement);
  unused facts fade.
- **Contradiction policies** (chosen at store creation): `supersede`
  (old fact marked superseded, kept for audit, excluded from retrieval),
  `keep_both` (both current, ranked by decay), `flag` (both current and
  marked conflicted so callers can surface the conflict). Every
  contradiction is logged as an event either way.
- **Context compression:** assembled context keeps the best fact per slot
  under a character budget — superseded versions never reach the prompt.
- **Cross-session sync:** the store serializes to JSON; a new session
  loads it and continues. A test verifies a save/load split mid-stream
  produces identical answers to an uninterrupted run.

## Quick start

```bash
pip install -e .
python benchmarks/run_benchmark.py   # writes results/results.json
python -m pytest tests/              # 15 tests
```

Embeddings use a local all-MiniLM-L6-v2 model (set `MEMWEAVE_MODEL_PATH`
if it lives elsewhere). CPU only, no LLM, no API keys, deterministic seed.

## Layout

- `src/memweave/store.py` — episodic store, decay scoring, contradiction policies, persistence
- `src/memweave/baselines.py` — buffer-only and vector-no-decay memories
- `src/memweave/benchgen.py` — deterministic planted-fact/contradiction/distractor streams
- `src/memweave/agent.py` — the conversational-agent harness used for all systems
- `benchmarks/run_benchmark.py` — the measured comparison above

## License

MIT © Habib Ur Rehman
