# memweave: decay + explicit contradiction policy for episodic agent memory, measured on stale-fact error

**Author:** Habib Ur Rehman · **Code:** `src/memweave/` · **Benchmark:** `benchmarks/run_benchmark.py` · **Raw results:** `results/results.json`

## 1. Claim

Long-term memory for conversational agents fails in a specific, measurable
way that standard recall benchmarks hide: when a user *updates* a fact
("I moved to Porto" after "I live in Lisbon"), a vector store keeps both
versions, their texts are near-identical, and retrieval returns the stale
version roughly as often as the current one. We call an answer that uses
the superseded value a **stale-fact error**. This work (a) quantifies the
stale-fact error rate of three memory designs on planted-contradiction
streams, and (b) shows that time-decay relevance scoring combined with an
explicit contradiction policy drives it to zero while improving overall
accuracy and halving assembled-context size.

## 2. Method

**Facts.** Structured `(user, slot, value)` triples over 10 single-valued
slots (city, occupation, employer, diet, hobby, pet, language, timezone,
instrument, sport), rendered as text (`"maria's city is Lisbon"`) and
embedded with all-MiniLM-L6-v2. Because slots are single-valued, a second
value for the same `(user, slot)` is a contradiction *by construction* —
detectable without an LLM judge.

**Streams.** Deterministic generator (seed 20261026): 16 users × 8
sessions. Per user: 10 planted facts (sessions 0–1), 11 distractor
statements on look-alike slots (`visited_city`, `dream_job`,
`considering_diet`; sessions 2, 3, 6, 7), 2 verbatim reaffirmations of
stable facts (sessions 2, 6), and 4 contradictions — new values for
planted slots (sessions 4–5). After the stream, one probe per
`(user, slot)` (160 probes; 64 on updated slots), answered by a mechanical
reader (value of the highest-ranked retrieved fact matching the slot), so
the benchmark measures the memory, not a language model.

**Systems.** (a) *Buffer-only*: last 20 facts, recency-ranked.
(b) *Vector-no-decay*: full store, pure cosine; nothing is ever resolved.
(c) *memweave*: full store, score
`cosine × 0.5^(age_sessions / half_life) × (1 + 0.15·ln(1 + recalls))`,
restated facts refresh their clock instead of duplicating; contradictions
resolved by policy — `supersede` (stale fact excluded from retrieval,
kept for audit), `keep_both` (decay only), `flag` (both current, marked
conflicted). Context assembly keeps the best fact per slot under a
1,200-char budget.

**Metrics.** Answer accuracy; stale-fact error rate on updated probes;
updated/stable accuracy split; missing rate; retrieval P/R@8 against the
current fact; context coverage and characters at that coverage.

## 3. Results

| System | Acc | Stale (upd) | Upd acc | Stable acc | Missing | P@8 | R@8 | Ctx chars |
|---|---|---|---|---|---|---|---|---|
| buffer_only | 0.300 | 0.000 | 0.500 | 0.167 | 0.700 | 0.038 | 0.300 | 683 (cov .71) |
| vector_no_decay | 0.788 | **0.531** | 0.469 | 1.000 | 0.000 | 0.125 | 1.000 | 878 |
| memweave supersede hl2 | 0.569 | 0.000 | 1.000 | 0.281 | 0.431 | 0.071 | 0.569 | 389 |
| memweave supersede hl4 | 0.762 | 0.000 | 1.000 | 0.604 | 0.238 | 0.095 | 0.762 | 389 |
| memweave keep_both hl4 | 0.625 | 0.000 | 0.984 | 0.385 | 0.375 | 0.078 | 0.625 | 389 |
| memweave supersede hl8 | **0.944** | **0.000** | 1.000 | 0.906 | 0.056 | 0.118 | 0.944 | 389 |

Reading of the table:

- **The vector baseline's failure is staleness, not forgetting.** It never
  fails to retrieve *a* version (missing 0.000, R@8 1.000) yet answers
  34/64 updated probes with the superseded value (stale rate 0.531;
  updated accuracy 0.469 — worse than a coin flip would be if the two
  versions tied exactly, because embedding similarity is not neutral
  between versions).
- **Buffer-only never answers stale and rarely answers at all.** Recency
  ordering means a surviving update wins, but 70% of probes find their
  fact evicted (stable accuracy 0.167). A short-term buffer is not a
  memory.
- **Decay alone (keep_both) already eliminates stale answers** (0.000):
  the newer statement outranks its predecessor on the decay term even
  though both remain retrievable. Its cost is recall crowding — recent
  distractors push stable facts below rank 8 (missing 0.375).
- **Supersede + decay (hl8) is the best overall system**: accuracy 0.944
  vs 0.788 for the vector baseline, stale rate 0.000, updated accuracy
  1.000, and — because superseded versions and per-slot duplicates never
  enter context assembly — the same 100% context coverage in 389 chars
  vs 878.
- **Half-life sensitivity is large and monotone here** (hl2 0.569 <
  hl4 0.762 < hl8 0.944): with only 8 sessions, aggressive decay mostly
  destroys stable-fact recall. The default in code (hl4) was fixed before
  the sweep was run; we report all three values rather than retuning
  silently.

## 4. Relation to prior art

Decay and contradiction handling each exist separately. MemoryBank
(AAAI-24) applies an Ebbinghaus forgetting curve to LLM companion memory
but has no contradiction policy and no stale-fact benchmark. Graphiti/Zep
invalidates contradicted facts in a bi-temporal graph but uses no decay
scoring, requires a graph database and an LLM in the loop, and publishes
no comparable stale-fact ablation. Mem0 offers supersede/merge and
decay reranking as platform features without an open measured comparison
of this form. The STALE benchmark (2026) measures whether *LLM agents*
notice implicit conflicts in long contexts — a model benchmark, not a
memory-store design with decay scoring. Hobby implementations
(YourMemory, VoltMem) explore Ebbinghaus decay or volatility classes with
small case studies. To our knowledge no prior open project combines
decay-scored retrieval, an explicit named contradiction policy, and a
planted-contradiction benchmark reporting stale-fact error rate across
buffer / no-decay / full ablations. That combination is memweave's
contribution; the components are deliberately simple.

## 5. Honest limits

- **Synthetic, structured facts.** Contradictions are explicit value
  changes on typed slots. Real conversation requires fact *extraction*
  (an LLM step, with its own errors) and contains *implicit* conflicts
  ("I started a new job" vs a stored employer) that slot equality cannot
  detect.
- **Small scale.** 16 users, 27 statements each, one seed reported.
  Numbers are exact for this stream and reproducible, but confidence
  intervals across seeds/stream shapes are future work.
- **Mechanical reader.** No LLM answers the probes; a real agent's
  reader could sometimes recover from a stale retrieval (or be misled
  despite a correct one). The benchmark isolates the memory layer by
  design.
- **Half-life is stream-dependent.** The right half-life scales with
  session cadence and stream length; hl8 winning on an 8-session stream
  does not transfer numerically to month-long deployments.
- **Precision@8 is low for every system** (≤0.125) because exactly one
  fact is relevant per probe and 8 are retrieved; it is reported for
  completeness, not as a headline.

## 6. Reproduce

```bash
pip install -e .
python benchmarks/run_benchmark.py   # ~4 min on 2 CPU cores
python -m pytest tests/              # 15 tests, incl. cross-session sync
```

Deterministic: fixed seed, fixed probe order, CPU inference.
