# Iteration Story

The eval framework produced a baseline (recall@5 coverage
0.60, faithfulness 1.00). This file documents the iteration experiments
run against that baseline — each one changes a single config knob,
measures the result, and explains what held and what surprised.

This file documents each iteration in sequence. Each iteration includes the
hypothesis (written before the experiment), what was changed, the result
versus baseline, and an honest reading of what held and what surprised.

---

## Iteration 1 — Chunk size: 800 → 1200

### Hypothesis

The baseline (recall@5 coverage 0.60, faithfulness 1.00) showed
retrieval was the bottleneck, not generation. Within retrieval, the
synthesis category had the largest any-hit/coverage gap (0.67 vs 0.54),
meaning retrieval often found *one* of the chunks needed to answer a
multi-sentence anchor but not all of them.

The matcher's per-sentence inspection had already shown why: of 30
fact-based anchors, 17 needed 2+ chunks because their sentences spanned
chunk boundaries. The hypothesis: increasing chunk size from 800 to 1200
characters would consolidate many of those multi-sentence anchors into
single chunks, raising synthesis coverage. Cross-doc was expected to be
roughly unchanged (chunk size doesn't help paper diversity). Faithfulness
was expected to hold (Haiku's grounded-reading behavior shouldn't depend
on chunk size at this scale).

### What changed

`build_store.py`: `target_size=800` → `target_size=1200`. No other code
changes. Re-ran the full pipeline: rebuild ChromaDB → re-run matcher →
re-run recall@k → re-run faithfulness.

### Results

**Chunk-level structural change.** Total chunks dropped from 219 to 141
(-36%), consistent across all three papers. The matcher confirmed 7 of
the 17 multi-chunk anchors consolidated into single chunks; 4 remained
multi-chunk; 1 partially consolidated (Q15 a2, from 3 chunks to 2); and
2 single-chunk anchors got *worse* (Q12 a2 and Q19 — chunk boundaries
happened to land mid-content). Net: structural improvement, but not
uniform.

**Recall@k at k=5:**

| Category | Baseline any@5 | Iter 1 any@5 | Baseline cov@5 | Iter 1 cov@5 |
|---|---|---|---|---|
| Single-fact (n=6) | 0.83 | 1.00 (+0.17) | 0.83 | 1.00 (+0.17) |
| Synthesis (n=12) | 0.67 | 0.83 (+0.16) | 0.54 | **0.79 (+0.25)** |
| Cross-doc (n=8) | 0.38 | 0.50 (+0.12) | 0.31 | 0.50 (+0.19) |
| Numerical (n=4) | 1.00 | 1.00 (=) | 1.00 | 0.88 (-0.12) |
| **Overall** | **0.67** | **0.80 (+0.13)** | **0.60** | **0.77 (+0.17)** |

**Faithfulness:** 1.00 → 0.98. The single ◐ came from a framing distinction
on Q17 (whether the reported metrics applied to the soprano pipistrelle
dataset specifically or to the bat detection task generally). The numerical
claims (F1=0.961, precision=0.994, recall=0.931) were directly traceable
to the passage. This is at the boundary of what LLM-as-judge can reliably
distinguish and is consistent with the noise floor for n=24 with same-model
judge (see NORMALIZER_NOTES.md). Practically: faithfulness held flat.

### Reading the result

**What worked as designed.** Synthesis coverage was the targeted improvement
and moved from 0.54 to 0.79 — directly inside the predicted range of
0.70–0.85. The any-hit/coverage gap in synthesis collapsed from 0.13 to
0.04, confirming the consolidation mechanism: when multi-sentence anchors
become single-chunk anchors, any-hit and coverage converge because there's
only one chunk to find.

**Wrong in the favorable direction (1): single-fact reached 1.00.** The
baseline miss was Q5 ("Raspberry Pi power-hungry"). The bigger chunk gave
retrieval more context around the named entity. Not a strong prediction
beforehand, but it cleared the prior known failure.

**Wrong in the favorable direction (2): cross-doc improved more than
expected.** Any@5 rose from 0.38 to 0.50; cov@5 from 0.31 to 0.50. The
original reasoning ("chunk size doesn't help paper diversity") was
incomplete. Fewer, more semantically rich chunks produced more distinctive
embeddings, which improved retrieval ranking at k=5 even for cross-paper
questions. The "top-k covers less ground" risk I had flagged didn't bite
at k=5; it might bite at smaller k.

**Honest regression: numerical coverage 1.00 → 0.88.** Q19's chunk boundary
fell mid-content under the new chunking. The answer is still findable at
k=10 (cov@10 = 1.00), but at k=5 only 1 of 2 needed chunks is retrieved.
The eval correctly registered this — bigger chunks helped most categories
but introduced new boundary problems for one specific question. Not hidden,
not minimized.

**The Day 3 planted failure (Q12 a2) resolved.** The 0.84 TPR / 500m
result that had been invisible to retrieval at k=5 in the baseline now
returned full coverage (2/2) at k=5. The same content was split across
two chunks under the new chunking, but retrieval surfaced both. The
documented limitation from Day 3 was confirmed by the eval framework on
Day 4 and resolved by Day 6's first iteration.

### What this iteration didn't fix

Three cross-doc anchors stayed at 0/2 at every k: Q14 a2, Q15 a2 (still
partially split: 2 chunks remaining), and Q16 a1. The matcher tells us
why: their sentences span chunk boundaries that even 1200 chars can't
bridge. These are not chunk-size problems. A different intervention is
needed — overlap, reranking, or hybrid retrieval. That diagnosis sets up
iteration 2.

### Decision

Keep `target_size=1200` as the production configuration. The experiment
showed it outperforms 800 on every meaningful metric except a small,
traceable numerical regression. Reverting would discard a clear net
improvement.

---

## Iteration 2 — Chunk overlap: 0 → 200 chars (chunk size held at 1200)

### Motivation from iteration 1

Iteration 1 raised overall recall@5 coverage from 0.60 to 0.77 and resolved
the Day 3 planted failure (Q12 a2). It also revealed three specific anchors
that did *not* improve and remained at 0/2 coverage at every k:

- **Q14 a2** (cross-doc, depthwise separable convolution on low-cost hardware)
- **Q15 a2** (cross-doc, HMM in non-biological detection — anchor sentences
  spread across chunks 32 and 35 in the 1200 chunking)
- **Q16 a1** (cross-doc, transition from earlier ML to deep learning)

The matcher diagnosis was clear: each of these anchors has sentences that
span chunk boundaries even at 1200 chars. The content the question needs is
present in the corpus, but no single chunk contains enough of it for
embedding-based retrieval to score the right chunks at the top of the
results list. Two near-boundary sentences each end up on the "wrong side"
of a chunk break, and the chunk that *would* match (if it existed) doesn't
exist as a single unit.

### Hypothesis

Adding 200 characters of overlap between adjacent chunks should make the
boundary region appear in *two* chunks rather than one. An anchor whose
sentences previously straddled a boundary now has at least partial
representation in either of the two neighboring chunks. Predicted effect:
cross-doc coverage rises from 0.50 to 0.60–0.75; the named cross-doc
failures (Q14 a2, Q15 a2) move from 0/2 to at least 1/2; Q16 a1 stays flat
(diagnosed as a ranking problem, not a boundary problem). Numerical
recovers (Q19's iter-1 regression should heal). Faithfulness holds at
0.95–1.00.

### What changes

Two files. `chunking.py` got a new `overlap` parameter (default 0,
preserving original behavior). The overlap implementation respects the
whole-sentence-only invariant: when starting a new chunk, the function
walks the previous chunk's sentences from the end backward, accumulating
complete sentences until the overlap budget is hit. An edge case handles
the situation where the overlap seed plus the triggering sentence would
exceed `target_size` — in that case the overlap is dropped for that
specific transition, rather than splitting a sentence. `build_store.py`
then calls the chunker with `overlap=200`.

### Structural change before retrieval ran

Total chunks rose from 141 to 154 (+9%). The increase was uneven across
papers:

| Paper | iter 1 chunks | iter 2 chunks | Change |
|---|---|---|---|
| neuroethology_review | 71 | 79 | +11% |
| opensoundscape_methods | 19 | 19 | 0% |
| audiomoth_deployment | 51 | 56 | +10% |

The opensoundscape paper held steady at 19 chunks. Initial interpretation
(that overlap kept falling back to "no overlap" via the edge case) turned
out to be wrong on inspection — chunk-size distribution showed iter 2's
opensoundscape chunks averaged 1117 chars (up from iter 1's lower average),
meaning overlap *was* applied; it filled previously-unused headroom inside
each chunk rather than forcing new chunks. The mechanism differs across
papers depending on each paper's sentence-length distribution, but overlap
was applied uniformly as a parameter.

The matcher then ran on the new chunks and confirmed structural
consolidation in some cases and *fragmentation* in others:

- **Q14 a2 consolidated** from 2 chunks to 1 (chunk_49 — overlap merged
  the boundary content; the targeted fix worked structurally)
- **Q11 a1 consolidated** from 2 chunks to 1
- **Q15 a1 fragmented** from 1 chunk to 2 (chunks 37, 38)
- **Q15 a2 fragmented** from 2 chunks to 4 (chunks 35, 36, 38, 39)
- **Q11 a2 fragmented** from 1 chunk to 2 (chunks 16, 17)

The fragmentation happened because anchor sentences near a boundary now
appear in *multiple* chunks due to overlap duplication. The matcher
correctly lists all chunks containing any anchor sentence as ground truth,
which inflates the denominator of coverage (a single anchor sentence
now requires retrieval to find 1 of 2 or 1 of 4 chunks). This was a
measurement issue I had not fully anticipated.

### Recall@k results

**Three-way comparison at k=5:**

| Category | Baseline (800) | Iter 1 (1200) | Iter 2 (1200+200) | Iter 2 vs Iter 1 |
|---|---|---|---|---|
| Single-fact any@5 | 0.83 | 1.00 | 0.83 | -0.17 |
| Single-fact cov@5 | 0.83 | 1.00 | 0.83 | -0.17 |
| Synthesis any@5 | 0.67 | 0.83 | 0.67 | -0.16 |
| Synthesis cov@5 | 0.54 | 0.79 | 0.67 | -0.12 |
| Cross-doc any@5 | 0.38 | 0.50 | 0.38 | -0.12 |
| Cross-doc cov@5 | 0.31 | 0.50 | 0.31 | -0.19 |
| Numerical any@5 | 1.00 | 1.00 | 1.00 | 0 |
| Numerical cov@5 | 1.00 | 0.88 | **1.00** | **+0.12** |
| **Overall any@5** | **0.67** | **0.80** | **0.67** | **-0.13** |
| **Overall cov@5** | **0.60** | **0.77** | **0.65** | **-0.12** |

A striking pattern: **iter 2's any@5 numbers are identical to baseline
across every category**. Not noise — every single any@5 number matches
baseline exactly. Iter 2 didn't fall below baseline; it landed precisely
back on baseline ranking performance, with slight residual coverage gains
from leftover consolidation.

### The mechanism — two distinct levers

Chunking changes affect retrieval through two separate mechanisms:

1. **Structural lever (which sentences live in which chunk).** Bigger
   chunks consolidate multi-sentence anchors into single chunks. Overlap
   extends this for boundary cases. Coverage at the consolidation level
   reflects this.

2. **Embedding-quality lever (how distinctive each chunk's embedding is).**
   Bigger chunks produce more distinctive embeddings — more context per
   chunk, more semantic richness, easier for the retriever to discriminate
   between chunks. Overlap *hurts* this lever — when chunks share content
   with their neighbors, their embeddings become less distinctive, and
   retrieval ranking gets less precise.

Iter 1 won on both levers, which is why it improved coverage *and* any-hit.
Iter 2 kept the structural gains (synthesis coverage stayed at 0.67,
above baseline's 0.54) but lost the embedding-quality gains (any-hit
returned to baseline across the board). The two levers don't have to move
together, and in iter 2's case they moved in opposite directions.

### Targeted failures — what actually happened

- **Q14 a2** (primary test of the overlap hypothesis): consolidated to a
  single chunk_49, but retrieval still scored 0/1 at every k. The
  *structural* problem was fixed; the *ranking* problem got worse. Embedding
  dilution made chunk_49 less findable than chunk_44+chunk_45 had been.
  **Prediction failed.**
- **Q15 a2**: predicted same or worse, got worse. 0/4 — anchor fragmented
  across more chunks, all still unfindable.
- **Q16 a1**: predicted unchanged, actually unchanged at k=10 (still 1/1
  there) but *dropped* at k=3 and k=5 due to embedding dilution. Prediction
  partially held.
- **Q19** (the iter-1 regression): recovered to 1/1, as predicted.
  Numerical coverage healed.
- **Q14 a1** (collateral damage I had not predicted): was 1/1 at k=10 in
  iter 1, now 0/1 at every k. A previously-findable cross-doc anchor
  became unfindable.
- **Q11 a1** (collateral damage): was 1/1 at k=3 in iter 1, now 0/1 at
  k=3 and k=5.

### Faithfulness results

| Run | Score | Composition |
|---|---|---|
| Baseline (800) | 1.00 | 24 ✓, 0 ◐ |
| Iter 1 (1200) | 0.98 | 23 ✓, 1 ◐ (Q17 — judge variance) |
| Iter 2 (1200+200) | **0.96** | 22 ✓, 2 ◐ (Q17 again, plus Q15 — real model error) |

**Q17** reproduced the same judge-variance pattern from iter 1 — same
nitpick about whether reported metrics are explicitly attributed to the
soprano pipistrelle dataset. Same answer pattern, same kind of judgment.
The fact that it appeared in both iter 1 and iter 2 strengthens the
judge-variance reading rather than indicating a real regression.

**Q15 was different.** The judge flagged a real generation error: the
model conflated MoSeq (a depth-camera behavioral analysis system) with
HMM-based vocal analysis tools, folding the source's "syllables, much
like birdsong" analogy into a claim that suggested MoSeq itself analyzes
vocalizations. The passages distinguish behavioral data from vocal data
explicitly; the answer blurred the distinction. The judge caught it
accurately.

Whether overlap *caused* this error isn't provable without re-running
under different configs, but it's plausible: overlap blends neighboring
content into each chunk, which makes adjacent topics less semantically
distinguishable. When the boundary between "MoSeq is behavioral analysis"
and "the analogy to birdsong" gets fuzzier in the retrieved chunks, the
model is more likely to merge them. Suggestive evidence, not proof.

The eval framework detected the issue regardless of the mechanism. This
is a real demonstration of the framework's value beyond retrieval — a
generation error caused (or potentially caused) by a chunking change,
surfaced by the judge with a specific, accurate rationale.

### Reading the result

**What worked.** Q19 recovered. Synthesis coverage retained partial gains
from iter 1's consolidation work (0.67 vs baseline's 0.54). The structural
effect of overlap was visible in the matcher's output.

**What didn't work.** The targeted cross-doc failures (Q14 a2 especially)
were not recovered. Embedding dilution from overlap pushed previously-findable
chunks out of the top-k. Any-hit performance returned to baseline across
every category. The cost-benefit was negative.

**What surprised me.** I had predicted "near-duplicate chunks waste top-k
slots" as a risk, but that turned out to be the *minor* effect. The larger
effect was embedding-quality degradation — overlap doesn't just create
slot redundancy, it actively makes each chunk's embedding less
discriminative. This was the lesson worth learning: ranking quality
depends on embedding distinctiveness in a way that even small content
duplication can disrupt.

The faithfulness finding (Q15) was a bonus surprise — the eval framework
catching a generation issue I would have missed if I had only looked at
retrieval metrics. Two distinct failure modes from one chunking change,
both caught.

### Decision

Revert to iter 1's configuration (`target_size=1200, overlap=0`) as
production. The two-iteration story is honest: iter 1 worked, iter 2
revealed why naive overlap doesn't help with this corpus and retrieval
strategy. The residual cross-doc failures (Q14 a1, Q14 a2, Q15 a2, Q16 a1)
are not chunking problems. They are ranking problems — the right chunks
exist in the corpus and at k=10 are usually findable, but at k=5 they're
crowded out. The natural next intervention would be a different retrieval
mechanism: hybrid keyword+semantic retrieval, or a reranker that scores
each candidate chunk against the question before final ordering. Both
target the ranking lever directly, which is where the iter-2 data points.

That intervention is left as future work for this project. The decision
to revert rather than push further chunking changes is itself a finding:
when the bottleneck moves from one lever to another, the right response
is to switch tools, not to keep tuning the same dial.

---

## Iteration 3 — Hybrid retrieval: BM25 + semantic with reciprocal rank fusion

### Motivation from iteration 2

Iteration 2 (chunk overlap) showed that not all retrieval failures are
chunking problems. Some chunks are structurally available but rank too
low — semantic retrieval finds them in candidate pools at k=10 but not
at k=5. Iter 2 identified Q16 a1 specifically as a ranking-bound failure:
the anchor's ground-truth chunk was a single chunk (no boundary issue),
yet retrieval consistently failed to rank it in the top-5. The iter 2
writeup ended with an explicit hypothesis: "the residual cross-doc
failures are ranking problems, not chunking problems," and identified
hybrid retrieval and reranking as candidate interventions.

This iteration tests that hypothesis. If hybrid retrieval (lexical +
semantic) recovers Q16 a1 cleanly, the iter 2 diagnosis was correct.
If it doesn't, the diagnosis was incomplete.

### Hypothesis

Semantic retrieval (MiniLM dense embeddings) handles paraphrase well
but can miss documents where the question's distinctive vocabulary
(proper nouns, rare technical terms) is the strongest signal. BM25
(lexical, term-frequency based with rare-word weighting) handles that
case directly. The two retrievers have complementary failure modes —
each catches what the other misses. Combining them via reciprocal rank
fusion (RRF) should improve cross-doc questions in particular, where
specific technical vocabulary in the question often appears in the
target chunk but the chunk's overall embedding doesn't dominate the
ranking.

**Predicted effect:** cross-doc coverage 0.50 → 0.62-0.88. Specific
named recoveries: Q14 a1, Q14 a2 (contain "depthwise separable
convolution" — distinctive technical term), Q15 a2 (contains "HMM,"
"Viterbi"), and Q16 a1 (the primary ranking-hypothesis test). Synthesis
and single-fact already near ceiling; modest gains. Faithfulness
expected to hold flat — different chunks retrieved but same generation
prompt and strict refusal behavior.

### What changed

Three engineering changes, no chunking change:

1. **`build_store.py`** refactored. The pipeline code moved into a
   `build()` function with an `if __name__ == "__main__":` guard,
   eliminating side-effect-on-import behavior. This was a structural
   fix discovered during iter 3 implementation: the first version of
   `retrieve_hybrid.py` imported `tokenize` from `build_store.py`, which
   triggered a full chroma rebuild as a side effect of the import.
   The refactor isolates the shared tokenizer (importable) from the
   pipeline (only runs when `build_store.py` is executed directly).

2. **BM25 index built alongside ChromaDB.** A new `tokenize()` function
   in `build_store.py` (simple whitespace+lowercase tokenization on
   non-word characters) feeds into `BM25Okapi` from the `rank_bm25`
   library. The index plus the chunk IDs (so internal BM25 positions
   can be mapped back to chunk IDs) is pickled to `bm25_index.pkl` at
   project root. Build is part of the standard `build_store.py` run,
   so the two indexes always reflect the same chunks.

3. **`retrieve_hybrid.py`** as a drop-in replacement for `retrieve.py`.
   Same public function signature: `retrieve(question, k)`. Internally
   it runs both retrievers independently, fetches K_FETCH=20 candidates
   from each, and fuses with RRF (constant k=60). Returns results in
   ChromaDB's native batch-nested format so downstream code
   (`generate.py`, `recall_at_k.py`, `faithfulness.py`) needed only a
   one-line import change to use the new retriever.

Chunks unchanged: same 141 chunks from iter 1's config
(`target_size=1200, overlap=0`). Ground truth `ground_truth.json`
unchanged. The only thing different between iter 1 and iter 3 is the
retrieval strategy. This is a clean A/B comparison.

### Recall@k results

**Three-way comparison at k=5:**

| Category | Baseline (800) | Iter 1 (1200) | Iter 3 (1200+hybrid) | Iter 3 vs Iter 1 |
|---|---|---|---|---|
| Single-fact any@5 | 0.83 | 1.00 | 1.00 | = |
| Single-fact cov@5 | 0.83 | 1.00 | 1.00 | = |
| Synthesis any@5 | 0.67 | 0.83 | **1.00** | **+0.17** |
| Synthesis cov@5 | 0.54 | 0.79 | **0.96** | **+0.17** |
| Cross-doc any@5 | 0.38 | 0.50 | **0.62** | **+0.12** |
| Cross-doc cov@5 | 0.31 | 0.50 | **0.62** | **+0.12** |
| Numerical any@5 | 1.00 | 1.00 | 1.00 | = |
| Numerical cov@5 | 1.00 | 0.88 | **1.00** | **+0.12** |
| **Overall any@5** | **0.67** | **0.80** | **0.90** | **+0.10** |
| **Overall cov@5** | **0.60** | **0.77** | **0.88** | **+0.11** |

Every category that had headroom moved up. Synthesis is the largest
absolute jump (cov@5 0.79 → 0.96 — nearly ceiling). Cross-doc moved
solidly (0.50 → 0.62) but landed at the lower end of the predicted
range. Numerical recovered the iter 1 regression on Q19. Overall
coverage now 0.88, up from 0.60 baseline.

### Targeted failures — what actually happened

This is where the iter 2 diagnosis gets tested. Going through each
prediction:

- **Q16 a1 (primary ranking-hypothesis test): RECOVERED.** Was 0/1 at
  k=3 and k=5 in iter 1 (1/1 only at k=10). Now 1/1 at every k.
  Hybrid retrieval put chunk_19 at rank 3. The iter 2 diagnosis
  for this case was correct.
- **Q14 a1: NOT RECOVERED.** Still 0/1 at every k. Hybrid retrieves
  10 CNN-related chunks but ground-truth chunk_7 (opensoundscape) is
  not among them.
- **Q14 a2: NOT RECOVERED.** Still 0/2 at every k. The
  depthwise-separable-convolution chunks (chunks 44, 45 in audiomoth)
  are not in hybrid's top-10.
- **Q15 a2: NOT RECOVERED.** Still 0/2 at every k. The HMM-for-gunshots
  chunks remain unreachable.

### Mechanism analysis

The iter 2 framing identified two retrieval levers: structural (which
sentences live in which chunk) and embedding quality (how distinctive
each chunk's embedding is). Iter 3 introduces a third lever: **lexical
matching** — whether retrieval can latch onto specific rare vocabulary
in the question.

Q16 a1 was a lexical-friendly case: the question contains the terms
"HMMs," "template-matching," "transition," and "deep learning," all of
which appear in chunk_19. BM25 boosted chunk_19 based on the rare-term
overlap; semantic also found it (in candidate pool at rank ~5);
RRF's consensus put it at top-3. Lexical signal closed the ranking gap.

Q14 a1, Q14 a2, and Q15 a2 reveal a **third failure mode** not predicted
by the iter 2 analysis: **query-document vocabulary mismatch**. The
questions ask about concepts at one level of abstraction
("practical constraints when deploying CNNs on low-cost hardware,"
"detecting non-biological acoustic events") while the answer chunks use
specific technical vocabulary at a lower level ("depthwise separable
convolution," "gunshots"). Neither BM25 nor semantic bridges this gap
reliably:

- BM25 can't help because the question doesn't contain the answer's
  vocabulary — there's no keyword overlap to weight.
- Semantic should theoretically bridge this via conceptual similarity,
  but MiniLM's embeddings represent each chunk as a single 384-dim
  vector. A chunk whose dominant content is "depthwise separable
  convolution math" doesn't embed as semantically close to "practical
  hardware constraints" — the abstraction gap is too wide.

So the iter 3 result decomposes the residual cross-doc failures into
two sub-types:
- **Lexical-recoverable (Q16 a1):** hybrid fixes
- **Vocabulary-gap (Q14 a1, Q14 a2, Q15 a2):** hybrid doesn't fix

This is a more refined diagnosis than the iter 2 framing produced.

### Faithfulness results

| Run | Score | Composition |
|---|---|---|
| Iter 1 (1200) | 0.98 | 23 ✓, 1 ◐ (Q17 — judge variance) |
| Iter 2 (1200+overlap) | 0.96 | 22 ✓, 2 ◐ (Q17 + Q15 model error) |
| Iter 3 (1200+hybrid) | 0.98 | 23 ✓, 1 ◐ (Q17 — same judge variance) |

Q17 reproduced for the third time in a row — same model, same answer,
same judge nitpick. The pattern is so consistent across configurations
that it's now best understood as Q17's specific claim being at the
boundary of LLM-as-judge reliability, regardless of retrieved context.
The Q15 ◐ from iter 2 did *not* reappear in iter 3 — suggesting iter 2's
Q15 model error was specifically caused by the overlap chunks blending
neighboring content. Reverting to iter 1's chunks (the foundation of
iter 3) restored faithful generation on Q15. Indirect confirmation of
the iter 2 mechanism.

Faithfulness is effectively 1.00 across iter 1 and iter 3. Hybrid
retrieval didn't degrade generation.

### Reading the result

**What worked.** The primary ranking-hypothesis test (Q16 a1) passed
cleanly — chunk_19 went from unreachable at k=5 to rank 3. Synthesis
coverage reached 0.96, essentially at ceiling. Numerical recovered
the iter 1 regression (Q19 1/2 → 2/2). Overall coverage 0.60 → 0.88
across the three iterations is a meaningful end-to-end improvement.

**What didn't work as predicted.** Three of the four named cross-doc
predictions failed (Q14 a1, Q14 a2, Q15 a2). I had predicted these as
ranking-bound, expecting hybrid to recover them. Inspection of what
hybrid actually retrieved showed they're not ranking failures at all —
they're query-document vocabulary-gap failures. The right chunks
aren't in the top-10 even at k=10. Hybrid retrieval can't recover them
because neither lexical nor dense embedding similarity is meaningful
when the question and the answer use disjoint vocabulary at different
levels of abstraction.

**What surprised me.** The iter 2 framing of "the residual failures
are ranking problems" was correct for ONE case (Q16 a1) and incorrect
for THREE cases (Q14 a1, Q14 a2, Q15 a2). The iter 3 data refined the
diagnosis: there are at least two distinct cross-doc failure types,
and they need different interventions. The right next move for the
vocabulary-gap failures would be a technique that bridges abstraction
levels — query expansion (asking the model to rewrite the question
with multiple phrasings) or HyDE (hypothetical document embeddings:
generate a hypothetical answer to the question, then retrieve chunks
similar to that hypothetical answer). Neither is a retrieval-strategy
improvement; both are query-side preprocessing.

### Decision

Adopt hybrid retrieval as the production configuration. Iter 3's
coverage gains over iter 1 (0.77 → 0.88) are meaningful and consistent
across categories. Faithfulness held flat. The three unrecovered
cross-doc failures aren't masked or hidden — they're correctly
identified as a separate failure mode (vocabulary-gap), and the
project documents what the next intervention would be (query
expansion or HyDE) without claiming this iteration solved them.

The chunking configuration stays at `target_size=1200, overlap=0` from
iter 1. The retrieval pipeline becomes hybrid. The eval framework
correctly characterized the system's performance across three
distinct interventions — chunking-structural (iter 1 helped),
chunking-overlap (iter 2 hurt), and hybrid retrieval (iter 3 helped,
but with refined diagnosis of the remaining failures).