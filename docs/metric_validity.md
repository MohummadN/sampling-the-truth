# Do the factuality numbers mean what they appear to mean?

Two threats to the support-rate metric, raised 14 September 2026, both resolved
from the 7,200 verdict records already on disk. **No new generation and no new
GPU pass was required.** Every claim below is a measurement, not an argument.

Companion to `DECISIONS.md`. Feeds Limitations and the Results discussion.

---

## Problem 1 — is the stratum effect a retrieval artifact?

### The concern

Support rate rises with entity prominence for both Llama models:

| model | stratum 0 | stratum 1 | stratum 2 |
|---|---|---|---|
| gpt2 | 0.219 | 0.229 | 0.226 |
| llama-1b | 0.199 | 0.264 | **0.426** |
| llama-3b | 0.195 | 0.248 | **0.376** |

(beam-4, macro-average of per-record support rate, 300 records per model.)

Three mechanisms could produce that rise, and only the third is a claim about
the model:

1. **Pool size.** The score is a max over 5 windows retrieved from the page.
   Stratum 2's pool is ~7x larger (mean 29 vs ~200 page sentences). A max over
   the best of 200 candidates beats a max over the best of 29 even if the
   generated text is identical. Pure evaluation artifact.
2. **Reference coverage.** Longer pages contain more facts, so a true claim is
   likelier to appear in the reference at all. The known FActScore limitation:
   it measures support against a reference, not truth.
3. **Model memorisation.** Prominent entities appear more in pretraining, so the
   model genuinely knows more. **This is the claim the paper wants to make.**

### Why the original control was not enough

GPT-2 is flat across strata while facing identical evidence pools, which argues
against (1). But GPT-2 has a floor effect — most of its sentences are so wrong
that no window would entail them at any pool size — so its flatness at a 0.5
threshold could hide a sub-threshold drift.

### Method

Measure the artifact directly instead of inferring it. Take only the sentences
the verifier marked **unsupported** (p_entail < 0.5) and ask whether their
scores drift upward with pool size. That is the verifier's noise floor. If a
larger pool produces better spurious matches, the floor rises.

Validated first on synthetic data with a known ground truth, to confirm the
diagnostic can tell the two cases apart:

| scenario | floor mean by stratum | 0.1-0.5 band |
|---|---|---|
| no artifact injected | 0.1490 / 0.1358 / 0.1448 | 56.6% / 49.6% / 50.0% |
| pool-size artifact injected | 0.2154 / 0.2157 / 0.2452 | 84.9% / 89.0% / 97.3% |

### Evidence

beam-4, non-degenerate records, all three seeds pooled:

| model | str | recs | macro rate | unsup n | floor mean | floor med | 0.1-0.5 | pool |
|---|---|---|---|---|---|---|---|---|
| gpt2 | 0 | 87 | 0.219 | 1026 | 0.0183 | 0.0061 | 2.6% | 29 |
| gpt2 | 1 | 90 | 0.229 | 990 | 0.0220 | 0.0088 | 4.2% | 61 |
| gpt2 | 2 | 81 | 0.226 | 1092 | **0.0402** | 0.0039 | **6.4%** | 197 |
| llama-1b | 0 | 102 | 0.199 | 984 | 0.0489 | 0.0113 | 11.9% | 29 |
| llama-1b | 1 | 96 | 0.264 | 816 | 0.0311 | 0.0068 | 7.0% | 60 |
| llama-1b | 2 | 96 | **0.426** | 705 | 0.0703 | 0.0135 | 10.7% | 226 |
| llama-3b | 0 | 102 | 0.195 | 879 | 0.0523 | 0.0092 | 15.9% | 29 |
| llama-3b | 1 | 99 | 0.248 | 681 | 0.0583 | 0.0062 | 13.1% | 60 |
| llama-3b | 2 | 99 | **0.376** | 687 | 0.0638 | 0.0169 | 14.9% | 222 |

### Findings

**The artifact is real and quantified.** GPT-2's noise floor more than doubles
across strata: 0.0183 -> 0.0402 mean, with the 0.1-0.5 band growing 2.6% ->
6.4%. A 7x larger pool buys about **+0.022 of entailment probability** on
unsupported sentences.

**It is an order of magnitude too small to explain the effect.** Llama-1b rises
**+0.227** across the same strata facing the same pools. GPT-2's macro rate
moves +0.007 — flat — because a floor sitting at 0.02-0.04 never approaches a
0.5 threshold. The median floor actually *falls* for GPT-2 (0.0061 -> 0.0039)
while the mean and band rise, so a bigger pool does not lift the whole
distribution; it adds occasional near-misses, and near-misses in 0.1-0.5 do not
cross 0.5.

**Mechanism (1) is therefore excluded as an explanation of the stratum effect.**

**GPT-2 is the only valid control here, for a reason worth stating.** Its
unsupported-sentence count is stable across strata (1026 / 990 / 1092), so the
floor comparison is like-for-like. Llama's shrinks (984 / 816 / 705) as
sentences cross the threshold, which changes the composition of what remains —
which is why Llama's floor is non-monotonic and cannot be read the same way.

**Mechanism (2) remains open, and this design cannot close it.** GPT-2's
flatness does not rule out coverage, because coverage only helps a model whose
claims are true. This is a genuine limitation of support-against-a-reference and
belongs in Limitations.

### Consequences

- **The equal-pool re-verification pass was cancelled.** It was one-directional:
  truncating every page to 39 sentences holds pool size constant but also
  removes content, suppressing (1) and (2) together. Survival would have
  excluded (1); shrinkage would have been uninterpretable. The floor measurement
  answers the same question with more precision and no GPU cost.
- **Mechanism (2) gets measured by the manual validation**, not by more compute.
  `DECISIONS.md` (2026-09-07) already specifies three labels: supported,
  unsupported, and *true but absent from the page*. Draw the 100 manual labels
  **stratified across the three strata**, and the "true but absent" rate per
  stratum is mechanism (2), measured directly.
- **Constraint on theta tuning.** The artifact lives precisely in the 0.1-0.5
  band, which grows 2.6% -> 6.4% with stratum. `DECISIONS.md` notes that moving
  theta from 0.15 to 0.95 reclassifies ~15% of sentences. **Keep theta at or
  above 0.5**, or report the stratum effect's sensitivity to it — tuning theta
  down imports the artifact into the headline result.

### A methodological note worth keeping

Macro and micro averaging disagree sharply here. Pooling all sentences gives
GPT-2 0.269 / 0.193 / 0.311, against 0.219 / 0.229 / 0.226 for the macro
average. Micro-averaging weights long generations more, and generation length
varies systematically by arm — so it lets a length difference masquerade as a
factuality difference. **The macro average over prompts is primary**, and it is
what the paired bootstrap over prompts requires.

---

## Problem 2 — degenerate generations are dropped at different rates per arm

### The concern

`score_generation` returns `support_rate = None` for any output with fewer than
two usable sentences. Those records are excluded from every mean, and exclusion
is far from uniform across arms:

| arm | excluded of 900 | rate |
|---|---|---|
| temp1.3 | 164 | 18.2% |
| greedy | 75 | 8.3% |
| beam4 | 48 | 5.3% |
| dola_nucleus | 25 | 2.8% |
| greedy_reppen | 21 | 2.3% |
| temp0.7 | 14 | 1.6% |
| nucleus0.9 | 13 | 1.4% |
| dola | 9 | 1.0% |

369 of 7,200 scored records, 5.1%. Each arm's score is computed on a different,
self-selected subsample: the outputs that arm managed to make coherent.

### Finding 1 — the ranking is unaffected

| arm | degen % | excluded | zeroed | rank |
|---|---|---|---|---|
| beam4 | 5.3% | 0.266 | 0.251 | 1 -> 1 |
| dola_nucleus | 2.8% | 0.132 | 0.128 | 2 -> 2 |
| greedy | 8.3% | 0.114 | 0.105 | 3 -> 3 |
| temp0.7 | 1.6% | 0.100 | 0.098 | 4 -> 4 |
| dola | 1.0% | 0.078 | 0.078 | 5 -> 5 |
| nucleus0.9 | 1.4% | 0.063 | 0.062 | 6 -> 6 |
| greedy_reppen | 2.3% | 0.045 | 0.044 | 7 -> 7 |
| temp1.3 | 18.2% | 0.030 | 0.024 | 8 -> 8 |

**Ordering identical under both conventions.** Exclusion was flattering
temp1.3, but not nearly enough to move it: it sits 0.015 below greedy_reppen,
and zeroing widens that gap rather than closing it. The confound is real and
inconsequential for the ranking, and this table says so with evidence.

### Finding 2 — exclusion breaks the paired bootstrap

Pairing requires the same prompts in both arms. Under exclusion, a temp1.3 vs
greedy comparison loses up to 18% + 8% of pairs, and the survivors are not a
random subset — they are the prompts each arm handled well.

**Resolution:** run the paired tests on the **zeroed** version, where all 900
pairs stay intact, and report the excluded numbers alongside a coverage column
in FActScore style (Min et al. report the response rate for exactly this
reason). Since the ordering is identical, this costs nothing and removes the
objection entirely.

### Finding 3 — "degenerate" is three different things

Only **1 of 369** excluded records is empty. The rest are full-length outputs:

| arm | degen | empty | median chars | loop severity | raw punkt sents |
|---|---|---|---|---|---|
| temp1.3 | 164 | 1 | 1110 | **0.01** | 1.1 |
| greedy | 75 | 0 | 1199 | 0.71 | 10.7 |
| beam4 | 48 | 0 | 602 | 0.85 | 6.6 |
| dola_nucleus | 25 | 0 | 944 | 0.59 | 7.2 |
| greedy_reppen | 21 | 0 | 265 | **0.00** | 1.3 |
| temp0.7 | 14 | 0 | 48 | 0.26 | 1.4 |
| nucleus0.9 | 13 | 0 | 36 | **0.00** | 1.3 |
| dola | 9 | 0 | 1006 | 0.58 | 18.0 |

Three populations, currently hidden inside one count:

**(a) Repetition loops** — beam4 0.85, greedy 0.71, dola/dola_nucleus ~0.58.
Unambiguous in the raw text: `"Abdullahi Mohammad Ahmad Hassan\n\n"` repeated to
the cap; `"Russian-born, Russian-born, ..."`; `"Okay?"` x20. One claim repeated.
Excluding them is honest.

**(b) Unsegmentable run-ons** — temp1.3, loop severity **0.01**, 158 records.
Not gibberish and not repetitive: non-repeating, comma-spliced prose with almost
no sentence-ending punctuation. *"He along with Kenneth Schultz, either fiction
writer Kyle Livingston, creative writer Jeph Wessler, field hometown
pharmacist-owners Landiki Nichols..."* These carry the most distinct content of
any excluded group and are dropped on a formatting property.

**(c) Valid single sentences** — greedy_reppen and nucleus0.9, loop 0.00, one
well-formed sentence each. *"I'm not sure if he's the best person to ask, but I
think it would be nice for him to know that we're all in this together..."*
These are not degenerate at all. The `< 2 sentences` rule (`DECISIONS.md`,
2026-09-09) was introduced because a denominator of 1 forces 0% or 100% — a
variance concern the bootstrap should absorb, not a reason to discard data.

### Finding 4 — MIN_CHARS is not the cause

`MIN_CHARS = 15` does discard fragments: on a run-on with short stops, punkt
finds 12 sentences and the filter keeps 3, dropping `"Yes."`, `"No."`, `"Ok."`.
But it is a minor contributor:

| threshold | records recovered of 369 |
|---|---|
| MIN_CHARS = 5 | 47 |
| MIN_CHARS = 0 | 57 |

About 15%. For temp1.3 specifically, punkt finds only **1.1 raw sentences**, so
148 of its 164 have no boundaries to recover at any threshold. Lowering the
threshold would also admit `"Yes."` and `"Ok."` as scoreable claims, which is
worse than the problem.

**Keep `MIN_CHARS = 15`.**

### Consequences

- **Keep the exclusion rule**, but report the three-way taxonomy per arm instead
  of one opaque count. A repetition loop and a well-formed sentence currently
  sit in the same bucket.
- **Use the zeroed variant for all paired tests**; report excluded plus a
  coverage column as the headline, FActScore style.
- **Category (c) is a real but small defect.** Those arms have 21 and 13
  exclusions out of 900, so recovering them could move their scores by at most
  ~2% — not worth a re-verification pass, but worth stating rather than hiding.

---

## What changed, and what did not

| | outcome |
|---|---|
| Generation | unchanged — no re-generation |
| Verification | unchanged — no re-verification pass |
| `MIN_CHARS`, theta, exclusion rule | unchanged |
| Equal-pool control pass | **cancelled**, superseded by the floor measurement |
| Paired tests | switch to the zeroed variant |
| Manual validation | must be **stratified** across the three strata |
| theta tuning | constrained to >= 0.5 |
| Reporting | three-way degeneracy taxonomy; coverage column beside every support rate |

## Residual risks, stated plainly

1. **Reference coverage (mechanism 2) is not excluded** and cannot be by this
   design. The stratified manual labels will bound it; until then the stratum
   effect is "memorisation and/or coverage", not memorisation alone.
2. **158 temp1.3 records carry content that is never scored.** Excluding them is
   defensible, but temp1.3's support rate is measured on 82% of its outputs and
   that number must appear next to it.
3. **The floor measurement is a bound, not a decomposition.** It shows the
   artifact is too small to matter at theta = 0.5. It would not hold at a lower
   threshold.

### Addendum — the bound, measured directly (14 September)

The argument above turns on "near-misses in 0.1-0.5 do not cross 0.5". That is
now measured rather than argued. GPT-2 beam-4 sentences with p_entail in
[0.4, 0.5) — the only ones a pool-size shift could plausibly flip:

| stratum | near-miss | of | share | pool |
|---|---|---|---|---|
| 0 | 9 | 1404 | 0.64% | 29 |
| 1 | 12 | 1227 | 0.98% | 61 |
| 2 | 3 | 1584 | **0.19%** | 197 |

Under 1% everywhere, and falling with pool size. Flipping every sentence in
stratum 2's band moves its support rate by 0.002; the measured pool-size shift
of +0.022 entailment probability is a fifth of the band's width, so the
achievable movement is smaller still.

**Bound: the pool-size artifact can move support rate by at most ~0.01 at
theta = 0.5, against an observed llama-1b stratum effect of +0.227.**

This supersedes the macro-rate control as the primary argument. That control is
weak alone: GPT-2's stratum 2 minus stratum 0 difference is +0.008 with a
bootstrap 95% interval of [-0.114, +0.129] (10,000 resamples, seed 0), bounding
the artifact only at ~0.12. The band measurement is an order of magnitude
tighter and does not depend on GPT-2 being free of a floor effect — which
answers the objection this document raises against its own control.

---

## Problem 3 — is "supported" just generic prose? (14 September)

### Method

Every **claim-like** sentence the verifier marked supported (8,162 of them) was
re-scored against a **different entity's page, drawn from the same stratum** so
the evidence pool size is matched. A real fact about one person should not be
entailed by another person's page. `scripts/mismatch_control.py`; deterministic
same-stratum rotation, seed-free.

Adjusted support rate counts a sentence only if the true page entails it AND
the wrong page does not.

**Corrected 17 September; the first run of this control was invalid.** Two
faults, both found by inspecting what the control was actually comparing:

1. **The hypothesis did not match the verifier's.** `score_generation` scores
   `"{entity}: {sentence}"`; the control scored the bare sentence. The two
   probabilities were therefore never comparable. Fixing it moved GPT-2 beam4
   genericity 45.5% -> 66.1% on one shard, and the entire swing came from
   name-bearing sentences (bare names +0.970, no-name sentences -0.012).
2. **Non-claims were being scored as claims.** A Wikipedia title echoed back
   ("Douglas Wood (engineer).") becomes a tautology once the entity prefix is
   prepended, so *any* page entails it. 41% of GPT-2 beam4's supported
   sentences were non-claims (title echoes, questions, truncation fragments)
   against 2% of Llama-3b's, so the bias was strongly model-dependent.
   `src.text.is_claim` now excludes them from both numerator and denominator,
   identically for every arm.

The raw column below is therefore over a **claim-only denominator** and is not
comparable term-for-term with the superseded table.

### Evidence

**Test split only** (80 entities x 3 seeds = 240 per row), per DECISIONS
2026-09-07. Coverage 6,717/6,717 claim-like supported sentences.

| model | arm | raw | adjusted | generic % |
|---|---|---|---|---|
| gpt2 | beam4 | 0.201 | **0.052** | **56.0** |
| gpt2 | greedy | 0.136 | **0.092** | 33.7 |
| gpt2 | dola_nucleus | 0.033 | 0.014 | 48.8 |
| gpt2 | nucleus0.9 | 0.010 | 0.008 | 16.7 |
| llama-1b | beam4 | 0.271 | 0.239 | 9.9 |
| llama-1b | greedy | 0.084 | 0.083 | 0.7 |
| llama-1b | nucleus0.9 | 0.075 | 0.068 | 10.7 |
| llama-3b | beam4 | 0.256 | 0.248 | **3.6** |
| llama-3b | greedy | 0.090 | 0.084 | 2.6 |
| llama-3b | nucleus0.9 | 0.074 | 0.064 | 18.5 |

### Findings

1. **GPT-2's beam-4 lead does not merely shrink — it inverts.** 56.0% of its
   supported claims fit a different person. Adjusted, beam4 falls to 0.052
   while greedy holds 0.092: greedy becomes the more factual arm by ~1.8x, and
   beam4 drops from first to second within GPT-2. The superseded table
   reported this as "level with greedy", which understated it.
2. **The Llama beam-4 advantage is real**, losing only 9.9% and 3.6%, and
   beam4 stays first in both models under either convention (0.239, 0.248).
3. **Genericity falls monotonically with scale: 56.0 -> 9.9 -> 3.6.** Larger
   models make more specific, more falsifiable claims. This explains GPT-2
   scoring above 0.9 on obscure entities without invoking copying, which
   `scripts/copy_check.py` had already excluded (max 8-gram overlap 0.061).
4. **Sampling is far more generic than greedy, and for the Llamas it is the
   most generic mode; for GPT-2 beam search is.** nucleus0.9 loses 10.7% and
   18.5% against greedy's 0.7% and 2.6%, and exceeds even beam4 on both Llamas
   (10.7 vs 9.9; 18.5 vs 3.6). GPT-2 is the exception, and by a wide margin:
   beam4 56.0 against nucleus 16.7. So "sampling is the generic mode" holds at
   1B and 3B but not at 124M, where beam search dominates it.
5. **`dola_nucleus` is the second-most generic GPT-2 arm** at 48.8%, dropping
   from 3rd to 5th place once adjusted.

### Consequence

Raw support rate is the headline (FActScore-comparable); adjusted is reported
beside it and is **primary for cross-model claims**, because genericity varies
systematically with scale and raw rates are not comparable across models
without it. Noise caveat: generic % for GPT-2's weakest arms rests on few
supported sentences.

## Labelling the 'absent' category (17 September)

`absent` means the claim is TRUE of this person but does not appear in the
page. A single retrieved evidence window cannot establish that, so
`scripts/sample_for_labeling.py` writes the full pinned page for every sampled
entity to `outputs/labeling_pages/` and the sheet carries a `page_file` column.

The pinned snapshot matters: the pages are `20231101.en`, and live Wikipedia
has moved on. Judging absence against the live site would measure a different
corpus than the metric used.
