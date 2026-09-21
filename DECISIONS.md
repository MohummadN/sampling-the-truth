# Decisions

Format: date · decision · one-sentence justification.
This file becomes the paper's Experimental Setup section.

2026-09-01  GPT-2 = `gpt2` (124M), not gpt2-large.
            The proposal says "GPT-2" unqualified; the small variant is cheapest
            for a sanity baseline and degeneration is more pronounced at small
            scale, which is what sanity experiment 1 tests.

2026-09-01  Base Llama-3.2-1B / 3B, not Instruct — same variant for both sizes.
            Alignment lowers output entropy, compressing the decoding differences
            we measure; mixing variants across sizes would confound scale with
            alignment. GPT-2 has no instruct variant, so base keeps all three
            models comparable.

2026-09-01  max_new_tokens = 256 for every arm.
            Long enough for a real biography, short enough that 6,300 generations
            finish in available queue time. Identical across arms — fairness of
            the comparison depends on it.

2026-09-05  All decoding arms explicitly set do_sample, temperature, top_p,
            and top_k. GPT-2 and Llama ship different generation defaults, so
            inherited values would make the arms non-comparable across models.

2026-09-05  Pinned transformers==5.16.1 because generation defaults and DoLa
            behavior differ across major Transformers versions.

2026-09-06  Fluency scorer = `Qwen/Qwen2.5-0.5B` (base), fp32.
            Disjoint from all three generators, so no model scores its own
            family's outputs. Ungated and small. One scorer for every arm and
            model, so perplexity is comparable across the whole grid.

2026-09-06  torch 2.14.0+cu126.
            The cluster's GPUs are TITAN Xp (sm_61) and CUDA 13 dropped Pascal,
            so the default cu130 wheel had no usable kernels and every run
            silently fell back to CPU. cu126 ships sm_60, binary-compatible
            upward to sm_61.

2026-09-06  Every loader returns `(tok, model)`.
            `src/models.py::load` and `src/metrics/fluency.py::load_scorer`
            alike, so call sites never have to remember which order applies.

2026-09-07  Prompt = "Tell me a bio of {entity}." verbatim, base-model raw text.
            FActScore's prompt, used unchanged for citability. A more
            instructive prompt would lower output entropy and shrink the very
            differences between decoders that the study measures.

2026-09-07  Verification unit = sentence, not atomic fact.
            Deliberate deviation from FActScore: no LM decomposition step, so
            it is deterministic, free and fast. Costs, stated in Limitations:
            mixed-truth sentences collapse to one label, factuality becomes
            weakly length-dependent, and absolute numbers are not comparable
            to published FActScore values.

2026-09-07  Metric name = "support rate", never "FActScore".
            It measures whether a claim is supported by the entity's Wikipedia
            page, not whether it is true. Mean sentence length is reported per
            arm so the length-dependence above can be checked.

2026-09-07  Manual validation uses three labels: supported, unsupported, and
            "true but absent from the page".
            NLI entailment is not truth. High-temperature arms wander into
            peripheral true facts the page omits, which would otherwise score
            as hallucination and bias the result toward our own hypothesis.
            Refusals are counted and reported as a separate column.
            SUPERSEDED 2026-09-18 by the Stage 7 scheme. The labels actually
            collected are entailment-style - supported / contradicted /
            not_addressed - and the labeller judges only what the shown
            evidence does with the claim, using no world knowledge. That makes
            the task objective and fast, but it merges "true but absent" into
            not_addressed. Consequence: reference coverage is NOT bounded by
            the manual validation. It is reported as an open limitation.

2026-09-07  Entities: FActScore's published list if obtainable; otherwise 100
            sampled in 3 strata by Wikipedia article length (~33 each).
            Disambiguation pages and redirects excluded; a minimum reference
            length required; the entity -> Wikipedia-title mapping stored
            explicitly. Frozen once in data/entities.json with a fixed seed and
            a split field, and never regenerated - choosing entities after
            seeing results is selection bias.

2026-09-07  All arm comparisons are paired (bootstrap over prompts). No
            unpaired tests; confidence intervals reported, not just means.
            Per-entity support rates have sigma ~ 0.25, so SE ~ 0.25/sqrt(100)
            = +/-2.5 points and an unpaired comparison would need 5-7 point
            gaps. Pairing cancels between-entity variance. sigma is measured on
            dev rather than assumed; gaps under ~3 points are treated as noise.

2026-09-07  Wikipedia snapshot pinned to 20231101.en, indexed in a single pass
            and cached to data/reference_pages.json.
            A later snapshot could contain facts the models never saw. The HW3
            linear scan walks 6M rows per page and would burn queue time. The
            cache is a build artifact: gitignored, with the build script
            committed, and every entity asserted to resolve.

2026-09-07  Length control: report mean +/- sd generated tokens per arm, and
            recompute all length-sensitive metrics on a fixed 128-token prefix.
            Greedy and beam run to the cap while nucleus terminates early, so
            part of any raw diversity gap is a length gap - and it flatters our
            hypothesis. The trailing incomplete sentence is dropped, by the
            same rule for every arm, noting this removes more text from capped
            arms.

2026-09-07  Log hit_cap = (gen_tokens >= max_new_tokens) on every record; no
            min_new_tokens floor.
            The per-arm cap rate measures degeneration directly and catches EOS
            misconfiguration. A floor would force models past a natural EOS and
            contaminate the abstention signal unevenly across arms; empty
            generations are instead counted, excluded from the denominator, and
            reported.

2026-09-07  Splits: 20 dev / 80 test, fixed seed, decided once.
            We tune theta, dola_layers, nucleus p and min_chars, so splits
            apply even though nothing is trained. All tuning, debugging and
            eyeballing happen on the 20; the grid runs on all 100 but only the
            80 are reported. theta is chosen to maximise agreement with the
            manual labels - never to maximise the gap between arms.

## Still open
- Entailment threshold theta          (tune on dev only)

2026-09-07  dtype = torch.float16 for all three generator models.
            Measured on TITAN Xp (sm_61, gpt2, 256 greedy tokens): fp32 114.6
            tok/s, fp16 117.0, bf16 77.1 — bf16 is emulated (not native below
            sm_80). Llama-3.2-3B needs ~12.8 GB in fp32 and the card has 12.7,
            so fp32 is impossible for the largest model; the dtype must be
            identical across scales or precision is confounded with scale.

2026-09-07  Slurm sharding is by (model, seed), never finer.
            Loading Llama-3.2-3B from the netapp share takes ~6 min warm
            (~11 min cold) — it is bandwidth-bound, not download-bound. One job
            therefore loads a model once and generates all 9 arms x 100 prompts
            inside it. Sharding per arm or per prompt would pay that 6 minutes
            again for every shard.

2026-09-07  Candidate pool = 454 of FActScore's 500 entities.
            46 titles are absent from wikimedia/wikipedia 20231101.en. Two full
            independent passes over all 6,407,814 rows found 0 of them, so the
            gap is systematic, not transient. The missing set is dominated by
            heavily-templated articles — major historical figures and athletes
            with large statistics tables — which the HTML-to-text conversion
            behind this dump drops. The pool is therefore tilted away from
            table-dense biographies and from the most famous entities; if
            anything this reduces ceiling effects. 100 entities are sampled
            from the 454 and the exclusion list is committed for reproducibility.

2026-09-07  data/reference_pages.json is committed, not gitignored.
            At ~6 MB it is small enough to version, and committing it makes the
            repo reproducible from a clone without re-streaming 6.4M Wikipedia
            rows. It also keeps the data tests runnable off-cluster.

2026-09-09  Generations with fewer than 2 sentences are excluded from the
            support-rate denominator and counted per arm.
            Observed at GATE 1: beam4 produced 256 tokens of newline-separated
            repetitions with no sentence-ending punctuation, which nltk splits
            into a single "sentence". A denominator of 1 makes the score 0% or
            100% on one label. Same rule as for empty generations.

2026-09-09  Correction to the 2026-09-07 hit_cap entry: hit_cap is per
            sequence, `any(c >= max_new_tokens)`, not `gen_tokens >=
            max_new_tokens`. Under the written formula every sc_k5 record would
            be True, since gen_tokens sums five sequences. The code was right;
            the entry was wrong.

2026-09-09  gen_tokens and compute_tokens are separate fields.
            beam4 returns one sequence but decodes num_beams in parallel, so
            returned tokens understate its cost 4x. Measured on gpt2, 24
            tokens: greedy 2.00 s, beam4 7.48 s, same gen_tokens.
            compute_tokens = gen_tokens x num_beams is the matched-compute axis.

2026-09-09  Timing is CUDA-synchronised and uses perf_counter.
            CUDA kernels are asynchronous; without a synchronize() bracket
            gen_time_s partly measures queueing, and gen_time_s is half the
            cost axis.

2026-09-09  dola_layers frozen at "low" for the full grid.
            Tuning it would require the verifier, which would delay the grid
            launch past the point where 43 GPU-hours still fit in the schedule.
            Low vs. high is instead compared on the 20 dev entities and
            reported as a sensitivity check; if high wins, only the dola arm is
            re-run (1 arm x 100 prompts x 3 seeds x 2 models).

2026-09-09  Seeds = 1234, 5678, 9012.
            Arbitrary by design and fixed before any result was seen.
            Three seeds is not where statistical power comes from — the
            paired bootstrap over prompts is — but it is what the proposal
            promised.

2026-09-10  HF_HOME = $STORE/.cache/huggingface (not $STORE/hf).
            The Llama weights were downloaded into the .cache path before
            HF_HOME took effect, leaving two caches with the models in the one
            HF_HOME did not point at. Batch jobs do not read ~/.bashrc, so
            every shard would have re-downloaded 9 GB against a gated repo with
            no token stored.

2026-09-10  DoLa runs via custom_generate="transformers-community/dola" with
            trust_remote_code=True.
            transformers 5.x moved DoLa out of the core library; dola_layers
            alone now raises. The implementation we run is therefore not the
            library's own and not the one the DoLa paper describes — it is an
            extraction of the 4.x code, pinned by the HF cache revision.
            Belongs in Limitations.

2026-09-10  Grid shards refuse to run on CPU (--allow-cpu overrides).
            The first launch put five of nine shards on CPU after CUDA failed
            to initialise, and they were producing records silently. CPU
            records are RNG- and numerically incomparable with GPU ones, and a
            3B shard would never finish. The array is also throttled to 3
            concurrent tasks to stop five jobs landing on one node.

2026-09-10  Decoding passes clean_up_tokenization_spaces=False explicitly.
            transformers already refuses that post-processing for BPE
            tokenizers, and both generator families are BPE, so output was
            byte-identical either way and no record was ever affected. Setting
            it explicitly pins the behaviour across library versions rather
            than relying on a default.

2026-09-10  DoLa remote code pinned at revision
            af6cdc351e7e0bd28a86ce32aac461494a09a9c1, pre-fetched with
            snapshot_download; grid jobs run with HF_HUB_OFFLINE=1.
            transformers rejects both `revision=` and `repo@sha` for
            custom_generate, so pre-fetch plus offline is the only available
            pin. It also guarantees no shard silently re-downloads the gated
            Llama weights.

2026-09-10  DoLa requires output_hidden_states=True despite transformers
            warning that the flag "may be ignored".
            That validation pass does not know about custom_generate kwargs.
            Removing the flag reproduces TypeError: 'NoneType' object is not
            subscriptable in _dola_decoding, which reads outputs.hidden_states.
            Verified: dola and greedy_reppen — identical in every kwarg except
            the layer contrast — produce different text, so DoLa is active and
            not silently skipped.

2026-09-10  DoLa cost on Llama-3.2-3B, measured: 6.5 GB peak GPU memory and
            ~10 s per 256-token generation on a TITAN Xp.
            Half the 12.7 GB card, so the arm is viable at the largest scale.
            A full (model, seed) shard is ~4.5 h at that rate; GPT-2 shards
            measured 37 min at 0.41 cells/s.

2026-09-11  Grid excludes node s-002.
            torch.cuda.is_available() returns False there ("CUDA unknown
            error") while s-003..s-006 work. Nine of nine shards landed on
            s-002 and refused; the GPU guard caught it in 40 seconds instead
            of producing an hour of CPU records.

2026-09-11  GATE 2 (sanity experiment 1) passed on current code.
            gpt2, seed 1234, first 30 entities, 3 arms, all 90 records on
            cuda:0. Metrics on the first 128 word tokens (Unicode tokenizer);
            95% CI across prompts, not seeds - greedy and beam-4 are
            deterministic, so a seed-wise error bar would be exactly zero.
                          rep-4            loop severity    hit cap
            greedy        0.833 +/- 0.041  0.759 +/- 0.083  100%
            beam-4        0.843 +/- 0.029  0.819 +/- 0.042   67%
            nucleus 0.9   0.001 +/- 0.002  0.018 +/- 0.004   90%
            Supersedes the 2026-09-10 run, whose shard predated the device
            field and whose loop severity was measured on full-length text.
            Nucleus reached the 256-token cap in 90% of generations - it did
            not terminate early on gpt2 - while beam-4 stopped early in a
            third. Lengths still differ across arms, so the fixed-prefix
            control stands, but the reason given in the 2026-09-07
            length-control entry ("nucleus terminates early") does not hold.

2026-09-12  hit_cap for sc_k5 is any() over its five sequences, so its cap rate
            is not comparable to single-sequence arms and is reported
            separately (or as a per-sequence mean) rather than in the same
            column.

2026-09-12  Measured per-arm means over all 8,100 records: cap rate ranges
            40% (greedy_reppen) to 89% (greedy, temp1.3); mean generated
            tokens 179-243. The length spread is why every length-sensitive
            metric is also reported on a fixed 128-token prefix.

2026-09-12  GATE 3 passed. Support rate falls monotonically with randomness
            (greedy 0.151, temp0.7 0.112, temp1.3 0.022 on llama-1b dev) and
            rises monotonically with entity popularity (stratum 0/1/2 =
            0.040/0.125/0.190), reproducing FActScore's popularity effect.
            p_entail is strongly bimodal (p50=0.018, p90=0.977), so theta is
            not knife-edge: moving it from 0.15 to 0.95 reclassifies ~15% of
            sentences. Absolute support rates are low because the stratified
            sample deliberately includes obscure entities a 1B model cannot
            recall — inspection confirms genuine fabrication, not verifier
            failure.

2026-09-12  Verifier dtype: fp16, measured rather than assumed.
            57,783 sentence scores across 4,950 verdicts contain zero
            non-finite values, range 0.0001-0.9998. DeBERTa-v3 has known fp16
            overflow reports and the fluency scorer is deliberately fp32, so
            this needed evidence, not a default.

2026-09-12  Repeated identical sentences: decided.
            verify.py scores every sentence and stores its p_entail, so both
            variants are computable at analysis time with no re-verification.
            Reported: distinct-claim (identical sentences collapsed) as the
            primary number, token-weighted (every sentence counted) as a
            secondary column. Token-weighted alone would flatter the
            degenerate arms - greedy and beam-4 repeat one sentence many
            times, and a biography's opening sentence is usually the one the
            page supports, so a loop inflates their support rate in the
            direction of our own hypothesis.

2026-09-12  "Still open" pruned to what is actually open. Seeds (decided
            2026-09-09), dola_layers (frozen "low", 2026-09-09) and the
            20/80 split (2026-09-07) were listed as open while being decided
            earlier in this same file.

2026-09-12  Verification complete. Support rate by model, mean over 3 seeds:
            gpt2 0.063 (+/-0.003), llama-1b 0.117 (+/-0.005), llama-3b 0.130
            (+/-0.004). Monotone in scale, with seed spread an order of
            magnitude below the between-model gaps. 124M->1B roughly doubles
            support; 1B->3B adds ~0.013, i.e. diminishing returns at small
            scale. Degenerate generations (<2 sentences): gpt2 ~8%, Llama 3-5%.

## Sentence deduplication: measured, rejected (14 September)

Repeated identical sentences are **not** collapsed before verification.
`scripts/dedup_probe.py` recomputes support rate over unique sentences across
all 7,200 verdicts. Every arm shifts by <= 0.026, and the shifts are *negative*
(raw below dedup): repeated sentences are disproportionately unsupported, so
repetition deflates the score rather than inflating it. The pre-registered
worry — that beam4's high support rate was repetition inflation — is therefore
wrong in sign.

Raw is kept as the headline metric: FActScore scores every atomic claim, a
model that emits a claim ten times has made ten claims, and deduplicating would
remove degeneracy from the factuality axis, which is the tradeoff under study.
The dedup table is reported as a robustness check.

Duplicate-sentence fraction, mean per arm: greedy 0.597, beam4 0.505,
dola_nucleus 0.240, dola 0.082, temp0.7 0.079, nucleus0.9 0.013, temp1.3 0.000,
greedy_reppen 0.000. The repetition-penalty control eliminates exact sentence
repeats entirely, establishing that the repetition is a decoding artifact.

## Metric validity (14 September) — see docs/metric_validity.md

Two threats measured from the existing verdicts; no re-generation, no
re-verification.

- **Pool-size artifact is real but bounded.** A ~7x larger evidence pool buys
  ~+0.022 entailment probability on unsupported sentences (GPT-2 noise floor
  0.0183 -> 0.0402 across strata). GPT-2's macro rate moves +0.007 across the
  same pools while Llama-1b moves +0.227. Reported as bounded, not excluded:
  GPT-2's difference carries a 95% interval of roughly +-0.11.
- **Reference coverage is NOT excluded, and is not bounded either.** The
  stratum effect is "memorisation and/or coverage". The plan was to bound it
  with a "true but absent from page" rate per stratum, but the Stage 7 labels
  are entailment-style and use no world knowledge, so that category was never
  collected - it is merged into `not_addressed`. Coverage remains an open
  limitation and the stratum effect is reported as memorisation and/or
  coverage, never as memorisation alone.
- **The equal-pool re-verification pass is cancelled** — truncating pages would
  suppress pool size and coverage together, so shrinkage would be
  uninterpretable.
- **theta >= 0.5 for the stratum analysis.** The artifact lives in the 0.1-0.5
  band, which grows with pool size. Stage 7 may tune theta for the headline
  metric, but the stratum effect is additionally reported at a fixed
  theta = 0.5.
- **Macro-average over prompts is primary.** Micro-averaging weights long
  generations more and generation length varies by arm, letting a length
  difference masquerade as a factuality difference. The paired bootstrap over
  prompts requires the macro form.
- **Paired tests run on the zeroed variant** (degenerate = 0.0 support) so all
  900 pairs survive; exclusion-based means are reported beside a coverage
  column, FActScore style. Arm ordering is identical under both conventions.
  Caveat to state: zeroing conflates "no scorable claims" with "all claims
  unsupported".
- **MIN_CHARS stays at 15**; lowering it recovers ~15% of degenerates while
  admitting "Yes." and "Ok." as claims.
- **Degeneracy is reported as a three-way taxonomy** (repetition loops,
  unsegmentable run-ons, valid single sentences), not one opaque count. Only
  1 of 369 excluded records is empty.

## Metric validity (14 September) — see docs/metric_validity.md

Two threats measured from the existing verdicts; no re-generation, no
re-verification pass.

- **Pool-size artifact is real and bounded at ~0.01.** A ~7x larger evidence
  pool buys ~+0.022 entailment probability on unsupported sentences (GPT-2
  noise floor 0.0183 -> 0.0402 across strata). The sentences that could cross
  theta = 0.5 are those in [0.4, 0.5): under 1% in every stratum, and falling
  with pool size (0.64% / 0.98% / 0.19%). Against llama-1b's +0.227 stratum
  effect the artifact is two orders of magnitude too small. The macro-rate
  control (GPT-2 flat, +0.008) corroborates but is weak alone: 95% CI
  [-0.114, +0.129].
- **Reference coverage is NOT excluded, and is not bounded either.** The
  stratum effect is "memorisation and/or coverage". The plan was to bound it
  with a "true but absent from page" rate per stratum, but the Stage 7 labels
  are entailment-style and use no world knowledge, so that category was never
  collected - it is merged into `not_addressed`. Coverage remains an open
  limitation and the stratum effect is reported as memorisation and/or
  coverage, never as memorisation alone.
- **The equal-pool re-verification pass is cancelled** — truncating pages would
  suppress pool size and coverage together, so shrinkage would be
  uninterpretable.
- **theta >= 0.5 for the stratum analysis.** The artifact lives in the 0.1-0.5
  band. Stage 7 may tune theta for the headline metric, but the stratum effect
  is additionally reported at a fixed theta = 0.5.
- **Macro-average over prompts is primary.** Micro-averaging weights long
  generations more and generation length varies by arm, letting a length
  difference masquerade as a factuality difference. The paired bootstrap over
  prompts requires the macro form.
- **Paired tests run on the zeroed variant** (degenerate = 0.0) so all 900 pairs
  survive; exclusion-based means are reported beside a coverage column,
  FActScore style. Arm ordering is identical under both conventions. Caveat to
  state: zeroing conflates "no scorable claims" with "all claims unsupported".
- **MIN_CHARS stays at 15**; lowering it recovers ~15% of degenerates while
  admitting "Yes." and "Ok." as claims.
- **Degeneracy is reported as a three-way taxonomy** (repetition loops,
  unsegmentable run-ons, valid single sentences), not one opaque count. Only
  1 of 369 excluded records is empty.

## Genericity control (14 September) — see docs/metric_validity.md Problem 3

SUPERSEDED 17 September — the first run of this control was invalid. It scored
a bare sentence while the verifier scores "{entity}: {sentence}", so the two
probabilities were never comparable; and it counted non-claims (Wikipedia title
echoes, questions, truncation fragments), which become tautologies under the
entity prefix and are entailed by any page at all. Both are fixed: the
hypothesis now matches, and src.text.is_claim excludes non-claims from
numerator and denominator alike.

Corrected result, reported on the test split only (80 entities x 3 seeds,
coverage 6,717/6,717 claim-like supported sentences). Generic share under
beam4: gpt2 56.0%, llama-1b 9.9%, llama-3b 3.6% — monotonic in scale.
**GPT-2's beam4 lead does not collapse to greedy's level, it inverts**:
adjusted beam4 0.052 against greedy 0.092, so greedy is the more factual GPT-2
arm by ~1.8x and beam4 falls from first to second. Both Llamas keep beam4 first
(0.239, 0.248). Raw rate is the headline; **adjusted is primary for cross-model
comparisons**, because genericity varies systematically with scale.

2026-09-17  Analysis scripts report the 80 test entities by default.
            DECISIONS 2026-09-07 says only the 80 are reported, but
            analyze_mismatch, compute_frontier, analyze_sc and copy_check were
            all averaging over all 100 - including the 20 we tune theta,
            dola_layers, nucleus p and min_chars on. They now filter through
            src.data.reported_entities(); SPLIT=dev or SPLIT=all overrides it
            for diagnostics, and every script prints which split it reported.
            Numbers produced before this change were over all 100 and must be
            regenerated before they go in the paper.

2026-09-17  Self-consistency aggregation = embedding medoid, encoder
            sentence-transformers/all-mpnet-base-v2. Closes the "SC selection
            embedder" question above.
            Wang et al. (2023) aggregate by MAJORITY VOTE over answers, which
            is undefined for open-ended text: five biographies are never
            string-identical, so the indicator is 0 for every pair and the
            argmax is meaningless. We replace equality with similarity and keep
            the sample with the highest mean cosine similarity to the other
            four. This is a real redefinition of the method and is declared as
            such in the paper - what we evaluate is "sample five, keep the
            medoid", not self-consistency as Wang et al. defined it.
            Encoder: the same all-mpnet-base-v2 already used for evidence
            retrieval - one embedding model in the project, already cached, and
            its checkpoint is already named in Experimental Setup. With L2
            normalisation the medoid and the geometric-centroid nearest
            neighbour rank identically, differing only by the constant 1/k
            self-similarity term.
            The medoid cannot beat the pool it is given, so the analysis
            reports the oracle best-of-five bound beside it.


## Stage 7 — verifier validation and calibration (18 September)

100 blind manual labels, entailment-style three-way (supported / contradicted /
not_addressed), stratified by entity stratum x p_entail band, zero `unsure`.
Raw distribution 25 / 7 / 68. Labels drawn by `scripts/sample_for_labeling.py`
(seed 20260914), collected with `scripts/label.py`, analysed by
`scripts/analyze_labels.py` and `scripts/calibrate.py`.

- **theta stays at 0.5.** Balanced accuracy peaks at 0.55 (0.956) versus 0.953
  at 0.50 — inside noise at n=100. The pre-registered value is kept; no
  re-verification.
- **Verifier characterisation: high recall, low precision.** Weighted TPR
  0.947, FPR 0.041, precision 0.582, balanced accuracy 0.953. One false
  negative in 100; 36 false positives, 34 of them `not_addressed`. This is the
  neutral trap, predicted in the study material and now measured.
- **False positives are confident (mean p_entail 0.754), not borderline.**
  This withdraws the [0.4, 0.5) band bound in docs/metric_validity.md.
- **All support rates are reported both raw and calibrated** by Rogan-Gladen,
  `T = (R - FPR)/(TPR - FPR)` ~ `1.10 R - 0.045`. Ranking is unchanged; levels
  fall. Arms at or below the false-positive floor of R = 0.041 are reported as
  **indistinguishable from zero true support**, not as small positive numbers.
- **The stratum effect survives calibration**: llama-1b beam4 goes
  0.185 / 0.239 / 0.414 calibrated against 0.199 / 0.264 / 0.426 raw, +0.229
  versus +0.227.
- **23 of 100 sentences are `partial`** — mixing established and unestablished
  facts. Sentence-level verification forces an all-or-nothing call on them;
  reported as a limitation.
- Remaining: `not_addressed` is not yet split into retrieval failure versus
  reference-coverage limit. One short GPU pass over all page windows for those
  rows would measure the retrieval share exactly.

## Retrieval depth (18 September) — see docs/metric_validity.md Problem 4

`TOP_M` stays at 5. All-window re-scoring of the 68 human-labelled
`not_addressed` sentences, with a same-stratum wrong page as noise floor, puts
retrieval failure at **4.4% of that population weighted to the frame** (theta =
0.9; the wrong page fires on 0.1%), i.e. roughly 3% of all sentences. Reported
as a measured limitation rather than re-running the grid. `slurm/topm20.sbatch`
re-verifies the dev entities at top-m = 20 as an independent check.

Also recorded: the two controls (`mismatch_control.py`, `split_not_addressed.py`)
score the same hypothesis form as the pipeline, `"<entity>: <sentence>"`, so
their numbers are comparable to the verifier's own.

## Retrieval depth sensitivity (18 September)

The dev entities were re-verified at `top_m = 20` (`slurm/topm20.sbatch`,
`scripts/compare_topm.py`), paired over the 20 dev entities with a 10,000-sample
bootstrap.

- **Mean shift +0.022 across the 24 model x arm cells, sd 0.014, every cell
  positive.** This agrees with the independent all-window estimate of ~3% of all
  sentences lost to top-5 retrieval (docs/metric_validity.md Problem 4), which
  was derived from human labels and sampling weights rather than a re-run.
- **Robust under both depths:** beam4 first and dola_nucleus second in all three
  models; temp1.3 last in all three.
- **Not robust:** the middle ranks reorder (llama-1b temp0.7 over greedy, dola
  over nucleus0.9; llama-3b greedy falls from 5th to 7th). Those arms differ by
  0.01-0.03, inside their own intervals, so they are reported as a band rather
  than an ordering.
- `TOP_M` stays at 5 for all headline numbers; this sweep is reported as the
  sensitivity check.
