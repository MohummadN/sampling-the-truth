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
- Deduplicate repeated identical sentences before verifying, or not?
  dedup = distinct-claim accuracy; no-dedup = token-weighted accuracy.
  Very different numbers for greedy and beam - decide deliberately.
- Entailment threshold theta          (tune on dev only)
- dola_layers: "low" vs "high"        (tune on dev only)
- Seeds (3 integers)
- Dev/test split size — currently 20/80
- SC selection embedder

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
