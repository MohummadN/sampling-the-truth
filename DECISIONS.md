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

## Still open
- Entailment threshold theta          (tune on dev only)
- dola_layers: "low" vs "high"        (tune on dev only)
- Seeds (3 integers)
- Dev/test split size — currently 20/80
- SC selection embedder
