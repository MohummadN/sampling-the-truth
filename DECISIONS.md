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

2026-09-05  max_new_tokens=256 for every arm: long enough for biography
            generation while keeping the full grid computationally feasible.

## Still open
- Scorer model for perplexity (must not be one of the 3 generators)
- Entailment threshold theta          (tune on dev only)
- dola_layers: "low" vs "high"        (tune on dev only)
- Seeds (3 integers)
- Dev/test split size — currently 20/80
- SC selection embedder

## Fluency scorer
`Qwen/Qwen2.5-0.5B` (base), fp32.
Disjoint from all three generators (gpt2, Llama-3.2-1B/3B), so no model scores
its own family's outputs. Ungated, 0.5B. One scorer for every arm and model,
so perplexity is comparable across the whole grid.
