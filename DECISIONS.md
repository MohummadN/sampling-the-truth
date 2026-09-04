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

## Still open
- Scorer model for perplexity (must not be one of the 3 generators)
- Entailment threshold theta          (tune on dev only)
- dola_layers: "low" vs "high"        (tune on dev only)
- Seeds (3 integers)
- Dev/test split size — currently 20/80
- SC selection embedder
