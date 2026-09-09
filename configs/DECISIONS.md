
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
