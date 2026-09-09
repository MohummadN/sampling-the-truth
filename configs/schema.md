# Generation JSONL contract

One line per `(model, decoding, seed, prompt_id)`. Written by `src/generate.py`,
one shard per `(model, seed)` at `outputs/gen_{model}_seed{seed}.jsonl`.
Append-only, fsynced per record, resumable, one writer per shard (lock file).

| field | type | notes |
|---|---|---|
| `task` | str | always `"factscore_bio"` |
| `prompt_id` | str | `p000`..`p099`, stable across all shards |
| `entity` | str | Wikipedia title; key into `data/reference_pages.json` |
| `split` | str | `dev` (20) or `test` (80) — report test only |
| `stratum` | int | 0/1/2, tertile of reference-page length |
| `model` | str | short key: `gpt2`, `llama-1b`, `llama-3b` |
| `model_id` | str | full HF id |
| `model_revision` | str | commit hash of the model snapshot |
| `dtype` | str | `torch.float16` for all generators |
| `decoding` | str | arm name, see `src/decoding.py` |
| `gen_kwargs` | obj | exact kwargs passed to `generate()` |
| `seed` | int | run seed; the per-cell seed is derived, see `cell_seed` |
| `prompt` | str | the exact resolved prompt string |
| `text` | str/null | the generation; **null for `sc_k5`** |
| `samples` | list/null | 5 strings for `sc_k5`; **null otherwise** |
| `selected_index` | int/null | `sc_k5` only — **the selector fills this** |
| `gen_tokens` | int | tokens returned |
| `compute_tokens` | int | tokens decoded = `gen_tokens x num_beams`; the cost axis |
| `gen_time_s` | float | CUDA-synchronised wall time |
| `hit_cap` | bool | any returned sequence reached `max_new_tokens` |
| `n_sentences` | int/null | null for `sc_k5` |
| `run_id`, `transformers_version`, `torch_version`, `git_commit` | str | provenance |

## Consumer notes

- `sc_k5` is the only arm with `samples`; `text` and `n_sentences` stay null
  until the selector writes `selected_index` and `text`.
- `gen_tokens` != `compute_tokens` for `beam4` (factor 4). Use
  `compute_tokens` for any matched-compute claim.
- Generations with fewer than 2 sentences are excluded from the support-rate
  denominator and counted per arm.
- Join key for everything downstream: `(model, decoding, seed, prompt_id)`.
