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
| `device` | str | device that produced the record; every grid record is `cuda:*` |
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

---

# Verdict JSONL contract

Written by `scripts/verify.py`, one shard per `(model, seed)` at
`outputs/verdicts_{model}_seed{seed}.jsonl`. Same append-only, fsynced,
resumable, lock-protected contract as generation.

One line per `(model, decoding, seed, prompt_id)` — the same join key as the
generation file, so the two join row-for-row.

| field | type | notes |
|---|---|---|
| `model`, `decoding`, `seed`, `prompt_id` | | join key, copied from the generation record |
| `entity`, `split`, `stratum` | | copied, so verdicts stand alone in analysis |
| `n_sentences` | int | sentences found by `src/text.py::split_sentences` |
| `n_supported` | int/null | sentences with `p_entail >= theta`; null when degenerate |
| `support_rate` | float/null | `n_supported / n_sentences`; **null when degenerate** |
| `degenerate` | bool | fewer than 2 sentences — excluded from the denominator, counted per arm |
| `sentences` | list/null | per-sentence detail, see below |
| `nli_model`, `retriever` | str | the two model ids |
| `theta`, `top_m`, `window` | | verifier settings in force for this record |
| `git_commit` | str | provenance |

Each entry of `sentences`:

| field | type | notes |
|---|---|---|
| `sentence` | str | the generated sentence, as split |
| `p_entail` | float | max P(entailment) over the retrieved premises, 4 dp |
| `supported` | bool | `p_entail >= theta` |
| `evidence` | str/null | the premise that produced the max — for the manual check |

## Consumer notes

- **`sc_k5` is absent.** Those records have `text: null` until the
  self-consistency selector runs, so `verify.py` skips them. They arrive in a
  second pass.
- **theta is re-tunable without re-running.** `p_entail` is stored per
  sentence, so a new threshold is a recomputation over this file, not another
  GPU job. The same is true of the dedup variant.
- **Two support rates are reported** (DECISIONS, 2026-09-12): distinct-claim
  (identical sentences collapsed) is primary, token-weighted (every sentence
  counted) is secondary. Both come from `sentences`.
- **Degenerate rows are not zeros.** Filter on `degenerate` before averaging;
  a null `support_rate` dropped into a mean silently becomes 0.

## Provenance gap in the first verification pass

Every verdict written by array 882671 carries `git_commit: "unknown"`: git is
not installed on compute nodes s-003 and s-006 (it is on s-004), and the
lookup swallowed the error. Those verdicts were produced by the code at
`50dc78b`, submitted 2026-09-12 11:01, on GPU nodes per `logs/verify_*.out`.
Later passes record the commit via a fallback that reads `.git` directly, and
also record a `device` field.
