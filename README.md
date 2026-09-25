# Sampling the Truth

Decoding Strategies and the Factuality–Diversity Tradeoff in Small Open LLMs.

Three models (GPT-2 124M, Llama-3.2-1B, Llama-3.2-3B) x 9 decoding
configurations x 3 seeds x 100 biography prompts = 8,100 generations, each
scored for factual support against the subject's Wikipedia page, for diversity,
repetition and fluency, and priced in inference compute.

Every number in the paper comes from `paper/numbers.txt`, which is produced by
the scripts below. `DECISIONS.md` records why each design choice was made.

## Environment

Torch must be installed first, from the cu126 index. The cluster's GPUs are
TITAN Xp (sm_61); the default cu130 wheel has no Pascal kernels and silently
falls back to CPU.

    conda create -p $STORE/venv python=3.11
    conda activate $STORE/venv
    pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
    pip install -r requirements.txt

All Slurm scripts export `HF_HUB_OFFLINE=1`, so model weights and the DoLa
remote code (revision `af6cdc35`) must be pre-fetched into `$HF_HOME` before
the first run. Nothing is downloaded at run time.

## Data

    python -m scripts.build_reference_pages     # pinned 20231101.en snapshot -> reference pages
    python -m scripts.fill_missing_pages        # retries pages the first pass missed

454 of FActScore's 500 entities resolve in the pinned snapshot; 100 are sampled
stratified by article length, split 20 dev / 80 test, and frozen.

**Dev/test discipline.** Every tunable quantity was chosen on the 20 dev
entities; every number in the paper is computed on the 80 test entities. The
split is applied by `src.data.reported_entities()` and can be overridden with
the `SPLIT` environment variable (`dev`, `test`, `all`) for dev-only sweeps.

## Pipeline

Each Slurm script is a 9-task array over the (model, seed) grid, throttled to 3
concurrent tasks. Run them in order; each stage consumes the previous stage's
output.

    sbatch slurm/generate.sbatch      # -> outputs/gen_{model}_seed{seed}.jsonl
    sbatch slurm/verify.sbatch        # -> outputs/verdicts_{model}_seed{seed}.jsonl
    sbatch slurm/metrics.sbatch       # -> outputs/metrics_{model}_seed{seed}.jsonl

Self-consistency is selected after generation, then verified as five separate
pseudo-records per entity so the pool, the selected sample and the oracle
best-of-five can all be measured:

    python -m scripts.select_sc --model gpt2 --seed 1234   # (repeat over the 9 cells)
                                                           # -> scsel_*, scchoice_*
    sbatch slurm/verify_sc.sbatch                          # -> scverdicts_*

## Validity controls

    sbatch slurm/mismatch.sbatch      # same claims vs a wrong same-stratum page -> mismatch_*
    sbatch slurm/topm20.sbatch        # retrieval depth m=20 (dev only)          -> topm20_*
    sbatch slurm/dola_sweep.sbatch    # DoLa layer selection (dev only)          -> dolasweep_*
    python -m scripts.copy_check      # n-gram overlap between generation and page
    python -m scripts.dedup_probe     # effect of collapsing repeated sentences
    python -m scripts.sanity1_repetition

## Human labels and calibration

100 sentences, stratified by entity stratum and entailment-probability band,
labelled blind by the authors without the model's verdict visible.

    python -m scripts.sample_for_labeling   # -> labeling_sheet.csv, labeling_key.csv, labeling_pages/
    python -m scripts.label                 # interactive labelling -> labels.csv
    python -m scripts.analyze_labels        # TPR, FPR, precision, balanced accuracy
    python -m scripts.calibrate             # Rogan-Gladen correction
    python -m scripts.split_not_addressed   # false-positive composition

## Analysis, tables and figures

    python -m scripts.analysis
    python -m scripts.compute_frontier
    python -m scripts.analyze_sc
    python -m scripts.analyze_mismatch
    python -m scripts.compare_topm
    python -m scripts.analyze_dola_sweep

    python -m scripts.tables       # -> paper/tables.tex   (Tables 1, 2)
    python -m scripts.table3       # -> paper/table3.tex   (paired bootstrap)
    python -m scripts.table4       # -> paper/table4.tex   (verifier validation)
    python -m scripts.f1_teaser    # -> paper/figures/f1_teaser.pdf
    python -m scripts.f3_frontier  # -> paper/figures/f3_frontier.pdf
    python -m scripts.plots

## Layout

    src/            generation and metric implementations
      data.py       entity pool, splits, reference pages
      decoding.py   the 9 decoding configurations
      generate.py   generation driver
      metrics/      factuality (NLI + retrieval), diversity, fluency,
                    consistency, calibration
      text.py       sentence splitting, claim filter
    scripts/        one script per analysis or output artifact
    slurm/          array jobs over the (model, seed) grid
    outputs/        all JSONL records and CSVs (156M, not in git)
    paper/          LaTeX source, generated tables and figures
    DECISIONS.md    every design decision and why

## Notes

- `srun --pty bash` is rejected on this cluster; run the command directly under
  `srun`, or submit with `sbatch`.
- `s-002` is excluded in every array: it fails CUDA initialisation after the
  shard has already been scheduled.
- Generations are stored with their git commit, device, dtype and decoding
  arguments, so any record can be traced to the code that produced it.
