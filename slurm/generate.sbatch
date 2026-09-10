#!/bin/bash
#SBATCH --job-name=gen
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/gen_%A_%a.out
#SBATCH --requeue

# One task per (model, seed). Sharding is by model because loading
# Llama-3.2-3B off the netapp share costs ~6 minutes; a finer shard would pay
# that again for every piece. See DECISIONS.md.
MODELS=(gpt2 gpt2 gpt2 llama-1b llama-1b llama-1b llama-3b llama-3b llama-3b)
SEEDS=(1234 5678 9012 1234 5678 9012 1234 5678 9012)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}

export STORE=/home/morg/NLP_2526b/mohummadn
export HF_HOME=$STORE/.cache/huggingface
export NLTK_DATA=$STORE/nltk_data
source $STORE/miniconda3/bin/activate
conda activate $STORE/venv
cd $STORE/NLPFinalProject

echo "shard: $M seed $S on $(hostname) at $(date)"
python -m src.generate --model "$M" --seed "$S"
