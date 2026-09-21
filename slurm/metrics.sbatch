#!/bin/bash
#SBATCH --job-name=metrics
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-8%3
#SBATCH --output=logs/metrics_%A_%a.out
#SBATCH --requeue
#SBATCH --exclude=s-002

MODELS=(gpt2 gpt2 gpt2 llama-1b llama-1b llama-1b llama-3b llama-3b llama-3b)
SEEDS=(1234 5678 9012 1234 5678 9012 1234 5678 9012)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}

export STORE=/home/morg/NLP_2526b/mohummadn
export HF_HOME=$STORE/.cache/huggingface
export NLTK_DATA=$STORE/nltk_data
export HF_HUB_OFFLINE=1
source $STORE/miniconda3/bin/activate
conda activate $STORE/venv
cd $STORE/NLPFinalProject

echo "metrics: $M seed $S on $(hostname) at $(date)"
python -m scripts.compute_metrics --model "$M" --seed "$S"
