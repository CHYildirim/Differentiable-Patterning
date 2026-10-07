#!/usr/bin/env bash
# Eddie (Grid Engine) array job: one manifest entry per array task.
#
# Submit from the repo root on an Eddie login node, e.g.:
#   N=$(grep -m1 '^count:' generated/cluster_test/manifest.yaml | awk '{print $2}')
#   mkdir -p logs/eddie
#   qsub -t 1-$N -tc 4 launch_eddie.sh Experiments/run_config.py generated/cluster_test/manifest.yaml
#
#$ -N nca_array
#$ -cwd
#$ -o logs/eddie/
#$ -e logs/eddie/
#$ -l h_rt=08:00:00
#$ -l h_vmem=64G
#$ -q gpu
#$ -pe gpu-a100 1

set -eo pipefail

PY_SCRIPT="${1:?Usage: qsub -t 1-N launch_eddie.sh PY_SCRIPT MANIFEST}"
MANIFEST="${2:?Usage: qsub -t 1-N launch_eddie.sh PY_SCRIPT MANIFEST}"

# --- Environment -----------------------------------------------------------
. /etc/profile.d/modules.sh
module load anaconda
eval "$(conda shell.bash hook)"
conda activate diffpat
unset PYTHONHOME PYTHONPATH

# --- Paths -----------------------------------------------------------------
CODE_ROOT="$(pwd)"                                   # -cwd = repo root
GROUP_ROOT="/exports/cmvm/eddie/scs/groups/schumacher-group/chyildirim"

export PVC_PATH="$CODE_ROOT/"
export DATA_PATH_BASE="${DATA_PATH_BASE:-$CODE_ROOT/demo/demo_data/}"   # expects .../Emojis/*.png
export MODEL_STORE_ROOT="${MODEL_STORE_ROOT:-$GROUP_ROOT/Models/}"

# Grid Engine task IDs start at 1, manifest indices at 0.
INDEX=$((SGE_TASK_ID - 1))
export JOB_WORKER_INDEX=0
export JOB_WORKER_COUNT=1
export JOB_COMPLETION_INDEX="$INDEX"

# W&B: keep local files off $HOME. If compute nodes cannot reach the internet,
# set WANDB_MODE=offline and later run `wandb sync` on a login node.
export WANDB_DIR="${WANDB_DIR:-$GROUP_ROOT/wandb/${JOB_ID}_${SGE_TASK_ID}}"
export WANDB_CACHE_DIR="$WANDB_DIR/cache"
export WANDB_DATA_DIR="$WANDB_DIR/data"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

mkdir -p "$MODEL_STORE_ROOT" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"

echo "Host: $(hostname)  Job: $JOB_ID  Task: $SGE_TASK_ID -> manifest index $INDEX"
nvidia-smi || true
python -c "import jax; print('jax.devices():', jax.devices())"

python -X faulthandler "$PY_SCRIPT" --manifest "$MANIFEST" --index "$INDEX"
