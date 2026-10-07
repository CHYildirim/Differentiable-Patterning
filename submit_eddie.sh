#!/usr/bin/env bash
# Submit one Eddie array job for every entry of a generated manifest.
# Usage (repo root, Eddie login node):
#   bash submit_eddie.sh generated/cluster_test/manifest.yaml [max_parallel]
set -euo pipefail
MANIFEST="${1:?Usage: bash submit_eddie.sh MANIFEST [max_parallel]}"
MAX_PARALLEL="${2:-4}"
[[ -f "$MANIFEST" ]] || { echo "Manifest not found: $MANIFEST"; exit 1; }

N=$(grep -m1 '^count:' "$MANIFEST" | awk '{print $2}')
NAME=$(grep -m1 '^experiment_name:' "$MANIFEST" | awk '{print $2}')
[[ "$N" =~ ^[0-9]+$ && "$N" -gt 0 ]] || { echo "Bad count in manifest: $N"; exit 1; }

mkdir -p logs/eddie
echo "Submitting $NAME: tasks 1-$N (max $MAX_PARALLEL parallel)"
qsub -N "${NAME:-nca_array}" -t "1-$N" -tc "$MAX_PARALLEL" \
    launch_eddie.sh Experiments/run_config.py "$MANIFEST"
