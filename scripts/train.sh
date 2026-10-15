#!/bin/bash
# Fine-tune SmolVLA (from lerobot/smolvla_base) on one of our datasets.
# usage: scripts/train.sh <run name> <dataset name> [seed=1000] [steps=6000]
# ~2 h on an RTX 4060 laptop GPU (8 GB) at batch 16.
set -eo pipefail
WORK=${P2R_WORK:-$HOME/humanoid/work}
run=$1; ds=$2; seed=${3:-1000}; steps=${4:-6000}
rm -rf "$WORK/runs/$run"
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.empty_cameras=1 \
  --dataset.repo_id=local/$ds \
  --dataset.root="$WORK/ds_$ds" \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
  --batch_size=16 \
  --steps=$steps \
  --save_freq=1000 \
  --log_freq=100 \
  --seed=$seed \
  --num_workers=2 \
  --output_dir="$WORK/runs/$run" \
  --job_name=$run \
  --wandb.enable=false
