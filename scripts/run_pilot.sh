#!/bin/bash
# Full pilot: 4 conditions x N paired episodes, two parallel queues, then analysis + media.
# Usage: scripts/run_pilot.sh <checkpoint.pth> [run_name=pilot] [num_episodes=20]
set -u
cd "$(dirname "$0")/.."
CK=$1; RUN=${2:-pilot}; N=${3:-20}
mkdir -p logs
q() { for c in "$@"; do
  scripts/run_eval_resilient.sh logs/${RUN}_$c.log --condition $c --checkpoint "$CK" --num_episodes $N --run_name $RUN
done; }
q C0 C1 & sleep 30; q C2 C3 & wait
source ~/isaac/activate.sh
python3 scripts/analyze.py results/$RUN | tee logs/${RUN}_analyze.log
for s in 1000 1001 1003; do python scripts/make_media.py --run $RUN --seed $s >> logs/${RUN}_media.log 2>&1; done
