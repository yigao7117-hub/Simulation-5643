#!/bin/bash
# Evaluate all conditions one after another (sequential = clean inference timing), then analysis + media.
# Usage: scripts/run_all.sh <checkpoint.pth> [run_name=main] [num_episodes=50] [conditions="C0 C1 C2 C3 C4 C5"]
set -u
cd "$(dirname "$0")/.."
CK=$1; RUN=${2:-main}; N=${3:-50}; CONDS=${4:-"C0 C1 C2 C3 C4 C5"}
mkdir -p logs
for c in $CONDS; do
  scripts/run_eval_resilient.sh logs/${RUN}_$c.log --condition $c --checkpoint "$CK" --num_episodes $N --run_name $RUN
done
source ~/isaac/activate.sh
python3 scripts/analyze.py results/$RUN > logs/${RUN}_analyze.log 2>&1
for s in 1000 1001 1003; do python scripts/make_media.py --run $RUN --seed $s >> logs/${RUN}_media.log 2>&1; done
echo ALL_DONE >> logs/${RUN}_analyze.log
