#!/bin/bash
# Re-run eval_distractors.py (which resumes from episodes.csv) until all episodes are done (max 40 process launches; the eval exits after every real cube drop).
# Usage: scripts/run_eval_resilient.sh <log> <eval args...>   (must include --num_episodes and --run_name, --condition)
cd "$(dirname "$0")/.."
log=$1; shift
source ~/isaac/activate.sh
for attempt in $(seq 1 40); do
  echo "=== attempt $attempt $(date)" >> "$log"
  python scripts/eval_distractors.py "$@" >> "$log" 2>&1; sleep 15
  n=$(grep -o -- '--num_episodes [0-9]*' <<<"$*" | awk '{print $2}')
  run=$(grep -o -- '--run_name [^ ]*' <<<"$*" | awk '{print $2}')
  c=$(grep -o -- '--condition [^ ]*' <<<"$*" | awk '{print $2}')
  f=results/$run/eval_$c/episodes.csv
  done_n=$([ -f "$f" ] && echo $(( $(wc -l < "$f") - 1 )) || echo 0)
  [ "$done_n" -ge "$n" ] && { echo "=== complete $done_n/$n" >> "$log"; exit 0; }
done
exit 1
