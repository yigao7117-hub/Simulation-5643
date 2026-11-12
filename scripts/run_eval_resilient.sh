#!/bin/bash
# Re-run eval_distractors.py (which resumes from episodes.csv) until all episodes are done
# (max 40 process launches; the eval exits after every real cube drop).
# Every launch is logged to <out_dir>/attempts.csv (exit code, episodes before/after, crash = no progress).
# Usage: scripts/run_eval_resilient.sh <log> <eval args...>   (must include --num_episodes, --run_name, --condition)
cd "$(dirname "$0")/.."
log=$1; shift
source ~/isaac/activate.sh
n=$(grep -o -- '--num_episodes [0-9]*' <<<"$*" | awk '{print $2}')
run=$(grep -o -- '--run_name [^ ]*' <<<"$*" | awk '{print $2}')
c=$(grep -o -- '--condition [^ ]*' <<<"$*" | awk '{print $2}')
dir=results/$run/eval_$c; f=$dir/episodes.csv; mkdir -p "$dir"
count() { [ -f "$f" ] && echo $(( $(wc -l < "$f") - 1 )) || echo 0; }
[ -f "$dir/attempts.csv" ] || echo "attempt,start,end,exit_code,episodes_before,episodes_after" > "$dir/attempts.csv"
for attempt in $(seq 1 40); do
  before=$(count); start=$(date '+%F %T')
  echo "=== attempt $attempt $start" >> "$log"
  EVAL_ATTEMPT=$attempt python scripts/eval_distractors.py "$@" >> "$log" 2>&1; rc=$?
  after=$(count)
  echo "$attempt,$start,$(date '+%F %T'),$rc,$before,$after" >> "$dir/attempts.csv"
  [ "$after" -ge "$n" ] && { echo "=== complete $after/$n" >> "$log"; exit 0; }
  sleep 15
done
exit 1
