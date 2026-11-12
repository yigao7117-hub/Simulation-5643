#!/bin/bash
# Wait until no other user's process is on the GPU for IDLE_MIN consecutive minutes, then run
# scripts/run_all.sh with the given args. While running, samples GPU usage of other users every 30 s
# into logs/<run>_gpu_others.csv so contended episodes can be identified afterwards.
# Usage: scripts/run_when_idle.sh <checkpoint.pth> <run_name> [num_episodes] [conditions]
cd "$(dirname "$0")/.."
IDLE_MIN=${IDLE_MIN:-10}; RUN=$2; me=$(id -un)
others() {  # other users' GPU memory in MiB
  local tot=0
  for line in $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | tr -d ' '); do
    pid=${line%,*}; mem=${line#*,}
    [ "$(ps -o user= -p $pid 2>/dev/null)" != "$me" ] && tot=$((tot + mem))
  done; echo $tot
}
echo "$(date '+%F %T') waiting for idle GPU" >> logs/${RUN}_driver.log
idle=0
while [ $idle -lt $((IDLE_MIN * 2)) ]; do
  [ "$(others)" -eq 0 ] && idle=$((idle + 1)) || idle=0
  sleep 30
done
echo "$(date '+%F %T') GPU idle for $IDLE_MIN min -> start" >> logs/${RUN}_driver.log
echo "time,other_users_gpu_mib,gpu_util" > logs/${RUN}_gpu_others.csv
( while :; do echo "$(date '+%F %T'),$(others),$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)" >> logs/${RUN}_gpu_others.csv; sleep 30; done ) &
sampler=$!
scripts/run_all.sh "$@" >> logs/${RUN}_driver.log 2>&1
kill $sampler
echo "$(date '+%F %T') finished" >> logs/${RUN}_driver.log
