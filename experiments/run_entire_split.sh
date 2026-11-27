#!/bin/bash
# MIMIC "entire" RGCN runs, split by memory need (2026-11-27 20:30).
# run_entire_resilient.sh got stuck: preprocessing the 30d post-discharge entire
# graphs (4.7-5.4 GB N-Triples) needs > 21 GB RSS and was memory-killed on every
# attempt. Phase 1 runs the 60 smaller graphs (<= 3.6 GB) under the usual 21G cap
# while the F2 jobs use GPU 1. Phase 2 runs the 7 large ones after the F2-scaled
# job has finished, with cap = MemAvailable - 3 GB (max 27G).
# Done runs are skipped; interrupted ones are resumed.
cd /mnt/storage/workspace/patient-centric-GNNs
LOG=/home/ubuntu/workspace/meds-to-owl-examples/MIMIC/exp-1k/logs
PY=/mnt/storage/conda/envs/gnn/bin/python
SMALL='^(?!post_hospital_discharge_mortality_30d__).*__entire__'
LARGE='^post_hospital_discharge_mortality_30d__.*__entire__'
ts() { echo "[$(date '+%F %T')] $*" | tee -a $LOG/pipeline.log; }
left() { $PY -m pcgnn run experiments/mimic_rerun.yaml --filter "$1" --gpus 2 --dry-run --retry-failed 2>/dev/null \
         | grep -cE '^[A-Za-z0-9_.-]+__entire__s[0-9]+$'; }
run_phase() {  # name regex cap
  for attempt in 1 2 3; do
    n=$(left "$2"); [ "${n:-1}" -eq 0 ] && return 0
    ts "entire-$1 attempt $attempt on GPU 2, cap $3 ($n runs left)"
    systemd-run --user --scope -q -p MemoryMax=$3 -p MemorySwapMax=0 \
      $PY -m pcgnn run experiments/mimic_rerun.yaml --filter "$2" --gpus 2 --retry-failed \
      >> $LOG/gnn_entire_resilient.log 2>&1 && return 0
    ts "entire-$1 attempt $attempt exited non-zero (memory kill?)"
  done
}

run_phase small "$SMALL" 21G

until grep -q "F2-scaled: ALLDONE" $LOG/pipeline.log; do sleep 120; done
avail_gb=$(( $(grep MemAvailable /proc/meminfo | awk '{print $2}') / 1048576 ))
cap=$(( avail_gb - 3 )); [ $cap -gt 27 ] && cap=27
ts "entire-large: MemAvailable=${avail_gb}G -> cap ${cap}G"
run_phase large "$LARGE" ${cap}G

$PY -m pcgnn status experiments/mimic_rerun.yaml >> $LOG/pipeline.log 2>&1
$PY -m pcgnn summarize experiments/mimic_rerun.yaml >> $LOG/pipeline.log 2>&1
ts "entire-split: ALLDONE"
