#!/bin/bash
# F2 follow-up 2 (2026-11-28): mimic_f2_notime (literal_mode scaled_no_time).
# Starts once run_entire_split.sh has finished (its large graphs need up to ~22 GB,
# so nothing else runs beside them). Then 2 workers (GPUs 1,3, ~10 GB each) under
# a 26G cap; after a memory kill, continue on 1 GPU (16G). Done runs are skipped.
cd /mnt/storage/workspace/patient-centric-GNNs
LOG=/home/ubuntu/workspace/meds-to-owl-examples/MIMIC/exp-1k/logs
PY=/mnt/storage/conda/envs/gnn/bin/python
ts() { echo "[$(date '+%F %T')] $*" | tee -a $LOG/pipeline.log; }

until grep -q "entire-split: ALLDONE" $LOG/pipeline.log; do sleep 120; done
GPUS=1,3; cap=26G
for attempt in 1 2 3 4; do
  ts "F2-notime attempt $attempt on GPUs $GPUS (cap $cap)"
  if systemd-run --user --scope -q -p MemoryMax=$cap -p MemorySwapMax=0 \
       $PY -m pcgnn run experiments/mimic_f2_notime.yaml --gpus $GPUS --retry-failed \
       >> $LOG/gnn_f2_notime.log 2>&1; then
    break
  fi
  ts "F2-notime attempt $attempt exited non-zero (memory kill?); next attempt on 1 GPU"
  GPUS=1; cap=16G
done
$PY -m pcgnn status experiments/mimic_f2_notime.yaml >> $LOG/pipeline.log 2>&1
$PY -m pcgnn summarize experiments/mimic_f2_notime.yaml >> $LOG/pipeline.log 2>&1
ts "F2-notime: ALLDONE"
