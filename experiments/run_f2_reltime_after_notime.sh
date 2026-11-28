#!/bin/bash
# F2 follow-up 3 (2026-11-28): mimic_f2_reltime (literal_mode scaled_relative).
# Starts once mimic_f2_notime has finished (run_f2_notime_after_entire.sh logs
# "F2-notime: ALLDONE"). 2 workers (GPUs 1,3, ~10 GB each) under a 26G cap; after
# a memory kill, continue on 1 GPU (16G). Done runs are skipped.
# Pre-launch data check (55/55 cohorts: ids, labels, no event after the
# prediction time) passed on 2026-11-28; preprocessing re-checks leakage per run.
cd /mnt/storage/workspace/patient-centric-GNNs
LOG=/home/ubuntu/workspace/meds-to-owl-examples/MIMIC/exp-1k/logs
PY=/mnt/storage/conda/envs/gnn/bin/python
ts() { echo "[$(date '+%F %T')] $*" | tee -a $LOG/pipeline.log; }

until grep -q "F2-notime: ALLDONE" $LOG/pipeline.log; do sleep 120; done
GPUS=1,3; cap=26G
for attempt in 1 2 3 4; do
  ts "F2-reltime attempt $attempt on GPUs $GPUS (cap $cap)"
  if systemd-run --user --scope -q -p MemoryMax=$cap -p MemorySwapMax=0 \
       $PY -m pcgnn run experiments/mimic_f2_reltime.yaml --gpus $GPUS --retry-failed \
       >> $LOG/gnn_f2_reltime.log 2>&1; then
    break
  fi
  ts "F2-reltime attempt $attempt exited non-zero (memory kill?); next attempt on 1 GPU"
  GPUS=1; cap=16G
done
$PY -m pcgnn status experiments/mimic_f2_reltime.yaml >> $LOG/pipeline.log 2>&1
$PY -m pcgnn summarize experiments/mimic_f2_reltime.yaml >> $LOG/pipeline.log 2>&1
ts "F2-reltime: ALLDONE"
