#!/bin/bash
# Plan A (2026-11-27): start the F2 follow-up (mimic_f2_scaled, literal_mode
# separate_scaled) on GPU 1 once the F2 ablation has finished. The entire runs
# keep running on GPU 2 alone (run_entire_resilient.sh, tmux entire2).
# Memory: entire worker ~12 GB (21G cap) + this job ~10 GB on a 31 GB machine,
# the same load as entire + F2 ablation today. Capped at 16G so a memory kill
# hits this job, not the entire runs; it then retries (done runs are skipped).
cd /mnt/storage/workspace/patient-centric-GNNs
LOG=/home/ubuntu/workspace/meds-to-owl-examples/MIMIC/exp-1k/logs
PY=/mnt/storage/conda/envs/gnn/bin/python
ts() { echo "[$(date '+%F %T')] $*" | tee -a $LOG/pipeline.log; }

until grep -q "F2 ablation end" $LOG/pipeline.log; do sleep 60; done
sleep 30
for attempt in 1 2 3; do
  ts "F2-scaled attempt $attempt on GPU 1 (cap 16G)"
  if systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 \
       $PY -m pcgnn run experiments/mimic_f2_scaled.yaml --gpus 1 --retry-failed \
       >> $LOG/gnn_f2_scaled.log 2>&1; then
    break
  fi
  ts "F2-scaled attempt $attempt exited non-zero (memory kill?)"
done
$PY -m pcgnn status experiments/mimic_f2_scaled.yaml >> $LOG/pipeline.log 2>&1
$PY -m pcgnn summarize experiments/mimic_f2_scaled.yaml >> $LOG/pipeline.log 2>&1
ts "F2-scaled: ALLDONE"
