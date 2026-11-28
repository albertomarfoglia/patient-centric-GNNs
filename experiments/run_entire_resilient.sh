#!/bin/bash
# MIMIC "entire" RGCN runs, resilient to memory kills.
# Entire graphs need up to ~9 GB RAM per worker (31 GB machine, F2 ablation running):
# 2 workers (GPUs 2,3) under a 21 GB cap; after an OOM kill, resume on 1 GPU.
# pcgnn skips done runs and resumes interrupted ones (same config hash).
cd /mnt/storage/workspace/patient-centric-GNNs
LOG=/home/ubuntu/workspace/meds-to-owl-examples/MIMIC/exp-1k/logs
PY=/mnt/storage/conda/envs/gnn/bin/python
ts() { echo "[$(date '+%F %T')] $*" | tee -a $LOG/pipeline.log; }
remaining() { $PY -m pcgnn status experiments/mimic_rerun.yaml 2>/dev/null | grep -oE "pending=[0-9]+|running=[0-9]+|failed=[0-9]+" | cut -d= -f2 | paste -sd+ | bc; }
GPUS=2,3
for attempt in 1 2 3 4 5 6; do
  n=$(remaining); [ "${n:-1}" -eq 0 ] && break
  ts "RGCN entire attempt $attempt on GPUs $GPUS ($n runs left)"
  if systemd-run --user --scope -q -p MemoryMax=21G -p MemorySwapMax=0 \
       $PY -m pcgnn run experiments/mimic_rerun.yaml --filter '__entire__' --gpus $GPUS --retry-failed \
       >> $LOG/gnn_entire_resilient.log 2>&1; then
    :
  else
    ts "entire attempt $attempt exited non-zero (memory kill?); next attempt on 1 GPU"
    GPUS=2
  fi
done
$PY -m pcgnn status experiments/mimic_rerun.yaml >> $LOG/pipeline.log 2>&1
$PY -m pcgnn summarize experiments/mimic_rerun.yaml >> $LOG/pipeline.log 2>&1
ts "entire-resilient: ALLDONE (remaining=$(remaining))"
