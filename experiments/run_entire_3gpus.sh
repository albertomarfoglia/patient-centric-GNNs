#!/bin/bash
# Move the MIMIC "entire" RGCN runs from 1 GPU (0) to 3 GPUs (0,2,3).
# Waits for the run in progress to finish, stops the single-GPU pcgnn process
# (the revision pipeline then just logs status/summarize), and relaunches
# --filter __entire__ on GPUs 0,2,3 as ONE pcgnn process (one queue, no run
# executed twice). Done runs are skipped; an interrupted run is resumed.
cd /mnt/storage/workspace/patient-centric-GNNs
LOG=/home/ubuntu/workspace/meds-to-owl-examples/MIMIC/exp-1k/logs
PY=/mnt/storage/conda/envs/gnn/bin/python
ts() { echo "[$(date '+%F %T')] $*" | tee -a $LOG/pipeline.log; }
main=$(pgrep -f "pcgnn run experiments/mimic_rerun.yaml --filter __entire__ --gpus 0$")
[ -n "$main" ] || { ts "entire-3gpu: single-GPU process not found"; }
running=$(grep -l '"running"' runs/mimic_rerun/*__entire__*/status.json)
ts "entire-3gpu: waiting for $(echo $running | xargs -n1 dirname | xargs -n1 basename | tr '\n' ' ')to finish"
while grep -q '"running"' $running 2>/dev/null; do sleep 5; done
if [ -n "$main" ]; then
  kids=$(pgrep -P $main); kill $main $kids 2>/dev/null
  for i in $(seq 60); do kill -0 $main 2>/dev/null || break; sleep 1; done
  kill -9 $main $kids 2>/dev/null
  ts "entire-3gpu: stopped single-GPU process $main"
fi
sleep 5
ts "RGCN entire (GPUs 0,2,3)"
systemd-run --user --scope -q -p MemoryMax=20G -p MemorySwapMax=0 \
  $PY -m pcgnn run experiments/mimic_rerun.yaml --filter '__entire__' --gpus 0,2,3 > $LOG/gnn_entire_3gpus.log 2>&1 \
  || ts "entire (3 GPUs): some runs failed (see pcgnn status)"
$PY -m pcgnn status experiments/mimic_rerun.yaml >> $LOG/pipeline.log 2>&1
$PY -m pcgnn summarize experiments/mimic_rerun.yaml >> $LOG/pipeline.log 2>&1
ts "entire-3gpu: ALLDONE"
