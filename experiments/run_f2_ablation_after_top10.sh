#!/usr/bin/env bash
# Starts the F2 ablation on GPU 1 once the mimic_rerun top10 batch has finished
# (RAM: 4 top10 workers leave ~1 GB free; afterwards only the 1-GPU entire stage runs).
LOG=/home/ubuntu/workspace/meds-to-owl-examples/MIMIC/exp-1k/logs
until grep -q "RGCN entire" $LOG/pipeline.log; do sleep 60; done
sleep 120   # let the entire stage load its graph first
cd /mnt/storage/workspace/patient-centric-GNNs
echo "[$(date '+%F %T')] F2 ablation start" >> $LOG/pipeline.log
/mnt/storage/conda/envs/gnn/bin/python -m pcgnn run experiments/mimic_f2_ablation.yaml --gpus 1 > $LOG/gnn_f2_ablation.log 2>&1
echo "[$(date '+%F %T')] F2 ablation end (exit $?)" >> $LOG/pipeline.log
