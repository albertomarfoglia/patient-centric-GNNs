# pipelines/train_pipeline.py

from models.node_pred_gnn import run_gnn


def run_train_pipeline(
    dataset_cfg,
    model_cfg,
    exp_cfg,
):

    for idx in range(exp_cfg.dataset_samples):
        run_gnn(
            num_patients=dataset_cfg.num_patients,
            mcfg=model_cfg,
            loader=dataset_cfg.generate(idx, exp_cfg),
            excfg=exp_cfg,
        )
