# pipelines/train_pipeline.py

from models.node_pred_rgcn import run_rgcn

def run_train_pipeline(
    dataset_cfg,
    model_cfg,
    exp_cfg,
):

    for idx in range(exp_cfg.dataset_samples):
        run_rgcn(
            num_patients=dataset_cfg.num_patients,
            folds=exp_cfg.folds,
            time_opt=exp_cfg.time_option,
            mcfg=model_cfg,
            loader=dataset_cfg.generate(idx, exp_cfg),
        )
