# pipelines/train_pipeline.py

#from models.node_pred_gnn_parallel import run_gnn
from models.node_pred_gnn import run_gnn
from utils.metrics import aggregate_subsample_results, save_aggregated_results

def run_train_pipeline(
    dataset_cfg,
    model_cfg,
    exp_cfg,
):

    subsample_results = []

    for idx in range(exp_cfg.dataset_samples):

        print(
            f"Running subsample "
            f"{idx + 1}/{exp_cfg.dataset_samples}"
        )

        loader = dataset_cfg.generate(idx, exp_cfg)

        result = run_gnn(
            num_patients=dataset_cfg.num_patients,
            mcfg=model_cfg,
            loader=loader,
            excfg=exp_cfg,
        )

        subsample_results.append(result)


    aggregated = aggregate_subsample_results(subsample_results)

    save_aggregated_results(
        aggregated=aggregated,
        result_dir=loader.root_results_dir, # type: ignore
        time_option=exp_cfg.time_option,
        num_patients=dataset_cfg.num_patients,
    )

    return aggregated