from pathlib import Path

from configs.datasets.mimic import MimicConfig
from configs.formats.MEDSFormat import MEDSFormat
from configs.model import ModelConfig
from configs.experiment import ExperimentConfig
import yaml

from pipelines.preprocess_pipeline import run_preprocess_pipeline
from pipelines.train_pipeline import run_train_pipeline


def main():
    model_cfg = ModelConfig(lr=1e-3)

    with open("experiments.yaml", "r") as f:
        exp_config = yaml.safe_load(f)["experiments"]

    for group_name, experiments in exp_config.items():
        print(f"\n=== Group: {group_name} ===")

        for exp in experiments:
            print(f"Running task: {exp['task']}")

            exp_cfg = ExperimentConfig(
                folds=5,
                dataset_samples=exp["num_of_samples"],
                time_option="TS",
                include_text=False,
                data_mode=MEDSFormat(),
                model_type="gcn",
                # enrich_events = MIMIC_ENHANCER_DICT,
            )

            dataset_cfg = MimicConfig(
                source_dir=Path("/home/ubuntu/workspace/meds-to-owl-examples/exports"),
                num_patients=exp["sample_size"],
                task=exp["task"],
            )

            run_preprocess_pipeline(
                dataset_cfg,
                exp_cfg,
                #bioportal_apikey="8b5b7825-538d-40e0-9e9e-5ab9274a9aeb",
            )

            run_train_pipeline(
                dataset_cfg,
                model_cfg,
                exp_cfg,
            )


if __name__ == "__main__":
    main()
