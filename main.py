from pathlib import Path

from configs.datasets.mimic import MimicConfig
from configs.formats.MEDSFormat import MEDSFormat
from configs.model import ModelConfig
from configs.experiment import ExperimentConfig

from pipelines.preprocess_pipeline import run_preprocess_pipeline
from pipelines.train_pipeline import run_train_pipeline
from utils.ontologies import MIMIC_ENHANCER_DICT


def main():
    model_cfg = ModelConfig()

    # exp1
    # exp_cfg = ExperimentConfig(
    #     folds = 5,
    #     dataset_samples = 5,
    #     time_option = "TS",
    #     include_text = True,
    #     enrich_events = MIMIC_ENHANCER_DICT,
    #     enrich_by_graphs = [],
    #     data_mode = MEDSFormat()
    # )

    # exp2
    exp_cfg = ExperimentConfig(
        folds=10,
        dataset_samples=5,
        time_option="TS",
        include_text=True,
        data_mode=MEDSFormat(),
    )

    for task in ["los_in_hospital_24h", "los_in_icu_24h", "mortality_in_hospital_48h", "mortality_in_icu_48h"]:
        
        dataset_cfg = MimicConfig(
            source_dir=Path("/home/ubuntu/workspace/meds-to-owl-examples/exports"),
            num_patients=1800,
            task=task
        )

        run_preprocess_pipeline(
            dataset_cfg, exp_cfg, bioportal_apikey="8b5b7825-538d-40e0-9e9e-5ab9274a9aeb"
        )

        run_train_pipeline(
            dataset_cfg,
            model_cfg,
            exp_cfg,
        )


if __name__ == "__main__":
    main()
