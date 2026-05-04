from pathlib import Path

from configs.datasets.neurovasc import NeurovascConfig
from configs.formats.SPHNFormat import SPHNFormat
from configs.formats.MEDSFormat import MEDSFormat
from configs.model import ModelConfig
from configs.experiment import ExperimentConfig

from generation.sphn_generation import gen_sphn_kg
from models.multiclass.rgcn import RGCNNet
from pipelines.preprocess_pipeline import run_preprocess_pipeline
from pipelines.train_pipeline import run_train_pipeline


def main():
    model_cfg = ModelConfig()

    exp_cfg = ExperimentConfig(
        folds=5,
        dataset_samples=1,  # exp["num_of_samples"],
        time_option="TS",
        include_text=False,
        data_mode=SPHNFormat(),
        model_type=RGCNNet.__name__
        # enrich_events = MIMIC_ENHANCER_DICT,
    )

    # dataset_cfg = NeurovascConfig(
    #     source_dir=Path("/home/ubuntu/workspace/meds-to-owl-examples/exports"),
    # )

    dataset_cfg = NeurovascConfig(
        source_dir=Path("data"),
    )

    gen_sphn_kg(
        dataset_cfg.num_patients,
        exp_cfg.time_option,
        data_path="https://raw.githubusercontent.com/TeamHeKA/neurovasc/refs/heads/main/exp/data/syn_data_10000.csv",
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
