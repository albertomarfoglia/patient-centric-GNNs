from pathlib import Path

from configs.datasets.neurovasc import NeurovascConfig
from configs.formats.SPHNFormat import SPHNFormat
from configs.formats.MEDSFormat import MEDSFormat
from configs.model import ModelConfig
from configs.experiment import ExperimentConfig

from generation.sphn_generation import gen_sphn_kg
from models.multiclass.rgcn import RGCNNet
from models.multiclass.gcn import GCNNet
from pipelines.preprocess_pipeline import run_preprocess_pipeline
from pipelines.train_pipeline import run_train_pipeline


MODEL_GRID = {
    "rgcnet": RGCNNet,
    "gcnet": GCNNet,
}

FORMAT_GRID = {
    "sphn": SPHNFormat(),
    "meds": MEDSFormat(),
}


def main():

    model_cfg = ModelConfig()

    dataset_cfg = NeurovascConfig(
        source_dir=Path("../meds-to-owl-examples/exports"),
        num_patients=503,
        name="neurovasc_v2",
        task="stroke-outcome2",
        classes=["DOMICILE", "REEDUC_TRANSFERT", "DECES"],
    )

    for model_name, model_cls in MODEL_GRID.items():
        for format_name, format_cfg in FORMAT_GRID.items():

            print(f"\n=== Running {model_name} with {format_name} ===\n")

            exp_cfg = ExperimentConfig(
                folds=5,
                dataset_samples=1,
                time_option="TS",
                include_text=False,
                data_mode=format_cfg,
                model_type=model_cls,
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
