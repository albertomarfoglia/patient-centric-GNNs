from pathlib import Path

from sentence_transformers import SentenceTransformer
import torch

from configs.datasets.neurovasc import NeurovascConfig
#from configs.formats.SPHNFormat import SPHNFormat
from configs.formats.MEDSFormat import MEDSFormat
from configs.model import ModelConfig
from configs.experiment import ExperimentConfig

from models.multiclass.rgcn import RGCNNet
from pipelines.preprocess_pipeline import run_preprocess_pipeline
from pipelines.train_pipeline import run_train_pipeline
#from utils.ontologies import NEUROVASC_ENHANCER_DICT


FORMAT_GRID = {
    #"sphn": SPHNFormat(),
    "meds": MEDSFormat(),
}


def main():

    model_cfg = ModelConfig(lr=5e-3, embed_dim=32, hidden_dim=32)

    dataset_cfg = NeurovascConfig(
        source_dir=Path("../meds-to-owl-examples/NEUROVASC2/exports-0.95"),
        num_patients=503,
        name="neurovasc_v2",
        task="stroke-outcome",
        classes=["BackHome", "Rehab", "Death"],
    )

    text_model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text_model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        device=device,
        cache_folder="__pycache__",
    )


    for format_name, format_cfg in FORMAT_GRID.items():

        print(f"\n=== Running rgcnet with {format_name} ===\n")

        exp_cfg = ExperimentConfig(
            folds=10,
            dataset_samples=1,
            time_option="TS",
            include_text=True,
            data_mode=format_cfg,
            model_type=RGCNNet,
            #enrich_events=NEUROVASC_ENHANCER_DICT
        )

        run_preprocess_pipeline(
            dataset_cfg,
            exp_cfg,
            text_model=text_model
            #bioportal_apikey="8b5b7825-538d-40e0-9e9e-5ab9274a9aeb",
        )

        run_train_pipeline(
            dataset_cfg,
            model_cfg,
            exp_cfg,
        )


if __name__ == "__main__":
    main()
