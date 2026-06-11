from pathlib import Path

from configs.datasets.mimic import MimicConfig
from configs.formats.MEDSFormat import MEDSFormat
from configs.model import ModelConfig
from configs.experiment import ExperimentConfig
import yaml
from sentence_transformers import SentenceTransformer
import torch

from models.binary.rgcn import RGCNet
from pipelines.preprocess_pipeline import run_preprocess_pipeline
from pipelines.train_pipeline import run_train_pipeline
#from utils.ontologies import MIMIC_ENHANCER_DICT

def main():
    model_cfg = ModelConfig(lr=5e-3, embed_dim=32)

    with open("experiments.yaml", "r") as f:
        exp_config = yaml.safe_load(f)["experiments"]

    for group_name, experiments in exp_config.items():
        print(f"\n=== Group: {group_name} ===")

        for exp in experiments:
            print(f"Running task: {exp['task']}")

            # import json
            # with open(f"../meds-to-owl-examples/MIMIC/exports/{exp['task']}/onto_features_dict.json", "r") as f:
            #     mimic_enhancer_dict = json.load(f)

            exp_cfg = ExperimentConfig(
                folds=10,
                dataset_samples=exp["num_of_samples"],
                time_option="TS",
                include_text=True,
                data_mode=MEDSFormat(),
                model_type=RGCNet,
                #enrich_events = MIMIC_ENHANCER_DICT,
                #enrich_events=mimic_enhancer_dict
            )

            dataset_cfg = MimicConfig(
                source_dir=Path("../meds-to-owl-examples/MIMIC/exp1-full"),
                labels_dir=Path("../meds-to-owl-examples/MIMIC/exp0-full"),
                num_patients=exp["sample_size"],
                task=exp["task"],
            )

            text_model = None
            if exp_cfg.include_text:
                device = "cuda" if torch.cuda.is_available() else "cpu"

                text_model = SentenceTransformer(
                    "all-MiniLM-L6-v2",
                    device=device,
                    cache_folder="__pycache__",
                )

            run_preprocess_pipeline(
                dataset_cfg,
                exp_cfg,
                #bioportal_apikey="8b5b7825-538d-40e0-9e9e-5ab9274a9aeb",
                text_model=text_model
            )

            run_train_pipeline(
                dataset_cfg,
                model_cfg,
                exp_cfg,
            )


if __name__ == "__main__":
    main()
