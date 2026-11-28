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

    text_model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # RDF schema variant -> directory of the SHAP-pruned (top 10%) graphs used
    # in the paper. Labels are shared and live in exp0-0.9.
    SCHEMA_DIRS = {
        "base": "exp1-0.9",
        "partial": "exp2-0.9",
        "full": "exp3-0.9",
    }
    MIMIC_ROOT = Path("../meds-to-owl-examples/MIMIC")

    text_model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        device=device,
        cache_folder="__pycache__",
    )

    for group_name, experiments in exp_config.items():
        print(f"\n=== Group: {group_name} ===")

        for exp in experiments:
          for schema_mode, schema_dir in SCHEMA_DIRS.items():
            print(f"Running task: {exp['task']} ({schema_mode})")

            # import json
            # with open(f"../onto_features_dict.json", "r") as f:
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

            # Separate processed and results folders per schema, so that
            # Base/Partial/Full and tasks never overwrite each other.
            dataset_cfg = MimicConfig(
                source_dir=MIMIC_ROOT / schema_dir,
                labels_dir=MIMIC_ROOT / "exp0-0.9",
                num_patients=exp["sample_size"],
                task=exp["task"],
                processed_dir=Path(f"processed_data/inference_timing/{schema_mode}"),
                run_name=f"{exp['task']}-{schema_mode.upper()}-inference",
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
