# pipelines/preprocess_pipeline.py

from pathlib import Path

from configs.experiment import ExperimentConfig
from generation.preprocess_lazy import preprocess_meds_kg

from utils.ontologies import (
    EXTERNAL_ONTOLOGIES,
    load_ontology_ancestors_stream,
    load_mimic_onto_concepts,
)

def run_preprocess_pipeline(
    dataset_cfg,
    exp_cfg: ExperimentConfig,
    bioportal_apikey: str | None = None,
):
    """
    Generate and preprocess KG datasets.
    """

    external_graph_paths: list[Path] = []

    # ----------------------------
    # Load ontology enrichments
    # ----------------------------
    if bioportal_apikey is not None:

        for onto, url in EXTERNAL_ONTOLOGIES.items():

            path = load_ontology_ancestors_stream(
                onto_code=onto,
                onto_url=url,
                apikey=bioportal_apikey,
                output_dir=Path("processed_data"),
                childs_concepts=load_mimic_onto_concepts(onto),
            )

            if path is not None:
                external_graph_paths.append(path)

    # ----------------------------
    # Generate dataset samples
    # ----------------------------
    for idx in range(exp_cfg.dataset_samples):
        if exp_cfg.data_mode.data_model == "meds":
            preprocess_meds_kg(
                dcfg=dataset_cfg.generate(idx, exp_cfg),
                ecfg=exp_cfg,
            )