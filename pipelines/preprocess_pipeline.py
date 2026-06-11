# pipelines/preprocess_pipeline.py

from pathlib import Path

from configs.experiment import ExperimentConfig
from configs.loader import LoaderConfig
from generation.preprocess_lazy import preprocess_meds_kg, preprocess_sphn_kg

from utils.ontologies import (
    EXTERNAL_ONTOLOGIES,
    load_ontology_ancestors_stream,
    load_mimic_onto_concepts,
)


def import_ontologies(bioportal_apikey: str, loader: LoaderConfig):
    external_graph_paths: list[Path] = []

    for onto, url in EXTERNAL_ONTOLOGIES.items():
        child_nodes = load_mimic_onto_concepts(onto, loader.onto_codes)
        path = load_ontology_ancestors_stream(
            onto_code=onto,
            onto_url=url,
            apikey=bioportal_apikey,
            output_dir=loader.sample_processed_dir,
            childs_concepts=child_nodes,
        )

        if path is not None:
            external_graph_paths.append(path)

    return external_graph_paths


def run_preprocess_pipeline(
    dataset_cfg,
    exp_cfg: ExperimentConfig,
    bioportal_apikey: str | None = None,
    text_model = None
):
    """
    Generate and preprocess KG datasets.
    """

    for idx in range(exp_cfg.dataset_samples):
        loader = dataset_cfg.generate(idx, exp_cfg)

        if bioportal_apikey is not None:
            exp_cfg.enrich_by_graphs = import_ontologies(bioportal_apikey, loader)

        if exp_cfg.data_mode.data_model == "meds":
            preprocess_meds_kg(
                dcfg=loader,
                ecfg=exp_cfg,
                text_model=text_model
            )
        elif exp_cfg.data_mode.data_model == "sphn_pc":
            preprocess_sphn_kg(
                dcfg=loader,
                ecfg=exp_cfg,
                text_model=text_model
            )
