# import argparse
# from pathlib import Path

# import numpy as np
# from sentence_transformers import SentenceTransformer
# import torch
# from torch.nn import Linear

# from generation.meds_generator import gen_meds_kg
# from generation.preprocess_lazy import preprocess_kg
# #from generation.preprocess_data import preprocess_kg
# from generation.sphn_generation import gen_sphn_kg
# from models.node_pred_rgcn import run_rgcn, LoaderConfig
# from utils.ontologies import (
#     ATC_BIOPORTAL_URL,
#     ICD10PCS,
#     ICD10PCS_BIOPORTAL_URL,
#     NEUROVASC_ATC_URIS,
#     NEUROVASC_ENHANCER_DICT,
#     MIMIC_ENHANCER_DICT,
#     NEUROVASC_ICD10PCS_URIS,
#     EXTERNAL_ONTOLOGIES,
#     load_ontolgy_ancestors,
#     load_mimic_onto_concepts,
#     load_ontology_ancestors_stream
# )

# parser = argparse.ArgumentParser()
# parser.add_argument("--num_patients", type=int, default=1800, help="number of patients")
# parser.add_argument(
#     "--time_opt",
#     type=str,
#     default="TS",
#     choices=["NT", "TS", "TR", "TS_TR"],
#     help="time information option",
# )
# parser.add_argument("--folds", type=int, default=5, help="number of folds for CV")
# parser.add_argument("--dr", type=float, default=0.0, help="dropout rate")
# parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
# parser.add_argument("--wd", type=float, default=5e-4, help="weight decay")
# parser.add_argument("--embed_dim", type=int, default=32, help="embedding dimension")
# parser.add_argument(
#     "--hidden_dim", type=int, default=32, help="hidden dimension of the model"
# )
# parser.add_argument(
#     "--data_model",
#     type=str,
#     default="meds",
#     help="data model used for representing the kg",
# )
# parser.add_argument(
#     "--bio_portal_apikey",
#     type=str,
#     default="8b5b7825-538d-40e0-9e9e-5ab9274a9aeb",
#     help="data model used for representing the kg",
# )
# args = parser.parse_args()

# PROJECT_ROOT = Path(__file__).resolve().parent
# SOURCE_DIR = Path(f"{PROJECT_ROOT}/data")
# PREPROCESS_DIR = Path(f"{PROJECT_ROOT}/processed_data")

# if __name__ == "__main__":
#     # gen_sphn_kg(args.num_patients, args.time_opt, data_path=Path(f"{PROJECT_ROOT}/data/syn_data_{args.num_patients}.csv"))

#     # atc_graph = load_ontolgy_ancestors(
#     #     output_path=f"{PREPROCESS_DIR}/atc_graph.ttl",
#     #     concepts=NEUROVASC_ATC_URIS,
#     #     ontology_url=ATC_BIOPORTAL_URL,
#     #     apikey=args.bio_portal_apikey
#     # )

#     # icd_graph = load_ontology_ancestors_stream(
#     #     output_path=f"{PREPROCESS_DIR}/mimic_icd_graph.ttl",
#     #     concepts=MIMIC_ICD10PCS_URIS,
#     #     ontology_url=ICD10PCS_BIOPORTAL_URL,
#     #     apikey=args.bio_portal_apikey
#     # )

#     # gen_meds_kg(
#     #     in_graph_path=f"{SOURCE_DIR}/meds_{args.time_opt}_{args.num_patients}_backup.nt",
#     #     out_graph_path=f"{SOURCE_DIR}/meds_{args.time_opt}_{args.num_patients}.nt",
#     #     enrich_events=MIMIC_ENHANCER_DICT,
#     #     #enrich_by_graphs=[atc_graph]
#     # )

#     # preprocess_kg(
#     #     num_patients=args.num_patients,
#     #     input_dir=SOURCE_DIR,
#     #     output_dir=PREPROCESS_DIR,
#     #     data_model=args.data_model,
#     #     time_opt=args.time_opt,
#     # )

#     # external_graph_paths: list[Path] = []

#     # for (onto, url) in EXTERNAL_ONTOLOGIES.items():
#     #     path = load_ontology_ancestors_stream(
#     #         onto_code=onto,
#     #         onto_url=url,
#     #         apikey=args.bio_portal_apikey,
#     #         output_dir=PREPROCESS_DIR,
#     #         childs_concepts=load_mimic_onto_concepts(onto)
#     #     )

#     #     if path is not None:
#     #         external_graph_paths.append(path)

#     dataset_samples = 5

#     for idx in range(dataset_samples):

#         MEDS_EXPORT_PATH = "/home/ubuntu/workspace/meds-to-owl-examples/exports"
#         TASK = "inhospital_mortality"
            
#         preprocess_kg(
#             args.num_patients,
#             input_dir=Path(f"{MEDS_EXPORT_PATH}/{TASK}/meds_{args.num_patients}_{idx}"),
#             output_dir=Path("processed_data"),
#             enrich_events=MIMIC_ENHANCER_DICT,
#             enrich_by_graphs=[],#external_graph_paths,
#             data_model="meds",
#             time_opt="TS",
#             inc_txt=True,
#             idd=idx
#         )

#         run_rgcn(
#             num_patients=args.num_patients,
#             folds=args.folds,
#             time_opt=args.time_opt,
#             dr=args.dr,
#             lr=args.lr,
#             wd=args.wd,
#             embed_dim=args.embed_dim,
#             hidden_dim=args.hidden_dim,
#             dcfg=data_config
#         )