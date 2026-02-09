import argparse
from pathlib import Path

from generation.sphn_generation import gen_sphn_kg
from generation.meds_generator import gen_meds_kg
from models.node_pred_rgcn import run_rgcn
from generation.preprocess_data import preprocess_kg
from utils import NEUROVASC_ENHANCER_DICT, load_ontolgy_ancestors, ATC_BIOPORTAL_URL, NEUROVASC_ATC_URIS

parser = argparse.ArgumentParser()
parser.add_argument('--num_patients', type=int, default=10000, help='number of patients')
parser.add_argument('--time_opt', type=str, default='TS', choices=['NT', 'TS', 'TR', 'TS_TR'], help='time information option')
parser.add_argument('--folds', type=int, default=10, help='number of folds for CV')
parser.add_argument('--dr', type=float, default=0.0, help='dropout rate')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
parser.add_argument('--embed_dim', type=int, default=32, help='embedding dimension')
parser.add_argument('--hidden_dim', type=int, default=32, help='hidden dimension of the model')
parser.add_argument('--data_model', type=str, default="meds", help='data model used for representing the kg')
parser.add_argument('--bio_portal_apikey', type=str, default="8b5b7825-538d-40e0-9e9e-5ab9274a9aeb", help='data model used for representing the kg')
args = parser.parse_args()

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = Path(f"{PROJECT_ROOT}/data")
PREPROCESS_DIR = Path(f"{PROJECT_ROOT}/processed_data")

if __name__ == "__main__":
    #gen_sphn_kg(args.num_patients, args.time_opt, data_path=Path(f"{PROJECT_ROOT}/data/syn_data_{args.num_patients}.csv"))

    atc_graph = load_ontolgy_ancestors(
        output_path=f"{PREPROCESS_DIR}/atc_graph.ttl", 
        concepts=NEUROVASC_ATC_URIS, 
        ontology_url=ATC_BIOPORTAL_URL, 
        apikey=args.bio_portal_apikey
    )

    gen_meds_kg(
        in_graph_path=f"{SOURCE_DIR}/meds_{args.time_opt}_{args.num_patients}_backup.nt",
        out_graph_path=f"{SOURCE_DIR}/meds_{args.time_opt}_{args.num_patients}.nt",
        enrich_events=NEUROVASC_ENHANCER_DICT,
        enrich_by_graphs=[atc_graph]
    )

    preprocess_kg(
        num_patients=args.num_patients,
        input_dir=SOURCE_DIR,
        output_dir=PREPROCESS_DIR,
        data_model=args.data_model,
        time_opt=args.time_opt,
    )
    
    run_rgcn(
        num_patients=args.num_patients, 
        folds=args.folds, 
        timeOpt=args.time_opt, 
        dr=args.dr, 
        lr=args.lr, 
        wd=args.wd, 
        embed_dim=args.embed_dim, 
        hidden_dim=args.hidden_dim,
        data_model=args.data_model,
    )

    