from pathlib import Path

from rdflib import Graph, Namespace
import zipfile

import polars as pl

from tqdm import tqdm
from io import BytesIO

import requests

import gzip

NS_DATA = Namespace("https://teamheka.github.io/meds-data/")
NS_ONTO = Namespace("https://teamheka.github.io/meds-ontology#")
NS_CODE = Namespace(f"{NS_DATA}code/")

BIOPORTAL_URI = "http://purl.bioontology.org/ontology"
BIOPORTAL_URL = "https://data.bioontology.org/ontologies"

ATC_BIOPORTAL_URL = f"{BIOPORTAL_URL}/ATC/submissions/23/download"
ICD10PCS_BIOPORTAL_URL = f"{BIOPORTAL_URL}/ICD10PCS/submissions/26/download"

ATC = Namespace(f"{BIOPORTAL_URI}/ATC/")
_NEUROVASC_ATC_CODES = ["C08CA06", "N02BE01", "C01CA03", "C01CE02", "N02AA01"]
NEUROVASC_ATC_URIS = list(
    f"<{c}>" for c in [ATC[code] for code in _NEUROVASC_ATC_CODES]
)

ICD10PCS = Namespace(f"{BIOPORTAL_URI}/ICD10PCS/")
_NEUROVASC_ICD10PCS_CODES = ["0BH17EZ", "00P6X0Z"]
NEUROVASC_ICD10PCS_URIS = list(
    f"<{c}>" for c in [ICD10PCS[code] for code in _NEUROVASC_ICD10PCS_CODES]
)

# _MIMIC_ICD10PCS_CODES = np.load(Path("./generation/mimic_external_codes") / "ICD10PCS_codes.npy", allow_pickle=True)
# MIMIC_ICD10PCS_URIS = list(f"<{c}>" for c in [ICD10PCS[code] for code in _MIMIC_ICD10PCS_CODES])

EXTERNAL_ONTOLOGIES = {
    "ICD10CM": f"{BIOPORTAL_URL}/ICD10CM/submissions/27/download",
    "ICD10PCS": f"{BIOPORTAL_URL}/ICD10PCS/submissions/26/download",
    "LOINC": f"{BIOPORTAL_URL}/LOINC/submissions/28/download",
    "RXNORM": f"{BIOPORTAL_URL}/RXNORM/submissions/28/download",
    "ICD9CM": f"{BIOPORTAL_URL}/ICD9CM/submissions/26/download",
    # "SNOMED": f"{BIOPORTAL_URL}/"
}


def load_mimic_onto_concepts(onto_code: str, external_path: Path):
    onto_uri = Namespace(f"{BIOPORTAL_URI}/")
    file = Path(external_path) / f"{onto_code}_codes.parquet"
    if file.exists():
        # codes = np.load(
        #     Path(external_path) / f"{onto_code}_codes.npy",
        #     allow_pickle=True,
        # )
        codes = pl.read_parquet(file).to_series()
        return list(f"<{c}>" for c in [onto_uri[code] for code in codes])
    return []


NEUROVASC_ENHANCER_DICT: dict[str, str] = {
    NS_CODE.Age_Years: NS_ONTO.hasAge,
    NS_CODE.Gender_M: NS_ONTO.hasMaleGender,
    NS_CODE.Gender_F: NS_ONTO.hasFemaleGender,
    NS_CODE.Number_of_Visited_Departments: NS_ONTO.hasVisitedDept,
    NS_CODE.Length_of_Stay: NS_ONTO.hasLengthOfStay,
    NS_CODE.Glasgow_Coma_Scale: NS_ONTO.hasGlasgowComaScale,
    NS_CODE.WFNS_Score: NS_ONTO.hasWFNSScore,
    NS_CODE.Fisher_Score: NS_ONTO.hasFisherScore,
    NS_CODE.Admission_Department_REA: NS_ONTO.hasAdmissionDepartment,
    NS_CODE.External_Ventricular_Drain_Details_True: NS_CODE.hasExternalVentricular,
    NS_CODE.Weight_Kg: NS_CODE.hasWeight,
    NS_CODE.Smoking_Kg: NS_CODE.hasSmoking,
    # NS_CODE.Hydrocephalus_true: NS_CODE.hasHydrocephalus,
    # NS_CODE.Norepinephrine: NS_CODE.hasNorepinephrine,
    # NS_CODE.Vasospasm_false: NS_CODE.hasVasospasm,
    # NS_CODE.Vasospasm_UNK: NS_CODE.hasVasospasm_UNK,
    # NS_CODE.Seizure_true: NS_CODE.hasSeizure
    # **{ URIRef(NS_CODE + f"ATC_{atc}"): NS_ONTO.hasAdministration for atc in _NEUROVASC_ATC_CODES },
}

MIMIC_ENHANCER_DICT: dict[str, str] = {
    #NS_CODE.MEDS_BIRTH: NS_ONTO.hasBirth,
    NS_CODE.GENDER_M: NS_ONTO.hasMale,
    NS_CODE.GENDER_F: NS_ONTO.hasFemale,
    #NS_CODE.TRANSFER_TO_discharge_UNKNOWN: NS_ONTO.hasUNKDischarge,
    NS_CODE["LAB_51277_%_max"]: NS_ONTO.hasLAB_51277_max,
    NS_CODE["LAB_51277_%_min"]: NS_ONTO.hasLAB_51277_min,
    NS_CODE[
        "MEDICATION_START_Acetaminophen"
    ]: NS_ONTO.MEDICATION_START_Acetaminophen,

    NS_CODE["LAB_51006_mg/dL_max"]: NS_ONTO["LAB_51006_mg/dL_max"],
    NS_CODE["LAB_225624_mg/dL_max"]: NS_ONTO["LAB_225624_mg/dL_max"],
    NS_CODE["LAB_50882_mEq/L_max"]: NS_ONTO["LAB_50882_mEq/L_max"],

    NS_CODE["MEDICATION_START_UNK"]: NS_ONTO["MEDICATION_START_UNK"],
    NS_CODE["LAB_50954_IU/L_max"]: NS_ONTO["LAB_50954_IU/L_max"],
    # NS_CODE["MEDICATION_START_Docusate Sodium"]: NS_ONTO["MEDICATION_START_Docusate_Sodium"],
    # NS_CODE["MEDICATION_START_Lorazepam"]: NS_ONTO["MEDICATION_START_Lorazepam"],
    # NS_CODE["LAB_51265_K/uL_max"]: NS_ONTO["LAB_51265_K/uL_max"],
    # NS_CODE["LAB_51265_K/uL_min"]: NS_ONTO["LAB_51265_K/uL_min"],
    # NS_CODE["LAB_51222_g_dL_min"]: NS_ONTO.LAB_51222_g_dL_min,
    # NS_CODE["DIAGNOSIS_ICD_9_4019"]: NS_ONTO.DIAGNOSIS_ICD_9_4019,
    # NS_CODE["LAB_50868_mEq_L_max"]: NS_ONTO.LAB_50868_mEq_L_max,
    # NS_CODE["LAB_50971_mEq_L_max"]: NS_ONTO.LAB_50971_mEq_L_max,
    # NS_CODE["LAB_50970_mg_dL_min"]: NS_ONTO.LAB_50970_mg_dL_min,
    # NS_CODE["LAB_50983_mEq_L_min"]: NS_ONTO.LAB_50983_mEq_L_min,
    # NS_CODE["LAB_50960_mg_dL_max"]: NS_ONTO.LAB_50960_mg_dL_max,
    # NS_CODE.HOSPITAL_DISCHARGE_UNK: NS_ONTO.hasHospitalDischargeUNK,
    # NS_CODE.TRANSFER_TO_admit_Hematology_Oncology: NS_ONTO.TRANSFER_TO_admit_Hematology_Oncology,
    # NS_CODE.MEDICATION_START_Acetaminophen: NS_ONTO.hasDocusate,
    # NS_CODE.MEDICATION_START_Lorazepam: NS_ONTO.hasLorazepam,
    # NS_CODE["LAB_50902_mEq_L_min"]: NS_ONTO.LAB_50902_mEq_L_min,
    # NS_CODE["LAB_51301_K_uL_min"]: NS_ONTO.LAB_51301_K_uL_min,
}


def ancestors_query(concepts: list[str]):
    return f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

    CONSTRUCT {{
        ?child rdfs:subClassOf ?parent .
        ?child skos:prefLabel ?label .
    }}
    WHERE {{
        VALUES ?start {{ {" ".join(concepts)} }}

        ?start rdfs:subClassOf* ?child .
        ?child rdfs:subClassOf ?parent .
        OPTIONAL {{
            ?child skos:prefLabel ?label .
        }}
    }}
"""


# def load_ontology_ancestors(
#     output_dir, childs_concepts: list[str], onto_url: str, apikey=""
# ) -> Graph:
#     if Path(output_dir).exists():
#         print("Start ontology parsing..")
#         return Graph().parse(source=output_dir, format="ttl")

#     g = (
#         Graph()
#         .parse(source=f"{onto_url}?apikey={apikey}", format="ttl")
#         .query(ancestors_query(childs_concepts))
#         .graph
#     )
#     if g is None:
#         raise Exception("Something went wrong during ontology parsing.")
#     g.serialize(destination=output_dir, format="ttl")  # cache
#     return g


def to_query(concept: str) -> str:
    return f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

    CONSTRUCT {{
        ?child rdfs:subClassOf ?parent .
    }}
    WHERE {{
        {concept} rdfs:subClassOf* ?child .
        ?child rdfs:subClassOf ?parent .
        OPTIONAL {{
            ?child skos:prefLabel ?label .
        }}
    }}
"""


def download_ontology_with_progress(
    ontology_url, apikey, output_path, desc="Downloading"
):
    url = f"{ontology_url}?apikey={apikey}"

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        buffer = BytesIO()

        with tqdm(total=total_size, unit="B", unit_scale=True, desc=desc) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                buffer.write(chunk)
                pbar.update(len(chunk))

    raw = buffer.getvalue()

    # -----------------------------
    # Handle ZIP archives
    # -----------------------------
    if zipfile.is_zipfile(BytesIO(raw)):
        with zipfile.ZipFile(BytesIO(raw)) as zf:
            # Find first .ttl file
            ttl_files = [f for f in zf.namelist() if f.endswith(".ttl")]

            if not ttl_files:
                raise ValueError("ZIP archive contains no .ttl file")

            with zf.open(ttl_files[0]) as f:
                text = f.read().decode("utf-8")

    else:
        # -----------------------------
        # Handle gzip if needed
        # -----------------------------
        try:
            raw = gzip.decompress(raw)
        except OSError:
            pass

        text = raw.decode("utf-8")

    # Parse RDF
    g = Graph()
    g.parse(data=text, format="ttl")
    g.serialize(output_path, format="nt")

    return g


def load_ontology_ancestors_stream(
    onto_code: str,
    onto_url: str,
    apikey: str,
    output_dir: Path,
    childs_concepts: list[str],
) -> Path | None:
    if len(childs_concepts) == 0:
        print(f"Skip {onto_code} ontology loading.")
        return

    output_file = output_dir / f"{onto_code}_graph.nt"
    if output_file.exists():
        print(f"Loading cached filtered {onto_code} ontology from {output_file}.")
        return

    onto_file = Path("processed_data") / f"full_{onto_code}_graph.nt"
    if onto_file.exists():
        print(f"Loading cached {onto_code} ontology from {onto_file}.")
        ontology_graph = Graph().parse(onto_file, format="nt")
    else:
        ontology_graph = download_ontology_with_progress(
            onto_url, apikey, output_path=onto_file, desc=f"Downloading {onto_code}"
        )
        print(f"Saved ontology graph to {onto_file}")

    with open(output_file, "w", encoding="utf-8") as f:
        for concept in tqdm(childs_concepts, desc=f"Processing {onto_code}'s concepts"):
            subgraph = ontology_graph.query(query_object=to_query(concept)).graph

            if subgraph is not None:
                for triple in subgraph:
                    f.write(f"{triple[0].n3()} {triple[1].n3()} {triple[2].n3()} .\n")


# with open(output_file, "wb") as f:
#     for concept in tqdm(concepts, desc="Processing concepts"):
#         subgraph = ontology.query(query_object=to_query(concept)).graph
#         if subgraph is not None:
#             subgraph.serialize(destination=f, format="nt")
#             return output_file
