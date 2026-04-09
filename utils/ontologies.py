from pathlib import Path

import numpy as np
from rdflib import Graph, Namespace, URIRef

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


def load_mimic_onto_concepts(onto_code: str):
    onto_uri = Namespace(f"{BIOPORTAL_URI}/{onto_code}/")
    file = Path("./generation/mimic_external_codes") / f"{onto_code}_codes.npy"
    if file.exists():
        codes = np.load(
            Path("./generation/mimic_external_codes") / f"{onto_code}_codes.npy",
            allow_pickle=True,
        )
        return list(f"<{c}>" for c in [onto_uri[code] for code in codes])
    return []


NEUROVASC_ENHANCER_DICT: dict[URIRef, URIRef] = {
    NS_CODE.AGE_Years: NS_ONTO.hasAge,
    NS_CODE.GENDER_0: NS_ONTO.hasGender,
    NS_CODE.GENDER_1: NS_ONTO.hasGender,
    # **{ URIRef(NS_CODE + f"ATC_{atc}"): NS_ONTO.hasAdministration for atc in _NEUROVASC_ATC_CODES },
}

MIMIC_ENHANCER_DICT: dict[str, str] = {
    str(NS_CODE.MEDS_BIRTH): NS_ONTO.hasBirth,
    str(NS_CODE.GENDER_M): NS_ONTO.hasMale,
    str(NS_CODE.GENDER_F): NS_ONTO.hasFemale,
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


def load_ontolgy_ancestors(
    output_path, concepts: list[str], ontology_url: str, apikey=""
) -> Graph:
    if Path(output_path).exists():
        print("Start ontology parsing..")
        return Graph().parse(source=output_path, format="ttl")

    g = (
        Graph()
        .parse(source=f"{ontology_url}?apikey={apikey}", format="ttl")
        .query(ancestors_query(concepts))
        .graph
    )
    if g is None:
        raise Exception("Something went wrong during ontology parsing.")
    g.serialize(destination=output_path, format="ttl")  # cache
    return g


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


def download_ontology_with_progress(ontology_url, apikey, desc="Downloading"):
    url = f"{ontology_url}?apikey={apikey}"

    # Start the request with streaming
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        buffer = BytesIO()
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=desc) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                buffer.write(chunk)
                pbar.update(len(chunk))

    buffer.seek(0)  # Reset pointer for reading
    g = Graph()
    g.parse(source=buffer, format="ttl")
    return g


def load_ontology_ancestors_stream(
    onto_code: str,
    onto_url: str,
    apikey: str,
    output_dir: Path,
    childs_concepts: list[str],
) -> Path | None:
    output_file = output_dir / f"{onto_code}_graph.nt.gz"
    if output_file.exists():
        print(f"Loading cached {onto_code} ontology from {output_file}.")
        return output_file

    if len(childs_concepts) == 0:
        print(f"Skip {onto_code} ontology loading.")
        return

    ontology_graph = download_ontology_with_progress(
        onto_url, apikey, desc=f"Downloading {onto_code}"
    )

    with gzip.open(output_file, "wt", encoding="utf-8") as f:
        for concept in tqdm(childs_concepts, desc=f"Processing {onto_code}'s concepts"):
            subgraph = ontology_graph.query(query_object=to_query(concept)).graph

            if subgraph is not None:
                for triple in subgraph:
                    f.write(
                        f"{triple[0].n3()} {triple[1].n3()} {triple[2].n3()} .\n"  # type: ignore
                    )


# with open(output_file, "wb") as f:
#     for concept in tqdm(concepts, desc="Processing concepts"):
#         subgraph = ontology.query(query_object=to_query(concept)).graph
#         if subgraph is not None:
#             subgraph.serialize(destination=f, format="nt")
#             return output_file
