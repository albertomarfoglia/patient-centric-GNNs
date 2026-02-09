from pathlib import Path
from rdflib import Namespace, Graph, URIRef

NS_DATA = Namespace("https://teamheka.github.io/meds-data/")
NS_ONTO = Namespace("https://teamheka.github.io/meds-ontology#")
NS_CODE = Namespace(f"{NS_DATA}code/")

BIOPORTAL_URL = "https://data.bioontology.org/ontologies"
ATC_BIOPORTAL_URL = f"{BIOPORTAL_URL}/ATC/submissions/23/download"


ATC = Namespace("http://purl.bioontology.org/ontology/ATC/")
_NEUROVASC_ATC_CODES = ["C08CA06", "N02BE01", "C01CA03", "C01CE02", "N02AA01"]
NEUROVASC_ATC_URIS = list(f"<{c}>" for c in [ATC[code] for code in _NEUROVASC_ATC_CODES])

NEUROVASC_ENHANCER_DICT: dict[URIRef, URIRef] = {
    NS_CODE.AGE_Years: NS_ONTO.hasAge,
    NS_CODE.GENDER_0: NS_ONTO.hasGender,
    NS_CODE.GENDER_1: NS_ONTO.hasGender,
    **{ URIRef(NS_CODE + f"ATC_{atc}"): NS_ONTO.hasAdministration for atc in _NEUROVASC_ATC_CODES },
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

def load_ontolgy_ancestors(output_path, concepts: list[str], ontology_url: str, apikey="") -> Graph :
    if  Path(output_path).exists():
        return Graph().parse(source=output_path, format="ttl")

    g = Graph().parse(source=f"{ontology_url}?apikey={apikey}", format="ttl").query(ancestors_query(concepts)).graph
    if g is None: 
        raise Exception("Something went wrong during ontology parsing.")
    g.serialize(destination=output_path, format="ttl") # cache
    return g