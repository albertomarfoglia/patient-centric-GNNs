import gzip
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import IO, cast

import numpy as np
import torch
from rdflib.plugins.parsers.ntriples import W3CNTriplesParser
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.preprocessing import QuantileTransformer

from configs.loader import LoaderConfig
from configs.experiment import ExperimentConfig
from utils.ontologies import NS_ONTO

NUMERIC_RELATION = f"<{NS_ONTO}numericValue>"
TEXT_RELATION = f"<{NS_ONTO}textValue>"
TIME_RELATION = f"<{NS_ONTO}time>"

def iter_nt_gz_files_fast(input_dir: Path, external_ontos: list[Path]):
    """
    True streaming N-Triples iterator.
    - No line parsing
    - No full-file buffering
    - One decompression pass
    - Incremental yields
    """
    files = list(input_dir.glob("*.nt.gz"))
    files.extend(external_ontos)

    for path in tqdm(files, desc="Processing .nt graph portion", dynamic_ncols=True):
        queue = deque()
        done = False
        triple_count = 0

        class StreamingSink:
            def triple(self, s, p, o):
                queue.append((str(s), str(p), str(o)))

        def parse_file():
            nonlocal done
            parser = W3CNTriplesParser(sink=StreamingSink())  # type: ignore
            with gzip.open(path, "rb") as f:
                parser.parse(cast(IO[bytes], f))
            done = True

        thread = Thread(target=parse_file)
        thread.start()

        while not done or queue:
            while queue:
                yield queue.popleft()
                triple_count += 1

        thread.join()


# --------------------------------------------------
# Main preprocessing
# --------------------------------------------------


def preprocess_meds_kg(
    dcfg: LoaderConfig,
    ecfg: ExperimentConfig,
):

    ent_to_id = {}
    rel_to_id = {}

    time_value_ids = []

    next_ent = 0
    next_rel = 0

    numeric_values = []
    text_values = []

    with open(dcfg.triples_path, "w", buffering=1024 * 1024) as out:
        iter = iter_nt_gz_files_fast(dcfg.dataset_dir, external_ontos=ecfg.enrich_by_graphs)

        for h, r, t in iter:
            # ---- invert hasSubject ----
            if r == str(NS_ONTO["hasSubject"]):
                # subjects_per_event[h] = t
                tmp = t
                t = h
                h = tmp

            if r == str(NS_ONTO["hasCode"]) and (code := ecfg.enrich_events.get(t)):
                r = code

            # ---- Entity mapping (dynamic)
            if h not in ent_to_id:
                ent_to_id[h] = next_ent
                numeric_values.append(0.0)
                text_values.append("")
                next_ent += 1

            if t not in ent_to_id:
                ent_to_id[t] = next_ent
                numeric_values.append(0.0)
                text_values.append("")
                next_ent += 1

            # ---- Relation mapping
            if r not in rel_to_id:
                rel_to_id[r] = next_rel
                next_rel += 1

            h_id = ent_to_id[h]
            r_id = rel_to_id[r]
            t_id = ent_to_id[t]

            # ---- Write triple
            out.write(f"{h_id}\t{r_id}\t{t_id}\n")

            # ---- Extract numeric literal
            if r == str(NS_ONTO["numericValue"]):
                try:
                    numeric_values[t_id] = round(float(t), 2)
                except Exception:
                    print("Exception during numericValue conversion")
                    pass

            elif r == str(NS_ONTO["time"]) and ecfg.time_option == "TS":
                try:
                    val = datetime.strptime(t, "%Y-%m-%dT%H:%M:%S")
                    numeric_values[t_id] = val.timestamp()
                    time_value_ids.append(t_id)
                except Exception:
                    print("Exception during time conversion")
                    pass

            elif r == str(NS_ONTO["textValue"]) and ecfg.include_text:
                try:
                    text_values[t_id] = t 
                except Exception:
                    print("Exception during textValue conversion")
                    pass

    print(f"Entities: {len(ent_to_id)} | Relations: {len(rel_to_id)}")

    # --------------------------------------------------
    # Compute time quantile transformation
    # --------------------------------------------------
    print("Computing time quantile transformation..")

    if ecfg.time_option == "TS":
        timestamps = np.array([numeric_values[i] for i in time_value_ids]).reshape(
            -1, 1
        )

        timestamps_scaled = QuantileTransformer(
            output_distribution="uniform"
        ).fit_transform(timestamps)

        for idx, val in zip(time_value_ids, timestamps_scaled):
            numeric_values[idx] = float(val)

    # --------------------------------------------------
    # Compute string embeddings
    # --------------------------------------------------

    if ecfg.include_text:
        print("Computing string embeddings..")
        model_name = "all-MiniLM-L6-v2"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        np.save(
            file=str(dcfg.text_values_path),
            arr=SentenceTransformer(model_name, device=device, cache_folder="__pycache__")
            .encode(text_values, batch_size=128, convert_to_tensor=True, device="cuda")
            .cpu()
            .numpy(),
        )

    # --------------------------------------------------
    # Save entity & relation mappings
    # --------------------------------------------------

    print("Writing entity file...")
    with open(dcfg.entities_path, "w", buffering=1024 * 1024) as f:
        for ent, idx in tqdm(ent_to_id.items(), total=len(ent_to_id), desc="Entities"):
            f.write(f"{idx}\t{ent}\n")

    print("Writing relation file...")
    with open(dcfg.relations_path, "w", buffering=1024 * 1024) as f:
        for rel, idx in tqdm(rel_to_id.items(), total=len(rel_to_id), desc="Relations"):
            f.write(f"{idx}\t{rel}\n")

    # --------------------------------------------------
    # Save numeric array
    # --------------------------------------------------

    numeric_arr = np.array(numeric_values, dtype=np.float32).reshape(-1, 1)

    np.save(
        dcfg.numeric_values_path,
        numeric_arr,
    )

    print("Processing complete.")
