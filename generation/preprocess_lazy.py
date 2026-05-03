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


def iter_nt_files_fast(input_dir: Path, external_ontos: list[Path]):
    """
    Streaming N-Triples iterator supporting:
    - .nt.gz (compressed)
    - .nt (plain text)

    Features:
    - No line parsing
    - No full-file buffering
    - One pass
    - Incremental yields
    """
    files = list(input_dir.glob("*.nt.gz"))
    files.extend(input_dir.glob("*.nt"))
    files.extend(external_ontos)

    for path in tqdm(files, desc="Processing .nt graph portion", dynamic_ncols=True):
        queue = deque()
        done = False
        triple_count = 0

        class StreamingSink:
            def triple(self, s, p, o):
                queue.append((str(s), str(p), str(o)))

        def open_file(p: Path):
            if p.suffix == ".gz":
                return gzip.open(p, "rb")
            return open(p, "rb")

        def parse_file():
            nonlocal done
            parser = W3CNTriplesParser(sink=StreamingSink())  # type: ignore
            with open_file(path) as f:
                parser.parse(cast(IO[bytes], f))
            done = True

        thread = Thread(target=parse_file)
        thread.start()

        while not done or queue:
            while queue:
                yield queue.popleft()
                triple_count += 1

        thread.join()


def preprocess_meds_kg(
    dcfg: LoaderConfig,
    ecfg: ExperimentConfig,
):

    ent_to_id = {}
    rel_to_id = {}

    next_ent, next_rel = 0, 0

    numeric_values = []
    text_values = []
    time_value_ids = []

    with open(dcfg.triples_path, "w", buffering=1024 * 1024) as out:
        iter = iter_nt_files_fast(
            dcfg.dataset_dir, external_ontos=ecfg.enrich_by_graphs
        )

        for h, r, t in iter:
            # swap subject/event
            if r == str(NS_ONTO["hasSubject"]):
                h, t = t, h

            if r == str(NS_ONTO["hasCode"]) and (code := ecfg.enrich_events.get(t)):
                r = code

            # ---- Entity mapping (dynamic)
            if h not in ent_to_id:
                ent_to_id[h] = next_ent
                numeric_values.append(np.nan)
                text_values.append(None)
                next_ent += 1

            if t not in ent_to_id:
                ent_to_id[t] = next_ent
                numeric_values.append(np.nan)
                text_values.append(None)
                next_ent += 1

            if r not in rel_to_id:
                rel_to_id[r] = next_rel
                next_rel += 1

            h_id = ent_to_id[h]
            r_id = rel_to_id[r]
            t_id = ent_to_id[t]
            #out.write(f"{h_id}\t{r_id}\t{t_id}\n")

            # ---- Extract numeric literal
            if r == str(NS_ONTO["numericValue"]):
                try:
                    # numeric_values[h_id] = float(t)
                    # continue

                    numeric_values[t_id] = float(t)
                except Exception:
                    print("An exception is occured during numeric conversion")
                    # numeric_values[h_id] = np.nan
                    # continue
                    numeric_values[t_id] = np.nan

            elif r == str(NS_ONTO["time"]) and ecfg.time_option == "TS":
                try:
                    dt = datetime.strptime(t, "%Y-%m-%dT%H:%M:%S")
                    # numeric_values[h_id] = dt.timestamp()
                    # time_value_ids.append(h_id)
                    # continue

                    numeric_values[t_id] = dt.timestamp()
                    time_value_ids.append(t_id)
                except Exception:
                    print("An exception is occured during timestamp conversion")
                    # numeric_values[h_id] = np.nan
                    # continue

                    numeric_values[t_id] = np.nan

            elif (r == str(NS_ONTO["textValue"])) and ecfg.include_text:
                # text_values[h_id] = t
                # continue

                text_values[t_id] = t

            # skip these nodes
            # elif (r == str(NS_ONTO["codeString"])) or (
            #     r == str(NS_ONTO["codeDescription"])
            # ):
            #     continue

            # r_id = rel_to_id[r]
            # t_id = ent_to_id[t]
            out.write(f"{h_id}\t{r_id}\t{t_id}\n")

    print(f"Entities: {len(ent_to_id)} | Relations: {len(rel_to_id)}")

    # --------------------------------------------------
    # Compute time quantile transformation
    # --------------------------------------------------
    print("Computing time quantile transformation..")

    if ecfg.time_option == "TS":
        ts = np.array([numeric_values[i] for i in time_value_ids]).reshape(-1, 1)

        ts_scaled = QuantileTransformer(output_distribution="uniform").fit_transform(ts)

        for idx, val in zip(time_value_ids, ts_scaled):
            numeric_values[idx] = float(val)

    # --------------------------------------------------
    # Compute string embeddings
    # --------------------------------------------------

    if ecfg.include_text:
        print("Computing string embeddings..")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        valid_idx = [i for i, t in enumerate(text_values) if t is not None]

        embeddings = np.zeros((len(text_values), 384), dtype=np.float32)

        if valid_idx:
            model = SentenceTransformer(
                "all-MiniLM-L6-v2", device=device, cache_folder="__pycache__"
            )
            encoded = model.encode(
                [text_values[i] for i in valid_idx],
                batch_size=128,
                convert_to_tensor=False,
            )

            for i, emb in zip(valid_idx, encoded):
                embeddings[i] = emb

        np.save(str(dcfg.text_values_path), embeddings)

    # --------------------------------------------------
    # Save numeric array
    # --------------------------------------------------
    np.save(
        dcfg.numeric_values_path,
        np.array(numeric_values, dtype=np.float32).reshape(-1, 1),
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

    print("Processing complete.")
