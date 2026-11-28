"""Convert a MEDS-RDF graph (N-Triples) into the arrays used by the model.

Graph construction is the same as in the first submission
(``generation/preprocess_lazy.py``):
  * ``meds:hasSubject`` triples are reversed (subject -> event); the loader
    later reverses every triple, so messages flow event -> subject,
    code -> event and class -> instance;
  * literal-valued ``meds:numericValue`` and ``meds:time`` triples (and the
    text properties when text is enabled) become node features instead of edges;
  * every other triple, including other literal-valued ones, becomes an edge.

Only the storage of literal features is configurable (``FeatureConfig``):
``literal_mode='legacy_shared'`` reproduces the original single numeric slot
(finding F2); ``'separate'`` keeps time and value apart; ``'separate_scaled'``
also quantile-scales the numeric values (pooled over all values);
``'scaled_no_time'`` scales the values in the same way and drops ``meds:time``.

The output is cached under a key derived from the inputs and the feature
settings, written atomically, and guarded by a file lock so that parallel
workers never build or overwrite the same cache entry.
"""

from __future__ import annotations

import fcntl
import gzip
import json
import logging
import shutil
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from rdflib.plugins.parsers.ntriples import W3CNTriplesParser
from sklearn.preprocessing import QuantileTransformer

from .config import RunConfig

log = logging.getLogger(__name__)

ONTO = "https://teamheka.github.io/meds-ontology#"
HAS_SUBJECT = ONTO + "hasSubject"
NUMERIC_VALUE = ONTO + "numericValue"
TIME = ONTO + "time"
TEXT_PROPS = {ONTO + "codeString", ONTO + "codeDescription"}
CODE_IRI_PREFIX = "https://teamheka.github.io/meds-data/code/"

ARRAYS = ("triples", "value", "time", "text")


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #

class _Collect:
    def __init__(self):
        self.items: list[tuple[str, str, str]] = []

    def triple(self, s, p, o):
        self.items.append((str(s), str(p), str(o)))


def iter_triples(graph_dir: Path):
    """Yield (subject, predicate, object) strings from every .nt / .nt.gz file,
    in sorted file order."""
    files = sorted(graph_dir.glob("*.nt.gz")) + sorted(graph_dir.glob("*.nt"))
    if not files:
        raise FileNotFoundError(f"no .nt or .nt.gz files in {graph_dir}")
    for path in files:
        sink = _Collect()
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as fh:
            W3CNTriplesParser(sink=sink).parse(fh)
        yield from sink.items


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #

def _timestamp(lexical: str) -> float:
    try:
        return datetime.strptime(lexical, "%Y-%m-%dT%H:%M:%S").timestamp()
    except ValueError:
        return float("nan")


def _float(lexical: str) -> float:
    try:
        return float(lexical)
    except ValueError:
        return float("nan")


def _quantile(seed: int) -> QuantileTransformer:
    # scikit-learn fits the quantiles on a random subsample of 10,000 values;
    # the first submission left it unseeded, so its time scaling varied
    # slightly between runs. Here the subsample is seeded.
    return QuantileTransformer(output_distribution="uniform", random_state=seed)


def build_arrays(run: RunConfig) -> dict:
    f = run.features
    ent: dict[str, int] = {}
    rel: dict[str, int] = {}
    value: list[float] = []
    time: list[float] = []
    text: list[list[str]] = []
    legacy_time_ids: list[int] = []
    triples: list[tuple[int, int, int]] = []

    def node(iri: str) -> int:
        idx = ent.get(iri)
        if idx is None:
            idx = ent[iri] = len(ent)
            value.append(np.nan)
            time.append(np.nan)
            text.append([])
        return idx

    for h, r, t in iter_triples(Path(run.graph_dir)):
        if r == HAS_SUBJECT:
            h, t = t, h
        h_id, t_id = node(h), node(t)  # literal objects become nodes, as originally
        r_id = rel.setdefault(r, len(rel))

        if r == NUMERIC_VALUE:
            value[h_id] = _float(t)
            continue
        if r == TIME:
            ts = _timestamp(t)
            if f.literal_mode == "legacy_shared":
                value[h_id] = ts          # shared slot: later literals overwrite it
                legacy_time_ids.append(h_id)
            elif f.literal_mode != "scaled_no_time":
                time[h_id] = ts
            # scaled_no_time: the time is dropped (still not an edge, as in all modes)
            continue
        if r in TEXT_PROPS and f.include_text:
            if f.text_mode == "legacy_last":
                text[h_id] = [t]
            else:
                text[h_id].append(t)
            continue
        triples.append((h_id, r_id, t_id))

    value_arr = np.asarray(value, dtype=np.float64)
    time_arr = np.asarray(time, dtype=np.float64)

    if f.time_scaling == "quantile":
        if f.literal_mode == "legacy_shared":
            # As originally: every slot registered by a time triple is rescaled,
            # including those later overwritten by a numeric value (F2).
            ids = np.asarray(legacy_time_ids, dtype=int)
            if len(ids):
                value_arr[ids] = _quantile(run.seed).fit_transform(
                    value_arr[ids].reshape(-1, 1)).ravel()
        else:
            mask = ~np.isnan(time_arr)
            if mask.any():
                time_arr[mask] = _quantile(run.seed).fit_transform(
                    time_arr[mask].reshape(-1, 1)).ravel()

    if f.literal_mode in ("separate_scaled", "scaled_no_time"):
        # Numeric values on one common [0, 1] scale (rank among all values),
        # independent of time_scaling.
        mask = ~np.isnan(value_arr)
        if mask.any():
            value_arr[mask] = _quantile(run.seed).fit_transform(
                value_arr[mask].reshape(-1, 1)).ravel()

    return {
        "entities": ent,
        "relations": rel,
        "triples": np.asarray(triples, dtype=np.int64).reshape(-1, 3),
        "value": value_arr.astype(np.float32),
        "time": time_arr.astype(np.float32),
        "text": [" | ".join(x) if x else None for x in text],
    }


def encode_text(texts: list[str | None], run: RunConfig, device: str) -> np.ndarray:
    """MiniLM sentence embeddings projected to ``embed_dim`` by a fixed random
    linear map (seeded), as in the first submission (where it was unseeded)."""
    from sentence_transformers import SentenceTransformer

    out = np.zeros((len(texts), run.model.embed_dim), dtype=np.float32)
    valid = [i for i, t in enumerate(texts) if t is not None]
    if not valid:
        return out
    encoder = SentenceTransformer(run.features.text_model, device=device)
    emb = encoder.encode([texts[i] for i in valid], batch_size=256, convert_to_numpy=True)
    gen = torch.Generator().manual_seed(run.seed)
    proj = torch.nn.Linear(run.features.text_dim, run.model.embed_dim)
    with torch.no_grad():
        bound = 1 / np.sqrt(run.features.text_dim)  # PyTorch's default Linear init, seeded
        proj.weight.uniform_(-bound, bound, generator=gen)
        proj.bias.uniform_(-bound, bound, generator=gen)
        out[valid] = proj(torch.from_numpy(emb).float()).numpy()
    return out


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #

@contextmanager
def _lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def cache_dir(run: RunConfig, cache_root: Path) -> Path:
    return cache_root / run.task / f"{run.schema}__{run.pruning}__s{run.subsample}__{run.preprocess_key}"


def ensure_preprocessed(run: RunConfig, cache_root: Path, device: str) -> Path:
    """Return the cache directory for ``run``, building it if needed."""
    out = cache_dir(run, cache_root)
    if (out / "DONE").exists():
        return out
    with _lock(out.with_suffix(".lock")):
        if (out / "DONE").exists():  # built by another worker while we waited
            return out
        log.info("preprocessing %s -> %s", run.graph_dir, out)
        arrays = build_arrays(run)
        tmp = out.with_name(out.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        np.save(tmp / "triples.npy", arrays["triples"])
        np.save(tmp / "value.npy", arrays["value"])
        np.save(tmp / "time.npy", arrays["time"])
        if run.features.include_text:
            np.save(tmp / "text.npy", encode_text(arrays["text"], run, device))
        for name in ("entities", "relations"):
            with open(tmp / f"{name}.tsv", "w") as fh:
                fh.writelines(f"{i}\t{iri}\n" for iri, i in arrays[name].items())
        meta = {
            "graph_dir": run.graph_dir, "labels_path": run.labels_path,
            "features": run.to_dict()["features"], "seed": run.seed,
            "num_entities": len(arrays["entities"]), "num_relations": len(arrays["relations"]),
            "num_edges": int(len(arrays["triples"])),
            "num_code_nodes": sum(i.startswith(CODE_IRI_PREFIX) for i in arrays["entities"]),
        }
        (tmp / "meta.json").write_text(json.dumps(meta, indent=2))
        (tmp / "DONE").touch()
        shutil.rmtree(out, ignore_errors=True)
        tmp.rename(out)
    return out
