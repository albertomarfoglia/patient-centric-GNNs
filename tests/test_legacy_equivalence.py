"""The refactored pipeline (pcgnn) must reproduce the first-submission pipeline
when run with the legacy settings.

1. Graph: pcgnn.preprocess.build_arrays (legacy_shared / legacy_last) yields the
   same nodes, edges and literal values as generation/preprocess_lazy.py.
2. Model: pcgnn.model.RGCNLit with the legacy settings computes the same
   outputs as models/binary/rgcn.RGCNet (2 layers) given the same weights.

Test 1 needs a graph preprocessed by the legacy code; it is skipped otherwise.
Run:  python -m pytest -q tests/test_legacy_equivalence.py
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from pcgnn.config import ExperimentSpec

ROOT = Path(__file__).resolve().parents[1]
LEGACY_DIR = ROOT / "processed_data/inference_timing/full/first_24_in_icu_mortality/0"


def _legacy_run():
    return ExperimentSpec.load(ROOT / "experiments/legacy_check.yaml").expand()[0]


def _read_tsv(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            i, iri = line.rstrip("\n").split("\t", 1)
            out[int(i)] = iri
    return out


@pytest.mark.skipif(not (LEGACY_DIR / "meds_TS_triples_1000.tsv").exists(), reason="no legacy graph")
def test_graph_matches_legacy():
    from pcgnn.preprocess import build_arrays

    run = _legacy_run()
    new = build_arrays(run)

    old_ent = _read_tsv(LEGACY_DIR / "meds_TS_entities_1000.tsv")
    old_rel = _read_tsv(LEGACY_DIR / "meds_TS_relations_1000.npy")  # tsv content, .npy name
    assert set(old_ent.values()) == set(new["entities"])
    assert set(old_rel.values()) == set(new["relations"])

    # Edges, compared as (IRI, IRI, IRI) multisets.
    new_ent = {i: iri for iri, i in new["entities"].items()}
    new_rel = {i: iri for iri, i in new["relations"].items()}
    old_tr = np.loadtxt(LEGACY_DIR / "meds_TS_triples_1000.tsv", dtype=np.int64, delimiter="\t")
    as_iri = lambda tr, ent, rel: sorted((ent[h], rel[r], ent[t]) for h, r, t in tr)
    assert as_iri(old_tr, old_ent, old_rel) == as_iri(new["triples"], new_ent, new_rel)

    # Literal values, by node IRI (legacy single numeric slot, after the time transform).
    old_num = np.load(LEGACY_DIR / "meds_TS_numeric_1000.npy").ravel()
    old_by_iri = {old_ent[i]: v for i, v in enumerate(old_num)}
    new_by_iri = {iri: new["value"][i] for iri, i in new["entities"].items()}
    a = np.array([old_by_iri[k] for k in sorted(old_by_iri)], dtype=np.float64)
    b = np.array([new_by_iri[k] for k in sorted(old_by_iri)], dtype=np.float64)
    assert np.array_equal(np.isnan(a), np.isnan(b))
    # The legacy quantile transform was fitted on an unseeded random subsample
    # of 10,000 values, so rescaled values differ by sampling noise (< 0.02 on
    # a [0, 1] scale, < 0.005 on average); values outside the time transform must match exactly.
    ok = ~np.isnan(a)
    np.testing.assert_allclose(a[ok], b[ok], rtol=0, atol=0.02)
    assert np.mean(np.abs(a[ok] - b[ok])) < 0.005
    exact = ok & (np.abs(a) > 1.0)
    np.testing.assert_allclose(a[exact], b[exact], rtol=1e-6)

    # Same nodes carry text.
    old_txt = np.load(LEGACY_DIR / "meds_TS_text_1000.npy")
    old_has = {old_ent[i] for i in np.nonzero(np.abs(old_txt).sum(1))[0]}
    new_has = {iri for iri, i in new["entities"].items() if new["text"][i] is not None}
    assert old_has == new_has


def test_model_matches_legacy():
    from torch_geometric.data import Data

    from models.binary.rgcn import RGCNet
    from pcgnn.model import RGCNLit

    torch.manual_seed(0)
    n, d, n_rel, n_edges = 300, 32, 5, 1500
    edge_index = torch.randint(0, n, (2, n_edges))
    edge_type = torch.randint(0, n_rel, (n_edges,))
    num_x = torch.randn(n, 1)
    num_mask = (torch.rand(n, 1) > 0.5).float()
    txt_x = torch.randn(n, d)
    txt_mask = (torch.rand(n) > 0.5).float()
    x = torch.randn(n, d)

    old = RGCNet(embed_dim=d, hidden_dim=d, num_relations=n_rel, dropout=0.0, include_text_features=True)
    old_data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=n, num_x=num_x,
                    num_mask=num_mask, txt_x=txt_x, txt_mask=txt_mask, x=x)

    run = _legacy_run()
    new = RGCNLit(n, n_rel, 1, torch.zeros(n, dtype=torch.bool), run.model, run.features,
                  torch.Generator().manual_seed(1))
    # Copy the legacy weights into the new module.
    new.inputs.value_proj.load_state_dict(old.num_proj.state_dict())
    new.inputs.act.load_state_dict(old.input_activation.state_dict())
    new.convs[0].load_state_dict(old.conv1.state_dict())
    new.acts[0].load_state_dict(old.act1.state_dict())
    new.convs[1].load_state_dict(old.conv3.state_dict())
    new.inputs.base.copy_(x)
    new_data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=n, value_x=num_x,
                    value_mask=num_mask, time_x=torch.zeros(n, 1), time_mask=torch.zeros(n, 1),
                    txt_x=txt_x, txt_mask=txt_mask)

    old.eval(), new.eval()
    with torch.no_grad():
        torch.testing.assert_close(new(new_data), old(old_data))
