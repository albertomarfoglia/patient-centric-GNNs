"""Literal encodings (finding F2) on a tiny hand-written graph.

  * legacy_shared:   the value overwrites the time slot; the time is lost.
  * separate:        time quantile-scaled, values unscaled.
  * separate_scaled: time as in "separate"; values quantile-scaled to [0, 1],
                     order preserved.
Run:  python -m pytest -q tests/test_literal_modes.py
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

from pcgnn.config import ExperimentSpec, ModelConfig

ROOT = Path(__file__).resolve().parents[1]
O = "https://teamheka.github.io/meds-ontology#"
D = "https://teamheka.github.io/meds-data/"
XSD = "http://www.w3.org/2001/XMLSchema#"

# (event, value or None, time); event e4 has no time.
EVENTS = [("e1", 0.7, "2110-01-01T00:00:00"), ("e2", 78.0, "2120-06-01T00:00:00"),
          ("e3", 8445.0, "2130-01-01T00:00:00"), ("e4", 15.0, None),
          ("e5", None, "2140-01-01T00:00:00")]


@pytest.fixture(scope="module")
def graph_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("g")
    lines = []
    for e, v, t in EVENTS:
        lines.append(f"<{D}event/{e}> <{O}hasSubject> <{D}subject/0> .")
        lines.append(f"<{D}event/{e}> <{O}hasCode> <{D}code/LAB_1> .")
        if t is not None:
            lines.append(f'<{D}event/{e}> <{O}time> "{t}"^^<{XSD}dateTime> .')
        if v is not None:
            lines.append(f'<{D}event/{e}> <{O}numericValue> "{v}"^^<{XSD}double> .')
    (d / "part-00000.nt").write_text("\n".join(lines) + "\n")
    return d


def _arrays(graph_dir, mode):
    from pcgnn.preprocess import build_arrays
    run = ExperimentSpec.load(ROOT / "experiments/mimic_f2_scaled.yaml").expand()[0]
    run = dataclasses.replace(run, graph_dir=str(graph_dir),
                              features=dataclasses.replace(run.features, literal_mode=mode, include_text=False))
    a = build_arrays(run)
    idx = [a["entities"][f"{D}event/{e}"] for e, _, _ in EVENTS]
    return a["value"][idx], a["time"][idx]


def test_spec_uses_new_mode():
    runs = ExperimentSpec.load(ROOT / "experiments/mimic_f2_scaled.yaml").expand()
    assert len(runs) == 55
    assert {r.features.literal_mode for r in runs} == {"separate_scaled"}


def test_separate_scaled_keeps_time_and_scales_values(graph_dir):
    v_sep, t_sep = _arrays(graph_dir, "separate")
    v_sc, t_sc = _arrays(graph_dir, "separate_scaled")
    # Time channel identical to "separate".
    np.testing.assert_array_equal(np.isnan(t_sc), np.isnan(t_sep))
    np.testing.assert_allclose(t_sc[~np.isnan(t_sc)], t_sep[~np.isnan(t_sep)])
    # "separate" leaves values raw; "separate_scaled" maps them into [0, 1], same order.
    raw = np.array([0.7, 78.0, 8445.0, 15.0, np.nan], dtype=np.float32)
    np.testing.assert_allclose(v_sep, raw, equal_nan=True)
    m = ~np.isnan(raw)
    assert np.array_equal(np.isnan(v_sc), ~m)
    assert v_sc[m].min() >= 0.0 and v_sc[m].max() <= 1.0
    assert list(np.argsort(v_sc[m])) == list(np.argsort(raw[m]))


def test_legacy_shared_loses_time_of_valued_events(graph_dir):
    v_leg, t_leg = _arrays(graph_dir, "legacy_shared")
    assert np.isnan(t_leg).all()
    assert not np.isnan(v_leg[4])  # e5: time only, the time sits in the value slot


def test_model_has_time_channel():
    from pcgnn.config import FeatureConfig
    from pcgnn.model import NodeInput
    g = torch.Generator().manual_seed(0)
    is_code = torch.zeros(3, dtype=torch.bool)
    for mode, has_time in [("legacy_shared", False), ("separate", True), ("separate_scaled", True)]:
        m = NodeInput(3, is_code, ModelConfig(), FeatureConfig(literal_mode=mode, include_text=False), g)
        assert hasattr(m, "time_proj") == has_time
