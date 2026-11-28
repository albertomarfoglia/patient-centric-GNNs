"""Literal encodings (finding F2) on a tiny hand-written graph.

  * legacy_shared:   the value overwrites the time slot; the time is lost.
  * separate:        time quantile-scaled, values unscaled.
  * separate_scaled: time as in "separate"; values quantile-scaled to [0, 1],
                     order preserved.
  * scaled_no_time:  values as in "separate_scaled"; no time at all.
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


def test_scaled_no_time_drops_time_keeps_scaled_values(graph_dir):
    v_sc, _ = _arrays(graph_dir, "separate_scaled")
    v_nt, t_nt = _arrays(graph_dir, "scaled_no_time")
    assert np.isnan(t_nt).all()
    np.testing.assert_allclose(v_nt, v_sc, equal_nan=True)


def test_legacy_shared_loses_time_of_valued_events(graph_dir):
    v_leg, t_leg = _arrays(graph_dir, "legacy_shared")
    assert np.isnan(t_leg).all()
    assert not np.isnan(v_leg[4])  # e5: time only, the time sits in the value slot


def test_model_has_time_channel():
    from pcgnn.config import FeatureConfig
    from pcgnn.model import NodeInput
    g = torch.Generator().manual_seed(0)
    is_code = torch.zeros(3, dtype=torch.bool)
    for mode, has_time in [("legacy_shared", False), ("separate", True), ("separate_scaled", True),
                           ("scaled_no_time", False), ("scaled_relative", True)]:
        m = NodeInput(3, is_code, ModelConfig(), FeatureConfig(literal_mode=mode, include_text=False), g)
        assert hasattr(m, "time_proj") == has_time


# --------------------------------------------------------------------------- #
# scaled_relative: time relative to the patient's prediction time
# --------------------------------------------------------------------------- #

def _relative_setup(tmp_path, future=False):
    """Two subjects; subject 0 predicted at 2110-01-02 00:00, subject 1 at 2150-06-01 12:00.
    Returns a run whose graph and MEDS labels follow the real directory layout."""
    import pandas as pd
    task = tmp_path / "task"
    g = tmp_path / "graph"; g.mkdir()
    lab = task / "meds" / "0" / "MEDS_cohort" / "labels" / "train"; lab.mkdir(parents=True)
    (task / "labels").mkdir()
    pd.DataFrame({"subject_id": [0, 1],
                  "prediction_time": pd.to_datetime(["2110-01-02 00:00:00", "2150-06-01 12:00:00"]),
                  "boolean_value": [0, 1]}).to_parquet(lab / "0.parquet")
    ev = [("0_1", 0, 7.0, "2110-01-01T18:00:00"),    # 6 h before prediction
          ("0_2", 0, None, "2110-01-01T23:00:00"),   # 1 h before
          ("1_3", 1, 3.0, "2150-06-01T00:00:00"),    # 12 h before
          ("1_4", 1, None, "2150-06-01T12:00:00" if not future else "2150-06-01T13:00:00")]
    lines = []
    for e, s, v, t in ev:
        lines.append(f"<{D}event/{e}> <{O}hasSubject> <{D}subject/{s}> .")
        lines.append(f'<{D}event/{e}> <{O}time> "{t}"^^<{XSD}dateTime> .')
        if v is not None:
            lines.append(f'<{D}event/{e}> <{O}numericValue> "{v}"^^<{XSD}double> .')
    (g / "part-00000.nt").write_text("\n".join(lines) + "\n")
    run = ExperimentSpec.load(ROOT / "experiments/mimic_f2_scaled.yaml").expand()[0]
    run = dataclasses.replace(run, graph_dir=str(g), labels_path=str(task / "labels" / "y_0.joblib"), subsample=0,
                              features=dataclasses.replace(run.features, literal_mode="scaled_relative",
                                                           include_text=False, time_scaling="none"))
    return run, [e for e, *_ in ev]


def test_scaled_relative_hours_before_prediction(tmp_path):
    from pcgnn.preprocess import build_arrays
    run, events = _relative_setup(tmp_path)
    a = build_arrays(run)
    t = a["time"][[a["entities"][f"{D}event/{e}"] for e in events]]
    np.testing.assert_allclose(t, [-6.0, -1.0, -12.0, 0.0])   # same scale for both patients


def test_scaled_relative_rejects_events_after_prediction(tmp_path):
    from pcgnn.preprocess import build_arrays
    run, _ = _relative_setup(tmp_path, future=True)
    with pytest.raises(ValueError, match="leakage"):
        build_arrays(run)


def test_scaled_relative_quantile_scaling(tmp_path):
    from pcgnn.preprocess import build_arrays
    run, events = _relative_setup(tmp_path)
    run = dataclasses.replace(run, features=dataclasses.replace(run.features, time_scaling="quantile"))
    a = build_arrays(run)
    t = a["time"][[a["entities"][f"{D}event/{e}"] for e in events]]
    assert t.min() >= 0 and t.max() <= 1
    assert list(np.argsort(t)) == [2, 0, 1, 3]   # -12 < -6 < -1 < 0
