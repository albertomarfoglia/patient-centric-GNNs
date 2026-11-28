"""Typed configuration for experiments.

An *experiment spec* (YAML) declares a grid of runs. A *run* is one
(dataset task, RDF schema, pruning level, subsample) combination with fully
resolved settings; it trains and evaluates one model per cross-validation fold.

All dataclasses are frozen so a resolved run cannot be modified after it has
been written to disk. Unknown YAML keys raise an error instead of being
silently ignored.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import yaml

# --------------------------------------------------------------------------- #
# Option types
# --------------------------------------------------------------------------- #

# How literal values reach the node encoder (finding F2):
#   "legacy_shared": time and numeric value share one slot per node, as in the
#                    runs reported in the first submission (the value overwrites
#                    the timestamp and is then rescaled with the time transform).
#   "separate":      time and numeric value are two independent channels; the
#                    time is quantile-scaled, the numeric value enters unscaled.
#   "separate_scaled": as "separate", and the numeric values are also
#                    quantile-scaled (one transform pooled over all numeric
#                    values, fitted on values only). Isolates the F2 effect:
#                    legacy_shared rank-scales values the same way but drops the
#                    time of every event that has a value.
#   "scaled_no_time": numeric values as in "separate_scaled"; meds:time is not
#                    used at all (no time channel). Tests whether the absolute,
#                    per-patient date-shifted MIMIC timestamps only add noise.
#   "scaled_relative": numeric values as in "separate_scaled"; the time channel
#                    holds the event time relative to the patient's prediction
#                    time (hours, <= 0), quantile-scaled. Prediction times are
#                    read from the cohort's MEDS labels next to labels_path;
#                    an event after its prediction time aborts preprocessing.
LiteralMode = Literal["legacy_shared", "separate", "separate_scaled", "scaled_no_time",
                      "scaled_relative"]

# Initial node representation (finding F6 and Reviewer 2, major comment 2):
#   "fixed_random":   one random Xavier vector per node, not trained (first submission).
#   "learnable_node": one trainable vector per node.
#   "learnable_code": trainable vectors for meds:Code nodes, fixed random elsewhere.
EmbeddingMode = Literal["fixed_random", "learnable_node", "learnable_code"]

# Text attached to a node (meds:codeString / meds:codeDescription):
#   "legacy_last":  the last text literal seen for a node wins (first submission).
#   "concat":       all text literals of a node are joined with " | ".
TextMode = Literal["legacy_last", "concat"]

TrackingMode = Literal["process", "machine"]


def _from_dict(cls, data: dict[str, Any] | None):
    """Build a (possibly nested) frozen dataclass from a dict, rejecting unknown keys."""
    data = dict(data or {})
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown key(s) {sorted(unknown)}; allowed: {sorted(known)}")
    kwargs = {}
    for name, f in known.items():
        if name not in data:
            continue
        value = data[name]
        sub = _NESTED.get((cls.__name__, name))
        kwargs[name] = _from_dict(sub, value) if sub is not None else value
    return cls(**kwargs)


@dataclass(frozen=True)
class ModelConfig:
    name: Literal["rgcn_lit"] = "rgcn_lit"
    embed_dim: int = 32
    hidden_dim: int = 32
    num_layers: int = 2  # number of RGCNConv layers, including the output layer
    num_bases: int = 8
    dropout: float = 0.0
    embeddings: EmbeddingMode = "fixed_random"

    def __post_init__(self):
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 5e-3
    weight_decay: float = 5e-4
    max_epochs: int = 2000
    patience: int = 50
    min_delta: float = 1e-4


@dataclass(frozen=True)
class FeatureConfig:
    include_text: bool = True
    text_model: str = "all-MiniLM-L6-v2"
    text_dim: int = 384
    literal_mode: LiteralMode = "separate"
    text_mode: TextMode = "concat"
    time_scaling: Literal["quantile", "none"] = "quantile"


@dataclass(frozen=True)
class EvalConfig:
    folds: int = 10
    val_fraction: float = 1 / 9  # of the non-test patients, as in the first submission
    split_seed: int = 77


@dataclass(frozen=True)
class CostConfig:
    track_emissions: bool = True
    tracking_mode: TrackingMode = "process"
    measure_power_secs: int = 1
    inference_repeats: int = 10
    # Grid of the machine (ISO 3166-1 alpha-3). Fixed instead of CodeCarbon's
    # IP geolocation, which alternated between ITA and CAN on this host.
    country_iso_code: str = "ITA"


@dataclass(frozen=True)
class SourceConfig:
    """Where the RDF graphs and labels of one (schema, pruning) variant live.

    Layout (as produced by meds-to-owl-examples):
        <root>/<graph_dir>/<task>/meds_<n>_<subsample>/*.nt[.gz]
        <root>/<labels_dir>/<task>/labels/outcomes_meds_TS_<n>_<subsample>.joblib
    """

    root: str
    graph_dirs: dict[str, dict[str, str]]  # schema -> pruning -> directory name
    labels_dir: str
    subject_iri_prefix: str = "https://teamheka.github.io/meds-data/subject/"
    classes: tuple[str, ...] = ("FALSE", "TRUE")


@dataclass(frozen=True)
class TaskConfig:
    name: str
    num_patients: int
    subsamples: tuple[int, ...] = (0, 1, 2, 3, 4)


_NESTED = {
    ("Defaults", "model"): ModelConfig,
    ("Defaults", "train"): TrainConfig,
    ("Defaults", "features"): FeatureConfig,
    ("Defaults", "evaluation"): EvalConfig,
    ("Defaults", "costs"): CostConfig,
}


@dataclass(frozen=True)
class Defaults:
    seed: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    costs: CostConfig = field(default_factory=CostConfig)


@dataclass(frozen=True)
class RunConfig:
    """One fully resolved run: a single subsample of one task/schema/pruning/variant."""

    experiment: str
    variant: str
    task: str
    schema: str
    pruning: str
    subsample: int
    num_patients: int
    graph_dir: str
    labels_path: str
    subject_iri_prefix: str
    classes: tuple[str, ...]
    seed: int
    model: ModelConfig
    train: TrainConfig
    features: FeatureConfig
    evaluation: EvalConfig
    costs: CostConfig

    @property
    def run_id(self) -> str:
        parts = [self.task, self.schema, self.pruning]
        if self.variant != "default":
            parts.append(self.variant)
        return "__".join(parts) + f"__s{self.subsample}"

    @property
    def config_hash(self) -> str:
        """Short hash of every setting that can change the result."""
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    @property
    def preprocess_key(self) -> str:
        """Identifies the preprocessed arrays; runs differing only in model or
        training settings share them."""
        f = self.features
        payload = json.dumps(
            [self.graph_dir, self.labels_path, self.num_patients, f.include_text, f.text_model,
             f.literal_mode, f.text_mode, f.time_scaling, self.seed],
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        out[k] = _merge(out.get(k, {}), v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


@dataclass(frozen=True)
class ExperimentSpec:
    """Parsed experiment spec. See ``experiments/README.md`` for the format."""

    name: str
    description: str
    source: SourceConfig
    tasks: tuple[TaskConfig, ...]
    schemas: tuple[str, ...]
    prunings: tuple[str, ...]
    defaults: dict[str, Any]
    variants: dict[str, dict[str, Any]]
    output_root: str
    cache_root: str

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentSpec":
        raw = yaml.safe_load(Path(path).read_text())
        allowed = {"name", "description", "source", "tasks", "schemas", "prunings",
                   "defaults", "variants", "output_root", "cache_root"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"spec {path}: unknown key(s) {sorted(unknown)}")
        src = raw["source"]
        source = SourceConfig(
            root=src["root"],
            graph_dirs=src["graph_dirs"],
            labels_dir=src["labels_dir"],
            subject_iri_prefix=src.get("subject_iri_prefix", SourceConfig.subject_iri_prefix),
            classes=tuple(src.get("classes", SourceConfig.classes)),
        )
        tasks = tuple(
            TaskConfig(name=t["name"], num_patients=int(t["num_patients"]),
                       subsamples=tuple(t.get("subsamples", (0, 1, 2, 3, 4))))
            for t in raw["tasks"]
        )
        spec = cls(
            name=raw["name"],
            description=raw.get("description", ""),
            source=source,
            tasks=tasks,
            schemas=tuple(raw["schemas"]),
            prunings=tuple(raw["prunings"]),
            defaults=raw.get("defaults", {}),
            variants=raw.get("variants") or {"default": {}},
            output_root=raw.get("output_root", "runs"),
            cache_root=raw.get("cache_root", "cache/preprocessed"),
        )
        spec.expand()  # validates every combination eagerly
        return spec

    def expand(self) -> list[RunConfig]:
        """All runs of the grid, in a deterministic order. Combinations whose
        (schema, pruning) has no graph directory are skipped."""
        runs: list[RunConfig] = []
        root = Path(self.source.root)
        for variant, override in self.variants.items():
            settings = _from_dict(Defaults, _merge(self.defaults, override or {}))
            for task in self.tasks:
                for schema in self.schemas:
                    for pruning in self.prunings:
                        gdir = self.source.graph_dirs.get(schema, {}).get(pruning)
                        if gdir is None:
                            continue
                        for s in task.subsamples:
                            n = task.num_patients
                            runs.append(RunConfig(
                                experiment=self.name, variant=variant, task=task.name,
                                schema=schema, pruning=pruning, subsample=s, num_patients=n,
                                graph_dir=str(root / gdir / task.name / f"meds_{n}_{s}"),
                                labels_path=str(root / self.source.labels_dir / task.name / "labels"
                                                / f"outcomes_meds_TS_{n}_{s}.joblib"),
                                subject_iri_prefix=self.source.subject_iri_prefix,
                                classes=self.source.classes,
                                seed=settings.seed + s,
                                model=settings.model, train=settings.train,
                                features=settings.features, evaluation=settings.evaluation,
                                costs=settings.costs,
                            ))
        ids = [r.run_id for r in runs]
        dup = {i for i in ids if ids.count(i) > 1}
        if dup:
            raise ValueError(f"duplicate run ids: {sorted(dup)[:5]}")
        return runs
