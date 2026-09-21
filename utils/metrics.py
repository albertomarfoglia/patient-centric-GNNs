from __future__ import annotations
from pathlib import Path

import pandas as pd

from utils.gcn_utils import mean_std_metrics

from dataclasses import dataclass

@dataclass
class SubsampleResult:
    """Results produced by one RGCN run on one patient subsample."""

    fold_metrics: pd.DataFrame
    fold_costs: pd.DataFrame

    @property
    def metrics_mean(self) -> pd.DataFrame:
        return self.fold_metrics.groupby(level=0).mean()

    @property
    def metrics_std(self) -> pd.DataFrame:
        return self.fold_metrics.groupby(level=0).std()

    @property
    def n_folds(self) -> int:
        return len(self.fold_costs)

def aggregate_subsample_results(
    results: list[SubsampleResult],
) -> dict:
    """
    Aggregate results from multiple independent patient subsamples.

    Returns:
        A dictionary containing:
        - all fold-level metrics
        - pooled mean/std metrics
        - subsample-level mean metrics
        - fold-level computational costs
        - pooled computational cost statistics
    """

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------
    fold_metrics = pd.concat(
        [result.fold_metrics for result in results]
    )

    pooled_mean = fold_metrics.groupby(level=0).mean()
    pooled_std = fold_metrics.groupby(level=0).std()

    # Mean for each individual subsample.
    # This is useful for paired statistical comparisons.
    subsample_means = pd.DataFrame(
        {
            i: result.metrics_mean.loc["MACRO"]
            for i, result in enumerate(results)
        }
    ).T

    # --------------------------------------------------------
    # Computational cost
    # --------------------------------------------------------
    fold_costs = pd.concat(
        [result.fold_costs for result in results],
        ignore_index=True,
    )

    return {
        "fold_metrics": fold_metrics,
        "mean": pooled_mean,
        "std": pooled_std,
        "subsample_means": subsample_means,
        "fold_costs": fold_costs,
    }

def save_aggregated_results(
    aggregated: dict,
    result_dir: Path,
    time_option: str,
    num_patients: int,
    classes = ["TRUE", "FALSE"],
) -> None:

    mean_df = aggregated["mean"]
    std_df = aggregated["std"]
    fold_costs = aggregated["fold_costs"]

    # Metrics
    metrics_path = (
        result_dir
        / f"metrics_{time_option}_{num_patients}_mean_std.csv"
    )

    mean_std_metrics(
        mean_df,
        std_df,
        classes=classes
    ).to_csv(
        metrics_path,
        sep="\t",
        index=False,
    )

    # Computational cost
    cost_summary = pd.DataFrame(
        [{
            "scope": "overall_all_folds",
            "n_folds": len(fold_costs),
            "duration_mean": fold_costs["duration"].mean(),
            "duration_std": fold_costs["duration"].std(ddof=1),
            "emissions_mean": fold_costs["emissions"].mean(),
            "emissions_std": fold_costs["emissions"].std(ddof=1),
        }]
    )

    cost_summary.to_csv(
        result_dir / "emissions_summary.csv",
        index=False,
    )