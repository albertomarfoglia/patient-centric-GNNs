# configs/experiment.py

from pathlib import Path

class ExperimentConfig:
    def __init__(
        self,
        folds: int,
        data_mode,
        enrich_by_graphs: list[Path] = [],
        enrich_events: dict[str, str] = dict(),
        dataset_samples: int = 1,
        time_option = "TS",
        include_text = False,
        model_type = "rgcn"
    ):
        self.folds = folds
        self.dataset_samples = dataset_samples
        self.time_option = time_option
        self.include_text = include_text
        self.enrich_events = enrich_events
        self.enrich_by_graphs = enrich_by_graphs
        self.data_mode = data_mode
        self.model_type = model_type