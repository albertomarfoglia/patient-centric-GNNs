from pathlib import Path
from typing import List, Literal
import os

class LoaderConfig:
    def __init__(
        self,
        dataset_dir: Path,
        outcomes_path: Path,
        triples_path: Path,
        entities_path: Path,
        relations_path: Path,
        numeric_values_path: Path,
        classes: List[str],
        results_dir: Path,
        onto_codes: Path,
        sample_processed_dir: Path,
        text_values_path: Path | None = None,
        data_mode: Literal["meds", "sphn_pc"] = "meds",
    ):
        self.triples_path = triples_path
        self.entities_path = entities_path
        self.relations_path = relations_path
        self.numeric_values_path = numeric_values_path
        self.text_values_path = text_values_path
        self.data_mode = data_mode
        self.classes = classes
        self.num_classes = len(classes)

        self.dataset_dir = dataset_dir
        self.outcomes_path = outcomes_path
        self.results_dir = results_dir
        self.onto_codes = onto_codes
        self.sample_processed_dir = sample_processed_dir

        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(f"{self.results_dir}/cm", exist_ok=True)

        self.sample_processed_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
