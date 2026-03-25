from pathlib import Path

from configs.experiment import ExperimentConfig
from configs.loader import LoaderConfig


class MimicConfig:
    def __init__(
        self,
        source_dir: Path = Path("data/mimic"),
        processed_dir: Path = Path("processed_data"),
        task = "inhospital_mortality",
        num_patients = 1000,
    ):
        self.name: str = "mimic"
        self.task: str = task
        self.num_patients: int = num_patients

        self.classes = ["FALSE", "TRUE"]

        self.source_dir: Path = source_dir
        self.processed_dir: Path = processed_dir

    def generate(self, idx: int, exp: ExperimentConfig) -> LoaderConfig:
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        return LoaderConfig(
            entities_path=self.processed_dir
            / f"{exp.data_mode.data_model}_{exp.time_option}_entities_{self.num_patients}_{idx}.tsv",
            numeric_values_path=self.processed_dir
            / f"{exp.data_mode.data_model}_{exp.time_option}_numeric_{self.num_patients}_{idx}.npy",
            relations_path=self.processed_dir
            / f"{exp.data_mode.data_model}_{exp.time_option}_relations_{self.num_patients}_{idx}.npy",
            text_values_path=self.processed_dir
            / f"{exp.data_mode.data_model}_{exp.time_option}_text_{self.num_patients}_{idx}.npy",
            triples_path=self.processed_dir
            / f"{exp.data_mode.data_model}_{exp.time_option}_triples_{self.num_patients}_{idx}.tsv",
            classes=self.classes,
            dataset_dir=self.source_dir
            / self.task
            / f"{exp.data_mode.data_model}_{self.num_patients}_{idx}",
            outcomes_path=self.source_dir
            / self.task
            / "labels"
            / f"outcomes_{exp.data_mode.data_model}_{exp.time_option}_{self.num_patients}_{idx}.joblib",
            results_dir=Path("results") / exp.data_mode.data_model / self.task
        )
