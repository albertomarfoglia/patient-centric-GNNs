from pathlib import Path

from configs.experiment import ExperimentConfig
from configs.loader import LoaderConfig


class MimicConfig:
    def __init__(
        self,
        source_dir: Path = Path("data/mimic"),
        task="inhospital_mortality",
        processed_dir: Path = Path("processed_data"),
        num_patients=1000,
    ):
        self.name: str = "mimic"
        self.task: str = task
        self.num_patients: int = num_patients

        self.classes = ["FALSE", "TRUE"]

        self.source_dir: Path = source_dir
        self.processed_dir: Path = processed_dir / task

    def generate(self, idx: int, exp: ExperimentConfig) -> LoaderConfig:
        sample_processed_dir = Path(f"{self.processed_dir}/{idx}")
        sample_processed_dir.mkdir(parents=True, exist_ok=True)
        sample_result_dir = Path(
            f"results/{exp.data_mode.data_model}/{exp.model_type}/{self.task}/{idx}"
        )
        sample_result_dir.mkdir(parents=True, exist_ok=True)
        return LoaderConfig(
            entities_path=sample_processed_dir
            / f"{exp.data_mode.data_model}_{exp.time_option}_entities_{self.num_patients}.tsv",
            numeric_values_path=sample_processed_dir
            / f"{exp.data_mode.data_model}_{exp.time_option}_numeric_{self.num_patients}.npy",
            relations_path=sample_processed_dir
            / f"{exp.data_mode.data_model}_{exp.time_option}_relations_{self.num_patients}.npy",
            text_values_path=sample_processed_dir
            / f"{exp.data_mode.data_model}_{exp.time_option}_text_{self.num_patients}.npy",
            triples_path=sample_processed_dir
            / f"{exp.data_mode.data_model}_{exp.time_option}_triples_{self.num_patients}.tsv",
            classes=self.classes,
            dataset_dir=self.source_dir
            / self.task
            / "graph"
            / f"{exp.data_mode.data_model}_{self.num_patients}_{idx}",
            outcomes_path=self.source_dir
            / self.task
            / "graph"
            / "labels"
            / f"outcomes_{exp.data_mode.data_model}_{exp.time_option}_{self.num_patients}_{idx}.joblib",
            results_dir=sample_result_dir,
        )
