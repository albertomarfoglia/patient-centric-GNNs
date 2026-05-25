from dataclasses import dataclass
from pathlib import Path

from configs.experiment import ExperimentConfig
from configs.loader import LoaderConfig

@dataclass
class NeurovascConfig:
    def __init__(
        self,
        source_dir: Path = Path("data"),
        name = "neurovasc",
        task="stroke_outcome",
        processed_dir: Path = Path("processed_data"),
        classes = ["B2H", "REHAB", "DEATH"],
        num_patients=10000,
        external_codes=Path("MEDS_cohort/neurovasc_external_codes")
    ):
        
        self.name = name
        self.task: str = task
        self.num_patients: int = num_patients
        self.external_codes = external_codes
        self.classes = classes
        self.source_dir: Path = source_dir
        self.processed_dir: Path = processed_dir / task

    def generate(self, idx: int, exp: ExperimentConfig) -> LoaderConfig:
        sample_processed_dir = Path(f"{self.processed_dir}")
        sample_result_dir = Path(
            f"results/{exp.data_mode.data_model}/{exp.model_type.__name__}/{self.task}"
        )
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
            #/ self.name / "graph"
            / f"{exp.data_mode.data_model}_{self.num_patients}",
            outcomes_path=self.source_dir
            #/ self.name / "graph"
            / "labels"
            / f"outcomes_{exp.data_mode.data_model}_{exp.time_option}_{self.num_patients}.joblib",
            results_dir=sample_result_dir,
            data_mode=exp.data_mode.data_model,
            onto_codes=self.source_dir / self.task / "meds" / self.external_codes,
            sample_processed_dir=sample_processed_dir
        )