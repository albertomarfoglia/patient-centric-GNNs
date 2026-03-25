from dataclasses import dataclass
from pathlib import Path

@dataclass
class NeurovascConfig:
    name: str = "neurovasc"
    task: str = "stroke_outcome"
    num_patients: int = 10000

    classes: list[str] = ["B2H", "REHAB", "DEATH"]

    raw_data_dir: Path = Path("data/neurovasc")
    processed_dir: Path = Path("processed_data")