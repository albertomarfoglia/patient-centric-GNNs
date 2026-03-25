# configs/model.py

from dataclasses import dataclass

@dataclass
class ModelConfig:
    embed_dim: int = 32
    hidden_dim: int = 32
    dropout: float = 0.0
    lr: float = 5e-4
    weight_decay: float = 5e-4