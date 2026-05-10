"""
config/config.py
Load and expose settings.yaml as typed dataclasses.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "settings.yaml"


@dataclass
class MLflowConfig:
    tracking_uri: str
    experiment_name: str


@dataclass
class DataConfig:
    path: str
    val_split: float
    random_seed: int


@dataclass
class ModelConfig:
    type: str
    input_channels: int
    num_classes: int
    artifact_path: str


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float


@dataclass
class GradCAMConfig:
    target_layer: str


@dataclass
class AppConfig:
    mlflow: MLflowConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    gradcam: GradCAMConfig


def load_config(path: Path = CONFIG_PATH) -> AppConfig:
    logger.info("Loading configuration from %s", path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return AppConfig(
        mlflow=MLflowConfig(**raw["mlflow"]),
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
        gradcam=GradCAMConfig(**raw["gradcam"]),
    )
