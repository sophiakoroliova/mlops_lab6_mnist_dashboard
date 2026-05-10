"""
src/model.py
CNN model definition and MLflow artifact loading.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MnistCNN(nn.Module):
    """
    Simple two-conv-block CNN for MNIST classification.
    conv1 → conv2 are named so Grad-CAM can hook them by name.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return self.classifier(x)


def load_model_from_path(path: str, num_classes: int = 10) -> MnistCNN:
    """Load a saved state-dict from a local .pt file."""
    logger.info("Loading model state-dict from %s", path)
    model = MnistCNN(num_classes=num_classes)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def load_model_from_mlflow(run_id: str, artifact_path: str = "model") -> MnistCNN:
    """
    Load a PyTorch model logged with mlflow.pytorch.log_model().
    Returns the model in eval mode on CPU.
    """
    import mlflow.pytorch

    uri = f"runs:/{run_id}/{artifact_path}"
    logger.info("Loading model from MLflow artifact URI: %s", uri)
    try:
        model = mlflow.pytorch.load_model(uri, map_location="cpu")
        model.eval()
        logger.info("Model loaded successfully from run %s", run_id)
        return model
    except Exception as exc:
        logger.error("Failed to load model from MLflow run %s: %s", run_id, exc)
        raise


@torch.no_grad()
def run_inference(
    model: nn.Module,
    image: torch.Tensor,
) -> tuple[int, float, list[float]]:
    """
    Run inference on a single (C,H,W) or (1,C,H,W) image tensor.
    Returns (predicted_class, confidence, all_probabilities).
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
    logits = model(image)
    probs = torch.softmax(logits, dim=1).squeeze().tolist()
    pred = int(torch.argmax(logits, dim=1).item())
    confidence = float(probs[pred])
    return pred, confidence, probs
