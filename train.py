"""
train.py
Standalone training script. Run BEFORE the dashboard.

    poetry run python train.py

Trains a CNN on MNIST, evaluates on the test set, and logs
all metrics, params, and the model artifact to MLflow.
"""

import logging
import sys

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Make sure src/ is importable when running from project root
sys.path.insert(0, ".")

from config.config import load_config
from src.data_loader import get_dataloader, load_datasets
from src.model import MnistCNN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("train")


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total


def main() -> None:
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ── Data ────────────────────────────────────────────────────────────────
    train_ds, val_ds, test_ds = load_datasets(
        cfg.data.path, cfg.data.val_split, cfg.data.random_seed
    )
    train_loader = get_dataloader(train_ds, cfg.training.batch_size, shuffle=True)
    val_loader   = get_dataloader(val_ds,   cfg.training.batch_size)
    test_loader  = get_dataloader(test_ds,  cfg.training.batch_size)

    # ── MLflow ──────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name="baseline-cnn"):
        run_id = mlflow.active_run().info.run_id
        logger.info("MLflow run started: %s", run_id)

        mlflow.log_params({
            "epochs":       cfg.training.epochs,
            "batch_size":   cfg.training.batch_size,
            "learning_rate": cfg.training.learning_rate,
            "optimizer":    "adam",
            "architecture": "MnistCNN",
        })

        # ── Model / optimiser / loss ─────────────────────────────────────────
        model     = MnistCNN(cfg.model.num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

        # ── Training loop ────────────────────────────────────────────────────
        for epoch in range(1, cfg.training.epochs + 1):
            model.train()
            train_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "val_acc":    val_acc,
            }, step=epoch)

            logger.info(
                "Epoch %d/%d — train_loss: %.4f  val_loss: %.4f  val_acc: %.4f",
                epoch, cfg.training.epochs, train_loss, val_loss, val_acc,
            )

        # ── Final test evaluation ─────────────────────────────────────────────
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        mlflow.log_metrics({"test_loss": test_loss, "test_acc": test_acc})
        logger.info("Test accuracy: %.4f", test_acc)

        # ── Log model artifact ───────────────────────────────────────────────
        mlflow.pytorch.log_model(model, artifact_path=cfg.model.artifact_path)
        logger.info("Model artifact logged to MLflow run %s", run_id)

    logger.info("Training complete. Run 'streamlit run app.py' to open the dashboard.")


if __name__ == "__main__":
    main()
