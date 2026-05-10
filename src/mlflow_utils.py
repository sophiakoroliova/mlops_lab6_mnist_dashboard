"""
src/mlflow_utils.py
Thin wrappers around the MLflow tracking client.
All MLflow queries go through here — never call mlflow directly from pages/.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import mlflow
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


@dataclass
class RunSummary:
    run_id: str
    run_name: str
    status: str
    start_time: Optional[int]
    metrics: dict[str, float]
    params: dict[str, str]


def set_tracking_uri(uri: str) -> None:
    mlflow.set_tracking_uri(uri)
    logger.info("MLflow tracking URI set to %s", uri)


def get_experiments(client: MlflowClient) -> list:
    try:
        experiments = client.search_experiments()
        # Filter out the default experiment (id "0") to avoid clutter
        experiments = [e for e in experiments if e.experiment_id != "0"]
        logger.info("Found %d experiments", len(experiments))
        return experiments
    except Exception as exc:
        logger.error("Failed to fetch experiments: %s", exc)
        return []


def get_runs(client: MlflowClient, experiment_name: str) -> list[RunSummary]:
    """Return all runs for an experiment as RunSummary objects."""
    try:
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            logger.warning("Experiment '%s' not found", experiment_name)
            return []

        runs: list[Run] = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
        )

        summaries = [
            RunSummary(
                run_id=r.info.run_id,
                run_name=r.info.run_name or r.info.run_id[:8],
                status=r.info.status,
                start_time=r.info.start_time,
                metrics=r.data.metrics,
                params=r.data.params,
            )
            for r in runs
        ]
        logger.info("Found %d runs in experiment '%s'", len(summaries), experiment_name)
        return summaries

    except Exception as exc:
        logger.error("Failed to fetch runs for experiment '%s': %s", experiment_name, exc)
        return []


def get_run_metrics(client: MlflowClient, run_id: str) -> dict[str, float]:
    """Fetch all logged metrics for a specific run."""
    try:
        run = client.get_run(run_id)
        return dict(run.data.metrics)
    except Exception as exc:
        logger.error("Failed to fetch metrics for run %s: %s", run_id, exc)
        return {}


def get_run_params(client: MlflowClient, run_id: str) -> dict[str, str]:
    """Fetch all logged params for a specific run."""
    try:
        run = client.get_run(run_id)
        return dict(run.data.params)
    except Exception as exc:
        logger.error("Failed to fetch params for run %s: %s", run_id, exc)
        return {}


def artifact_exists(client: MlflowClient, run_id: str, artifact_path: str) -> bool:
    """Check whether a model artifact exists for the given run."""
    try:
        artifacts = client.list_artifacts(run_id, path=artifact_path)
        return len(artifacts) > 0
    except Exception as exc:
        logger.warning("Could not verify artifact '%s' for run %s: %s", artifact_path, run_id, exc)
        return False
