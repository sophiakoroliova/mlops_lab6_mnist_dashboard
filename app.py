"""
app.py
Entry point for the MNIST MLOps Dashboard.

Run with:
    streamlit run app.py
"""

import logging
import sys

import streamlit as st

# ── Logging setup (must be before any src imports) ───────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("app")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MNIST MLOps Dashboard",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports (after page config) ───────────────────────────────────────────────
from mlflow.tracking import MlflowClient

from config.config import load_config
from src.data_loader import load_datasets
from src.mlflow_utils import get_experiments, get_runs, set_tracking_uri

import tabs.dataset_exploration as tab_data
import tabs.error_analysis as tab_errors
import tabs.prediction_explainability as tab_explain


# ── Cached resource loaders ───────────────────────────────────────────────────

@st.cache_resource
def get_config():
    logger.info("Loading app configuration")
    return load_config()


@st.cache_resource
def get_datasets(_cfg):
    logger.info("Loading MNIST datasets")
    return load_datasets(_cfg.data.path, _cfg.data.val_split, _cfg.data.random_seed)


@st.cache_resource
def get_mlflow_client(_cfg):
    logger.info("Initialising MLflow client at %s", _cfg.mlflow.tracking_uri)
    set_tracking_uri(_cfg.mlflow.tracking_uri)
    return MlflowClient()


# ── Main app ──────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = get_config()
    train_ds, val_ds, test_ds = get_datasets(cfg)
    client = get_mlflow_client(cfg)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🔢 MNIST Dashboard")
        st.divider()

        # Experiment selector
        experiments = get_experiments(client)

        if not experiments:
            st.warning("No experiments found. Run `python train.py` first.")
            selected_run = None
        else:
            exp_names = [e.name for e in experiments]

            # Pre-select the experiment defined in settings.yaml if it exists
            default_exp_idx = (
                exp_names.index(cfg.mlflow.experiment_name)
                if cfg.mlflow.experiment_name in exp_names
                else 0
            )

            selected_experiment_name = st.selectbox(
                "Experiment",
                exp_names,
                index=default_exp_idx,
            )

            # Run selector — updates automatically when experiment changes
            runs = get_runs(client, selected_experiment_name)

            if not runs:
                st.warning("No runs in this experiment.")
                selected_run = None
            else:
                run_labels = [f"{r.run_name} ({r.run_id[:8]})" for r in runs]
                selected_run_idx = st.selectbox(
                    "Run",
                    range(len(run_labels)),
                    format_func=lambda i: run_labels[i],
                )
                selected_run = runs[selected_run_idx]

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📊 Dataset exploration",
        "❌ Error analysis",
        "🔍 Prediction & explainability",
    ])

    with tab1:
        tab_data.render(train_ds, val_ds, test_ds)

    with tab2:
        # selected_run is passed in directly — no run selector inside the tab
        tab_errors.render(client, cfg, test_ds, selected_run)

    with tab3:
        tab_explain.render(client, cfg, train_ds, val_ds, test_ds, selected_run)


if __name__ == "__main__":
    main()
