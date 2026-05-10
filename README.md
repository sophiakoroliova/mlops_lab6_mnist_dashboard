# MNIST MLOps Dashboard — Lab 6

Interactive Streamlit dashboard for dataset exploration, model error analysis,
and Grad-CAM explainability. Integrates with MLflow for experiment tracking.

---

## Project structure

```
mnist-dashboard/
├── app.py                        # Entry point — streamlit run app.py
├── train.py                      # Standalone training script
├── pyproject.toml                # Poetry dependencies
├── config/
│   ├── settings.yaml             # All configuration (no hard-coded paths in code)
│   └── config.py                 # Typed config loader
├── src/
│   ├── data_loader.py            # MNIST loading & split utilities
│   ├── model.py                  # CNN architecture + MLflow artifact loader
│   ├── mlflow_utils.py           # MLflow client wrappers
│   ├── explainability.py         # Grad-CAM implementation
│   └── viz_utils.py              # Matplotlib / Plotly figure builders
└── tabs/
    ├── dataset_exploration.py    # Tab 1
    ├── error_analysis.py         # Tab 2
    └── prediction_explainability.py  # Tab 3
```

---

## Setup

### 1. Install Poetry (if not already installed)

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 2. Install dependencies

```bash
cd mnist-dashboard
poetry install
```

### 3. Start MLflow tracking server

Open a separate terminal and run:

```bash
poetry run mlflow ui --port 5000
```

Keep this running. The dashboard connects to `http://127.0.0.1:5000` by default
(configurable in `config/settings.yaml`).

### 4. Train a model (creates your first MLflow run)

```bash
poetry run python train.py
```

This downloads MNIST, trains a CNN for 5 epochs, logs metrics and the model
artifact to MLflow, and exits. You can run it multiple times with different
settings to create multiple runs for comparison.

### 5. Launch the dashboard

```bash
poetry run streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Configuration

All runtime parameters live in `config/settings.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `mlflow.tracking_uri` | `http://127.0.0.1:5000` | MLflow server URL |
| `mlflow.experiment_name` | `mnist-experiment` | Experiment name |
| `data.path` | `./data` | Dataset download location |
| `data.val_split` | `0.1` | Fraction of training set used for validation |
| `training.epochs` | `5` | Training epochs |
| `training.batch_size` | `64` | Batch size |
| `training.learning_rate` | `0.001` | Adam learning rate |
| `gradcam.target_layer` | `conv2` | Conv layer name for Grad-CAM hooks |

---

## Dashboard tabs

### Tab 1 — Dataset exploration
- Total sample counts and train/val/test split proportions
- Class distribution bar chart (per split)
- Sample inspector: browse by index or filter by class

### Tab 2 — Error analysis
- Select any MLflow run
- Runs inference on the full test set
- Confusion matrix and per-class error counts
- Paginated grid of misclassified examples with true label, predicted label, and confidence
- Sortable by predicted class, true class, or confidence

### Tab 3 — Prediction & explainability
- Select any MLflow run
- Run inference on a dataset sample or an uploaded image
- Probability distribution across all 10 classes
- Grad-CAM heatmap with class selection (explain any class, not just the top prediction)

---

## Notes

- The dashboard is **read-only** — it never triggers retraining. This is intentional.
- Models are cached in `st.session_state` to avoid re-downloading on every interaction.
- All configuration is loaded once and cached with `@st.cache_resource`.
- Logging goes to stdout; check the terminal running `streamlit run` for debug output.
