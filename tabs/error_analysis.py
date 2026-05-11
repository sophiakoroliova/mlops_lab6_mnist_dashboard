"""
tabs/error_analysis.py
Tab 2 — Model Error Analysis.
Experiment and run are selected in the sidebar (app.py) and passed in.
"""

import logging
from typing import Optional

import streamlit as st
import torch

from src.data_loader import CLASS_NAMES, get_dataloader, tensor_to_numpy
from src.mlflow_utils import RunSummary, artifact_exists
from src.model import load_model_from_mlflow, run_inference
from src.viz_utils import (
    plot_confusion_matrix,
    plot_mnist_image,
    plot_per_class_errors,
)

logger = logging.getLogger(__name__)


def _run_error_extraction(model, test_ds) -> tuple[list, list, list]:
    """
    Run inference on the full test set.
    Returns (y_true, y_pred, misclassified_indices).
    """
    loader = get_dataloader(test_ds, batch_size=256)
    y_true, y_pred = [], []

    with st.spinner("Running inference on test set…"):
        for images, labels in loader:
            with torch.no_grad():
                logits = model(images)
                preds = logits.argmax(dim=1).tolist()
            y_true.extend(labels.tolist())
            y_pred.extend(preds)

    misclassified = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != p]
    logger.info(
        "Inference complete — %d errors out of %d samples (%.1f%%)",
        len(misclassified), len(y_true), 100 * len(misclassified) / len(y_true),
    )
    return y_true, y_pred, misclassified


def render(client, cfg, test_ds, selected_run: Optional[RunSummary]) -> None:
    st.header("Model error analysis")

    # ── Guard: no run selected ───────────────────────────────────────────────
    if selected_run is None:
        st.info("Select an experiment and run from the sidebar to begin.")
        st.stop()

    # ── Run info banner ──────────────────────────────────────────────────────
    st.caption(f"Run: **{selected_run.run_name}** · `{selected_run.run_id}`")

    with st.expander("Run details"):
        col_m, col_p = st.columns(2)
        with col_m:
            st.markdown("**Metrics**")
            for k, v in selected_run.metrics.items():
                st.markdown(f"- `{k}`: {v:.4f}")
        with col_p:
            st.markdown("**Params**")
            for k, v in selected_run.params.items():
                st.markdown(f"- `{k}`: {v}")

    st.divider()

    # ── Load model ───────────────────────────────────────────────────────────
    if not artifact_exists(client, selected_run.run_id, cfg.model.artifact_path):
        st.error(
            f"No model artifact found at path `{cfg.model.artifact_path}` "
            f"for run `{selected_run.run_id}`. The run may not have logged a model."
        )
        st.stop()

    cache_key = f"model_{selected_run.run_id}"
    if cache_key not in st.session_state:
        try:
            st.session_state[cache_key] = load_model_from_mlflow(
                selected_run.run_id, cfg.model.artifact_path
            )
        except Exception as exc:
            st.error(f"Failed to load model: {exc}")
            logger.error("Model load failed for run %s: %s", selected_run.run_id, exc)
            st.stop()

    model = st.session_state[cache_key]

    # ── Inference & error extraction ─────────────────────────────────────────
    results_key = f"results_{selected_run.run_id}"
    if results_key not in st.session_state:
        y_true, y_pred, misclassified = _run_error_extraction(model, test_ds)
        st.session_state[results_key] = (y_true, y_pred, misclassified)

    y_true, y_pred, misclassified = st.session_state[results_key]

    # ── Summary metrics ──────────────────────────────────────────────────────
    total = len(y_true)
    n_errors = len(misclassified)
    accuracy = (total - n_errors) / total

    c1, c2, c3 = st.columns(3)
    c1.metric("Test samples", total)
    c2.metric("Errors",       n_errors)
    c3.metric("Accuracy",     f"{accuracy:.2%}")

    st.divider()

    # ── Confusion matrix + per-class errors ──────────────────────────────────
    col_cm, col_err = st.columns(2)
    with col_cm:
        st.subheader("Confusion matrix")
        st.pyplot(plot_confusion_matrix(y_true, y_pred), use_container_width=True)
    with col_err:
        st.subheader("Per-class errors")
        st.plotly_chart(plot_per_class_errors(y_true, y_pred), use_container_width=True)

    st.divider()

    # ── Individual misclassified examples ────────────────────────────────────
    st.subheader(f"Misclassified examples ({n_errors} total)")

    sort_by = st.selectbox(
        "Sort by",
        ["Index", "Predicted class", "True class", "Confidence (desc)"],
        key="sort_errors",
    )

    error_records = []
    for idx in misclassified:
        img, label = test_ds[idx]
        pred, conf, _ = run_inference(model, img)
        error_records.append({"idx": idx, "true": label, "pred": pred, "conf": conf, "img": img})

    if sort_by == "Predicted class":
        error_records.sort(key=lambda r: r["pred"])
    elif sort_by == "True class":
        error_records.sort(key=lambda r: r["true"])
    elif sort_by == "Confidence (desc)":
        error_records.sort(key=lambda r: r["conf"], reverse=True)

    page_size = 12
    total_pages = max(1, (len(error_records) + page_size - 1) // page_size)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1) - 1
    page_records = error_records[page * page_size: (page + 1) * page_size]

    cols = st.columns(6)
    for i, record in enumerate(page_records):
        with cols[i % 6]:
            img_np = tensor_to_numpy(record["img"])
            st.pyplot(
                plot_mnist_image(
                    img_np,
                    title=f"True: {record['true']} | Pred: {record['pred']}",
                ),
                use_container_width=True,
            )
            st.caption(f"Confidence: {record['conf']:.1%}  |  idx: {record['idx']}")
