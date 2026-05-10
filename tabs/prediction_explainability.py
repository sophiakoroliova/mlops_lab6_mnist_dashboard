"""
tabs/prediction_explainability.py
Tab 3 — Prediction & Explainability (Grad-CAM).
Experiment and run are selected in the sidebar (app.py) and passed in.
"""

import logging
from typing import Optional

import numpy as np
import streamlit as st
import torch
from PIL import Image

from src.data_loader import CLASS_NAMES, get_sample, get_transform, tensor_to_numpy
from src.explainability import GradCAM
from src.mlflow_utils import RunSummary, artifact_exists
from src.model import load_model_from_mlflow, run_inference
from src.viz_utils import (
    plot_gradcam_overlay,
    plot_mnist_image,
    plot_probability_distribution,
)

logger = logging.getLogger(__name__)


def _load_model_cached(client, run_id: str, artifact_path: str):
    cache_key = f"expl_model_{run_id}"
    if cache_key not in st.session_state:
        if not artifact_exists(client, run_id, artifact_path):
            st.error(f"No model artifact found at `{artifact_path}` for run `{run_id}`.")
            st.stop()
        try:
            st.session_state[cache_key] = load_model_from_mlflow(run_id, artifact_path)
        except Exception as exc:
            st.error(f"Failed to load model: {exc}")
            logger.error("Model load error for run %s: %s", run_id, exc)
            st.stop()
    return st.session_state[cache_key]


def _gradcam_section(model, image_tensor: torch.Tensor, image_np: np.ndarray, pred: int, probs: list):
    st.subheader("Grad-CAM explanation")

    explain_class = st.selectbox(
        "Explain for class",
        options=list(range(10)),
        format_func=lambda i: f"{i}  (prob: {probs[i]:.1%})",
        index=pred,
        key="gradcam_class",
    )

    try:
        cam = GradCAM(model, target_layer_name="conv2")
        heatmap = cam.generate(image_tensor.clone(), class_idx=explain_class)
        overlay = cam.overlay(heatmap, image_np)
        cam.remove_hooks()

        st.pyplot(plot_gradcam_overlay(image_np, overlay), use_container_width=True)

        with st.expander("What is Grad-CAM?"):
            st.markdown(
                "**Gradient-weighted Class Activation Mapping (Grad-CAM)** highlights "
                "the regions of the input image that most influenced the model's prediction "
                "for the selected class. Warm colours (red/yellow) indicate high importance."
            )
    except Exception as exc:
        st.error(f"Grad-CAM failed: {exc}")
        logger.error("Grad-CAM error: %s", exc)


def render(client, cfg, train_ds, val_ds, test_ds, selected_run: Optional[RunSummary]) -> None:
    st.header("Prediction & explainability")

    # ── Guard: no run selected ───────────────────────────────────────────────
    if selected_run is None:
        st.info("Select an experiment and run from the sidebar to begin.")
        st.stop()

    st.caption(f"Run: **{selected_run.run_name}** · `{selected_run.run_id}`")
    st.divider()

    model = _load_model_cached(client, selected_run.run_id, cfg.model.artifact_path)

    # ── Input source ──────────────────────────────────────────────────────────
    source = st.radio(
        "Input source",
        ["Dataset sample", "Upload image"],
        horizontal=True,
        key="expl_source",
    )

    image_tensor = None
    image_np = None

    if source == "Dataset sample":
        ds_map = {"Train": train_ds, "Validation": val_ds, "Test": test_ds}
        col_split, col_idx = st.columns([1, 3])
        with col_split:
            split = st.selectbox("Split", list(ds_map.keys()), key="expl_split")
        with col_idx:
            idx = st.slider("Sample index", 0, len(ds_map[split]) - 1, 0, key="expl_idx")

        image_tensor, true_label = get_sample(ds_map[split], idx)
        image_np = tensor_to_numpy(image_tensor)
        logger.info("Selected dataset sample: split=%s idx=%d label=%d", split, idx, true_label)

    else:
        uploaded = st.file_uploader(
            "Upload a 28×28 grayscale image (PNG/JPG)",
            type=["png", "jpg", "jpeg"],
            key="expl_upload",
        )
        true_label = None

        if uploaded is None:
            st.info("Upload an image to run inference.")
            return

        try:
            pil_img = Image.open(uploaded).convert("L").resize((28, 28))
            image_np = np.array(pil_img)
            transform = get_transform()
            image_tensor = transform(pil_img)
            logger.info("User-uploaded image loaded and preprocessed")
        except Exception as exc:
            st.error(f"Failed to process uploaded image: {exc}")
            logger.error("Upload processing error: %s", exc)
            return

    # ── Inference ─────────────────────────────────────────────────────────────
    pred, confidence, probs = run_inference(model, image_tensor)
    logger.info("Inference result: pred=%d confidence=%.4f", pred, confidence)

    col_img, col_pred = st.columns([1, 2])

    with col_img:
        title = f"True: {true_label}" if true_label is not None else "Input"
        st.pyplot(plot_mnist_image(image_np, title=title), use_container_width=False)

    with col_pred:
        st.markdown(f"### Predicted: **{pred}**")
        st.markdown(f"Confidence: **{confidence:.1%}**")
        if true_label is not None:
            correct = pred == true_label
            st.markdown(
                f"Result: {'✅ Correct' if correct else f'❌ Wrong (true: {true_label})'}"
            )
        st.plotly_chart(
            plot_probability_distribution(probs, pred, true_label),
            use_container_width=True,
        )

    st.divider()

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    _gradcam_section(model, image_tensor, image_np, pred, probs)
