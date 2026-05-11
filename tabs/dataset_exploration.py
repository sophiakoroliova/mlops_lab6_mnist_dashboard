"""
tabs/dataset_exploration.py
Tab 1 — Dataset Exploration.
Strictly analysis and visualisation; no model inference.
"""

import logging

import streamlit as st

from src.data_loader import (
    CLASS_NAMES,
    get_class_distribution,
    get_sample,
    get_samples_by_class,
    get_split_info,
    tensor_to_numpy,
)
from src.viz_utils import (
    plot_class_distribution,
    plot_mnist_image,
    plot_split_sizes,
)

logger = logging.getLogger(__name__)


def render(train_ds, val_ds, test_ds) -> None:
    st.header("Dataset Overview")

    # ── Overview ────────────────────────────────────────────────────────────
    split_info = get_split_info(train_ds, val_ds, test_ds)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total samples", split_info.train_size + split_info.val_size + split_info.test_size)
    col2.metric("Training",      split_info.train_size)
    col3.metric("Validation",    split_info.val_size)
    col4.metric("Test",          split_info.test_size)

    st.divider()

    col_pie, col_bar = st.columns(2)
    with col_pie:
        st.plotly_chart(
            plot_split_sizes(split_info.train_size, split_info.val_size, split_info.test_size),
            use_container_width=True,
        )
    with col_bar:
        split_label = st.selectbox(
            "Show class distribution for split",
            ["Train", "Validation", "Test"],
            key="dist_split",
        )
        ds_map = {"Train": train_ds, "Validation": val_ds, "Test": test_ds}
        dist = get_class_distribution(ds_map[split_label])
        st.plotly_chart(
            plot_class_distribution(dist, title=f"{split_label} class distribution"),
            use_container_width=True,
        )

    st.divider()

    # ── Sample inspection ─────────────────────────────────────────────────────
    st.subheader("Sample inspection")

    # All controls in one row
    col_split, col_class, col_idx = st.columns([2, 2, 3])

    with col_split:
        selected_split = st.selectbox(
            "Split",
            ["Train", "Validation", "Test"],
            key="insp_split",
        )

    active_ds = ds_map[selected_split]

    with col_class:
        filter_class = st.selectbox(
            "Filter by class",
            ["All"] + CLASS_NAMES,
            key="insp_class",
        )

    with col_idx:
        if filter_class == "All":
            max_idx = len(active_ds) - 1
            sample_idx = st.number_input(
                "Sample index",
                min_value=0,
                max_value=max_idx,
                value=0,
                step=1,
                key="insp_idx",
            )
        else:
            valid_indices = get_samples_by_class(active_ds, int(filter_class))
            if not valid_indices:
                st.warning(f"No samples found for class {filter_class} in {selected_split}.")
                return
            max_pos = len(valid_indices) - 1
            position = st.number_input(
                f"Sample index (class {filter_class})",
                min_value=0,
                max_value=max_pos,
                value=0,
                step=1,
                key="insp_pos",
            )
            sample_idx = valid_indices[int(position)]

    # ── Image + metadata side by side ─────────────────────────────────────────
    image_tensor, label = get_sample(active_ds, int(sample_idx))
    image_np = tensor_to_numpy(image_tensor)

    logger.info("Displaying sample idx=%d label=%d from %s", sample_idx, label, selected_split)

    col_img, col_meta = st.columns([1, 2])

    with col_img:
        st.pyplot(plot_mnist_image(image_np, title=""), use_container_width=True)

    with col_meta:
        st.markdown("<br><br>", unsafe_allow_html=True)

        st.markdown(
            f"""
                <div style="background-color:#cce5ff; padding:12px 18px; border-radius:6px;
                            color:#003d7a; font-weight:700; font-size:18px; margin-bottom:12px;">
                    True label: {label}
                </div>
                """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
                <div style="background-color:#f0f2f6; padding:12px 18px; border-radius:6px;
                            font-size:15px; line-height:2;">
                    <b>Shape:</b> {image_np.shape[0]} × {image_np.shape[1]} px<br>
                    <b>Channels:</b> 1 (grayscale)<br>
                    <b>Pixel min:</b> {int(image_np.min())}<br>
                    <b>Pixel max:</b> {int(image_np.max())}<br>
                    <b>Intensity range:</b> [{int(image_np.min())}, {int(image_np.max())}]
                </div>
                """,
            unsafe_allow_html=True,
        )

