"""
src/viz_utils.py
Visualisation utilities — return figure objects, never call plt.show().
All figures are compatible with st.pyplot() and st.plotly_chart().
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)

CLASS_NAMES = [str(i) for i in range(10)]


def plot_class_distribution(
    distribution: dict[str, int],
    title: str = "Class distribution",
) -> go.Figure:
    """Horizontal bar chart of class counts."""
    labels = list(distribution.keys())
    counts = list(distribution.values())

    fig = px.bar(
        x=counts,
        y=labels,
        orientation="h",
        title=title,
        labels={"x": "Samples", "y": "Class"},
        color=counts,
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=40, r=20, t=40, b=20),
        height=320,
    )
    logger.debug("Class distribution chart created")
    return fig


def plot_split_sizes(train: int, val: int, test: int) -> go.Figure:
    """Donut chart of dataset split proportions."""
    fig = go.Figure(
        go.Pie(
            labels=["Train", "Validation", "Test"],
            values=[train, val, test],
            hole=0.5,
            marker_colors=["#3B82F6", "#10B981", "#F59E0B"],
        )
    )
    fig.update_layout(
        title="Dataset splits",
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
        showlegend=True,
    )
    return fig


def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str] = CLASS_NAMES,
) -> plt.Figure:
    """Annotated confusion matrix heatmap using matplotlib."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=9,
            )

    fig.tight_layout()
    logger.debug("Confusion matrix plot created")
    return fig


def plot_per_class_errors(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str] = CLASS_NAMES,
) -> go.Figure:
    """Bar chart of per-class error counts."""
    error_counts = [0] * len(class_names)
    for t, p in zip(y_true, y_pred):
        if t != p:
            error_counts[t] += 1

    fig = px.bar(
        x=class_names,
        y=error_counts,
        title="Errors per class (true label)",
        labels={"x": "Class", "y": "Misclassified samples"},
        color=error_counts,
        color_continuous_scale="Reds",
    )
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=40, r=20, t=40, b=20),
        height=300,
    )
    return fig


def plot_probability_distribution(
    probs: list[float],
    predicted: int,
    true_label: int | None = None,
    class_names: list[str] = CLASS_NAMES,
) -> go.Figure:
    """Bar chart of softmax probabilities for all classes."""
    colors = []
    for i in range(len(class_names)):
        if i == predicted:
            colors.append("#3B82F6")
        elif true_label is not None and i == true_label:
            colors.append("#10B981")
        else:
            colors.append("#CBD5E1")

    fig = go.Figure(
        go.Bar(
            x=class_names,
            y=probs,
            marker_color=colors,
            text=[f"{p:.1%}" for p in probs],
            textposition="auto",
        )
    )
    fig.update_layout(
        title="Prediction probabilities",
        xaxis_title="Class",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        margin=dict(l=40, r=20, t=40, b=20),
        height=300,
    )
    return fig


def plot_mnist_image(
    image_np: np.ndarray,
    title: str = "",
) -> plt.Figure:
    """Display a single MNIST image (H, W) uint8."""
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(image_np, cmap="gray", vmin=0, vmax=255)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout(pad=0.5)
    return fig


def plot_gradcam_overlay(
    original_np: np.ndarray,
    overlay_np: np.ndarray,
) -> plt.Figure:
    """Side-by-side original and Grad-CAM overlay."""
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(original_np, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Input image", fontsize=10)
    axes[0].axis("off")
    axes[1].imshow(overlay_np)
    axes[1].set_title("Grad-CAM overlay", fontsize=10)
    axes[1].axis("off")
    fig.tight_layout(pad=0.5)
    return fig
