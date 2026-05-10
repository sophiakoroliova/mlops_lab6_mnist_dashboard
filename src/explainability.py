"""
src/explainability.py
Grad-CAM implementation for MNIST CNN.
Returns heatmaps as numpy arrays ready for overlay.
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Grad-CAM for any model with a named convolutional layer.

    Usage:
        cam = GradCAM(model, target_layer_name="conv2")
        heatmap = cam.generate(image_tensor, class_idx=predicted_class)
        overlay = cam.overlay(heatmap, original_image_np)
    """

    def __init__(self, model: nn.Module, target_layer_name: str = "conv2"):
        self.model = model
        self.target_layer_name = target_layer_name
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self._hook_handles: list = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        target = self._find_layer(self.target_layer_name)
        if target is None:
            raise ValueError(
                f"Layer '{self.target_layer_name}' not found in model. "
                f"Available: {[n for n, _ in self.model.named_modules()]}"
            )

        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self._hook_handles.append(target.register_forward_hook(forward_hook))
        self._hook_handles.append(target.register_full_backward_hook(backward_hook))
        logger.info("Grad-CAM hooks registered on layer '%s'", self.target_layer_name)

    def _find_layer(self, name: str) -> Optional[nn.Module]:
        for module_name, module in self.model.named_modules():
            if module_name == name:
                return module
        return None

    def generate(
        self,
        image: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap.

        Args:
            image: (1, C, H, W) or (C, H, W) tensor.
            class_idx: Target class. If None, uses the predicted class.

        Returns:
            heatmap: (H, W) float32 array in [0, 1].
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.requires_grad_(True)
        self.model.zero_grad()

        logits = self.model(image)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        score = logits[0, class_idx]
        score.backward()

        # Pool gradients over spatial dimensions
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Upsample to input size
        cam = F.interpolate(
            cam,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().numpy()

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        logger.info("Grad-CAM generated for class %d", class_idx)
        return cam.astype(np.float32)

    def overlay(
        self,
        heatmap: np.ndarray,
        original_image: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Overlay a heatmap (H, W) on a grayscale image (H, W) uint8.
        Returns an (H, W, 3) RGB uint8 array.
        """
        import matplotlib.cm as cm

        # Apply colormap to heatmap
        colormap = cm.get_cmap("jet")
        heatmap_rgb = colormap(heatmap)[:, :, :3]  # (H, W, 3) float
        heatmap_rgb = (heatmap_rgb * 255).astype(np.uint8)

        # Convert grayscale to RGB
        if original_image.ndim == 2:
            original_rgb = np.stack([original_image] * 3, axis=-1)
        else:
            original_rgb = original_image

        blended = (alpha * heatmap_rgb + (1 - alpha) * original_rgb).astype(np.uint8)
        return blended

    def remove_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
