"""Class-conditioned TF evidence extraction.

This module is optional at import time. PyTorch and torchvision are only needed
when the learned evidence extractor is used.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class PerImageStandardize:
    """Normalize each image tensor by its own mean and standard deviation."""

    def __call__(self, tensor: Any) -> Any:
        mean = tensor.mean()
        std = tensor.std()
        return (tensor - mean) / (std + 1e-6)


def build_mobilenetv3_small(num_classes: int, pretrained: bool = False) -> Any:
    """Build the classifier backbone used by the TF evidence front end."""
    import torch.nn as nn
    from torchvision import models

    if pretrained:
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=weights)
    else:
        model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, int(num_classes))
    return model


def load_checkpoint_class_to_idx(checkpoint: dict[str, Any]) -> dict[str, int]:
    """Read class mapping metadata from a checkpoint dictionary."""
    for key in ("class_to_idx", "meta", "metadata"):
        value = checkpoint.get(key)
        if isinstance(value, dict) and all(isinstance(k, str) for k in value):
            if all(isinstance(v, int) for v in value.values()):
                return dict(value)
            nested = value.get("class_to_idx")
            if isinstance(nested, dict):
                return {str(k): int(v) for k, v in nested.items()}
    return {}


class TFEvidenceExtractor:
    """Compute a normalized class-conditioned evidence map from a classifier."""

    def __init__(self, model: Any, target_layer: Any):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._handles = [
            target_layer.register_forward_hook(self._forward_hook),
            target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _forward_hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        self.activations = output.detach()

    def _backward_hook(self, _module: Any, _grad_inputs: Any, grad_outputs: Any) -> None:
        self.gradients = grad_outputs[0].detach()

    def __call__(self, x: Any, class_index: Any) -> Any:
        import torch
        import torch.nn.functional as functional

        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        cls = class_index.to(logits.device).long().view(-1, 1)
        score = logits.gather(1, cls).sum()
        score.backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("target layer did not produce evidence tensors")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * self.activations).sum(dim=1))
        heatmap = functional.interpolate(
            heatmap[:, None, :, :],
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[:, 0, :, :]
        flat = heatmap.flatten(start_dim=1)
        min_v = flat.min(dim=1).values[:, None, None]
        max_v = flat.max(dim=1).values[:, None, None]
        return (heatmap - min_v) / (max_v - min_v + 1e-8)


def evidence_to_numpy(evidence: Any) -> np.ndarray:
    """Move a tensor-like evidence map to a NumPy array."""
    if hasattr(evidence, "detach"):
        evidence = evidence.detach().cpu().numpy()
    return np.asarray(evidence, dtype=np.float32)
