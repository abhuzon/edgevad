"""MobileNetV3-Small embedder and crop preprocessing."""

from __future__ import annotations

import sys
from typing import List

import cv2
import numpy as np
import torch

try:
    import torchvision
    _torchvision_import_error = None
except Exception as e:
    torchvision = None
    _torchvision_import_error = e


class MobileNetV3SmallEmbedder(torch.nn.Module):
    """MobileNetV3-Small feature extractor (D=576) with built-in ImageNet normalization.

    Args:
        pretrained: If True, load torchvision ImageNet pretrained weights (default).
        weights_path: If set, load weights from this local path instead.
    """
    out_dim: int = 576

    def __init__(self, pretrained: bool = True, weights_path: str = ""):
        super().__init__()
        if torchvision is None:
            raise RuntimeError(f"Failed to import torchvision. Error: {_torchvision_import_error}")

        if weights_path:
            backbone = torchvision.models.mobilenet_v3_small(weights=None)
            sd = torch.load(weights_path, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            cleaned = {}
            for k, v in sd.items():
                nk = k[7:] if k.startswith("module.") else k
                cleaned[nk] = v
            missing, unexpected = backbone.load_state_dict(cleaned, strict=False)
            if missing:
                print(f"[WARN] Missing keys in embedder weights: {len(missing)}", file=sys.stderr)
            if unexpected:
                print(f"[WARN] Unexpected keys in embedder weights: {len(unexpected)}", file=sys.stderr)
        else:
            if pretrained:
                try:
                    weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
                    backbone = torchvision.models.mobilenet_v3_small(weights=weights)
                except Exception as e:
                    print(f"[WARN] Could not load pretrained MobileNetV3 weights: {e}", file=sys.stderr)
                    print("[WARN] Falling back to random init.", file=sys.stderr)
                    backbone = torchvision.models.mobilenet_v3_small(weights=None)
            else:
                backbone = torchvision.models.mobilenet_v3_small(weights=None)

        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # ImageNet normalization applied inside forward() for consistency
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std", std, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,3,224,224) float in [0,1]
        x = (x - self._mean) / self._std
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def preprocess_crops_to_tensor(
    crops_bgr: List[np.ndarray], device: torch.device, fp16: bool
) -> torch.Tensor:
    """Convert BGR uint8 crops to (B,3,224,224) float tensor in [0,1]."""
    batch = []
    for c in crops_bgr:
        if c is None or c.size == 0:
            continue
        r = cv2.resize(c, (224, 224), interpolation=cv2.INTER_LINEAR)
        r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(r).to(torch.float32) / 255.0  # (H,W,C)
        t = t.permute(2, 0, 1).contiguous()  # (C,H,W)
        batch.append(t)
    if not batch:
        return torch.empty((0, 3, 224, 224), device=device, dtype=torch.float16 if fp16 else torch.float32)
    x = torch.stack(batch, dim=0).to(device=device)
    if fp16:
        x = x.half()
    return x


def get_embedder(
    feature_mode: str = "mobilenet",
    yolo_model=None,
    device: torch.device = torch.device("cpu"),
    pretrained: bool = True,
    weights_path: str = "",
) -> torch.nn.Module:
    """Factory function to get embedder based on feature mode.

    Provides a unified interface for creating embedders in different feature extraction modes.
    Supports both the original MobileNetV3 pixel-crop approach and the new YOLO FPN approach.

    Args:
        feature_mode: Feature extraction mode, one of:
            - "mobilenet": Crop pixels → resize → MobileNetV3 embedder (Phase 1 baseline)
            - "fpn": YOLO FPN features → RoI-Align → projection (Phase 2a innovation)
        yolo_model: YOLO model instance (required if feature_mode="fpn")
        device: torch device for computation
        pretrained: For MobileNetV3, whether to load ImageNet pretrained weights (default: True)
        weights_path: For MobileNetV3, custom weights path (default: "" uses ImageNet)

    Returns:
        Embedder module with:
            - .forward() method: processes input and returns (N, D) embeddings
            - .out_dim attribute: output dimension (576 for both modes)

    Raises:
        ValueError: If feature_mode is invalid or if fpn mode is requested without yolo_model

    Example:
        >>> # MobileNetV3 mode (Phase 1 baseline)
        >>> embedder = get_embedder(feature_mode="mobilenet", device="cuda")
        >>> crops = [frame[y1:y2, x1:x2] for (x1, y1, x2, y2) in boxes]
        >>> batch = preprocess_crops_to_tensor(crops, device="cuda", fp16=False)
        >>> features = embedder(batch)  # (N, 576)

        >>> # FPN mode (Phase 2a innovation)
        >>> from ultralytics import YOLO
        >>> yolo = YOLO('yolo11n.pt')
        >>> embedder = get_embedder(feature_mode="fpn", yolo_model=yolo, device="cuda")
        >>> features = embedder(frame, boxes_xyxy, imgsz=640)  # (N, 576)
    """
    if feature_mode == "mobilenet":
        return (
            MobileNetV3SmallEmbedder(pretrained=pretrained, weights_path=weights_path)
            .to(device)
            .eval()
        )

    elif feature_mode == "fpn":
        if yolo_model is None:
            raise ValueError(
                "yolo_model required for feature_mode='fpn'. "
                "Pass the YOLO model instance when using FPN feature extraction."
            )

        from edgevad.core.fpn_extractor import FPNFeatureExtractor

        return FPNFeatureExtractor(yolo_model=yolo_model, output_dim=576, device=device)

    else:
        raise ValueError(
            f"Unknown feature_mode: '{feature_mode}'. "
            f"Must be one of: 'mobilenet' (Phase 1 baseline), 'fpn' (Phase 2a innovation)."
        )
