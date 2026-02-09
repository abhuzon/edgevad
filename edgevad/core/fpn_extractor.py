"""YOLO FPN-based feature extractor using RoI-Align.

This module implements multi-scale feature extraction from YOLO's Feature Pyramid Network (FPN)
for video anomaly detection. Instead of cropping pixels and embedding via MobileNetV3, this
approach reuses YOLO's internal FPN features (P3, P4, P5) and applies RoI-Align to extract
fixed-size feature vectors for each detection box.

Key advantages over pixel-crop + MobileNetV3:
    1. Detection-aligned features (YOLO trained on COCO person detection)
    2. Multi-scale context (P3 fine details + P5 semantic features)
    3. Computational efficiency (reuse features from detection pass)
    4. Better performance (expected +5-10% AUC)

Architecture:
    Frame → YOLO forward (captures P3, P4, P5 via hooks)
         → Assign boxes to scales based on area
         → RoI-Align pooling (7×7 per level)
         → Flatten + concat multi-scale features
         → Linear projection → 576-dim embedding
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from torchvision.ops import roi_align

    _torchvision_import_error = None
except Exception as e:
    roi_align = None
    _torchvision_import_error = e


class FPNFeatureExtractor(nn.Module):
    """Multi-scale FPN feature extraction via RoI-Align.

    Extracts features from YOLO FPN layers (P3, P4, P5) for given detection boxes.
    Routes boxes to appropriate scale based on area thresholds, applies RoI-Align pooling,
    concatenates multi-scale features, and projects to fixed output dimension.

    Args:
        yolo_model: Ultralytics YOLO model instance (already loaded with weights)
        output_dim: Final embedding dimension (default: 576 for backward compatibility)
        roi_output_size: RoI-Align output grid size per level (default: 7 → 7×7 pooling)
        area_thresholds: Tuple of (small_max, medium_max) area thresholds in px²
            - boxes with area < small_max → route to P3 (high-res)
            - boxes with area ∈ [small_max, medium_max) → route to P4 (mid-res)
            - boxes with area >= medium_max → route to P5 (low-res, semantic)
        device: torch device for computation

    Architecture:
        - P3 (80×80, stride=8):  small objects (<96² px) - fine spatial details
        - P4 (40×40, stride=16): medium objects (96²-224² px) - balanced
        - P5 (20×20, stride=32): large objects (>224² px) - semantic features
        - RoI-Align (7×7 per level) → flatten → concat → linear → 576-dim

    Example:
        >>> from ultralytics import YOLO
        >>> yolo = YOLO('yolo11n.pt')
        >>> extractor = FPNFeatureExtractor(yolo, output_dim=576, device='cuda')
        >>> frame = cv2.imread('frame.jpg')  # (H, W, 3) BGR
        >>> boxes = np.array([[100, 100, 200, 300], [300, 200, 500, 450]])  # (N, 4) xyxy
        >>> features = extractor(frame, boxes, imgsz=640)  # (N, 576) L2-normalized
    """

    out_dim: int = 576  # Class variable for compatibility with MobileNetV3SmallEmbedder

    def __init__(
        self,
        yolo_model,
        output_dim: int = 576,
        roi_output_size: int = 7,
        area_thresholds: Tuple[float, float] = (96**2, 224**2),
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        if roi_align is None:
            raise RuntimeError(
                f"torchvision.ops.roi_align not available. "
                f"Install torchvision>=0.10. Error: {_torchvision_import_error}"
            )

        self.yolo = yolo_model
        self.output_dim = output_dim
        self.roi_size = roi_output_size
        self.area_thresholds = area_thresholds
        self.device = device

        # FPN layer indices for YOLO v8/v11/v26
        # These correspond to the neck layers before the detection head:
        # Layer 15: P3/small (stride=8, 80×80 for 640×640 input)
        # Layer 18: P4/medium (stride=16, 40×40)
        # Layer 21: P5/large (stride=32, 20×20)
        self.fpn_indices = [15, 18, 21]

        # Feature hook storage (populated during forward pass)
        self.fpn_features: Dict[int, torch.Tensor] = {}
        self._hooks: List = []

        # Auto-detect FPN channel dimensions by running dummy forward pass
        # This handles different YOLO architectures (yolo11n, yolo8n, yolo26n)
        self._detect_fpn_channels()

        # Build projection layers: concat all scales → output_dim
        # Total features = sum(C_i * roi_size^2) for i in {P3, P4, P5}
        total_features = sum(
            c * roi_output_size * roi_output_size for c in self.fpn_channels.values()
        )

        self.projection = nn.Sequential(
            nn.Linear(total_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, output_dim),
        ).to(device)

        # Register persistent forward hooks for FPN feature extraction
        self._register_hooks()

    def _detect_fpn_channels(self):
        """Auto-detect FPN channel dimensions by running dummy inference.

        Runs a single forward pass with a dummy 640×640 input to probe the channel
        dimensions of each FPN level. This ensures compatibility across different YOLO
        architectures without hard-coding channel counts.

        Populates:
            self.fpn_channels: Dict[int, int] mapping layer_index → num_channels
            self.fpn_strides: Dict[int, float] mapping layer_index → spatial_stride

        Raises:
            RuntimeError: If FPN features cannot be extracted (wrong layer indices)
        """
        dummy_input = torch.zeros((1, 3, 640, 640), device=self.device)

        def hook_fn(idx):
            def fn(module, input, output):
                self.fpn_features[idx] = output

            return fn

        # Temporarily register hooks
        temp_hooks = []
        model = self.yolo.model
        for idx in self.fpn_indices:
            if idx < len(model.model):
                h = model.model[idx].register_forward_hook(hook_fn(idx))
                temp_hooks.append(h)

        # Run dummy forward pass
        with torch.no_grad():
            _ = model(dummy_input)

        # Extract channel dimensions and infer strides
        self.fpn_channels = {}
        self.fpn_strides = {}

        for idx in self.fpn_indices:
            if idx in self.fpn_features:
                feat = self.fpn_features[idx]
                self.fpn_channels[idx] = feat.shape[1]  # C dimension (B, C, H, W)

                # Infer stride from spatial size: stride = input_size / feature_size
                spatial_size = feat.shape[2]  # H dimension (assuming H==W)
                self.fpn_strides[idx] = 640.0 / spatial_size

        # Remove temporary hooks
        for h in temp_hooks:
            h.remove()

        self.fpn_features.clear()

        if not self.fpn_channels:
            raise RuntimeError(
                f"Failed to detect FPN channels. Check YOLO model architecture. "
                f"Expected layers {self.fpn_indices} to contain FPN features."
            )

        print(
            f"[FPNFeatureExtractor] Detected FPN channels: {self.fpn_channels}",
            file=sys.stderr,
        )
        print(
            f"[FPNFeatureExtractor] Detected FPN strides: {self.fpn_strides}",
            file=sys.stderr,
        )

    def _register_hooks(self):
        """Register persistent forward hooks for FPN layer feature extraction.

        These hooks capture intermediate feature maps during YOLO forward pass.
        The hooks remain active until the extractor is destroyed (__del__ is called).
        """

        def hook_fn(idx):
            def fn(module, input, output):
                self.fpn_features[idx] = output

            return fn

        model = self.yolo.model
        for idx in self.fpn_indices:
            if idx < len(model.model):
                h = model.model[idx].register_forward_hook(hook_fn(idx))
                self._hooks.append(h)

    def _assign_boxes_to_scales(self, boxes_xyxy: np.ndarray) -> Dict[int, np.ndarray]:
        """Assign detection boxes to FPN scales based on area thresholds.

        Uses box area (width × height) to route boxes to appropriate FPN level:
            - Small boxes (< small_thresh) → P3 (highest resolution, fine details)
            - Medium boxes ([small_thresh, medium_thresh)) → P4 (balanced)
            - Large boxes (>= medium_thresh) → P5 (semantic features)

        Args:
            boxes_xyxy: (N, 4) array of boxes in [x1, y1, x2, y2] format

        Returns:
            Dict mapping fpn_layer_index → array of box indices assigned to that level
            Example: {15: [0, 2], 18: [1], 21: [3, 4]} means boxes 0,2→P3, box 1→P4, boxes 3,4→P5
        """
        areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (
            boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
        )

        small_thresh, medium_thresh = self.area_thresholds

        assignments = {}

        # P3: smallest boxes (highest spatial resolution)
        p3_mask = areas < small_thresh
        if p3_mask.any():
            assignments[self.fpn_indices[0]] = np.where(p3_mask)[0]

        # P4: medium boxes (mid-level features)
        p4_mask = (areas >= small_thresh) & (areas < medium_thresh)
        if p4_mask.any():
            assignments[self.fpn_indices[1]] = np.where(p4_mask)[0]

        # P5: largest boxes (most semantic features)
        p5_mask = areas >= medium_thresh
        if p5_mask.any():
            assignments[self.fpn_indices[2]] = np.where(p5_mask)[0]

        return assignments

    @torch.no_grad()
    def forward(
        self,
        frame: np.ndarray,
        boxes_xyxy: np.ndarray,
        imgsz: int = 640,
    ) -> torch.Tensor:
        """Extract multi-scale FPN features for detection boxes.

        Runs YOLO forward pass (if needed) to populate FPN features, applies RoI-Align
        to extract fixed-size features for each box from appropriate FPN levels,
        concatenates multi-scale features, projects to output dimension, and L2-normalizes.

        Args:
            frame: (H, W, 3) BGR uint8 frame (original image, NOT preprocessed)
            boxes_xyxy: (N, 4) detection boxes in [x1, y1, x2, y2] format at image resolution
            imgsz: YOLO input size (default: 640, must match model's expected input)

        Returns:
            features: (N, output_dim) tensor of L2-normalized embeddings (float32)
                     Empty (0, output_dim) tensor if no boxes provided

        Note:
            This method assumes YOLO forward pass has been run recently to populate
            FPN features. In typical usage, YOLO.predict() is called first to get boxes,
            which triggers the hooks that populate self.fpn_features. If FPN features are
            stale, we trigger a fresh forward pass.
        """
        if len(boxes_xyxy) == 0:
            # No detections: return empty tensor
            return torch.empty((0, self.output_dim), device=self.device)

        # Ensure FPN features are fresh: trigger YOLO forward if not already populated
        # In practice, this happens automatically during detection pass, but we ensure here
        if not self.fpn_features:
            with torch.no_grad():
                _ = self.yolo.predict(
                    source=frame, imgsz=imgsz, verbose=False, device=str(self.device)
                )

        # Check if FPN features were captured
        if not self.fpn_features:
            raise RuntimeError(
                "FPN features not captured. Check hook registration and YOLO model architecture."
            )

        # Get frame dimensions for scaling boxes to YOLO input size
        H, W = frame.shape[:2]

        # Scale boxes from image coordinates to YOLO input size (typically 640×640)
        # YOLO resizes input to square, so we scale both dimensions appropriately
        scale_x = imgsz / W
        scale_y = imgsz / H

        boxes_scaled = boxes_xyxy.copy()
        boxes_scaled[:, [0, 2]] *= scale_x
        boxes_scaled[:, [1, 3]] *= scale_y

        # Assign boxes to FPN scales based on area
        assignments = self._assign_boxes_to_scales(boxes_scaled)

        # Extract RoI features from each scale
        all_features = []
        box_order = []  # Track original order for reassembly

        for fpn_idx in self.fpn_indices:
            if fpn_idx not in assignments:
                continue

            box_indices = assignments[fpn_idx]
            box_order.extend(box_indices.tolist())

            # Prepare boxes for roi_align: (N, 5) with batch_idx in column 0
            boxes_this_scale = boxes_scaled[box_indices]

            # Adjust box coordinates for FPN stride
            # FPN features are at lower resolution than input image
            stride = self.fpn_strides[fpn_idx]
            boxes_fpn = boxes_this_scale / stride

            # Add batch index (always 0 for single-frame inference)
            batch_indices = torch.zeros((len(boxes_fpn), 1), device=self.device)
            boxes_tensor = torch.from_numpy(boxes_fpn).float().to(self.device)
            roi_boxes = torch.cat([batch_indices, boxes_tensor], dim=1)  # (N, 5)

            # Get FPN feature map for this level
            fpn_feat = self.fpn_features[fpn_idx]  # (1, C, H, W)

            # RoI-Align: extract fixed-size features for each box
            roi_features = roi_align(
                fpn_feat,
                roi_boxes,
                output_size=(self.roi_size, self.roi_size),
                spatial_scale=1.0,  # Already adjusted boxes for stride
                aligned=True,  # PyTorch 1.10+ default, more accurate
            )  # (N, C, roi_size, roi_size)

            # Flatten spatial dimensions
            roi_features = roi_features.flatten(
                start_dim=1
            )  # (N, C*roi_size*roi_size)
            all_features.append(roi_features)

        # Concatenate features from all scales
        if not all_features:
            # Edge case: no boxes assigned (shouldn't happen if boxes_xyxy is non-empty)
            return torch.empty((0, self.output_dim), device=self.device)

        concat_features = torch.cat(all_features, dim=0)  # (N, total_features)

        # Restore original box order (RoI-Align processes by scale, not original order)
        inverse_order = np.argsort(box_order)
        concat_features = concat_features[inverse_order]

        # Project to output dimension via learned MLP
        embeddings = self.projection(concat_features)  # (N, output_dim)

        # L2 normalize to unit vectors (standard for k-NN similarity)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Clear FPN features for next frame (prevent memory leak)
        self.fpn_features.clear()

        return embeddings

    def __del__(self):
        """Clean up forward hooks when extractor is destroyed."""
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass  # Hook might already be removed
