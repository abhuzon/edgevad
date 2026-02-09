"""Tests for edgevad.core.fpn_extractor (YOLO FPN feature extraction + RoI-Align)."""

import numpy as np
import pytest
import torch


@pytest.mark.skipif(
    torch.cuda.is_available() is False, reason="YOLO models require CUDA for reliable testing"
)
def test_fpn_extractor_output_shape():
    """Test FPN extractor produces correct output shape (N, 576)."""
    pytest.importorskip("ultralytics")

    from ultralytics import YOLO

    from edgevad.core.fpn_extractor import FPNFeatureExtractor

    # Use yolo11n for testing (smallest model)
    # This will download the model if not cached (~6MB)
    yolo = YOLO("yolo11n.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extractor = FPNFeatureExtractor(yolo, output_dim=576, device=device)

    # Dummy frame and boxes
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    boxes = np.array(
        [
            [100, 100, 200, 300],  # Small box (100x200 = 20000 px²) → P3
            [300, 200, 500, 450],  # Large box (200x250 = 50000 px²) → P5
        ]
    )

    features = extractor(frame, boxes, imgsz=640)

    assert features.shape == (2, 576), f"Expected (2, 576), got {features.shape}"
    assert features.dtype == torch.float32, f"Expected float32, got {features.dtype}"

    # Check L2 normalization: each row should have norm ≈ 1.0
    norms = torch.linalg.norm(features, dim=1)
    assert torch.allclose(
        norms, torch.ones_like(norms), atol=1e-5
    ), f"Features not L2-normalized: norms={norms}"


@pytest.mark.skipif(
    torch.cuda.is_available() is False, reason="YOLO models require CUDA for reliable testing"
)
def test_fpn_extractor_empty_boxes():
    """Test FPN extractor handles no detections gracefully."""
    pytest.importorskip("ultralytics")

    from ultralytics import YOLO

    from edgevad.core.fpn_extractor import FPNFeatureExtractor

    yolo = YOLO("yolo11n.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = FPNFeatureExtractor(yolo, output_dim=576, device=device)

    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    boxes = np.empty((0, 4))  # No detections

    features = extractor(frame, boxes, imgsz=640)

    assert features.shape == (0, 576), f"Expected (0, 576), got {features.shape}"
    assert features.dtype == torch.float32


def test_fpn_extractor_multi_scale_assignment():
    """Test boxes correctly assigned to FPN scales based on area."""
    pytest.importorskip("ultralytics")

    from ultralytics import YOLO

    from edgevad.core.fpn_extractor import FPNFeatureExtractor

    yolo = YOLO("yolo11n.pt")
    device = torch.device("cpu")  # CPU is fine for box assignment logic

    extractor = FPNFeatureExtractor(
        yolo, device=device, area_thresholds=(96**2, 224**2)
    )

    # Create boxes with known areas
    # Box 0: 50×50 = 2500 px² < 96² → should go to P3 (fpn_indices[0] = 15)
    # Box 1: 150×150 = 22500 px², 96² < area < 224² → should go to P4 (fpn_indices[1] = 18)
    # Box 2: 300×300 = 90000 px² >= 224² → should go to P5 (fpn_indices[2] = 21)
    boxes = np.array(
        [
            [0, 0, 50, 50],  # Small
            [0, 0, 150, 150],  # Medium
            [0, 0, 300, 300],  # Large
        ]
    )

    assignments = extractor._assign_boxes_to_scales(boxes)

    # Verify all three levels are assigned
    assert 15 in assignments, "P3 (layer 15) should have assignments"
    assert 18 in assignments, "P4 (layer 18) should have assignments"
    assert 21 in assignments, "P5 (layer 21) should have assignments"

    # Verify correct box indices
    assert (
        assignments[15].tolist() == [0]
    ), f"Small box should be assigned to P3, got {assignments[15]}"
    assert (
        assignments[18].tolist() == [1]
    ), f"Medium box should be assigned to P4, got {assignments[18]}"
    assert (
        assignments[21].tolist() == [2]
    ), f"Large box should be assigned to P5, got {assignments[21]}"


def test_fpn_extractor_edge_case_all_one_scale():
    """Test all boxes assigned to single scale (edge case)."""
    pytest.importorskip("ultralytics")

    from ultralytics import YOLO

    from edgevad.core.fpn_extractor import FPNFeatureExtractor

    yolo = YOLO("yolo11n.pt")
    device = torch.device("cpu")

    extractor = FPNFeatureExtractor(yolo, device=device)

    # All small boxes (< 96²)
    boxes = np.array(
        [
            [0, 0, 50, 50],  # 2500 px²
            [100, 100, 140, 150],  # 2000 px²
            [200, 200, 230, 240],  # 1200 px²
        ]
    )

    assignments = extractor._assign_boxes_to_scales(boxes)

    # Only P3 should have assignments
    assert len(assignments) == 1, f"Expected 1 scale, got {len(assignments)}"
    assert 15 in assignments, "All small boxes should go to P3"
    assert len(assignments[15]) == 3, f"Expected 3 boxes in P3, got {len(assignments[15])}"


def test_get_embedder_factory_mobilenet():
    """Test embedder factory function with mobilenet mode."""
    pytest.importorskip("torchvision")
    from edgevad.core.embedder import get_embedder

    embedder = get_embedder(feature_mode="mobilenet", device=torch.device("cpu"))

    assert embedder.out_dim == 576, f"Expected out_dim=576, got {embedder.out_dim}"
    assert hasattr(embedder, "forward"), "Embedder should have forward() method"


def test_get_embedder_factory_fpn_requires_yolo():
    """Test embedder factory raises error when fpn mode lacks yolo_model."""
    from edgevad.core.embedder import get_embedder

    with pytest.raises(ValueError, match="yolo_model required"):
        get_embedder(feature_mode="fpn", device=torch.device("cpu"))


def test_get_embedder_factory_invalid_mode():
    """Test embedder factory raises error for invalid mode."""
    from edgevad.core.embedder import get_embedder

    with pytest.raises(ValueError, match="Unknown feature_mode"):
        get_embedder(feature_mode="invalid_mode", device=torch.device("cpu"))


@pytest.mark.skipif(
    torch.cuda.is_available() is False, reason="YOLO models require CUDA for reliable testing"
)
def test_fpn_extractor_deterministic_output():
    """Test FPN extractor produces deterministic outputs for same input."""
    pytest.importorskip("ultralytics")

    from ultralytics import YOLO

    from edgevad.core.fpn_extractor import FPNFeatureExtractor

    yolo = YOLO("yolo11n.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    extractor = FPNFeatureExtractor(yolo, output_dim=576, device=device)

    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    boxes = np.array([[100, 100, 200, 300]])

    # Extract features twice
    features1 = extractor(frame, boxes, imgsz=640)
    features2 = extractor(frame, boxes, imgsz=640)

    # Should be identical (deterministic forward pass)
    assert torch.allclose(
        features1, features2, atol=1e-5
    ), "FPN extractor should produce deterministic outputs"
