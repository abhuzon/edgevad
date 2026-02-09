import logging

import numpy as np
import torch

from edgevad.core import (
    l2_normalize,
    sanitize_array,
    set_seed,
    smooth_moving_average,
    validate_metrics_schema,
)
from edgevad.scripts.score_avenue import (
    score_dataset,
    volLabel_to_frame_labels,
)


def test_vollabel_to_frame_labels_object_array():
    vol = np.empty((1, 3), dtype=object)
    vol[0, 0] = np.zeros((2, 2), dtype=np.uint8)
    vol[0, 1] = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    vol[0, 2] = np.zeros((2, 2), dtype=np.uint8)
    labels = volLabel_to_frame_labels(vol)
    assert labels.tolist() == [0, 1, 0]


def test_smooth_moving_average_length_and_finite():
    x = np.linspace(0.0, 1.0, 11, dtype=np.float32)
    y = smooth_moving_average(x, win=5)
    assert len(y) == len(x)
    assert np.isfinite(y).all()


def test_set_seed_reproducible_numpy_torch():
    logger = logging.getLogger("test_seed")
    set_seed(123, deterministic=False, logger=logger)
    a1 = np.random.rand(5)
    t1 = torch.rand(5)

    set_seed(123, deterministic=False, logger=logger)
    a2 = np.random.rand(5)
    t2 = torch.rand(5)

    np.testing.assert_allclose(a1, a2, rtol=0.0, atol=0.0)
    assert torch.equal(t1, t2)


def test_validate_metrics_schema_required_keys():
    metrics = {
        "timestamp_utc": "2024-01-01T00:00:00Z",
        "git_hash": None,
        "args": {},
        "env": {},
        "memory_bank": {},
        "overall": {"auc_raw": None, "auc_smooth": None, "ap_raw": None, "ap_smooth": None},
        "runtime": {"wall_seconds": 1.0, "proc_fps": 30.0, "num_videos": 1, "frames_processed": 10},
        "per_video": [],
    }
    validate_metrics_schema(metrics)

    metrics.pop("overall")
    try:
        validate_metrics_schema(metrics)
    except ValueError:
        pass
    else:
        raise AssertionError("validate_metrics_schema should raise on missing keys")


def test_score_dataset_uses_gt_and_produces_metrics(monkeypatch):
    # Minimal dataset with two frames and GT spanning both classes.
    fake_samples = [
        {"video": "v1.avi", "frame_idx": 0, "gt": 0, "image": np.zeros((2, 2, 3), dtype=np.uint8)},
        {"video": "v1.avi", "frame_idx": 1, "gt": 1, "image": np.zeros((2, 2, 3), dtype=np.uint8)},
    ]

    class FakeDataset(list):
        pass

    class FakeYOLO:
        def predict(self, **kwargs):
            class Pred:
                boxes = []

            return [Pred()]

    class FakeEmbedder:
        def __call__(self, x):
            return x

    bank = torch.zeros((1, 1))
    results = score_dataset(
        dataset=FakeDataset(fake_samples),
        yolo=FakeYOLO(),
        embedder=FakeEmbedder(),
        bank=bank,
        device=torch.device("cpu"),
        imgsz=640,
        conf=0.25,
        classes=None,
        max_dets_per_frame=3,
        fp16=False,
        smooth=1,
        max_frames=0,
        expand=1.05,
        topk=5,
        smooth_sigma=0.0,
        min_box_area=400.0,
        logger=logging.getLogger("test_score_dataset"),
    )

    assert len(results) == 1
    vr = results[0]
    assert vr.video == "v1.avi"
    assert vr.gt == [0, 1]
    # With one positive and one negative, AUC/AP should be computed (not None).
    assert vr.auc_raw is not None
    assert vr.ap_raw is not None
    assert np.isfinite(vr.scores).all()
    assert np.isfinite(vr.scores_smooth).all()


def test_l2_normalize_fp16_zero_vector_finite():
    x = torch.zeros((2, 3), dtype=torch.float16)
    y = l2_normalize(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert torch.equal(y, torch.zeros_like(y))


def test_sanitize_array_replaces_nonfinite():
    arr = np.array([0.5, np.nan, np.inf, -np.inf], dtype=np.float32)
    cleaned = sanitize_array(arr, "test", logging.getLogger("sanitize"))
    assert cleaned.tolist() == [0.5, 0.0, 0.0, 0.0]
    assert np.isfinite(cleaned).all()


def test_score_dataset_replaces_nan_scores(monkeypatch):
    sample = {"video": "v1.avi", "frame_idx": 0, "gt": 0, "image": np.zeros((4, 4, 3), dtype=np.uint8)}

    class FakeDataset(list):
        pass

    class FakeBoxes:
        def __init__(self):
            self.xyxy = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)
            self.conf = torch.tensor([0.9], dtype=torch.float32)
        def __len__(self):
            return 1

    class FakeYOLO:
        def predict(self, **kwargs):
            class Pred:
                boxes = FakeBoxes()

            return [Pred()]

    class FakeEmbedder:
        def __call__(self, x):
            # Force NaN embedding to trigger sanitization path.
            return torch.full((1, 1), float("nan"))

    bank = torch.zeros((1, 1))
    results = score_dataset(
        dataset=FakeDataset([sample]),
        yolo=FakeYOLO(),
        embedder=FakeEmbedder(),
        bank=bank,
        device=torch.device("cpu"),
        imgsz=640,
        conf=0.25,
        classes=None,
        max_dets_per_frame=3,
        fp16=False,
        smooth=3,
        max_frames=0,
        expand=1.05,
        topk=5,
        smooth_sigma=0.0,
        min_box_area=0.0,
        logger=logging.getLogger("test_score_dataset_nan"),
    )

    assert len(results) == 1
    vr = results[0]
    assert np.isfinite(vr.scores).all()
    assert np.isfinite(vr.scores_smooth).all()
