import logging

import numpy as np
import torch

from scripts.score_avenue import (
    set_seed,
    smooth_moving_average,
    validate_metrics_schema,
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
