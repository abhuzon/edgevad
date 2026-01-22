from pathlib import Path

import numpy as np
import scipy.io as sio

from edgevad.data.avenue_dataset import (
    list_videos,
    load_avenue_gt_per_frame,
    resolve_gt_mat_path,
)


def test_list_videos_numeric_sort(tmp_path):
    (tmp_path / "10.avi").write_text("a")
    (tmp_path / "2.avi").write_text("b")
    (tmp_path / "1.avi").write_text("c")
    vids = list_videos(str(tmp_path))
    assert [Path(p).name for p in vids] == ["1.avi", "2.avi", "10.avi"]


def test_resolve_gt_mat_path_supports_zero_and_nonzero(tmp_path):
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    (gt_dir / "1_label.mat").write_text("gt")
    video_path = str(tmp_path / "01.avi")
    resolved = resolve_gt_mat_path(str(gt_dir), video_path)
    assert resolved.endswith("1_label.mat")


def test_resolve_gt_mat_path_prefers_exact_zero_padded(tmp_path):
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    (gt_dir / "01_label.mat").write_text("gt")
    video_path = str(tmp_path / "01.avi")
    resolved = resolve_gt_mat_path(str(gt_dir), video_path)
    assert resolved.endswith("01_label.mat")


def test_load_avenue_gt_per_frame_object_array(tmp_path):
    mat_path = tmp_path / "obj.mat"
    vol = np.empty((1, 2), dtype=object)
    vol[0, 0] = np.zeros((2, 2), dtype=np.uint8)
    vol[0, 1] = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    sio.savemat(mat_path, {"volLabel": vol})
    labels = load_avenue_gt_per_frame(str(mat_path))
    assert labels.tolist() == [0, 1]


def test_load_avenue_gt_per_frame_numeric_hwt(tmp_path):
    mat_path = tmp_path / "num.mat"
    vol = np.zeros((2, 2, 3), dtype=np.uint8)
    vol[:, :, 1] = 1
    vol[:, :, 2] = 2
    sio.savemat(mat_path, {"volLabel": vol})
    labels = load_avenue_gt_per_frame(str(mat_path))
    assert labels.tolist() == [0, 1, 1]


def test_load_avenue_gt_per_frame_row_vector(tmp_path):
    mat_path = tmp_path / "vec.mat"
    vol = np.array([[0, 1, 0, 1]], dtype=np.uint8)
    sio.savemat(mat_path, {"volLabel": vol})
    labels = load_avenue_gt_per_frame(str(mat_path))
    assert labels.tolist() == [0, 1, 0, 1]
