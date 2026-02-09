from __future__ import annotations

from pathlib import Path

import numpy as np

from edgevad.data.shanghaitech_dataset import (
    get_gt_per_frame,
    load_pixel_mask,
    list_videos,
    resolve_shanghaitech_mask_path,
)


def test_resolve_shanghaitech_mask_path_variants(tmp_path) -> None:
    mask_dir = tmp_path / "test_pixel_mask"
    mask_dir.mkdir()

    stripped = mask_dir / "1_14.npy"
    np.save(stripped, np.zeros((1,), dtype=np.uint8))

    path = resolve_shanghaitech_mask_path(str(mask_dir), "01_0014")
    assert path == stripped

    exact = mask_dir / "01_0014.npy"
    np.save(exact, np.zeros((1,), dtype=np.uint8))

    path_exact = resolve_shanghaitech_mask_path(str(mask_dir), "01_0014")
    assert path_exact == exact


def test_load_pixel_mask_accepts_hwt_and_thw(tmp_path) -> None:
    mask_dir = tmp_path / "test_pixel_mask"
    mask_dir.mkdir()

    mask_thw = np.zeros((3, 4, 5), dtype=np.uint8)
    mask_thw[1, 2, 3] = 1
    np.save(mask_dir / "01_0001.npy", mask_thw)

    mask_hwt = mask_thw.transpose(1, 2, 0)
    np.save(mask_dir / "01_0002.npy", mask_hwt)

    loaded_thw = load_pixel_mask(str(mask_dir), "01_0001")
    loaded_hwt = load_pixel_mask(str(mask_dir), "01_0002")

    assert loaded_thw.shape == (3, 4, 5)
    assert loaded_hwt.shape == (3, 4, 5)
    assert np.array_equal(loaded_thw, mask_thw.astype(bool))
    assert np.array_equal(loaded_hwt, mask_thw.astype(bool))


def test_get_gt_per_frame_from_pixel_mask(tmp_path) -> None:
    root = tmp_path / "dataset_root"
    pixel_dir = root / "testing" / "test_pixel_mask"
    pixel_dir.mkdir(parents=True)

    pixel_mask = np.zeros((4, 3, 3), dtype=np.uint8)
    pixel_mask[1, 0, 0] = 1
    pixel_mask[3, 2, 2] = 1
    np.save(pixel_dir / "01_0001.npy", pixel_mask)

    labels = get_gt_per_frame(str(root), "01_0001")
    assert labels is not None
    assert labels.tolist() == [0, 1, 0, 1]


def test_list_videos_supports_frames_layout(tmp_path) -> None:
    frames_root = tmp_path / "frames"
    clip1 = frames_root / "01"
    clip2 = frames_root / "02"
    clip1.mkdir(parents=True)
    clip2.mkdir(parents=True)

    (clip1 / "001.jpg").write_bytes(b"fake")
    (clip2 / "001.jpg").write_bytes(b"fake")

    paths = list_videos(str(frames_root), ext=".avi")
    assert [Path(p).name for p in paths] == ["01", "02"]


def test_list_videos_prefers_avi_layout(tmp_path) -> None:
    root = tmp_path / "videos"
    root.mkdir()

    (root / "03.avi").write_bytes(b"fake")
    (root / "01.avi").write_bytes(b"fake")
    (root / "frames").mkdir()
    (root / "frames" / "001.jpg").write_bytes(b"fake")

    paths = list_videos(str(root), ext=".avi")
    assert [Path(p).name for p in paths] == ["01.avi", "03.avi"]
