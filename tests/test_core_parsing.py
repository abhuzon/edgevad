"""Tests for edgevad.core.parsing."""

from edgevad.core.parsing import parse_classes, extract_scene_id


def test_parse_classes_single():
    assert parse_classes("0") == [0]


def test_parse_classes_multiple():
    assert parse_classes("0,1,2") == [0, 1, 2]


def test_parse_classes_none_string():
    assert parse_classes("none") is None
    assert parse_classes("null") is None


def test_parse_classes_empty():
    assert parse_classes("") is None


# --- extract_scene_id tests ---


def test_extract_scene_id_training_video():
    assert extract_scene_id("01_001.avi") == "01"


def test_extract_scene_id_test_clip():
    assert extract_scene_id("01_0014") == "01"


def test_extract_scene_id_double_digit():
    assert extract_scene_id("12_003.avi") == "12"


def test_extract_scene_id_no_match():
    assert extract_scene_id("something") is None


def test_extract_scene_id_preserves_leading_zeros():
    assert extract_scene_id("01_001") == "01"
