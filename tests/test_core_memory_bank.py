"""Tests for edgevad.core.memory_bank (coreset + score_knn)."""

import numpy as np
import torch

from edgevad.core.math_utils import l2_normalize
from edgevad.core.memory_bank import coreset_farthest_point, score_knn, build_scene_banks


def test_coreset_returns_correct_size():
    rng = np.random.RandomState(42)
    emb = rng.randn(100, 16).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.maximum(norms, 1e-6)

    selected, indices = coreset_farthest_point(emb, target_size=20, seed=0)
    assert selected.shape == (20, 16)
    assert indices.shape == (20,)
    assert len(set(indices.tolist())) == 20  # all unique


def test_coreset_noop_when_smaller():
    emb = np.random.randn(10, 8).astype(np.float32)
    selected, indices = coreset_farthest_point(emb, target_size=50, seed=0)
    assert selected.shape == emb.shape
    np.testing.assert_array_equal(indices, np.arange(10))


def test_coreset_deterministic():
    rng = np.random.RandomState(7)
    emb = rng.randn(200, 32).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.maximum(norms, 1e-6)

    _, idx1 = coreset_farthest_point(emb, 30, seed=42)
    _, idx2 = coreset_farthest_point(emb, 30, seed=42)
    np.testing.assert_array_equal(idx1, idx2)


def test_score_knn_basic():
    # Bank: 3 unit vectors along axes
    bank = torch.eye(3, dtype=torch.float32)  # (3, 3)
    # Query: exactly axis 0 -> max sim=1.0 with bank[0], 0 with others
    query = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)  # (1, 3)

    scores = score_knn(query, bank, topk=1)
    # top-1 sim = 1.0, so anomaly = 1.0 - 1.0 = 0.0
    assert scores.shape == (1,)
    np.testing.assert_allclose(scores.numpy(), [0.0], atol=1e-6)


def test_score_knn_topk_mean():
    bank = torch.eye(3, dtype=torch.float32)
    query = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)

    # top-3: sims are [1.0, 0.0, 0.0], mean = 1/3
    scores = score_knn(query, bank, topk=3)
    expected = 1.0 - (1.0 / 3.0)
    np.testing.assert_allclose(scores.numpy(), [expected], atol=1e-6)


def test_score_knn_anomalous_vector():
    # Bank of "normal" vectors all pointing in +x direction
    bank = torch.tensor([[1.0, 0.0], [0.9, 0.1]], dtype=torch.float32)
    bank = l2_normalize(bank)

    # Anomalous query perpendicular to all bank entries
    query = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

    scores = score_knn(query, bank, topk=2)
    # Should be high anomaly (close to 1.0) since low similarity
    assert scores.item() > 0.8


def test_score_knn_multiple_queries():
    bank = torch.randn(50, 16)
    bank = l2_normalize(bank)
    queries = torch.randn(5, 16)
    queries = l2_normalize(queries)

    scores = score_knn(queries, bank, topk=3)
    assert scores.shape == (5,)
    assert (scores >= 0).all()
    assert (scores <= 2.0).all()  # theoretical max for cosine-based


# --- build_scene_banks tests ---


def test_build_scene_banks_partitions_correctly():
    # 10 embeddings: 4 from scene "01", 3 from "02", 3 from "03"
    bank = torch.randn(10, 8)
    scene_ids = ["01"] * 4 + ["02"] * 3 + ["03"] * 3

    result = build_scene_banks(bank, scene_ids)

    assert set(result.keys()) == {"01", "02", "03"}
    assert result["01"].shape == (4, 8)
    assert result["02"].shape == (3, 8)
    assert result["03"].shape == (3, 8)
    # Verify content matches original bank rows
    torch.testing.assert_close(result["01"], bank[:4])
    torch.testing.assert_close(result["02"], bank[4:7])
    torch.testing.assert_close(result["03"], bank[7:10])


def test_build_scene_banks_single_scene():
    bank = torch.randn(5, 4)
    scene_ids = ["01"] * 5

    result = build_scene_banks(bank, scene_ids)

    assert len(result) == 1
    assert "01" in result
    assert result["01"].shape == (5, 4)
    torch.testing.assert_close(result["01"], bank)
