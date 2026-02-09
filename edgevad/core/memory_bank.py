"""Memory bank loading, coreset selection, and k-NN scoring."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from edgevad.core.math_utils import l2_normalize


def load_memory_bank_npz(
    path: str, device: torch.device, fp16: bool
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    d = np.load(path, allow_pickle=True)
    if "embeddings" not in d:
        raise RuntimeError(f"Memory bank npz missing 'embeddings': {path}")
    E = d["embeddings"]
    E = E.astype(np.float16 if fp16 else np.float32, copy=False)
    E = np.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0)
    meta: Dict[str, Any] = {}
    if "config" in d:
        try:
            meta["config"] = str(d["config"])
        except Exception:
            meta["config"] = None
    if "scene_ids" in d:
        try:
            meta["scene_ids"] = d["scene_ids"].astype(str).tolist()
        except Exception:
            pass
    bank = torch.from_numpy(E).to(device=device)
    bank = l2_normalize(bank)
    return bank, meta


def build_scene_banks(
    bank: torch.Tensor,
    scene_ids: List[str],
) -> Dict[str, torch.Tensor]:
    """Partition a memory bank into per-scene sub-banks.

    Args:
        bank: (N, D) full memory bank tensor on device.
        scene_ids: list of N scene ID strings, one per embedding.

    Returns:
        Dict mapping scene_id -> (n_i, D) sub-bank tensor.
    """
    ids_arr = np.array(scene_ids, dtype=object)
    unique = sorted(set(scene_ids))
    scene_banks: Dict[str, torch.Tensor] = {}
    for sid in unique:
        mask = ids_arr == sid
        indices = np.where(mask)[0]
        scene_banks[sid] = bank[indices]
    return scene_banks


def coreset_farthest_point(
    embeddings: np.ndarray,
    target_size: int,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy farthest-point sampling for coreset selection.

    Returns (selected_embeddings, selected_indices).
    For L2-normalized vectors: dist = 2 - 2*cos_sim.
    """
    N, D = embeddings.shape
    if N <= target_size:
        return embeddings.copy(), np.arange(N, dtype=np.int64)

    rng = np.random.RandomState(seed)
    selected = [rng.randint(0, N)]
    min_dist = np.full(N, np.inf, dtype=np.float64)

    for _ in range(target_size - 1):
        last = embeddings[selected[-1]].astype(np.float64)  # (D,)
        # Squared L2 distance for unit vectors: 2 - 2*cos_sim
        sim = embeddings.astype(np.float64) @ last  # (N,)
        dist = 2.0 - 2.0 * sim
        np.minimum(min_dist, dist, out=min_dist)
        # Mask already selected
        for s in selected:
            min_dist[s] = -np.inf
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)

    indices = np.array(selected, dtype=np.int64)
    return embeddings[indices].copy(), indices


def score_knn(
    emb: torch.Tensor,
    bank: torch.Tensor,
    topk: int = 5,
) -> torch.Tensor:
    """k-NN anomaly scoring: mean of top-k cosine similarities, inverted.

    Args:
        emb: (num_crops, D) L2-normalized query embeddings
        bank: (bank_size, D) L2-normalized bank embeddings
        topk: number of nearest neighbors

    Returns:
        scores: (num_crops,) anomaly scores in [0, 1] (higher = more anomalous)
    """
    sim = emb @ bank.T  # (num_crops, bank_size)
    k = min(topk, sim.shape[1])
    topk_sim = sim.topk(k, dim=1).values  # (num_crops, k)
    mean_sim = topk_sim.mean(dim=1)  # (num_crops,)
    return 1.0 - mean_sim
