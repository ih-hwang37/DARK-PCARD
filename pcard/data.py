import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# Filename heuristics
# ---------------------------------------------------------------------------

def infer_label_from_name(p: Path):
    name = p.stem.lower()
    if "fem" in name:
        return 0
    if "patella" in name or "pat" in name:
        return 1
    if "tib" in name or "tibia" in name:
        return 2
    return 3


def infer_position_from_name(p: Path):
    name = p.stem.lower()
    match = re.search(r"pos\d+", name)
    if match:
        return match.group(0)
    return name


# ---------------------------------------------------------------------------
# Geometry-preserving subsampling
# ---------------------------------------------------------------------------

def _random_subsample(pos: torch.Tensor,
                      y: torch.Tensor,
                      num_points: int,
                      generator: torch.Generator | None = None):
    """
    Uniform random subsample without replacement.

    Across many epochs the model sees a different view of the same cloud
    each pass, so the *distribution* of local neighbourhoods that the
    covariance / kNN losses act on is preserved — which is the relevant
    notion of "intrinsic geometry" for this pipeline.
    """
    N = pos.size(0)
    if N == 0:
        raise ValueError("Cannot sample an empty point cloud.")
    if N <= num_points:
        # Pad by sampling with replacement so the batch shape stays uniform.
        extra_idx = torch.randint(0, N, (num_points - N,), generator=generator)
        idx = torch.cat([torch.arange(N), extra_idx])
    else:
        idx = torch.randperm(N, generator=generator)[:num_points]
    return pos[idx], y[idx]


@torch.no_grad()
def _fps_subsample(pos: torch.Tensor,
                   y: torch.Tensor,
                   num_points: int):
    """
    Farthest point sampling — geometry-preserving downsample.

    Slower than random, but produces a *coverage-preserving* subset which
    keeps the global manifold shape even with very aggressive downsampling.
    Use when num_points is small relative to the cloud (e.g. <2k from 200k).
    """
    N = pos.size(0)
    if N <= num_points:
        return _random_subsample(pos, y, num_points)

    selected = torch.empty(num_points, dtype=torch.long)
    # start from a random seed point
    selected[0] = torch.randint(0, N, (1,)).item()
    dist = torch.full((N,), float("inf"))
    last = pos[selected[0]]

    for i in range(1, num_points):
        d = (pos - last).pow(2).sum(dim=1)
        dist = torch.minimum(dist, d)
        nxt = torch.argmax(dist).item()
        selected[i] = nxt
        last = pos[nxt]

    return pos[selected], y[selected]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CATMAUS(torch.utils.data.Dataset):
    """
    Lazy, memory-efficient point-cloud dataset.

    Key changes vs. the previous version
    ------------------------------------
    1. **Lazy loading.** __init__ only walks the directory and groups files;
       it does not read CSVs. The first time a sample is requested, its
       concatenated point cloud is parsed and cached to disk as a .pt file.
       Subsequent epochs / runs reload the cached tensor in milliseconds.

    2. **Per-call subsampling.** Every __getitem__ returns at most
       `num_points` points, drawn fresh on each access. With 100 epochs and
       a stochastic random subsample, the model effectively sees the full
       cloud distribution — but never holds more than `num_points` per
       sample in memory.

    3. **Multiple views per group via `iterations`.** If you set
       iterations > 1, each (subject, position) group is exposed as
       `iterations` virtual samples per epoch, each a different random
       subsample. This is how you reclaim training signal that you would
       otherwise lose by downsampling — without any memory cost, because
       every view is generated on the fly.

    4. **Configurable sampling strategy.** sampling="random" (default,
       fast, distributionally lossless across epochs) or "fps" (slower,
       coverage-preserving in a single shot). Random is the right choice
       for the loss structure in train.py.
    """

    def __init__(self, root, monte_carlo=True, **kwargs):
        super().__init__()
        self.root           = root
        self.monte_carlo    = monte_carlo
        self.num_points     = int(kwargs.get("num_points", 4096))
        self.iterations     = int(kwargs.get("iterations", 1))
        self.sampling       = kwargs.get("sampling", "random")
        self.cache_dir      = kwargs.get("cache_dir", None)
        self.translation_range = kwargs.get("translation_range", 0.02)
        self.jitter_std     = kwargs.get("jitter_std", 0.01)
        self.jitter_clip    = kwargs.get("jitter_clip", 0.02)

        # Resolve paths
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = (
            self.root if os.path.isabs(self.root)
            else os.path.join(BASE_DIR, self.root)
        )
        if self.cache_dir is None:
            self.cache_dir = os.path.join(self.data_dir, ".pt_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Light-weight directory walk: just file paths grouped by sample.
        self.groups = self._index_groups()
        if not self.groups:
            raise ValueError(f"No CSV files found under {self.data_dir!r}")

        # Print a summary that mirrors the old "Loaded subject=..." log,
        # but without actually loading anything yet.
        for (subject, position), files in self.groups:
            print(
                f"Indexed subject={subject}, position={position}, "
                f"files={len(files)}"
            )
        print(
            f"Total grouped samples: {len(self.groups)}  "
            f"(iterations={self.iterations} → "
            f"{len(self.groups) * self.iterations} effective samples per epoch)"
        )

    # ---- indexing -------------------------------------------------------

    def _index_groups(self):
        csv_files = list(Path(self.data_dir).rglob("*.csv"))
        groups = defaultdict(list)
        for csv_path in csv_files:
            subject  = csv_path.parent.name
            position = infer_position_from_name(csv_path)
            groups[(subject, position)].append(csv_path)
        # Stable order so train/test split is reproducible
        return sorted(
            ((k, sorted(v)) for k, v in groups.items()),
            key=lambda kv: kv[0],
        )

    # ---- caching --------------------------------------------------------

    def _cache_path(self, subject, position):
        return os.path.join(self.cache_dir, f"{subject}__{position}.pt")

    def _load_group(self, subject, position, files):
        """
        Read the CSVs for one (subject, position) group, concatenate, and
        cache to disk. Returns (pos: (N,3) float32, y: (N,) int64).
        """
        cache = self._cache_path(subject, position)
        if os.path.exists(cache):
            try:
                blob = torch.load(cache, map_location="cpu", weights_only=True)
            except TypeError:  # PyTorch < 2.0
                blob = torch.load(cache, map_location="cpu")
            return blob["pos"], blob["y"]

        all_X, all_y = [], []
        for csv_path in files:
            df = pd.read_csv(csv_path, header=None)
            # Strip optional x,y,z header
            first_row = df.iloc[0, :3].astype(str).str.strip().str.lower().tolist()
            if first_row == ["x", "y", "z"]:
                df = df.iloc[1:].reset_index(drop=True)
            if df.shape[1] < 3:
                raise ValueError(
                    f"Expected at least 3 columns in {csv_path}, "
                    f"got {df.shape[1]}"
                )
            df = df.iloc[:, :3]
            df.columns = ["x", "y", "z"]

            X = df.values.astype(np.float32)
            label = infer_label_from_name(csv_path)
            yy = np.full(len(df), label, dtype=np.int64)

            all_X.append(X)
            all_y.append(yy)

        pos = torch.from_numpy(np.vstack(all_X)).float()
        y   = torch.from_numpy(np.hstack(all_y)).long()

        torch.save({"pos": pos, "y": y}, cache)
        print(
            f"Cached subject={subject}, position={position}, "
            f"files={len(files)}, points={pos.size(0)}  → {cache}"
        )
        return pos, y

    # ---- dataset protocol ----------------------------------------------

    def __len__(self):
        return len(self.groups) * self.iterations

    def __getitem__(self, idx):
        group_idx = idx // self.iterations
        (subject, position), files = self.groups[group_idx]

        pos_full, y_full = self._load_group(subject, position, files)

        if self.sampling == "fps":
            pos, y = _fps_subsample(pos_full, y_full, self.num_points)
        else:
            pos, y = _random_subsample(pos_full, y_full, self.num_points)

        # Free reference so peak memory stays bounded.
        del pos_full, y_full

        # Centre and scale to the unit sphere.
        # Why: DGCNN's first EdgeConv layer operates on raw 3D positions,
        # and knn_graph uses raw Euclidean distance. Without
        # normalisation, two clouds spanning e.g. 50mm and 200mm look
        # geometrically different to the model even though they share
        # the same underlying shape — the same k=20 neighbours mean
        # different things at different scales, and downstream features
        # cascade differently. Centring removes absolute position;
        # scaling to unit-sphere removes absolute scale. The model then
        # sees geometrically equivalent inputs regardless of the
        # source-CSV's coordinate units.
        pos = pos - pos.mean(dim=0, keepdim=True)
        scale = pos.norm(dim=1).max().clamp(min=1e-6)
        pos = pos / scale

        return Data(
            pos=pos,
            y=y,
            subject=subject,
            position=position,
        )

    # ---- legacy augmentation helpers (kept for compatibility) -----------

    def apply_translation(self, X):
        translation = np.random.uniform(
            -self.translation_range, self.translation_range,
            size=(X.shape[0], 3),
        )
        return X + translation

    def apply_jitter(self, X):
        noise = np.random.normal(0, self.jitter_std, size=(X.shape[0], 3))
        noise = np.clip(noise, -self.jitter_clip, self.jitter_clip)
        return X + noise
