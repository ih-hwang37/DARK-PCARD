import argparse
import copy
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import knn_graph
from pcard.model import DGCNNWithKNN
from pcard.data import CATMAUS
from pcard.filtering import (
    build_local_covariance_masked,
    compute_scatter,
    edge_index_to_knn_idx,
)
import yaml


def _merge(base, override):
    """Recursively merge a YAML mapping into the release defaults."""
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path):
    defaults = {
        "data": {
            "root": "data",
            "num_points": 1024,
            "iterations": 40,
            "sampling": "random",
            "test_fraction": 0.2,
            "cache_dir": "outputs/cache",
        },
        "model": {"k": 20, "feature_dim": 128},
        "training": {
            "batch_size": 32,
            "learning_rate": 0.0005,
            "num_epochs": 100,
            "tau_nce": 0.2,
            "num_neg": 5,
            "lambda_smooth": 0.1,
            "lambda_cov": 1.0,
            "lambda_scatter": 0.5,
            "scatter_threshold": 0.25,
            "edge_drop_rate": 0.1,
            "cov_warmup_epochs": 5,
            "patience": 10,
            "seed": 42,
        },
        "output": {
            "checkpoint": "outputs/pcard_pretrained.pth",
            "split_manifest": "outputs/split_manifest.json",
        },
        "wandb": {
            "mode": "disabled",
            "project_name": "PCARD",
            "entity": None,
            "run_name": None,
        },
    }
    with open(path, "r", encoding="utf-8") as stream:
        supplied = yaml.safe_load(stream) or {}
    return _merge(defaults, supplied)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_by_subject(dataset, test_fraction, seed, manifest_path):
    """Split complete subjects, including every position/view, only once."""
    subjects = sorted({group[0][0] for group in dataset.groups})
    if len(subjects) < 2:
        raise ValueError("At least two subjects are required for a train/test split.")

    rng = random.Random(seed)
    rng.shuffle(subjects)
    n_test = min(len(subjects) - 1, max(1, round(len(subjects) * test_fraction)))
    test_subjects = set(subjects[:n_test])
    train_subjects = set(subjects[n_test:])

    train_indices, test_indices = [], []
    for group_idx, ((subject, _), _) in enumerate(dataset.groups):
        target = test_indices if subject in test_subjects else train_indices
        first = group_idx * dataset.iterations
        target.extend(range(first, first + dataset.iterations))

    manifest = {
        "seed": seed,
        "test_fraction": test_fraction,
        "train_subjects": sorted(train_subjects),
        "test_subjects": sorted(test_subjects),
    }
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return (
        torch.utils.data.Subset(dataset, train_indices),
        torch.utils.data.Subset(dataset, test_indices),
        manifest,
    )


class RunLogger:
    """Optional W&B logger; disabled runs need no W&B installation or key."""

    def __init__(self, settings, run_config):
        self.run = None
        mode = str(settings.get("mode", "disabled")).lower()
        if mode == "disabled":
            return
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "Install the optional 'tracking' dependencies to use W&B."
            ) from exc
        self.run = wandb.init(
            project=settings.get("project_name") or "PCARD",
            entity=settings.get("entity"),
            name=settings.get("run_name"),
            mode=mode,
            config=run_config,
        )

    def log(self, values):
        if self.run is not None:
            self.run.log(values)

    def finish(self):
        if self.run is not None:
            self.run.finish()


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 10, delta: float = 0.0,
                 verbose: bool = False,
                 save_path: str | None = None):

        self.patience   = patience
        self.delta      = delta
        self.counter    = 0
        self.best_loss  = None
        self.early_stop = False
        self.verbose    = verbose
        self.save_path  = save_path
        self.best_model_state = None

    def __call__(self, val_loss: float, model: nn.Module):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            if self.verbose and self.best_loss is not None:
                print(f"  Val loss improved "
                      f"({self.best_loss:.6f} → {val_loss:.6f}). Saving.")
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())

            if self.save_path is not None:
                import os
                os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                torch.save(self.best_model_state, self.save_path)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  No improvement for "
                      f"{self.counter}/{self.patience} epochs.")
            if self.counter >= self.patience:
                self.early_stop = True


# ---------------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------------

def drop_edges(edge_index: torch.Tensor,
               drop_rate: float = 0.1,
               training: bool = True) -> torch.Tensor:
    """
    Randomly drop edges from the dynamic graph during training.

    The dynamic graph can overfit to noise by consistently routing through
    the same noise point positions across batches. Feature dropout does not
    address this because the overfitting is structural, not
    parametric. DropEdge prevents the graph from memorising specific noise
    edges — any noise edge that survives must genuinely reduce the loss.

    No-op at test time so inference is deterministic.
    """
    if not training or drop_rate == 0.0:
        return edge_index
    keep = torch.rand(edge_index.size(1), device=edge_index.device) > drop_rate
    return edge_index[:, keep]


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def edge_nce_loss(features: torch.Tensor,
                  edge_index: torch.Tensor,
                  num_neg: int = 5,
                  tau: float = 0.2) -> torch.Tensor:
    """
    Binary NCE loss over Euclidean kNN edges (input-space graph).

    Pulls positional neighbours together in feature space. This is the
    only loss that operates on the input-space graph — it provides the
    anchor so the representation does not drift away from 3D geometry
    entirely while the covariance repulsion reshapes it.

    Positive pairs : edges in edge_index_input (Euclidean neighbours)
    Negative pairs : uniformly sampled random pairs
    """
    z        = F.normalize(features, p=2, dim=1)
    src, dst = edge_index

    pos_sim  = (z[src] * z[dst]).sum(dim=1) / tau
    pos_loss = -F.logsigmoid(pos_sim).mean()

    N        = z.size(0)
    neg_src  = src.repeat_interleave(num_neg)
    neg_dst  = torch.randint(0, N, (neg_src.size(0),), device=z.device)
    neg_sim  = (z[neg_src] * z[neg_dst]).sum(dim=1) / tau
    neg_loss = -F.logsigmoid(-neg_sim).mean()

    return pos_loss + neg_loss


def local_smoothness_loss(features: torch.Tensor,
                           edge_index: torch.Tensor) -> torch.Tensor:
    """
    Penalise feature disagreement along dynamic graph edges.

    Cheap baseline regulariser on the feature field. Weaker than the
    covariance repulsion but stabilises early training.
    """
    z        = F.normalize(features, p=2, dim=1)
    src, dst = edge_index
    return (z[src] - z[dst]).pow(2).sum(dim=1).mean()


def covariance_repulsion_loss(features: torch.Tensor,
                               pos: torch.Tensor,
                               knn_idx_input: torch.Tensor,
                               edge_index_feat: torch.Tensor,
                               scatter_threshold: float = 0.25,
                               eps: float = 1e-6) -> torch.Tensor:
    """
    Covariance-based geometric repulsion over dynamic graph edges
    (one-sided ReLU form).

    What this does
    --------------
    For every edge (i, j) in the current dynamic graph, compute a
    geometric target similarity from the Frobenius distance between
    local 3D covariance matrices, and penalise feat_sim *only when it
    exceeds the target*:

        target_sim = 2 * exp(-cov_diff_norm) - 1
        loss       = mean over edges of  relu(feat_sim - target_sim)²

    Behavioural roles
    -----------------
    - Geometrically incoherent pair (large cov_diff, target ≈ -1):
      almost always feat_sim > target_sim, so the loss fires and pushes
      feat_sim down. **Repulsion**.
    - Geometrically coherent pair (small cov_diff, target ≈ +1):
      feat_sim ≤ +1 by construction, so the residual feat_sim - target
      is ≤ 0, ReLU is zero, no force. **No attraction from this loss.**

    Why the loss is repulsion-only
    ------------------------------
    NCE on G_input already provides spatial attraction: Euclidean
    neighbours are pulled together, random pairs are pushed apart.
    Adding a separate geometric-attraction term inside L_cov was
    redundant for spatially-close geometrically-similar pairs, and
    actively harmful for spatially-distant geometrically-similar pairs
    (it produced global type-clustering — e.g. noise blobs on opposite
    sides of the cloud being pulled together in feature space, breaking
    the locality assumption that the inference-time filter relies on).

    By restricting L_cov to repulsion only, we decouple the two forces:
      - NCE: spatial attraction (Euclidean kNN).
      - L_cov: geometric repulsion (high cov_diff pairs in the dynamic
        graph).
    Distant geometrically-similar pairs receive no attractive force from
    either loss, so they stay dissimilar in feature space and the
    dynamic graph at inference does not connect them across the cloud.

    Why this does not re-introduce the original collapse
    ----------------------------------------------------
    The collapse-prone earlier version was
        loss_old = cov_diff * (feat_sim + 1),
    whose gradient on feat_sim was always positive (cov_diff is
    mean-normalised, always ≥ 0). Every edge pushed feat_sim down,
    overwhelming NCE.

    The ReLU form pushes down only on edges where feat_sim genuinely
    exceeds its geometric target — i.e. only on the geometrically
    incoherent pairs that the loss is supposed to act on. Coherent
    pairs contribute zero. NCE's attractive pull on spatial neighbours
    is therefore not fought across the whole graph, only on the
    contested edges where geometry and spatial proximity disagree.

    Self-correcting behaviour (unchanged)
    -------------------------------------
    Because we operate on the dynamic graph's own edges, the loss is
    self-correcting across training: when an incoherent pair (i, j)
    appears in G_feat, the loss pushes feat_sim down; by the next
    iteration the pair may have dropped out of G_feat and replaced by
    a different one. The loss always acts on the currently-hardest
    cases.

    Covariance computation (unchanged)
    ----------------------------------
    C is computed from 3D positions via build_local_covariance_masked,
    which softly downweights high-scatter neighbours before computing
    C_i. Gradients flow only through features z; C is under no_grad.

    Args:
        features:          (N, F)  learned point features
        pos:               (N, 3)  input positions
        knn_idx_input:     (N, k)  input-space kNN (for covariance scaffold)
        edge_index_feat:   (2, E)  dynamic graph edges
        scatter_threshold: soft boundary for noise-aware covariance masking
        eps:               numerical stabiliser

    Returns:
        Scalar loss.
    """
    # C computed from 3D positions only — no gradient through geometry
    with torch.no_grad():
        _, _, C = build_local_covariance_masked(
            pos, knn_idx_input,
            scatter_threshold=scatter_threshold,
            eps=eps
        )                                                          # (N, 3, 3)

    z        = F.normalize(features, p=2, dim=1)
    src, dst = edge_index_feat

    # Geometric signal — no gradient through this side.
    with torch.no_grad():
        cov_diff = (C[src] - C[dst]).pow(2).sum(dim=(-2, -1))      # (E,)
        cov_diff = cov_diff / (cov_diff.mean() + eps)              # unit mean
        # Map cov_diff ∈ [0, ∞) → target_sim ∈ (-1, +1].
        # exp(-0) = 1, exp(-1) ≈ 0.37, exp(-3) ≈ 0.05.
        target_sim = 2.0 * torch.exp(-cov_diff) - 1.0              # (E,)

    # Feature similarity carries the gradient.
    feat_sim = (z[src] * z[dst]).sum(dim=1)                        # (E,) in [-1,1]

    # One-sided: penalise only when feat_sim exceeds its geometric
    # target. Coherent pairs (target ≈ +1) contribute zero — NCE
    # handles spatial attraction. See docstring for rationale.
    return torch.relu(feat_sim - target_sim).pow(2).mean()


def scatter_separation_loss(features: torch.Tensor,
                             pos: torch.Tensor,
                             knn_idx_input: torch.Tensor,
                             edge_index_feat: torch.Tensor,
                             scatter_threshold: float = 0.25,
                             margin: float = 1.0,
                             eps: float = 1e-6) -> torch.Tensor:
    """
    Scatter-based geometric repulsion — explicit noise/surface
    separation in feature space using the scatter index (one-sided
    ReLU form).

    Same role as covariance_repulsion_loss but using the scalar scatter
    index `λ₃ / (λ₁+λ₂+λ₃)` instead of the full Frobenius distance
    between covariance matrices. Scatter is in [0, 1/3]: 0 for planar
    surfaces, 1/3 for isotropic noise.

        target_sim = 1 - 6 * |scatter_i - scatter_j|   (clamped to [-1, +1])
        loss       = mean over edges of  relu(feat_sim - target_sim)²

    Behavioural roles
    -----------------
    - Cross-type pair (e.g. surface ↔ noise, scatter_diff large,
      target ≈ -1): feat_sim > target_sim, ReLU fires, repulsion.
    - Same-type pair (scatter_diff small, target ≈ +1): feat_sim ≤ +1
      always, ReLU is zero, no force from this loss. NCE handles
      same-type attraction implicitly via spatial proximity (same-type
      points tend to be locally clustered).

    Why one-sided
    -------------
    See covariance_repulsion_loss for the full argument. In short: NCE
    already provides spatial attraction on G_input; an attractive
    component in L_scatter would either duplicate that (for spatially
    close same-type pairs) or actively create long-range type
    clustering (for spatially distant same-type pairs). The latter
    breaks the inference-time filter, which assumes feature-space
    neighbours are also spatial neighbours. The one-sided ReLU
    decouples the forces: NCE pulls spatially, L_scatter pushes on
    cross-type pairs, nothing pulls cross-cloud same-type pairs
    together.

    Earlier formulations
    --------------------
    - Original hinge form `scatter_diff * relu(feat_sim + margin)` with
      margin=1: the hinge never engaged because feat_sim + 1 ≥ 0
      always. Reduced to a one-sided downward push on every edge,
      caused feature collapse.
    - Symmetric quadratic `(feat_sim - target_sim)²`: avoided collapse
      but introduced global type-clustering by actively pulling
      same-type pairs together anywhere in the cloud.
    - Current form (this one): symmetric quadratic with a ReLU on the
      residual, so only the repulsion half remains.

    The `margin` parameter is retained in the signature for backward
    compatibility but is no longer used.
    """
    with torch.no_grad():
        evals, _, _ = build_local_covariance_masked(
            pos, knn_idx_input,
            scatter_threshold=scatter_threshold,
            eps=eps
        )
        scatter = compute_scatter(evals, eps=eps)                  # (N,)

    z        = F.normalize(features, p=2, dim=1)
    src, dst = edge_index_feat

    with torch.no_grad():
        scatter_diff = (scatter[src] - scatter[dst]).abs()         # (E,) in [0, 1/3]
        # Linear map scatter_diff ∈ [0, 1/3] → target_sim ∈ [-1, +1].
        target_sim = (1.0 - 6.0 * scatter_diff).clamp(-1.0, 1.0)   # (E,)

    feat_sim = (z[src] * z[dst]).sum(dim=1)                        # (E,)

    # One-sided: penalise only when feat_sim exceeds its scatter-derived
    # target. Same-type pairs (target ≈ +1) contribute zero.
    return torch.relu(feat_sim - target_sim).pow(2).mean()


# ---------------------------------------------------------------------------
# Shared forward pass
# ---------------------------------------------------------------------------

def _forward_pass(model, data, k, num_neg, tau_nce,
                  lambda_smooth, lambda_cov, lambda_scatter,
                  scatter_threshold, edge_drop_rate, training):
    """
    Single forward pass shared by train() and test().

    Loss structure
    --------------
    loss_nce     : pulls Euclidean neighbours together (input-space graph)
    loss_smooth  : soft regulariser on dynamic graph feature field
    loss_cov     : covariance-weighted repulsion over dynamic graph edges —
                   the primary geometric signal; pushes apart feature-space
                   neighbours whose 3D covariance matrices disagree
    loss_scatter : explicit noise/surface separation via scatter index

    The tension between loss_nce (attraction) and loss_cov + loss_scatter
    (repulsion) is the core mechanism. Noise points satisfy the NCE
    attraction (they are Euclidean neighbours of surface points) but
    violate the covariance repulsion (their C_i is isotropic vs planar).
    The optimiser resolves this by moving noise points to a region of
    feature space where neither force acts strongly — isolated from
    surface points, visible as a separate cluster in PCA.

    Diagnostic: scatter_consistency
    --------------------------------
    Mean scatter dissimilarity over dynamic graph edges. Not a loss term.
    Should decrease over training as the graph increasingly connects
    geometrically similar points.
    """
    features, knn_graphs = model(data)

    # Input-space graph: Euclidean kNN, used only for NCE
    edge_index_input = knn_graph(data.pos, k=k, batch=data.batch)

    # Dynamic graph: 20 nearest feature-space neighbours, last DGCNN layer.
    # Edge dropout prevents topology overfitting to noise positions.
    edge_index_feat = drop_edges(
        knn_graphs[-1], drop_rate=edge_drop_rate, training=training
    )

    # knn_idx from input-space graph — used as the 3D geometry scaffold
    # for covariance computation in both repulsion losses
    knn_idx_input = edge_index_to_knn_idx(
        edge_index_input, num_points=data.pos.size(0), k=k
    )

    loss_nce     = edge_nce_loss(
                       features, edge_index_input,
                       num_neg=num_neg, tau=tau_nce)

    loss_smooth  = local_smoothness_loss(features, edge_index_feat)

    loss_cov     = covariance_repulsion_loss(
                       features, data.pos,
                       knn_idx_input, edge_index_feat,
                       scatter_threshold=scatter_threshold)

    loss_scatter = scatter_separation_loss(
                       features, data.pos,
                       knn_idx_input, edge_index_feat,
                       scatter_threshold=scatter_threshold)

    loss = (loss_nce
            + lambda_smooth  * loss_smooth
            + lambda_cov     * loss_cov
            + lambda_scatter * loss_scatter)

    # Diagnostics — not loss terms
    with torch.no_grad():
        evals, _, _ = build_local_covariance_masked(
            data.pos, knn_idx_input,
            scatter_threshold=scatter_threshold
        )
        scatter = compute_scatter(evals)
        src, dst = edge_index_feat
        scatter_consistency = (scatter[src] - scatter[dst]).abs().mean()

        # Feature norm — collapse early-warning. After removing the
        # final ReLU the natural scale is much smaller than O(1); any
        # value comfortably above F.normalize's eps (1e-12) is fine.
        # The signal we care about is *stability*, not magnitude.
        feature_norm = features.norm(dim=1).mean()

        # Edge label purity — diagnostic only. Measures the fraction
        # of dynamic-graph edges whose two endpoints share the same
        # anatomical label (femur/patella/tibia/fallback). The labels
        # are never used by any loss; this is purely a window into
        # whether the geometric losses are organising features by
        # anatomical structure. Rising = the dynamic graph is
        # increasingly connecting same-anatomy points, which would
        # mean the embedding is learning structurally meaningful
        # neighbourhoods independent of the noise/surface axis your
        # scatter_consistency metric tracks.
        edge_label_purity = (data.y[src] == data.y[dst]).float().mean()

    return (loss, loss_nce, loss_smooth, loss_cov, loss_scatter,
            scatter_consistency, feature_norm, edge_label_purity)


# ---------------------------------------------------------------------------
# Train / test loops
# ---------------------------------------------------------------------------

def _run_epoch(model, loader, optimizer, k, num_neg, tau_nce,
               lambda_smooth, lambda_cov, lambda_scatter,
               scatter_threshold, edge_drop_rate, training,
               device, grad_clip=1.0):
    """
    grad_clip: max global gradient norm. Clipping is essential here because
    the random subsampling makes per-batch loss variance high — a single
    unlucky batch can otherwise kick the optimiser out of a good basin
    (we observed this: model recovered to test loss 2.61 at epoch 9, then
    a bad step at epoch 10 sent it back to 5.59 and it never recovered).
    """
    if training:
        model.train()
    else:
        model.eval()

    totals = dict(loss=0., nce=0., smooth=0., cov=0.,
                  scatter=0., scatter_consistency=0., feature_norm=0.,
                  edge_label_purity=0.)

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for data in loader:
            data = data.to(device)

            if training:
                optimizer.zero_grad()

            (loss, loss_nce, loss_smooth, loss_cov,
             loss_scatter, scatter_consistency,
             feature_norm, edge_label_purity) = _forward_pass(
                model, data, k, num_neg, tau_nce,
                lambda_smooth, lambda_cov, lambda_scatter,
                scatter_threshold, edge_drop_rate, training=training
            )

            if training:
                loss.backward()
                # Stabilise against rare high-norm batches caused by
                # adverse random subsamples / dynamic-graph realisations.
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip
                )
                optimizer.step()

            totals["loss"]                += loss.item()
            totals["nce"]                 += loss_nce.item()
            totals["smooth"]              += loss_smooth.item()
            totals["cov"]                 += loss_cov.item()
            totals["scatter"]             += loss_scatter.item()
            totals["scatter_consistency"] += scatter_consistency.item()
            totals["feature_norm"]        += feature_norm.item()
            totals["edge_label_purity"]   += edge_label_purity.item()

    n = len(loader)
    return {key: val / n for key, val in totals.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train the PCARD point encoder")
    parser.add_argument(
        "--config", default="configs/pcard.yaml",
        help="Path to a YAML configuration file.",
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto",
        help="Training device (default: auto).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    data_cfg = SimpleNamespace(**config["data"])
    model_cfg = SimpleNamespace(**config["model"])
    cfg = SimpleNamespace(**config["training"])
    output_cfg = SimpleNamespace(**config["output"])

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(
        "cuda" if args.device == "cuda" or
        (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    seed_everything(cfg.seed)
    print(f"Using device: {device}")

    root = data_cfg.root
    # num_points: per-sample subsample size. 4096 is a good balance for k=20
    #   covariance neighbourhoods — enough density for local geometry, light
    #   enough that batch_size=4 fits comfortably in RAM.
    # iterations: number of random subsamples per (subject,position) per
    #   epoch. Each iteration draws a fresh random view, so over training
    #   the model sees the full point distribution.
    # sampling="random" is distributionally lossless across epochs; switch
    #   to "fps" if you need coverage-preserving views in a single pass.
    # iterations bumped 20 → 40. Each iteration is a fresh random subsample
    # of a (subject,position) cloud. More iterations = lower variance per
    # batch (the optimiser sees a smoother loss landscape), which is the
    # root cause of the bouncing observed in earlier runs. The on-disk
    # cache makes this essentially free after the first epoch.
    dataset = CATMAUS(
        root=root,
        num_points=data_cfg.num_points,
        iterations=data_cfg.iterations,
        sampling=data_cfg.sampling,
        cache_dir=data_cfg.cache_dir,
    )
    print("Dataset indexed successfully")

    train_ds, test_ds, split_manifest = split_by_subject(
        dataset,
        test_fraction=data_cfg.test_fraction,
        seed=cfg.seed,
        manifest_path=output_cfg.split_manifest,
    )
    print(
        f"Subject split: {len(split_manifest['train_subjects'])} train / "
        f"{len(split_manifest['test_subjects'])} test"
    )

    train_generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        generator=train_generator,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
    )
    print(f"Train batches: {len(train_loader)}  "
          f"Test batches:  {len(test_loader)}")

    model = DGCNNWithKNN(
        k=model_cfg.k, feature_dim=model_cfg.feature_dim
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs, eta_min=1e-5
    )

    early_stopping = EarlyStopping(
        patience=cfg.patience,
        verbose=True,
        save_path=output_cfg.checkpoint,
    )
    logger = RunLogger(config["wandb"], config)

    for epoch in range(1, cfg.num_epochs + 1):

        # ------------------------------------------------------------
        # Geometric-loss curriculum
        # ------------------------------------------------------------
        # lambda_cov and lambda_scatter are the repulsion forces that
        # reshape feature space according to 3D geometry. Applying them
        # at full strength from epoch 1 — when features are essentially
        # random — pushes the model into degenerate configurations
        # before NCE has a chance to form a coherent embedding. We
        # linearly ramp them from 0 → full over `cov_warmup_epochs`.
        warmup = min(1.0, epoch / float(cfg.cov_warmup_epochs))
        lambda_cov_eff     = cfg.lambda_cov     * warmup
        lambda_scatter_eff = cfg.lambda_scatter * warmup

        train_m = _run_epoch(
            model, train_loader, optimizer,
            k=model_cfg.k, num_neg=cfg.num_neg, tau_nce=cfg.tau_nce,
            lambda_smooth=cfg.lambda_smooth,
            lambda_cov=lambda_cov_eff,
            lambda_scatter=lambda_scatter_eff,
            scatter_threshold=cfg.scatter_threshold,
            edge_drop_rate=cfg.edge_drop_rate,
            training=True, device=device,
        )

        test_m = _run_epoch(
            model, test_loader, optimizer=None,
            k=model_cfg.k, num_neg=cfg.num_neg, tau_nce=cfg.tau_nce,
            lambda_smooth=cfg.lambda_smooth,
            lambda_cov=lambda_cov_eff,
            lambda_scatter=lambda_scatter_eff,
            scatter_threshold=cfg.scatter_threshold,
            edge_drop_rate=cfg.edge_drop_rate,
            training=False, device=device,
        )

        scheduler.step()

        print(
            f"Epoch {epoch:03d} | "
            f"warmup {warmup:.2f} | "
            f"Train loss {train_m['loss']:.4f} "
            f"(nce {train_m['nce']:.4f}  "
            f"smooth {train_m['smooth']:.4f}  "
            f"cov {train_m['cov']:.4f}  "
            f"scatter {train_m['scatter']:.4f}) | "
            f"feat_norm {train_m['feature_norm']:.3f} | "
            f"sc_tr {train_m['scatter_consistency']:.4f} "
            f"sc_te {test_m['scatter_consistency']:.4f} | "
            f"purity {train_m['edge_label_purity']:.3f} | "
            f"Test loss {test_m['loss']:.4f}"
        )

        logger.log({
            "epoch":              epoch,
            "lr":                 scheduler.get_last_lr()[0],
            "warmup":             warmup,
            "lambda_cov_eff":     lambda_cov_eff,
            "lambda_scatter_eff": lambda_scatter_eff,
            **{f"train_{k}": v for k, v in train_m.items()},
            **{f"test_{k}":  v for k, v in test_m.items()},
        })

        early_stopping(test_m['loss'], model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    if early_stopping.best_model_state is None:
        raise RuntimeError("Training finished without producing a checkpoint.")
    checkpoint_path = Path(output_cfg.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(early_stopping.best_model_state, checkpoint_path)
    print(f"Best model saved to {checkpoint_path}.")
    logger.finish()


if __name__ == "__main__":
    main()
