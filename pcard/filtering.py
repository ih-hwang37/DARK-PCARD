import torch
import torch.nn.functional as F
import h5py


# ---------------------------------------------------------------------------
# Scatter index utility
# ---------------------------------------------------------------------------

def compute_scatter(evals: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Scatter (isotropy) index from eigenvalues in descending order.

        scatter_i = λ₃ / (λ₁ + λ₂ + λ₃)

      scatter ≈ 0   →  planar / surface point  (λ₃ ≈ 0)
      scatter ≈ 1/3 →  isotropic blob / noise  (all λ equal)

    Exposed as a standalone utility so train.py can use it for both the
    scatter separation loss and the scatter_consistency diagnostic without
    recomputing covariances from scratch.

    Args:
        evals: (N, 3) eigenvalues in descending order
        eps:   numerical stabiliser

    Returns:
        scatter: (N,) in [0, 1/3]
    """
    return evals[:, 2] / (evals.sum(dim=1) + eps)


# ---------------------------------------------------------------------------
# Covariance primitives
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_local_covariance(pos: torch.Tensor,
                            knn_idx: torch.Tensor,
                            eps: float = 1e-6):
    """
    Per-point local covariance matrices — uniform neighbour weights.

    Inference-time version used by filter_point_cloud. At inference the
    cloud has already been filtered so neighbourhood corruption from noise
    is limited and uniform weighting is appropriate.

    train.py calls build_local_covariance_masked instead, which softly
    downweights high-scatter neighbours during training when the cloud is
    still noisy.

    Args:
        pos:     (N, 3)
        knn_idx: (N, k)

    Returns:
        evals: (N, 3)     descending eigenvalues
        evecs: (N, 3, 3)  eigenvectors as columns
        cov:   (N, 3, 3)  covariance matrices
    """
    k        = knn_idx.shape[1]
    neigh    = pos[knn_idx]                                       # (N, k, 3)
    mu       = neigh.mean(dim=1, keepdim=True)                    # (N, 1, 3)
    centered = neigh - mu                                         # (N, k, 3)
    cov      = (centered.transpose(1, 2) @ centered) / float(k)  # (N, 3, 3)
    cov      = cov + eps * torch.eye(3, device=pos.device,
                                     dtype=pos.dtype).unsqueeze(0)

    evals, evecs = torch.linalg.eigh(cov)
    evals = torch.flip(evals, dims=[1])
    evecs = torch.flip(evecs, dims=[2])

    return evals, evecs, cov


@torch.no_grad()
def build_local_covariance_masked(pos: torch.Tensor,
                                   knn_idx: torch.Tensor,
                                   scatter_threshold: float = 0.25,
                                   eps: float = 1e-6):
    """
    Noise-aware covariance estimation with soft scatter-based masking.

    Motivation
    ----------
    During training, noise points are intentionally kept in the graph so
    the covariance repulsion loss can act on them. But if noisy neighbours
    are included with uniform weight in covariance computation, they
    corrupt C_i for surface points — the repulsion loss then fires on a
    corrupted signal.

    This function resolves that: noise neighbours are softly downweighted
    before computing C_i. Surface points get clean covariance estimates
    from their surface-like neighbours. Noise points get C_i from whatever
    they have — typically isotropic — making their covariance structurally
    distinct from surface C_i (planar). The repulsion loss then has a
    reliable geometric signal even in noisy training batches.

    Masking mechanism (two-pass)
    ----------------------------
    Pass 1: uniform-weight covariance → real scatter index λ₃/(λ₁+λ₂+λ₃)
            for every point j, using the actual eigendecomposition.

    Pass 2: for each point i, look up the pass-1 scatter score of each
            neighbour j and form a soft weight:

                w_j = sigmoid( -(scatter_j - scatter_threshold) * 10 )

            High-scatter (noisy) neighbours → w_j ≈ 0
            Low-scatter  (surface) neighbours → w_j ≈ 1

    This eliminates the old d_min/d_max proxy which was unreliable:
      - Surface points with ring-shaped neighbourhoods scored ≈ 1 (false noise)
      - Noise blobs with spread centroids scored ≈ 0 (false surface)
    The eigenvalue-derived scatter is geometrically correct by construction.

    scatter_threshold = 0.25 sits between pure surface (0) and pure
    isotropic (1/3 ≈ 0.33). Intentionally loose — we only exclude clear
    volumetric blobs, not high-curvature surface points which can have
    moderate scatter.

    Args:
        pos:               (N, 3)
        knn_idx:           (N, k)
        scatter_threshold: soft boundary between surface and noise
        eps:               numerical stabiliser

    Returns:
        evals: (N, 3)     descending eigenvalues
        evecs: (N, 3, 3)  eigenvectors as columns
        cov:   (N, 3, 3)  noise-masked weighted covariance matrices
    """
    k        = knn_idx.shape[1]
    neigh    = pos[knn_idx]                                       # (N, k, 3)
    mu       = neigh.mean(dim=1, keepdim=True)                    # (N, 1, 3)
    centered = neigh - mu                                         # (N, k, 3)

    # --- Pass 1: uniform covariance → real scatter scores ----------------
    cov_pass1 = (centered.transpose(1, 2) @ centered) / float(k) # (N, 3, 3)
    cov_pass1 = cov_pass1 + eps * torch.eye(3, device=pos.device,
                                            dtype=pos.dtype).unsqueeze(0)
    evals_pass1, _ = torch.linalg.eigh(cov_pass1)                # ascending
    evals_pass1    = torch.flip(evals_pass1, dims=[1])            # descending
    scatter_all    = compute_scatter(evals_pass1, eps=eps)        # (N,) ∈ [0, 1/3]

    # --- Pass 2: neighbour scatter → soft weights ------------------------
    neigh_scatter = scatter_all[knn_idx]                          # (N, k)
    weights = torch.sigmoid(
        -(neigh_scatter - scatter_threshold) * 10.0
    )                                                             # (N, k) ∈ (0,1)
    weights = weights / (weights.sum(dim=1, keepdim=True) + eps)  # normalise rows

    # --- weighted covariance ---------------------------------------------
    w_centered = centered * weights.unsqueeze(-1)                 # (N, k, 3)
    cov        = w_centered.transpose(1, 2) @ centered            # (N, 3, 3)
    cov        = cov + eps * torch.eye(3, device=pos.device,
                                       dtype=pos.dtype).unsqueeze(0)

    evals, evecs = torch.linalg.eigh(cov)
    evals = torch.flip(evals, dims=[1])
    evecs = torch.flip(evecs, dims=[2])

    return evals, evecs, cov


# ---------------------------------------------------------------------------
# Edge-index → knn_idx  (fully vectorised, no Python for-loop)
# ---------------------------------------------------------------------------

@torch.no_grad()
def edge_index_to_knn_idx(knn_graphs, num_points: int, k: int) -> torch.Tensor:
    """
    Robustly convert knn_graphs into knn_idx (N, k).

    Accepts:
      - knn_idx tensor  (N, k)
      - edge_index tensor (2, E)
      - tuple/list whose last element is one of the above

    For edge_index inputs:
      1. Symmetrise  — add reverse edges so no node has degree 0
      2. Remove self-loops
      3. Sort by source node; scatter into (N, k) via cumsum offsets
      4. Pad isolated / low-degree nodes with their own index
    """
    if isinstance(knn_graphs, (tuple, list)):
        knn_graphs = knn_graphs[-1]

    if not torch.is_tensor(knn_graphs):
        raise TypeError(
            f"knn_graphs must be a Tensor or tuple/list of Tensors, "
            f"got {type(knn_graphs)}"
        )

    # Case 1: already (N, k)
    if (knn_graphs.dim() == 2
            and knn_graphs.shape[0] == num_points
            and knn_graphs.shape[1] == k):
        return knn_graphs.long()

    # Case 2: edge_index (2, E)
    if knn_graphs.dim() == 2 and knn_graphs.shape[0] == 2:
        src, dst = knn_graphs.long()

        # Symmetrise
        src = torch.cat([src, dst], dim=0)
        dst = torch.cat([dst, knn_graphs[0].long()], dim=0)

        # Remove self-loops
        mask = src != dst
        src, dst = src[mask], dst[mask]

        # Sort by source node
        order = torch.argsort(src, stable=True)
        src_s = src[order]
        dst_s = dst[order]

        counts  = torch.bincount(src_s, minlength=num_points)
        offsets = torch.zeros(num_points + 1, dtype=torch.long,
                              device=src_s.device)
        offsets[1:] = counts.cumsum(0)

        # Default = self-index (handles isolated nodes)
        knn_idx = (torch.arange(num_points, device=src_s.device)
                       .unsqueeze(1).expand(num_points, k).clone())

        ranks = (torch.arange(src_s.size(0), device=src_s.device)
                 - offsets[src_s])
        valid = ranks < k
        knn_idx[src_s[valid], ranks[valid]] = dst_s[valid]

        return knn_idx

    raise ValueError(
        f"Unrecognized knn_graphs shape: {tuple(knn_graphs.shape)}"
    )


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

@torch.no_grad()
def filter_point_cloud(pos, labels, knn_graphs, k=20, verbose=False,
                       tau=1.5, eps=1e-6, use_ratio=True, scatter_tau=None):
    """
    Geometry-aware outlier filter based on eigenvalue consistency.

    A point is kept if:
      (a) Its eigenvalue signature is consistent with its neighbours
          (robust Z-score of eigenvalue residual ≤ tau), AND
      (b) Optionally, it is not a volumetric / isotropic blob
          (scatter index λ₃/(λ₁+λ₂+λ₃) ≤ scatter_tau).

    Uses build_local_covariance (uniform weights) because at inference
    the cloud has already been filtered and neighbourhood corruption is
    limited. The masked version is only needed during training.

    The dynamic graph passed in as knn_graphs should be the one produced
    by the trained model — the covariance repulsion loss has ensured that
    its edges connect geometrically consistent points, so the covariance
    estimates computed here are clean and the eigenvalue residuals give
    an unambiguous noise/surface separation.

    Args:
        pos:         (N, 3)
        labels:      (N,) or None
        knn_graphs:  edge_index (2, E) or knn_idx (N, k)
        k:           neighbourhood size
        tau:         robust Z-score threshold
        eps:         covariance regularisation
        use_ratio:   whether to apply the volumetric blob filter
        scatter_tau: quantile threshold; defaults to 95th percentile

    Returns:
        filtered_pos:    (M, 3)
        filtered_labels: (M,) or None
        keep_indices:    (M,) indices into original N
    """
    num_points = pos.size(0)
    knn_idx    = edge_index_to_knn_idx(knn_graphs, num_points=num_points, k=k)

    # Use uniform-weight covariance at inference — cloud is pre-filtered
    evals, _, _ = build_local_covariance(pos, knn_idx, eps=eps)   # (N, 3)

    # Eigenvalue residual
    neigh_evals = evals[knn_idx]                                   # (N, k, 3)
    pred        = neigh_evals.mean(dim=1)                          # (N, 3)
    residual    = torch.linalg.norm(evals - pred, dim=1)           # (N,)

    # Robust Z-score — resistant to the outliers being removed
    med      = residual.median()
    mad      = (residual - med).abs().median() + eps
    robust_z = (residual - med).abs() / mad
    keep_z   = robust_z <= tau

    if use_ratio:
        scatter    = compute_scatter(evals, eps=eps)
        if scatter_tau is None:
            scatter_tau = torch.quantile(scatter, 0.95)
        keep_ratio = scatter <= scatter_tau
        keep_mask  = keep_z & keep_ratio
    else:
        keep_mask = keep_z

    filtered_pos    = pos[keep_mask]
    filtered_labels = labels[keep_mask] if labels is not None else None

    # squeeze(1) not squeeze() — avoids collapsing single-point result
    # to a scalar which breaks downstream indexing
    keep_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)

    if verbose:
        kept = int(keep_mask.sum().item())
        print(f"[EigenFilter] Kept {kept}/{num_points} points  "
              f"(tau={tau}, scatter_tau={scatter_tau:.4f})")

    return filtered_pos, filtered_labels, keep_indices


# ---------------------------------------------------------------------------
# Joint filter — scatter + covariance + feature coherence
# ---------------------------------------------------------------------------

@torch.no_grad()
def filter_point_cloud_joint(
    pos,                    # (N, 3)  raw positions (unit-sphere normalised)
    labels,                 # (N,)    bone labels (0=fem,1=pat,2=tib,3=other)
    features,               # (N, F)  PCARD embeddings from forward pass
    dynamic_edge_index,     # (2, E)  last EdgeConv layer's feature-space kNN
    k: int = 20,
    alpha: float = 1.0,     # scatter signal weight
    beta:  float = 1.0,     # eigenvalue residual weight
    gamma: float = 1.0,     # dynamic-graph covariance Frobenius weight
    delta: float = 1.0,     # feature incoherence weight  (1 − cosine sim)
    scatter_threshold: float = 0.25,  # hard pre-gate: same value used in training
    score_tau: float = 2.0,           # robust Z-score threshold on joint score
    eps: float = 1e-6,
    verbose: bool = False,
) -> tuple:
    """
    Four-signal joint noise filter that uses everything PCARD learned.

    Signal origins
    --------------
    alpha · scatter        — λ_min/Σλ from spatial-kNN covariance.
                             Directly reflects L_scatter's training target:
                             low = planar surface, high = isotropic noise.

    beta  · residual_z     — robust Z-score of how much each point's
                             eigenvalue signature deviates from its spatial
                             neighbours. The original filter criterion;
                             kept here as a geometric sanity check.

    gamma · cov_frob_dyn   — mean Frobenius distance ||C_i − C_j||_F
                             averaged over the DYNAMIC (feature-space) kNN.
                             L_cov trained the model to route only
                             covariance-coherent pairs into the dynamic
                             graph. High Frobenius on a dynamic edge means
                             the model failed to separate that pair —
                             a strong noise indicator.

    delta · feat_incohere  — 1 − mean cosine similarity to SPATIAL
                             neighbours in feature space. L_scatter pushed
                             noise points away from their spatial neighbours
                             in feature space. Low cosine similarity to
                             spatial neighbours = noise point that the model
                             correctly isolated.

    All four signals are min-max normalised to [0, 1] and summed.
    A hard scatter pre-gate (scatter > scatter_threshold) immediately
    removes clear volumetric blobs before the joint scoring step.

    Among the pre-gated candidates, a robust Z-score (MAD-based) is
    computed on the joint score. Points are kept if their robust Z-score
    is ≤ score_tau. This is data-adaptive — the cut moves with the
    score distribution rather than discarding a fixed fraction.

    Returns
    -------
    filtered_pos    : (M, 3)
    filtered_labels : (M,)
    keep_indices    : (M,) indices into original N
    scores          : (N,) joint noise score for every input point
    """
    N      = pos.size(0)
    device = pos.device

    # ------------------------------------------------------------------
    # Step 0: spatial kNN indices (used by signals 1, 2, 4)
    # ------------------------------------------------------------------
    if N <= k:
        raise ValueError(f"Filtering requires more than k={k} points; received {N}.")
    # torch_geometric delegates to torch-cluster's chunked kNN routine and
    # avoids allocating the quadratic N×N matrix used by torch.cdist.
    from torch_geometric.nn import knn_graph as pyg_knn
    sp_edge = pyg_knn(pos, k=k)
    spatial_knn_idx = edge_index_to_knn_idx(sp_edge, N, k)

    # ------------------------------------------------------------------
    # Step 1 (alpha): scatter index from SPATIAL covariance
    # Uses the masked version because the cloud is still raw/unfiltered —
    # noisy neighbours corrupt covariance estimates just as during training.
    # ------------------------------------------------------------------
    evals_sp, _, C_sp = build_local_covariance_masked(
        pos, spatial_knn_idx, scatter_threshold=scatter_threshold, eps=eps
    )
    scatter = compute_scatter(evals_sp, eps=eps)                       # (N,)

    # Hard pre-gate: immediately discard clear volumetric blobs.
    # This mirrors the scatter_threshold used in training so the
    # hard boundary is semantically consistent with L_scatter.
    pre_gate = scatter <= scatter_threshold                             # (N,) bool

    # ------------------------------------------------------------------
    # Step 2 (beta): eigenvalue residual Z-score from SPATIAL kNN
    # ------------------------------------------------------------------
    neigh_evals = evals_sp[spatial_knn_idx]                            # (N, k, 3)
    pred_evals  = neigh_evals.mean(dim=1)                              # (N, 3)
    residual    = torch.linalg.norm(evals_sp - pred_evals, dim=1)     # (N,)
    med = residual.median()
    mad = (residual - med).abs().median() + eps
    residual_z  = (residual - med).abs() / mad                        # (N,)

    # ------------------------------------------------------------------
    # Step 3 (gamma): Frobenius covariance distance on DYNAMIC graph
    # ------------------------------------------------------------------
    # L_cov trained the model to keep only covariance-coherent pairs in
    # the dynamic graph. High ||C_i − C_j||_F on a surviving dynamic
    # edge = the model couldn't separate this pair = likely contested noise.
    src_d, dst_d = dynamic_edge_index.long()
    cov_frob     = (C_sp[src_d] - C_sp[dst_d]).pow(2).sum(dim=(-2, -1))  # (E,)

    # Accumulate per source node — mean Frobenius across dynamic neighbours
    cov_frob_node  = torch.zeros(N, device=device)
    cov_frob_count = torch.zeros(N, device=device)
    cov_frob_node.scatter_add_(0, src_d, cov_frob)
    cov_frob_count.scatter_add_(0, src_d, torch.ones(src_d.size(0), device=device))
    cov_frob_node = cov_frob_node / (cov_frob_count + eps)             # (N,)

    # ------------------------------------------------------------------
    # Step 4 (delta): feature incoherence with SPATIAL neighbours
    # ------------------------------------------------------------------
    # L_scatter pushed noise points away from their spatial neighbours
    # in feature space. 1 − cosine_similarity is the noise signal.
    z = F.normalize(features, p=2, dim=1)                              # (N, F)
    # Build spatial edge list from knn_idx
    src_s = torch.arange(N, device=device).unsqueeze(1).expand(N, k).reshape(-1)
    dst_s = spatial_knn_idx.reshape(-1)                                # (N*k,)
    feat_sim = (z[src_s] * z[dst_s]).sum(dim=1)                        # (N*k,)

    feat_coh_node  = torch.zeros(N, device=device)
    feat_coh_count = torch.zeros(N, device=device)
    feat_coh_node.scatter_add_(0, src_s, feat_sim)
    feat_coh_count.scatter_add_(0, src_s, torch.ones(src_s.size(0), device=device))
    feat_coh_node  = feat_coh_node / (feat_coh_count + eps)            # (N,) in [-1,1]
    feat_incohere  = 1.0 - feat_coh_node                               # (N,) high = noise

    # ------------------------------------------------------------------
    # Joint score: normalise each signal to [0,1], weighted sum
    # ------------------------------------------------------------------
    def _norm01(x: torch.Tensor) -> torch.Tensor:
        x = x - x.min()
        rng = x.max()
        return x / (rng + eps)

    score = (alpha * _norm01(scatter)
           + beta  * _norm01(residual_z)
           + gamma * _norm01(cov_frob_node)
           + delta * _norm01(feat_incohere))                           # (N,)

    # Among pre-gated candidates, compute a robust Z-score on the joint score
    # and keep points whose score is not an outlier (robust_z <= score_tau).
    # This is data-adaptive: the cut follows the score distribution rather
    # than discarding a fixed fraction.
    score_robust_z = torch.zeros(N, device=device)
    cand_scores = score[pre_gate]                                        # (C,)
    if cand_scores.numel() > 0:
        med_s = cand_scores.median()
        mad_s = (cand_scores - med_s).abs().median() + eps
        score_robust_z[pre_gate] = (cand_scores - med_s).abs() / mad_s

    # Hard-excluded points get infinite robust-z so they never pass
    score_robust_z[~pre_gate] = float('inf')
    keep_mask = (pre_gate) & (score_robust_z <= score_tau)

    filtered_pos    = pos[keep_mask]
    filtered_labels = labels[keep_mask] if labels is not None else None
    keep_indices    = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)

    if verbose:
        kept       = int(keep_mask.sum().item())
        n_gated    = int((~pre_gate).sum().item())
        n_cands    = int(pre_gate.sum().item())
        n_rejected = n_cands - kept
        print(
            f"[JointFilter] {N} pts → {n_gated} hard-gated (scatter>{scatter_threshold:.2f}) "
            f"→ {n_rejected} score-rejected (robust_z>{score_tau}) "
            f"→ {kept} kept | "
            f"scatter μ={scatter.mean():.3f}  "
            f"cov_frob μ={cov_frob_node.mean():.4f}  "
            f"feat_coh μ={feat_coh_node.mean():.3f}"
        )

    return filtered_pos, filtered_labels, keep_indices, score


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_filtered_point_cloud_h5(filtered_pos, filtered_labels,
                                  output_path, num_points=1024):
    """
    Save filtered point clouds and labels to HDF5 (DCP/DGCNN format).

    Points that don't fill a complete cloud are dropped; a warning is
    printed so the caller is aware rather than silently losing data.
    """
    pos_np    = filtered_pos.cpu().numpy()
    labels_np = filtered_labels.cpu().numpy()

    total       = len(pos_np)
    num_samples = total // num_points
    remainder   = total % num_points

    if remainder != 0:
        print(
            f"[save_filtered_point_cloud_h5] Warning: {remainder}/{total} "
            f"points do not fill a complete cloud and will be dropped. "
            f"Consider adjusting num_points or your filtering threshold."
        )

    reshaped_pos    = pos_np[:num_samples * num_points].reshape(
                          num_samples, num_points, 3)
    reshaped_labels = labels_np[:num_samples * num_points].reshape(
                          num_samples, num_points)

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('data',  data=reshaped_pos.astype('float32'))
        f.create_dataset('label', data=reshaped_labels[:, 0].astype('int64'))

    print(f"[save_filtered_point_cloud_h5] Saved {num_samples} clouds "
          f"→ {output_path}")
