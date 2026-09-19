import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.data import Data
from torch_geometric.nn import EdgeConv, knn_graph
from utils.util import quat2mat

# Part of the code is referred from: http://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(
        query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn


def nearest_neighbor(src, dst):
    # src, dst (num_dims, num_points)
    inner = -2 * torch.matmul(src.transpose(1, 0).contiguous(), dst)
    distances = -torch.sum(src ** 2, dim=0, keepdim=True).transpose(1, 0).contiguous() - inner - torch.sum(dst ** 2,
                                                                                                           dim=0,
                                                                                                           keepdim=True)
    distances, indices = distances.topk(k=1, dim=-1)
    return distances, indices

def knn(x, k):
    """
    x: [B, D, N] tensor (channels first)
    returns: [B, N, k] tensor of indices
    """
    B, D, N = x.size()
    x = x.transpose(2, 1).contiguous()  # [B, N, D]

    k = min(k, N)
    x_i = x.unsqueeze(2)  # [B, N, 1, D]
    x_j = x.unsqueeze(1)  # [B, 1, N, D]
    pairwise_distance = torch.sum((x_i - x_j) ** 2, dim=3)  # [B, N, N]

    idx = pairwise_distance.topk(k=k, dim=-1, largest=False)[1]  # [B, N, k]
    return idx


def get_graph_feature(x, k=20):
    """
    x: [B, D, N] input
    returns: [B, 2D, N, k] graph features
    """
    B, D, N = x.size()
    k = min(k, N)
    idx = knn(x, k=k)  # [B, N, k]

    device = x.device
    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()  # [B, N, D]
    feature = x.view(B * N, -1)[idx, :]  # [B*N*k, D]
    feature = feature.view(B, N, k, D)
    x = x.view(B, N, 1, D).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)  # [B, 2D, N, k]
    return feature

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.generator(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask))


class Generator(nn.Module):
    def __init__(self, emb_dims):
        super(Generator, self).__init__()
        self.nn = nn.Sequential(nn.Linear(emb_dims, emb_dims // 2),
                                nn.BatchNorm1d(emb_dims // 2),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 2, emb_dims // 4),
                                nn.BatchNorm1d(emb_dims // 4),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 4, emb_dims // 8),
                                nn.BatchNorm1d(emb_dims // 8),
                                nn.ReLU())
        self.proj_rot = nn.Linear(emb_dims // 8, 4)
        self.proj_trans = nn.Linear(emb_dims // 8, 3)

    def forward(self, x):
        x = self.nn(x.max(dim=1)[0])
        rotation = self.proj_rot(x)
        translation = self.proj_trans(x)
        rotation = rotation / torch.norm(rotation, p=2, dim=1, keepdim=True)
        return rotation, translation


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=None):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = None

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.norm = nn.Sequential()  # nn.BatchNorm1d(d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = None

    def forward(self, x):
        return self.w_2(self.norm(F.relu(self.w_1(x)).transpose(2, 1).contiguous()).transpose(2, 1).contiguous())


class PCARDEncoder(nn.Module):
    """
    Drop-in replacement for DARK's DGCNN encoder using the pre-trained
    dynamic-graph model from PCARD training (DGCNNWithKNN).

    Format bridge
    -------------
    DARK input : (B, 3, N)           — batched point clouds, channels-first
    PyG  format: pos=(B*N, 3), batch=(B*N,)
    DARK output: (B, feature_dim, N) — expected by Transformer and MLPHead

    The architecture is a verbatim copy of PCARD/models/dgcnn.py so that
    state-dict keys match exactly and pre-trained weights load strict=True.
    """

    def __init__(self, k: int = 20, feature_dim: int = 128):
        super(PCARDEncoder, self).__init__()
        self.k = k
        self.feature_dim = feature_dim

        self.conv1 = EdgeConv(self._mlp(6,   64),  aggr='max')
        self.conv2 = EdgeConv(self._mlp(128, 64),  aggr='max')
        self.conv3 = EdgeConv(self._mlp(128, 64),  aggr='max')
        self.conv4 = EdgeConv(self._mlp(128, 128), aggr='max')

        # No final ReLU — features are compared via cosine similarity
        self.mlp1 = nn.Sequential(nn.Linear(320, 256), nn.ReLU())
        self.mlp2 = nn.Linear(256, feature_dim)

    def _mlp(self, in_dim: int, out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim,  out_dim), nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, N) — DARK-format point cloud

        Returns:
            (B, feature_dim, N) — per-point embeddings, DARK-format
        """
        B, _, N = x.size()
        device  = x.device

        # Convert to PyG flat format
        pos       = x.permute(0, 2, 1).reshape(-1, 3)
        batch_idx = torch.arange(B, device=device).repeat_interleave(N)

        # Dynamic graph forward — mirrors DGCNNWithKNN.forward
        edge1 = knn_graph(pos,  self.k, batch=batch_idx)
        x1    = self.conv1(pos, edge1)

        edge2 = knn_graph(x1,  self.k, batch=batch_idx)
        x2    = self.conv2(x1, edge2)

        edge3 = knn_graph(x2,  self.k, batch=batch_idx)
        x3    = self.conv3(x2, edge3)

        edge4 = knn_graph(x3,  self.k, batch=batch_idx)
        x4    = self.conv4(x3, edge4)

        feats = torch.cat([x1, x2, x3, x4], dim=1)       # (B*N, 320)
        feats = self.mlp2(self.mlp1(feats))                # (B*N, feature_dim)

        # Reshape back to DARK (B, feature_dim, N)
        return feats.view(B, N, self.feature_dim).permute(0, 2, 1)


class DGCNN(nn.Module):
    def __init__(self, emb_dims=512):
        super(DGCNN, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(512, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(emb_dims)

    def forward(self, x):
        batch_size, num_dims, num_points = x.size()
        x = get_graph_feature(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn2(self.conv2(x)))
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn4(self.conv4(x)))
        x4 = x.max(dim=-1, keepdim=True)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = F.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)
        return x


class MLPHead(nn.Module):
    def __init__(self, args):
        super(MLPHead, self).__init__()
        emb_dims = args.emb_dims
        self.emb_dims = emb_dims
        self.nn = nn.Sequential(nn.Linear(emb_dims * 2, emb_dims // 2),
                                nn.BatchNorm1d(emb_dims // 2),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 2, emb_dims // 4),
                                nn.BatchNorm1d(emb_dims // 4),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 4, emb_dims // 8),
                                nn.BatchNorm1d(emb_dims // 8),
                                nn.ReLU())
        self.proj_rot = nn.Linear(emb_dims // 8, 4)
        self.proj_trans = nn.Linear(emb_dims // 8, 3)

    def forward(self, src_embedding, tgt_embedding, src=None, tgt=None,
                src_weights=None, tgt_weights=None):
        """
        Args:
            src_embedding: (B, D, N)
            tgt_embedding: (B, D, N)
            src, tgt:      unused (kept for signature parity with SVDHead)
            src_weights:   (B, 1, N) scatter-based surface confidence for src
            tgt_weights:   (B, 1, N) scatter-based surface confidence for tgt

        When weights are provided, the max-pool is done SEPARATELY for src
        and tgt embeddings using their own respective weights, then the two
        pooled vectors are concatenated.  This is important: src_weights[i]
        reflects whether source point i is a surface point — it says nothing
        about target point i (which is a different physical location after
        rotation), so applying src_weights to tgt features would be wrong.
        """
        if src_weights is not None and tgt_weights is not None:
            # Separate scatter-weighted max-pool for each cloud
            src_pooled = (src_embedding * src_weights).max(dim=-1)[0]  # (B, D)
            tgt_pooled = (tgt_embedding * tgt_weights).max(dim=-1)[0]  # (B, D)
            embedding  = torch.cat((src_pooled, tgt_pooled), dim=1)    # (B, 2D)
        else:
            embedding = torch.cat((src_embedding, tgt_embedding), dim=1).max(dim=-1)[0]

        embedding   = self.nn(embedding)
        rotation    = self.proj_rot(embedding)
        rotation = rotation / torch.norm(
            rotation, p=2, dim=1, keepdim=True
        ).clamp_min(1e-8)
        translation = self.proj_trans(embedding)
        return quat2mat(rotation), translation


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, *input):
        return input


class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.emb_dims = args.emb_dims
        self.N = args.n_blocks
        self.dropout = args.dropout
        self.ff_dims = args.ff_dims
        self.n_heads = args.n_heads
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.n_heads, self.emb_dims)
        ff = PositionwiseFeedForward(self.emb_dims, self.ff_dims, self.dropout)
        self.model = EncoderDecoder(Encoder(EncoderLayer(self.emb_dims, c(attn), c(ff), self.dropout), self.N),
                                    Decoder(DecoderLayer(self.emb_dims, c(attn), c(
                                        attn), c(ff), self.dropout), self.N),
                                    nn.Sequential(),
                                    nn.Sequential(),
                                    nn.Sequential())

    def forward(self, *input):
        src = input[0]
        tgt = input[1]

        src = src.transpose(2, 1).contiguous()
        tgt = tgt.transpose(2, 1).contiguous()
        tgt_embedding = self.model(
            src, tgt, None, None).transpose(2, 1).contiguous()
        src_embedding = self.model(
            tgt, src, None, None).transpose(2, 1).contiguous()
        return src_embedding, tgt_embedding


@torch.no_grad()
def scatter_surface_weights(pos: torch.Tensor,
                            k: int = 10,
                            scatter_threshold: float = 0.15,
                            eps: float = 1e-6) -> torch.Tensor:
    """
    Per-point surface confidence weights from the local 3D scatter index.

    scatter_i = λ_min / (λ₁ + λ₂ + λ₃)
      ≈ 0   for planar surface points (smallest eigenvalue near 0)
      ≈ 1/3 for isotropic noise (all eigenvalues equal)

    weight_i = exp(-scatter_i / scatter_threshold):
      → 1.0 for a perfect planar surface point
      → near 0 for an isotropic noise point

    This is the same geometric primitive used by PCARD's
    scatter_separation_loss during encoder pre-training.  By applying it
    at registration time we ensure that the rotation estimate is dominated
    by surface-to-surface correspondences and is robust to scanner noise.

    Args:
        pos:               (B, 3, N) — batched point clouds, channels-first
        k:                 neighbourhood size for local covariance (10 is
                           enough to estimate planarity)
        scatter_threshold: scatter value that maps to weight ≈ 0.37

    Returns:
        weights: (B, 1, N) in (0, 1], mean-normalised per sample so that
                 replacing uniform weights with these preserves the scale of H.
    """
    B, _, N = pos.shape
    device = pos.device
    k = min(k, N - 1)

    pts = pos.permute(0, 2, 1)                          # (B, N, 3)

    # Pairwise squared distances — (B, N, N)
    sq_dist = torch.cdist(pts, pts).pow(2)

    # k nearest neighbours, excluding self (self-distance = 0 → first col)
    _, nn_idx = sq_dist.topk(k + 1, dim=-1, largest=False)
    nn_idx = nn_idx[:, :, 1:]                           # (B, N, k)

    # Gather neighbour positions then centre around the query point
    B_idx   = torch.arange(B, device=device)[:, None, None].expand(B, N, k)
    nn_pts  = pts[B_idx, nn_idx]                        # (B, N, k, 3)
    centred = nn_pts - pts.unsqueeze(2)                 # (B, N, k, 3)

    # Local 3×3 covariance matrices: C_i = (1/k) Σ_j δ_ij δ_ij^T
    c_flat = centred.reshape(B * N, k, 3)
    C = torch.bmm(c_flat.transpose(1, 2), c_flat) / k  # (B*N, 3, 3)

    # Eigenvalues in ascending order (eigvalsh is numerically stable)
    try:
        evals = torch.linalg.eigvalsh(C).clamp(min=0.0)   # (B*N, 3)
    except RuntimeError:
        return torch.ones(B, 1, N, device=device)

    # scatter = λ_min / Σλ  (ascending order: evals[:,0] is smallest)
    scatter = evals[:, 0] / (evals.sum(dim=1) + eps)       # (B*N,)
    w = torch.exp(-scatter / scatter_threshold).view(B, N) # (B, N)

    # Mean-normalise per sample: keeps the absolute scale of H unchanged
    w = w / (w.mean(dim=1, keepdim=True) + eps)
    return w.unsqueeze(1)                                   # (B, 1, N)


class SVDHead(nn.Module):
    def __init__(self, args):
        super(SVDHead, self).__init__()
        self.emb_dims = args.emb_dims
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1

    def forward(self, src_embedding, tgt_embedding, src, tgt,
                src_weights=None, tgt_weights=None):
        """
        Args:
            src_embedding: (B, D, N)
            tgt_embedding: (B, D, N)
            src:           (B, 3, N)  source point cloud
            tgt:           (B, 3, N)  target point cloud
            src_weights:   (B, 1, N) optional per-point surface confidence
                           weights (from scatter_surface_weights).  When
                           provided, the weighted centroid and weighted
                           cross-covariance are used so that high-scatter
                           noise points contribute less to the rotation
                           estimate.  None → standard unweighted SVD.
        """
        batch_size = src.size(0)
        d_k = src_embedding.size(1)

        # Soft correspondence scores: (B, N_src, N_tgt)
        scores = torch.softmax(
            torch.matmul(src_embedding.transpose(2, 1).contiguous(),
                         tgt_embedding) / math.sqrt(d_k),
            dim=2)

        # Soft-assigned target position for each source point
        src_corr = torch.matmul(tgt, scores.transpose(2, 1).contiguous())  # (B, 3, N)

        # --- Weighted (or standard) centroid + cross-covariance ---
        if src_weights is not None:
            w     = src_weights                                   # (B, 1, N)
            w_sum = w.sum(dim=2, keepdim=True).clamp(min=1e-8)
            src_mean      = (src      * w).sum(dim=2, keepdim=True) / w_sum
            src_corr_mean = (src_corr * w).sum(dim=2, keepdim=True) / w_sum

            src_centred      = src      - src_mean
            src_corr_centred = src_corr - src_corr_mean

            # Weighted cross-covariance H
            w_norm = w / w_sum                                    # (B, 1, N)
            H = torch.matmul(src_centred * w_norm,
                             src_corr_centred.transpose(2, 1).contiguous())
        else:
            src_mean      = src.mean(dim=2, keepdim=True)
            src_corr_mean = src_corr.mean(dim=2, keepdim=True)

            src_centred      = src      - src_mean
            src_corr_centred = src_corr - src_corr_mean

            H = torch.matmul(src_centred,
                             src_corr_centred.transpose(2, 1).contiguous())

        # --- SVD ---
        R = []
        for i in range(batch_size):
            try:
                u, s, vh = torch.linalg.svd(H[i].float(), full_matrices=False)
                v = vh.transpose(-2, -1)
                r = torch.matmul(v, u.transpose(-2, -1))
                if torch.det(r) < 0:
                    r = torch.matmul(
                        torch.matmul(v, self.reflect),
                        u.transpose(-2, -1),
                    )
            except RuntimeError:
                r = torch.eye(3, device=H.device)
            R.append(r)
        R = torch.stack(R, dim=0)

        t = torch.matmul(-R, src_mean) + src_corr_mean
        return R, t.view(batch_size, 3)


class DCP(nn.Module):
    def __init__(self, args):
        super(DCP, self).__init__()
        self.emb_dims = args.emb_dims
        self.cycle = args.cycle
        if args.emb_nn == 'dgcnn':
            self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        elif args.emb_nn == 'pcard':
            # Pre-trained dynamic-graph encoder from PCARD.
            # emb_dims must equal PCARDEncoder's feature_dim (128).
            assert self.emb_dims == 128, (
                "When using emb_nn='pcard', set --emb_dims 128 to match "
                "the PCARD checkpoint's feature dimension."
            )
            self.emb_nn = PCARDEncoder(k=20, feature_dim=128)
        else:
            raise Exception('Not implemented')

        if args.pointer == 'identity':
            self.pointer = Identity()
        elif args.pointer == 'transformer':
            self.pointer = Transformer(args=args)
        else:
            raise Exception("Not implemented")

        if args.head == 'mlp':
            self.head = MLPHead(args=args)
        elif args.head == 'svd':
            self.head = SVDHead(args=args)
        else:
            raise Exception('Not implemented')

    def forward(self, src, tgt):
        """
        Args:
            src: (B, 3, N) source point cloud
            tgt: (B, 3, N) target point cloud
        """
        src_embedding = self.emb_nn(src)
        tgt_embedding = self.emb_nn(tgt)

        src_embedding_p, tgt_embedding_p = self.pointer(
            src_embedding, tgt_embedding)

        src_embedding = src_embedding + src_embedding_p
        tgt_embedding = tgt_embedding + tgt_embedding_p

        # --- Scatter-based surface weights (always computed) ---
        # scatter_surface_weights operates purely on raw 3D positions via
        # local covariance eigenvalues — it is encoder-agnostic. Surface
        # points (low λ_min/Σλ) get high weight; isotropic noise points
        # (high scatter) get low weight. This guides the MLP max-pool and
        # SVD cross-covariance to focus on reliable surface geometry
        # regardless of which encoder produced the features.
        src_weights = scatter_surface_weights(src)
        tgt_weights = scatter_surface_weights(tgt)

        rotation_ab, translation_ab = self.head(
            src_embedding, tgt_embedding, src, tgt,
            src_weights=src_weights, tgt_weights=tgt_weights)

        if self.cycle:
            rotation_ba, translation_ba = self.head(
                tgt_embedding, src_embedding, tgt, src,
                src_weights=tgt_weights, tgt_weights=src_weights)
        else:
            rotation_ba = rotation_ab.transpose(2, 1).contiguous()
            translation_ba = -torch.matmul(
                rotation_ba, translation_ab.unsqueeze(2)).squeeze(2)

        return rotation_ab, translation_ab, rotation_ba, translation_ba
