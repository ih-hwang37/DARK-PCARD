"""Create DARK-ready CSV point clouds with a pretrained PCARD encoder."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data

from pcard.data import CATMAUS
from pcard.filtering import filter_point_cloud_joint
from pcard.model import DGCNNWithKNN


LABEL_TO_BONE = {0: "fem", 1: "pat", 2: "tib", 3: "other"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter CSV point clouds with the pretrained PCARD encoder."
    )
    parser.add_argument("--input", required=True, help="Private input CSV directory.")
    parser.add_argument("--output", required=True, help="Generated output directory.")
    parser.add_argument(
        "--checkpoint", default="weights/pcard_pretrained.pth",
        help="PCARD state-dict checkpoint.",
    )
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--score-tau", type=float, default=2.0)
    parser.add_argument("--scatter-threshold", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument(
        "--max-points", type=int, default=16384,
        help=(
            "Maximum points processed per subject/position. Larger groups are "
            "deterministically subsampled to control kNN memory use."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_state_dict(path: Path, device):
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 2.0
        state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError("Checkpoint must contain a PyTorch state dictionary.")
    return state


def select_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    use_cuda = requested == "cuda" or (
        requested == "auto" and torch.cuda.is_available()
    )
    return torch.device("cuda" if use_cuda else "cpu")


def deterministic_subsample(pos, labels, max_points, seed):
    if max_points <= 0 or pos.size(0) <= max_points:
        return pos, labels
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(pos.size(0), generator=generator)[:max_points]
    return pos[indices], labels[indices]


@torch.no_grad()
def forward_group(model, pos, labels, device):
    mean = pos.mean(dim=0)
    centred = pos - mean
    scale = centred.norm(dim=1).max().clamp(min=1e-6)
    normalised = centred / scale

    normalised = normalised.to(device)
    labels = labels.to(device)
    batch = torch.zeros(normalised.size(0), dtype=torch.long, device=device)
    features, graphs = model(Data(pos=normalised, y=labels, batch=batch))
    return normalised, labels, features, graphs[-1], mean, scale


def run(args):
    input_root = Path(args.input).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_root}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if output_root == input_root or input_root in output_root.parents:
        raise ValueError("Output must not be inside the private input directory.")

    device = select_device(args.device)
    model = DGCNNWithKNN(k=args.k, feature_dim=args.feature_dim).to(device)
    model.load_state_dict(load_state_dict(checkpoint, device), strict=True)
    model.eval()
    print(f"Using device: {device}")
    print(f"Checkpoint SHA-256: {file_sha256(checkpoint)}")

    dataset = CATMAUS(
        root=str(input_root),
        monte_carlo=False,
        iterations=1,
        num_points=min(args.max_points, 4096),
        cache_dir=str(output_root / ".pt_cache"),
    )
    output_root.mkdir(parents=True, exist_ok=True)

    total_in = 0
    total_kept = 0
    for group_index, ((subject, position), files) in enumerate(dataset.groups):
        pos, labels = dataset._load_group(subject, position, files)
        original_count = pos.size(0)
        pos, labels = deterministic_subsample(
            pos, labels, args.max_points, args.seed + group_index
        )
        if pos.size(0) <= args.k:
            print(f"Skipping {subject}/{position}: only {pos.size(0)} points.")
            continue

        normalised, labels_device, features, dynamic_edges, mean, scale = (
            forward_group(model, pos, labels, device)
        )
        filtered, filtered_labels, _, _ = filter_point_cloud_joint(
            pos=normalised,
            labels=labels_device,
            features=features,
            dynamic_edge_index=dynamic_edges,
            k=args.k,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            delta=args.delta,
            scatter_threshold=args.scatter_threshold,
            score_tau=args.score_tau,
            verbose=True,
        )
        restored = filtered * scale.to(device) + mean.to(device)
        save_dir = output_root / subject
        save_dir.mkdir(parents=True, exist_ok=True)
        for label_id, bone_name in LABEL_TO_BONE.items():
            mask = filtered_labels == label_id
            if not bool(mask.any()):
                continue
            points = restored[mask].cpu().numpy()
            output_path = save_dir / f"{position}_{bone_name}.csv"
            pd.DataFrame(points, columns=["x", "y", "z"]).to_csv(
                output_path, index=False
            )

        total_in += pos.size(0)
        total_kept += filtered.size(0)
        sampling_note = (
            f" (sampled from {original_count})" if original_count != pos.size(0) else ""
        )
        print(
            f"Saved {subject}/{position}: {filtered.size(0)}/{pos.size(0)} "
            f"points{sampling_note}."
        )

    print(
        f"Finished: {total_kept}/{total_in} processed points kept in {output_root}."
    )


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
