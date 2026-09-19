
from __future__ import print_function
import os
import shutil
import gc
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from utils.data import CATMAUS, CATMAUSFromCSV
from models.model import DCP
from utils.util import transform_point_cloud, npmat2euler
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from scipy.spatial.transform import Rotation as ScipyRotation

# Part of the code is referred from: https://github.com/floodsung/LearningToCompare_FSL

# Runs Pytorch on CUDA GPU if present
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SCRIPT_DIR = Path(__file__).resolve().parent


def load_state_dict_safely(path, map_location):
    """Load tensor-only checkpoints where supported by the PyTorch version."""
    try:
        state = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # PyTorch < 2.0
        state = torch.load(path, map_location=map_location)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    return state

class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def _init_(args):
    # Make checkpoints directory
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    # Make subdirectory within checkpoints for experiment name
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    # Make another subdirectory within checkpoints for model name
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')

    shutil.copyfile(SCRIPT_DIR / 'main.py', f"checkpoints/{args.exp_name}/main.py.backup")
    shutil.copyfile(SCRIPT_DIR / 'models/model.py',
                    f"checkpoints/{args.exp_name}/model.py.backup")
    shutil.copyfile(SCRIPT_DIR / 'utils/data.py',
                    f"checkpoints/{args.exp_name}/data.py.backup")




def random_rotate_batch(src, device):
    """Apply independent random rotations + translations to each point cloud in a batch.

    Unlike repeat_interleave (which duplicates the same rotation N times), this
    function generates a genuinely fresh random rotation for each copy, giving the
    model true angular diversity within a single training step.

    Args:
        src    : (B, 3, N) tensor already on `device` — source point clouds.
        device : torch device.

    Returns:
        target   : (B, 3, N) freshly rotated targets.
        R_ab     : (B, 3, 3) rotation matrices  A→B.
        t_ab     : (B, 3)    translation vectors A→B.
        R_ba     : (B, 3, 3) rotation matrices  B→A.
        t_ba     : (B, 3)    translation vectors B→A.
        euler_ab : (B, 3) numpy array  [anglez, angley, anglex] in radians.
        euler_ba : (B, 3) numpy array.
    """
    B = src.size(0)
    angles = np.random.uniform(np.pi / 4, 3 * np.pi / 4, size=(B, 3))   # rows: [ax, ay, az]
    R_np = np.stack([
        ScipyRotation.from_euler('zyx', [a[2], a[1], a[0]]).as_matrix()
        for a in angles
    ]).astype(np.float32)                                                  # (B, 3, 3)
    t_np = np.random.uniform(-0.5, 0.5, size=(B, 3)).astype(np.float32)   # (B, 3)

    R_ab = torch.tensor(R_np, device=device)
    t_ab = torch.tensor(t_np, device=device)
    target = torch.bmm(R_ab, src) + t_ab.unsqueeze(2)
    R_ba   = R_ab.transpose(1, 2)
    t_ba   = -torch.bmm(R_ba, t_ab.unsqueeze(2)).squeeze(2)

    euler_ab = np.stack([np.array([a[2], a[1], a[0]]) for a in angles])   # [z,y,x] conv.
    euler_ba = -euler_ab[:, ::-1].copy()
    return target, R_ab, t_ab, R_ba, t_ba, euler_ab, euler_ba


def project_to_so3(R: torch.Tensor) -> torch.Tensor:
    """Projects a batch of 3x3 matrices to the closest valid SO(3) rotations using SVD."""
    U, _, Vt = torch.linalg.svd(R)
    # Enforce right-handed coordinate system
    S = torch.eye(3, device=R.device).unsqueeze(0).repeat(R.shape[0], 1, 1)
    S[:, 2, 2] = torch.det(U @ Vt)
    R_proj = U @ S @ Vt
    return R_proj



class GeodesicLoss(nn.Module):
    r"""Creates a criterion that measures the distance between rotation matrices, which is
    useful for pose estimation problems.
    The distance ranges from 0 to :math:`pi`.
    See: http://www.boris-belousov.net/2016/12/01/quat-dist/#using-rotation-matrices and:
    "Metrics for 3D Rotations: Comparison and Analysis" (https://link.springer.com/article/10.1007/s10851-009-0161-2).

    Both `input` and `target` consist of rotation matrices, i.e., they have to be Tensors
    of size :math:`(minibatch, 3, 3)`.

    The loss can be described as:

    .. math::
        \text{loss}(R_{S}, R_{T}) = \arccos\left(\frac{\text{tr} (R_{S} R_{T}^{T}) - 1}{2}\right)

    Args:
        eps (float, optional): term to improve numerical stability (default: 1e-7). See:
            https://github.com/pytorch/pytorch/issues/8069.

        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Default: ``'mean'``

    Shape:
        - Input: Shape :math:`(N, 3, 3)`.
        - Target: Shape :math:`(N, 3, 3)`.
        - Output: If :attr:`reduction` is ``'none'``, then :math:`(N)`. Otherwise, scalar.
    """

    def __init__(self, eps: float = 1e-7, reduction: str = "mean") -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        R_diffs = input @ target.permute(0, 2, 1)
        traces = R_diffs.diagonal(dim1=-2, dim2=-1).sum(-1)
        value = (traces - 1) / 2
        value = torch.clamp(value, min=-1 + self.eps, max=1 - self.eps)

        # Debugging NaNs
        if torch.any(torch.isnan(value)):
            print("⚠️ NaN detected in geodesic loss input. Values:", value)
            value[torch.isnan(value)] = 0.0

        dists = torch.acos(value)

        if self.reduction == "none":
            return dists
        elif self.reduction == "mean":
            return dists.mean()
        elif self.reduction == "sum":
            return dists.sum()

geodesic_loss_fn = GeodesicLoss().to(device)


class EarlyStopping:
    """Stop training when the monitored metric stops improving.

    Args:
        patience  (int)  : How many epochs to wait after the last improvement.
        verbose   (bool) : Print a message each time the counter increments.
        save_path (str)  : If given, save the best model state-dict here.
    """

    def __init__(self, patience: int = 10, verbose: bool = True, save_path: str = None):
        self.patience = patience
        self.verbose = verbose
        self.save_path = save_path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module):
        if self.best_loss is None or val_loss < self.best_loss:
            if self.verbose and self.best_loss is not None:
                print(f"  [EarlyStopping] Val loss improved ({self.best_loss:.6f} → {val_loss:.6f}). Saving model.")
            self.best_loss = val_loss
            self.counter = 0
            if self.save_path is not None:
                os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save(state, self.save_path)
        else:
            self.counter += 1
            if self.verbose:
                print(f"  [EarlyStopping] No improvement for {self.counter}/{self.patience} epochs.")
            if self.counter >= self.patience:
                self.early_stop = True


def test_one_epoch(args, net, test_loader, geodesic_loss_fn):
    net.eval()
    total_loss = 0
    total_cycle_loss = 0
    num_examples = 0


    rotations_ab = []
    translations_ab = []
    rotations_ab_pred = []
    translations_ab_pred = []

    rotations_ba = []
    translations_ba = []
    rotations_ba_pred = []
    translations_ba_pred = []

    eulers_ab = []
    eulers_ba = []

    for src, target, rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab, euler_ba in tqdm(test_loader):
        src = src.to(device)
        target = target.to(device)

        with torch.no_grad():
            rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred = net(src, target)

            # Now safe to project
            rotation_ab_pred = project_to_so3(rotation_ab_pred)
            rotation_ba_pred = project_to_so3(rotation_ba_pred)

        # (continue with loss calc, logging, etc.)

        rotation_ab = rotation_ab.to(device)
        translation_ab = translation_ab.to(device)
        rotation_ba = rotation_ba.to(device)
        translation_ba = translation_ba.to(device)

        batch_size = src.size(0)
        num_examples += batch_size

        translation_ab_pred = torch.clamp(translation_ab_pred, -10.0, 10.0)
        translation_ba_pred = torch.clamp(translation_ba_pred, -10.0, 10.0)

        # Save rotation and translation values
        rotations_ab.append(rotation_ab.detach().cpu().numpy())
        translations_ab.append(translation_ab.detach().cpu().numpy())
        rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
        translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())

        rotations_ba.append(rotation_ba.detach().cpu().numpy())
        translations_ba.append(translation_ba.detach().cpu().numpy())
        rotations_ba_pred.append(rotation_ba_pred.detach().cpu().numpy())
        translations_ba_pred.append(translation_ba_pred.detach().cpu().numpy())

        eulers_ab.append(euler_ab.numpy())
        eulers_ba.append(euler_ba.numpy())

        # Identity matrix
        identity = torch.eye(3).to(device).unsqueeze(0).repeat(batch_size, 1, 1)


        #rotation_loss
        rotation_loss_ab = geodesic_loss_fn(rotation_ab_pred, rotation_ab)
        rotation_loss_ba = geodesic_loss_fn(rotation_ba_pred, rotation_ba)
        rotation_loss = rotation_loss_ab + rotation_loss_ba


        # Symmetric translation and orthogonality terms.
        translation_loss = (
            F.smooth_l1_loss(translation_ab_pred, translation_ab)
            + F.smooth_l1_loss(translation_ba_pred, translation_ba)
        )
        orthogonality_loss = (
            ((rotation_ab_pred.transpose(1, 2) @ rotation_ab_pred - identity) ** 2).mean()
            + ((rotation_ba_pred.transpose(1, 2) @ rotation_ba_pred - identity) ** 2).mean()
        )
        loss = (
            rotation_loss
            + args.translation_weight * translation_loss
            + args.orthogonality_weight * orthogonality_loss
        )

        # If cycle consistency is enabled, apply cycle loss
        if args.cycle:
            cycle_rotation_loss = geodesic_loss_fn(torch.matmul(rotation_ba_pred, rotation_ab_pred), identity)


            cycle_translation_loss = torch.mean((torch.matmul(rotation_ba_pred,
                                            translation_ab_pred.view(batch_size, 3, 1)).view(batch_size, 3)
                                     + translation_ba_pred) ** 2, dim=[0, 1])

            # Total cycle loss
            cycle_loss = cycle_rotation_loss + cycle_translation_loss
            loss = loss + args.cycle_weight * cycle_loss
            total_cycle_loss += cycle_loss.item() * batch_size

        total_loss += loss.item() * batch_size

    # Aggregate predictions and ground truth values

    rotations_ab = np.concatenate(rotations_ab, axis=0)
    translations_ab = np.concatenate(translations_ab, axis=0)
    rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
    translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)

    rotations_ba = np.concatenate(rotations_ba, axis=0)
    translations_ba = np.concatenate(translations_ba, axis=0)
    rotations_ba_pred = np.concatenate(rotations_ba_pred, axis=0)
    translations_ba_pred = np.concatenate(translations_ba_pred, axis=0)

    eulers_ab = np.concatenate(eulers_ab, axis=0)
    eulers_ba = np.concatenate(eulers_ba, axis=0)

    return total_loss / num_examples, total_cycle_loss / num_examples, \
        rotation_loss_ab, rotation_loss_ba, \
        rotations_ab, translations_ab, rotations_ab_pred, translations_ab_pred, \
        rotations_ba, translations_ba, rotations_ba_pred, translations_ba_pred, \
        eulers_ab, eulers_ba



def train_one_epoch(args, net, train_loader, opt, geodesic_loss_fn):
    net.train()

    mse_ab = 0
    mae_ab = 0
    mse_ba = 0
    mae_ba = 0


    total_loss = 0
    total_cycle_loss = 0
    num_examples = 0
    rotations_ab = []
    translations_ab = []
    rotations_ab_pred = []
    translations_ab_pred = []

    rotations_ba = []
    translations_ba = []
    rotations_ba_pred = []
    translations_ba_pred = []

    eulers_ab = []
    eulers_ba = []

    for src, target, rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab, euler_ba in tqdm(train_loader):
        src = src.to(device)
        target = target.to(device)
        rotation_ab = rotation_ab.to(device)
        translation_ab = translation_ab.to(device)
        rotation_ba = rotation_ba.to(device)
        translation_ba = translation_ba.to(device)

        # --- Multi-angle augmentation ---
        # Each extra view gets a genuinely fresh random rotation applied to the
        # same source cloud, giving the model true angular diversity per step.
        if args.num_angles_per_sample > 1:
            src_all    = [src]
            tgt_all    = [target]
            R_ab_all   = [rotation_ab]
            t_ab_all   = [translation_ab]
            R_ba_all   = [rotation_ba]
            t_ba_all   = [translation_ba]
            eul_ab_all = [euler_ab]          # CPU tensors
            eul_ba_all = [euler_ba]

            for _ in range(args.num_angles_per_sample - 1):
                tgt_new, R_ab_new, t_ab_new, R_ba_new, t_ba_new, eul_ab_np, eul_ba_np = \
                    random_rotate_batch(src, device)
                src_all.append(src)
                tgt_all.append(tgt_new)
                R_ab_all.append(R_ab_new)
                t_ab_all.append(t_ab_new)
                R_ba_all.append(R_ba_new)
                t_ba_all.append(t_ba_new)
                eul_ab_all.append(torch.tensor(eul_ab_np, dtype=torch.float32))
                eul_ba_all.append(torch.tensor(eul_ba_np, dtype=torch.float32))

            src            = torch.cat(src_all,    dim=0)
            target         = torch.cat(tgt_all,    dim=0)
            rotation_ab    = torch.cat(R_ab_all,   dim=0)
            translation_ab = torch.cat(t_ab_all,   dim=0)
            rotation_ba    = torch.cat(R_ba_all,   dim=0)
            translation_ba = torch.cat(t_ba_all,   dim=0)
            euler_ab       = torch.cat(eul_ab_all, dim=0)
            euler_ba       = torch.cat(eul_ba_all, dim=0)

        batch_size = src.size(0)
        identity = torch.eye(3).to(device).unsqueeze(0).repeat(batch_size, 1, 1)

        opt.zero_grad()
        num_examples += batch_size
        rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred = net(
            src, target)

        translation_ab_pred = torch.clamp(translation_ab_pred, -10.0, 10.0)
        translation_ba_pred = torch.clamp(translation_ba_pred, -10.0, 10.0)

        # === Compute geodesic loss (raw predictions — orthogonality loss keeps them near SO(3)) ===
        rotation_loss_ab = geodesic_loss_fn(rotation_ab_pred, rotation_ab)
        rotation_loss_ba = geodesic_loss_fn(rotation_ba_pred, rotation_ba)
        rotation_loss = rotation_loss_ab + rotation_loss_ba

        # Compute translation loss symmetrically in both directions.
        translation_loss = (
            F.smooth_l1_loss(translation_ab_pred, translation_ab)
            + F.smooth_l1_loss(translation_ba_pred, translation_ba)
        )

        # Compute orthogonality loss (keeps raw predictions close to SO(3) during training)
        ortho_loss_ab = ((rotation_ab_pred.transpose(1, 2) @ rotation_ab_pred - identity)**2).mean()
        ortho_loss_ba = ((rotation_ba_pred.transpose(1, 2) @ rotation_ba_pred - identity)**2).mean()
        orthogonality_loss = ortho_loss_ab + ortho_loss_ba

        # --- Total loss ---
        loss = (
            rotation_loss
            + args.translation_weight * translation_loss
            + args.orthogonality_weight * orthogonality_loss
        )

        if args.cycle:
            cycle_rotation_loss = geodesic_loss_fn(
                torch.matmul(rotation_ba_pred, rotation_ab_pred), identity
            )
            cycle_translation_loss = torch.mean(
                (
                    torch.matmul(
                        rotation_ba_pred, translation_ab_pred.unsqueeze(2)
                    ).squeeze(2)
                    + translation_ba_pred
                ) ** 2
            )
            cycle_loss = cycle_rotation_loss + cycle_translation_loss
            loss = loss + args.cycle_weight * cycle_loss
            total_cycle_loss += cycle_loss.item() * batch_size
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
        opt.step()

        total_loss += loss.item() * batch_size

        # Save rotation and translation
        rotations_ab.append(rotation_ab.detach().cpu().numpy())
        translations_ab.append(translation_ab.detach().cpu().numpy())
        rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
        translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
        eulers_ab.append(euler_ab.numpy())

        rotations_ba.append(rotation_ba.detach().cpu().numpy())
        translations_ba.append(translation_ba.detach().cpu().numpy())
        rotations_ba_pred.append(rotation_ba_pred.detach().cpu().numpy())
        translations_ba_pred.append(translation_ba_pred.detach().cpu().numpy())
        eulers_ba.append(euler_ba.numpy())

    rotations_ab = np.concatenate(rotations_ab, axis=0)
    translations_ab = np.concatenate(translations_ab, axis=0)
    rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
    translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)

    rotations_ba = np.concatenate(rotations_ba, axis=0)
    translations_ba = np.concatenate(translations_ba, axis=0)
    rotations_ba_pred = np.concatenate(rotations_ba_pred, axis=0)
    translations_ba_pred = np.concatenate(translations_ba_pred, axis=0)

    eulers_ab = np.concatenate(eulers_ab, axis=0)
    eulers_ba = np.concatenate(eulers_ba, axis=0)

    return total_loss / num_examples, total_cycle_loss / num_examples, \
        rotation_loss_ab.item(), rotation_loss_ba.item(), \
        rotations_ab, translations_ab, rotations_ab_pred, translations_ab_pred, \
        rotations_ba, translations_ba, rotations_ba_pred, translations_ba_pred, \
        eulers_ab, eulers_ba

def test(args, net, test_loader, boardio, textio, geodesic_loss_fn):

    test_loss, test_cycle_loss, \
        rotation_loss_ab, rotation_loss_ba, \
        test_rotations_ab, test_translations_ab, test_rotations_ab_pred, test_translations_ab_pred, \
        test_rotations_ba, test_translations_ba, test_rotations_ba_pred, test_translations_ba_pred, \
        test_eulers_ab, test_eulers_ba = test_one_epoch(args, net, test_loader, geodesic_loss_fn)

    # Compute geodesic distance as rotation error
    test_r_geodesic_ab = geodesic_loss_fn(
        torch.tensor(test_rotations_ab_pred).to(device),
        torch.tensor(test_rotations_ab).to(device)
    ).item()

    test_r_geodesic_ba = geodesic_loss_fn(
        torch.tensor(test_rotations_ba_pred).to(device),
        torch.tensor(test_rotations_ba).to(device)
    ).item()

    test_t_mae_ab = np.mean(np.abs(test_translations_ab - test_translations_ab_pred))
    test_t_mae_ba = np.mean(np.abs(test_translations_ba - test_translations_ba_pred))

    textio.cprint('==FINAL TEST==')
    textio.cprint('A--------->B')
    textio.cprint('EPOCH:: %d, Loss: %f, Cycle Loss: %f, rot_Geodesic: %f, trans_MAE: %f'
                  % (-1, test_loss, test_cycle_loss, test_r_geodesic_ab, test_t_mae_ab))
    textio.cprint('B--------->A')
    textio.cprint('EPOCH:: %d, Loss: %f, rot_Geodesic: %f, trans_MAE: %f'
                  % (-1, test_loss, test_r_geodesic_ba, test_t_mae_ba))

def train(args, net, train_loader, test_loader, boardio, textio, geodesic_loss_fn):
    # Optimizer parameter groups:
    #   --freeze_encoder : only train Transformer + head (encoder excluded)
    #   default          : differential LR — encoder at 0.1× base LR,
    #                      Transformer + head at full base LR
    if args.emb_nn == 'pcard' and not args.freeze_encoder:
        encoder_params = list(net.emb_nn.parameters())
        encoder_ids    = {id(p) for p in encoder_params}
        other_params   = [p for p in net.parameters() if id(p) not in encoder_ids]
        param_groups = [
            {'params': encoder_params, 'lr': args.lr * 0.1},
            {'params': other_params,   'lr': args.lr},
        ]
        print(f"[train] Differential LR — encoder: {args.lr * 0.1:.2e}, rest: {args.lr:.2e}")
    else:
        # Frozen encoder: only optimise params that still require gradients
        trainable = [p for p in net.parameters() if p.requires_grad]
        param_groups = [{'params': trainable, 'lr': args.lr}]
        print(f"[train] Single LR — {args.lr:.2e} ({len(trainable)} param tensors)")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(param_groups, momentum=args.momentum, weight_decay=1e-4)
        # SGD convention: multiply base LR by 10 (matches original DCP paper)
        for pg in opt.param_groups:
            pg['lr'] *= 10
    else:
        print("Use Adam")
        opt = optim.Adam(param_groups, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    best_test_loss = np.inf
    best_test_cycle_loss = np.inf
    best_test_geodesic_ab = np.inf
    best_test_geodesic_ba = np.inf

    early_stopping = EarlyStopping(
        patience=args.patience,
        verbose=True,
        save_path=f'checkpoints/{args.exp_name}/models/model.best.t7',
    )

    for epoch in range(args.epochs):
        train_loss, train_cycle_loss,rotation_loss_ab, rotation_loss_ba,\
            rotations_ab, translations_ab, rotations_ab_pred, translations_ab_pred, \
            rotations_ba, translations_ba, rotations_ba_pred, translations_ba_pred, \
            eulers_ab, eulers_ba = train_one_epoch(args, net, train_loader, opt, geodesic_loss_fn)

        train_geodesic_ab = geodesic_loss_fn(
            torch.tensor(rotations_ab_pred).to(device),
        torch.tensor(rotations_ab).to(device)
        ).item()

        train_geodesic_ba = geodesic_loss_fn(
            torch.tensor(rotations_ba_pred).to(device),
            torch.tensor(rotations_ba).to(device)
        ).item()

        test_loss, test_cycle_loss,rotation_loss_ab, rotation_loss_ba,\
            test_rotations_ab, test_translations_ab, test_rotations_ab_pred, test_translations_ab_pred,\
            test_rotations_ba, test_translations_ba, test_rotations_ba_pred, test_translations_ba_pred, \
            test_eulers_ab, test_eulers_ba = test_one_epoch(args, net, test_loader, geodesic_loss_fn)

        # Compute mean geodesic error for logging
        test_geodesic_ab = geodesic_loss_fn(
            torch.tensor(test_rotations_ab_pred).to(device),
            torch.tensor(test_rotations_ab).to(device)
        ).item()

        test_geodesic_ba = geodesic_loss_fn(
            torch.tensor(test_rotations_ba_pred).to(device),
            torch.tensor(test_rotations_ba).to(device)
        ).item()

        # --- Euler angle MAE (degrees) ---
        try:
            ep_tr = ScipyRotation.from_matrix(rotations_ab_pred).as_euler('xyz', degrees=True)
            eg_tr = ScipyRotation.from_matrix(rotations_ab).as_euler('xyz', degrees=True)
            d_tr  = (ep_tr - eg_tr + 180) % 360 - 180
            train_euler_mae = np.abs(d_tr).mean(axis=0)   # (3,) per-axis x,y,z

            ep_te = ScipyRotation.from_matrix(test_rotations_ab_pred).as_euler('xyz', degrees=True)
            eg_te = ScipyRotation.from_matrix(test_rotations_ab).as_euler('xyz', degrees=True)
            d_te  = (ep_te - eg_te + 180) % 360 - 180
            test_euler_mae = np.abs(d_te).mean(axis=0)    # (3,)
        except Exception:
            train_euler_mae = test_euler_mae = np.array([float('nan')] * 3)

        # Update learning rate based on test loss (once per epoch)
        scheduler.step(test_loss)

        # Track best metrics for logging
        if best_test_loss >= test_loss:
            best_test_loss = test_loss
            best_test_cycle_loss = test_cycle_loss
            best_test_geodesic_ab = test_geodesic_ab
            best_test_geodesic_ba = test_geodesic_ba

        # Logging
        textio.cprint('==TRAIN==')
        textio.cprint('EPOCH:: %d, Loss: %f, Cycle Loss: %f, Geodesic AB: %f, Euler MAE (x,y,z): %.2f, %.2f, %.2f deg'
                      % (epoch, train_loss, train_cycle_loss, train_geodesic_ab,
                         train_euler_mae[0], train_euler_mae[1], train_euler_mae[2]))
        textio.cprint('==TEST==')
        textio.cprint('EPOCH:: %d, Loss: %f, Cycle Loss: %f, Geodesic AB: %f, Geodesic BA: %f, Euler MAE (x,y,z): %.2f, %.2f, %.2f deg'
                      % (epoch, test_loss, test_cycle_loss, test_geodesic_ab, test_geodesic_ba,
                         test_euler_mae[0], test_euler_mae[1], test_euler_mae[2]))
        textio.cprint('==BEST TEST==')
        textio.cprint('EPOCH:: %d, Loss: %f, Cycle Loss: %f, Geodesic AB: %f, Geodesic BA: %f'
                      % (epoch, best_test_loss, best_test_cycle_loss, best_test_geodesic_ab, best_test_geodesic_ba))

        boardio.add_scalar('train/loss', train_loss, epoch)
        boardio.add_scalar('train/cycle_loss', train_cycle_loss, epoch)
        boardio.add_scalar('train/geodesic_ab', train_geodesic_ab, epoch)
        boardio.add_scalar('train/euler_mae_x', train_euler_mae[0], epoch)
        boardio.add_scalar('train/euler_mae_y', train_euler_mae[1], epoch)
        boardio.add_scalar('train/euler_mae_z', train_euler_mae[2], epoch)
        boardio.add_scalar('test/loss', test_loss, epoch)
        boardio.add_scalar('test/cycle_loss', test_cycle_loss, epoch)
        boardio.add_scalar('test/geodesic_ab', test_geodesic_ab, epoch)
        boardio.add_scalar('test/geodesic_ba', test_geodesic_ba, epoch)
        boardio.add_scalar('test/euler_mae_x', test_euler_mae[0], epoch)
        boardio.add_scalar('test/euler_mae_y', test_euler_mae[1], epoch)
        boardio.add_scalar('test/euler_mae_z', test_euler_mae[2], epoch)

        # Best-model checkpoints are sufficient for normal runs. Opt in to
        # per-epoch snapshots only when an experiment explicitly needs them.
        if args.save_every_epoch:
            state = net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict()
            torch.save(
                state,
                'checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch),
            )

        # Early stopping — also saves model.best.t7 on improvement
        early_stopping(test_loss, net)
        if early_stopping.early_stop:
            textio.cprint(f'Early stopping triggered at epoch {epoch}. Best loss: {early_stopping.best_loss:.6f}')
            break

        gc.collect()


def main():
    global device
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dcp', metavar='N',
                        choices=['dcp'],
                        help='Model to use, [dcp]')
    parser.add_argument('--emb_nn', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn', 'pcard'],
                        help='Embedding nn to use: [dgcnn] (static) or [pcard] (pre-trained dynamic graph)')
    parser.add_argument('--pcard_checkpoint', type=str, default='', metavar='PATH',
                        help='Path to PCARD pre-trained model (.pth). '
                             'Required when --emb_nn pcard. '
                             'Example: ../outputs/trained_models/trained_model.pth')
    parser.add_argument('--pointer', type=str, default='transformer', metavar='N',
                        choices=['identity', 'transformer'],
                        help='Attention-based pointer generator to use, [identity, transformer]')
    parser.add_argument('--head', type=str, default='mlp', metavar='N',
                        choices=['mlp', 'svd', ],
                        help='Head to use, [mlp, svd]')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--n_blocks', type=int, default=1, metavar='N',
                        help='Num of blocks of encoder&decoder')
    parser.add_argument('--n_heads', type=int, default=4, metavar='N',
                        help='Num of heads in multiheadedattention')
    parser.add_argument('--ff_dims', type=int, default=1024, metavar='N',
                        help='Num of dimensions of fc in transformer')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='N',
                        help='Dropout ratio in transformer')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action='store_true', default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Force CPU execution even when CUDA is available')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the model')
    parser.add_argument('--cycle', action='store_true',
                        help='Whether to use cycle consistency')
    parser.add_argument('--gaussian_noise', action='store_true',
                        help='Whether to add Gaussian noise')
    parser.add_argument('--num_points', type=int, default=1024, metavar='N',
                        help='Num of points to use')
    parser.add_argument('--dataset', type=str, default='catmaus',
                        choices=['catmaus', 'catmaus_csv'], metavar='N',
                        help='dataset to use: catmaus (h5 files) or catmaus_csv (PCARD CSV scans)')
    parser.add_argument('--factor', type=float, default=4, metavar='N',
                        help='Divided factor for rotations')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--root', type=str, default='./data')
    parser.add_argument('--num_angles_per_sample', type=int, default=1, metavar='N',
                        help='Number of independently-rotated views per sample per training step '
                             '(1 = standard, >1 = multi-angle augmentation). '
                             'Extra views get fresh random rotations applied to the same source cloud.')
    parser.add_argument('--freeze_encoder', action='store_true', default=False,
                        help='Freeze the PCARD encoder weights and only train the Transformer + head. '
                             'When False (default), encoder fine-tunes at 0.1x base LR (differential LR). '
                             'Only has effect when --emb_nn pcard.')
    parser.add_argument('--patience', type=int, default=10, metavar='N',
                        help='EarlyStopping patience in epochs (default: 10). '
                             'Increase to 20-30 when training with small datasets '
                             'to allow the scheduler more time to reduce LR.')
    parser.add_argument('--translation_weight', type=float, default=1e-7,
                        help='Weight for the symmetric translation loss.')
    parser.add_argument('--orthogonality_weight', type=float, default=0.01,
                        help='Weight for rotation-matrix orthogonality loss.')
    parser.add_argument('--cycle_weight', type=float, default=0.1,
                        help='Weight for cycle consistency when --cycle is set.')
    parser.add_argument('--save_every_epoch', action='store_true',
                        help='Also keep every epoch checkpoint (uses substantially more disk).')

    args = parser.parse_args()

    device = torch.device(
        'cpu' if args.no_cuda else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    print(f'Using device: {device}')

    # PCARD encoder outputs 128-dim features — enforce this automatically
    # so the user doesn't need to remember --emb_dims 128 every time.
    if args.emb_nn == 'pcard' and args.emb_dims != 128:
        print(f"[INFO] --emb_nn pcard requires --emb_dims 128 "
              f"(was {args.emb_dims}). Auto-correcting.")
        args.emb_dims = 128

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)
    _init_(args)

    textio = IOStream('checkpoints/' + args.exp_name + '/run.log')
    textio.cprint(str(args))

    if args.dataset == 'catmaus':
        train_loader = DataLoader(
            CATMAUS(num_points=args.num_points, partition='train',
                    gaussian_noise=args.gaussian_noise, factor=args.factor, root=args.root),
            batch_size=args.batch_size, shuffle=True, drop_last=False)
        test_loader = DataLoader(
            CATMAUS(num_points=args.num_points, partition='test',
                    gaussian_noise=args.gaussian_noise, factor=args.factor, root=args.root),
            batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    elif args.dataset == 'catmaus_csv':
        train_loader = DataLoader(
            CATMAUSFromCSV(num_points=args.num_points, partition='train',
                           gaussian_noise=args.gaussian_noise, root=args.root),
            batch_size=args.batch_size, shuffle=True, drop_last=False)
        test_loader = DataLoader(
            CATMAUSFromCSV(num_points=args.num_points, partition='test',
                           gaussian_noise=args.gaussian_noise, root=args.root),
            batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    else:
        raise Exception("not implemented")

    if args.model == 'dcp':
        net = DCP(args).to(device)

        # ----------------------------------------------------------------
        # PCARD pre-trained encoder loading
        # ----------------------------------------------------------------
        # When --emb_nn pcard, load the geometry-aware dynamic-graph weights
        # into net.emb_nn as initialisation. The encoder is NOT frozen —
        # it fine-tunes end-to-end alongside the Transformer and head,
        # but at 0.1× base LR (differential learning rates, set in train()).
        # ----------------------------------------------------------------
        if args.emb_nn == 'pcard':
            if not args.pcard_checkpoint:
                raise ValueError(
                    "--pcard_checkpoint is required when using --emb_nn pcard.\n"
                    "Example: --pcard_checkpoint ../outputs/trained_models/trained_model.pth"
                )
            if not os.path.exists(args.pcard_checkpoint):
                raise FileNotFoundError(
                    f"PCARD checkpoint not found: {args.pcard_checkpoint}"
                )
            pcard_state = load_state_dict_safely(args.pcard_checkpoint, device)
            net.emb_nn.load_state_dict(pcard_state, strict=True)
            if args.freeze_encoder:
                for param in net.emb_nn.parameters():
                    param.requires_grad = False
                trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
                total     = sum(p.numel() for p in net.parameters())
                print(f"[PCARD] Loaded encoder weights from: {args.pcard_checkpoint}")
                print(f"[PCARD] Encoder FROZEN. Training Transformer + head only.")
                print(f"[PCARD] Trainable: {trainable:,} / {total:,}")
            else:
                total = sum(p.numel() for p in net.parameters())
                print(f"[PCARD] Loaded encoder weights from: {args.pcard_checkpoint}")
                print(f"[PCARD] Encoder will fine-tune at 0.1× LR; Transformer+Head at full LR.")
                print(f"[PCARD] Total trainable params: {total:,}")

        if args.eval:
            if args.model_path == '':
                model_path = 'checkpoints' + '/' + args.exp_name + '/models/model.best.t7'
            else:
                model_path = args.model_path
                print(model_path)
            if not os.path.exists(model_path):
                print("can't find pretrained model")
                return
            net.load_state_dict(load_state_dict_safely(model_path, device), strict=True)
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
            print("Let's use", torch.cuda.device_count(), "GPUs!")
    else:
        raise Exception('Not implemented')
    if args.eval:
        test(args, net, test_loader, boardio, textio, geodesic_loss_fn)
    else:
        train(args, net, train_loader, test_loader, boardio, textio, geodesic_loss_fn)

    print('FINISH')
    boardio.close()


if __name__ == '__main__':
    main()
