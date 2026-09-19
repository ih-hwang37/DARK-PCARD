import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from scipy.spatial.transform import Rotation
import os
import h5py
import glob
import hashlib
import json
import warnings
from torch.utils.data import DataLoader


class CATMAUS(Dataset):
    def __init__(self, num_points, partition='train', gaussian_noise=False, factor=4, root='data'):
        self.num_points = num_points
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.factor = factor
        self.root = root

        if self.partition not in {'train', 'test'}:
            raise ValueError(
                f"Invalid partition '{self.partition}'. Must be 'train' or 'test'.")
        self.data, self.label = self.load_data()

    def load_data(self):
        """
        Load all .h5 files, combine the data, and return them as NumPy arrays.
        """

        DATA_DIR = os.path.abspath(self.root)

        all_data = []
        all_label = []

        # Read each .h5 file and concatenate into one dataset
        h5_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.h5')))

        if not h5_files:
            raise ValueError("No .h5 files found in the data directory!")

        # Keep every sample from one source HDF5 file in the same partition.
        # For a patient-safe evaluation, callers should store each subject in
        # a separate file. A single legacy file necessarily falls back to a
        # deterministic sample-level split and emits a warning.
        fallback_sample_split = len(h5_files) == 1
        if fallback_sample_split:
            selected_files = h5_files
            warnings.warn(
                "Only one HDF5 file was found; using a sample-level split. "
                "Use one HDF5 file per subject to prevent subject leakage.",
                RuntimeWarning,
            )
        else:
            train_files, test_files = train_test_split(
                h5_files, test_size=0.2, random_state=42
            )
            selected_files = train_files if self.partition == 'train' else test_files

        for file_path in selected_files:
            with h5py.File(file_path, 'r') as f:
                data = f['data'][:].astype('float32')
                label = f['label'][:].astype('int64')
                all_data.append(data)
                all_label.append(label)

        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        if fallback_sample_split:
            train_data, test_data, train_label, test_label = train_test_split(
                all_data, all_label, test_size=0.2, random_state=42,
                stratify=all_label,
            )
            if self.partition == 'train':
                return train_data, train_label
            return test_data, test_label
        return all_data, all_label

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]

        if self.partition == 'test':
            np.random.seed(item)  # Seed based on index for reproducibility

        if self.gaussian_noise:
            pointcloud = jitter_pointcloud(pointcloud)

        # Generate deterministic rotations/translations for test
        if self.partition == 'test':
            anglex = np.random.uniform(np.pi / 4, 3 * np.pi / 4)
            angley = np.random.uniform(np.pi / 4, 3 * np.pi / 4)
            anglez = np.random.uniform(np.pi / 4, 3 * np.pi / 4)
        else:
            if np.random.rand() < 0.8:
                anglex = np.random.uniform(np.pi / 4, 3 * np.pi / 4)
                angley = np.random.uniform(np.pi / 4, 3 * np.pi / 4)
                anglez = np.random.uniform(np.pi / 4, 3 * np.pi / 4)
            else:
                anglex = np.random.uniform(-np.pi, np.pi)
                angley = np.random.uniform(-np.pi, np.pi)
                anglez = np.random.uniform(-np.pi, np.pi)

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)

        Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])
        R_ab = Rz.dot(Ry).dot(Rx)
        R_ba = R_ab.T

        translation_ab = np.array(
            [np.random.uniform(-0.5, 0.5) for _ in range(3)])
        translation_ba = -R_ba.dot(translation_ab)

        pointcloud1 = pointcloud.T
        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pointcloud2 = rotation_ab.apply(
            pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)

        euler_ab = np.array([anglez, angley, anglex])
        euler_ba = -euler_ab[::-1]

        # Use deterministic permutations in test mode
        if self.partition == 'test':
            perm1 = np.random.permutation(pointcloud1.shape[1])
            perm2 = np.random.permutation(pointcloud2.shape[1])
            pointcloud1 = pointcloud1[:, perm1]
            pointcloud2 = pointcloud2[:, perm2]
        else:
            pointcloud1 = np.random.permutation(pointcloud1.T).T
            pointcloud2 = np.random.permutation(pointcloud2.T).T

        return (
            pointcloud1.astype('float32'),
            pointcloud2.astype('float32'),
            R_ab.astype('float32'),
            translation_ab.astype('float32'),
            R_ba.astype('float32'),
            translation_ba.astype('float32'),
            euler_ab.astype('float32'),
            euler_ba.astype('float32')
        )

    def __len__(self):
        return len(self.data)


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


class CATMAUSFromCSV(Dataset):
    """Registration dataset built from private PCARD-format CSV scans.

    Each CSV file (one bone scan) becomes one dataset sample.  On every
    __getitem__ call a fresh random subsample of `num_points` is drawn from
    the full cloud, then a random rotation + translation is applied to
    produce a source/target registration pair.  The point clouds are
    normalised to zero-mean / unit-sphere before rotation — matching the
    pre-processing used when training the PCARD encoder.

    Output tuple (identical to CATMAUS):
        pointcloud1  (3, N)  – source
        pointcloud2  (3, N)  – rotated target
        R_ab         (3, 3)
        translation_ab (3,)
        R_ba         (3, 3)
        translation_ba (3,)
        euler_ab     (3,)    – [anglez, angley, anglex] in radians
        euler_ba     (3,)
    """

    def __init__(self, num_points=1024, partition='train',
                 gaussian_noise=False, root='data'):
        self.num_points     = num_points
        self.partition      = partition
        self.gaussian_noise = gaussian_noise

        data_dir = os.path.abspath(root)

        # Discover all CSV files
        csv_files = sorted(glob.glob(os.path.join(data_dir, '**', '*.csv'),
                                     recursive=True))
        if not csv_files:
            raise ValueError(f"No CSV files found under {data_dir!r}")

        # Split by subject directory, keeping every bone/position CSV from a
        # subject in one partition. Flat directories treat each CSV as an
        # independent group because no subject identity is available.
        grouped = {}
        for csv_file in csv_files:
            relative = os.path.relpath(csv_file, data_dir)
            parts = relative.split(os.sep)
            subject = parts[0] if len(parts) > 1 else relative
            grouped.setdefault(subject, []).append(csv_file)
        subjects = sorted(grouped)
        if len(subjects) < 2:
            raise ValueError(
                "At least two subject groups are required. Organise CSVs as "
                "root/<subject>/*.csv to prevent subject leakage."
            )
        train_subjects, test_subjects = train_test_split(
            subjects, test_size=0.2, random_state=42
        )
        selected_subjects = train_subjects if partition == 'train' else test_subjects
        self.files = sorted(
            path for subject in selected_subjects for path in grouped[subject]
        )

        # Per-file cache: stores the normalised (N,3) float32 array
        cache_dir = os.path.abspath(os.path.join('outputs', '.npy_cache'))
        os.makedirs(cache_dir, exist_ok=True)
        self._cache_dir = cache_dir

        print(f"[CATMAUSFromCSV] {partition}: {len(self.files)} files "
              f"(from {len(csv_files)} total)")

    # ------------------------------------------------------------------

    def _load_cloud(self, csv_path):
        """Load a CSV, normalise, and return (N,3) float32 array.
        Cached to disk so repeated epochs are fast."""
        key = hashlib.sha256(os.path.abspath(csv_path).encode('utf-8')).hexdigest()
        cache = os.path.join(self._cache_dir, key + '.npy')
        if os.path.exists(cache):
            return np.load(cache)

        df = pd.read_csv(csv_path, header=None)
        first_row = df.iloc[0, :3].astype(str).str.strip().str.lower().tolist()
        if first_row == ['x', 'y', 'z']:
            df = df.iloc[1:].reset_index(drop=True)
        if df.shape[1] < 3:
            raise ValueError(f"Expected at least three columns in {csv_path}")
        pts = df.iloc[:, :3].values.astype(np.float32)

        # Normalise: zero-mean, unit-sphere — matches PCARD encoder training
        pts -= pts.mean(axis=0, keepdims=True)
        scale = np.linalg.norm(pts, axis=1).max()
        if scale > 1e-6:
            pts /= scale

        np.save(cache, pts)
        return pts

    # ------------------------------------------------------------------

    def __getitem__(self, item):
        csv_path = self.files[item]
        cloud    = self._load_cloud(csv_path)   # (N, 3)

        # Seed before sampling so the complete test example is reproducible.
        if self.partition == 'test':
            np.random.seed(item)

        # Random subsample to num_points
        N = cloud.shape[0]
        if N >= self.num_points:
            idx = np.random.choice(N, self.num_points, replace=False)
        else:
            idx = np.random.choice(N, self.num_points, replace=True)
        pointcloud = cloud[idx]                 # (num_points, 3)

        if self.gaussian_noise:
            pointcloud = jitter_pointcloud(pointcloud)

        # Random rotation angles
        if np.random.rand() < 0.8:
            anglex = np.random.uniform(np.pi / 4, 3 * np.pi / 4)
            angley = np.random.uniform(np.pi / 4, 3 * np.pi / 4)
            anglez = np.random.uniform(np.pi / 4, 3 * np.pi / 4)
        else:
            anglex = np.random.uniform(-np.pi, np.pi)
            angley = np.random.uniform(-np.pi, np.pi)
            anglez = np.random.uniform(-np.pi, np.pi)

        cosx, sinx = np.cos(anglex), np.sin(anglex)
        cosy, siny = np.cos(angley), np.sin(angley)
        cosz, sinz = np.cos(anglez), np.sin(anglez)

        Rx = np.array([[1,0,0],[0,cosx,-sinx],[0,sinx,cosx]])
        Ry = np.array([[cosy,0,siny],[0,1,0],[-siny,0,cosy]])
        Rz = np.array([[cosz,-sinz,0],[sinz,cosz,0],[0,0,1]])
        R_ab = Rz @ Ry @ Rx
        R_ba = R_ab.T

        translation_ab = np.random.uniform(-0.5, 0.5, size=3)
        translation_ba = -R_ba @ translation_ab

        # Build pointcloud1 (source) and pointcloud2 (rotated target)
        pointcloud1 = pointcloud.T                              # (3, N)
        rot         = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pointcloud2 = rot.apply(pointcloud1.T).T + translation_ab[:, None]  # (3, N)

        euler_ab = np.array([anglez, angley, anglex])
        euler_ba = -euler_ab[::-1].copy()

        # Shuffle point order
        pointcloud1 = np.random.permutation(pointcloud1.T).T
        pointcloud2 = np.random.permutation(pointcloud2.T).T

        return (
            pointcloud1.astype('float32'),
            pointcloud2.astype('float32'),
            R_ab.astype('float32'),
            translation_ab.astype('float32'),
            R_ba.astype('float32'),
            translation_ba.astype('float32'),
            euler_ab.astype('float32'),
            euler_ba.astype('float32'),
        )

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    train = CATMAUS(1024, partition='train')
    test = CATMAUS(1024, partition='test')
    print(f"Dataset size: {len(train)}")
    print(f"Sample data: {train.data[0].shape}")
    print(f"Sample shape: {train.data[0][:5]}")
    print(f"Sample labels: {train.label[:5]}")
    print(f"Test dataset size: {len(test)}")
