# DARK + PCARD

Research code for **DARK: Dynamic Graphs Based Angle-Aware Registration of
Knee Ultrasound Point Clouds**, together with the PCARD dynamic-graph encoder,
its training pipeline, an inference-time point-cloud filter, and a small
pretrained PCARD checkpoint.

DARK estimates coarse aligment matrices between sparse, noisy 3D point clouds
reconstructed from freehand knee ultrasound. PCARD learns geometric
features from dynamic k-nearest-neighbour graphs and can be used either to
filter a point cloud (likewise to DG-PPU) or help registration as DARK's embedding network.

> Hwang, I., Mellon, S., and Tu, S. J. (2026). **DARK: Dynamic Graphs Based
> Angle-Aware Registration of Knee Ultrasound Point Clouds.** In *Simplifying
> Medical Ultrasound: ASMUS 2025*, LNCS 16165, pp. 87–97.
> [doi:10.1007/978-3-032-06329-8_9](https://doi.org/10.1007/978-3-032-06329-8_9)

## What is included

- `dark/`: DARK training and evaluation, including standard, multi-angle, and
  pretrained-PCARD variants.
- `pcard/`: the PCARD encoder, covariance utilities, and
  geometry-aware filtering code.
- `train_pcard.py`: the complete YAML-configured PCARD training pipeline.
- `filter_with_pcard.py`: a command-line tool that produces DARK-ready CSVs.
- `weights/pcard_pretrained.pth`: the released PCARD encoder state dictionary.
- `configs/pcard.yaml`: documented PCARD training defaults.

Patient data, generated filtered point clouds, W&B histories, and bulk
experiment checkpoints are intentionally not included.

## Published DARK result

The paper evaluates independently transformed and Monte Carlo-sampled 3D
freehand-ultrasound point-cloud pairs without pointwise correspondence. Table 1
reports a mean geodesic rotation error of **33.8°** for DARK over 32 challenging
test cases.

| Method | Rx | Ry | Rz | Geodesic error |
|---|---:|---:|---:|---:|
| ICP | 96.2° | 51.8° | 89.4° | 150.9° |
| DCP + Geo | 65.7° | 41.5° | 72.6° | 85.4° |
| DARK − Geo | 87.3° | 19.8° | 60.4° | 48.4° |
| **DARK** | **30.4°** | **9.17°** | **39.5°** | **33.8°** |

These values are transcribed from the paper. Exact reproduction also depends on
the original governed dataset, its saved split, preprocessing, and the original
software/hardware environment, none of which is distributed here.

## Installation

Python 3.10 or 3.11 is recommended. Create an isolated environment, install a
PyTorch build for your CPU/CUDA platform, then install PyTorch Geometric and
this repository:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install torch
python -m pip install torch-geometric
python -m pip install -e .
```

`knn_graph` also needs the PyTorch Geometric `torch-cluster` extension on
platforms where it is not bundled. Install the wheel matching your PyTorch and
CUDA versions using the
[official PyG instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).
For optional W&B logging, run `python -m pip install -e ".[tracking]"`.

## Three DARK commands

Run these commands from the repository root and replace the example private
data paths with your own.

### 1. Standard DARK

```bash
python dark/main.py --exp_name dark_standard --dataset catmaus --root path/to/private_h5 --emb_nn dgcnn --pointer transformer --head mlp --emb_dims 512 --ff_dims 1024 --batch_size 32 --test_batch_size 32 --epochs 100 --lr 0.001 --num_points 1024
```

### 2. DARK with multiple sampled angles

The wrapper defaults to five independently transformed views per input sample.
More views use proportionally more accelerator memory.

```bash
python dark/main_multiple_angles.py --exp_name dark_multi_angle --dataset catmaus --root path/to/private_h5 --emb_nn dgcnn --pointer transformer --head mlp --emb_dims 512 --ff_dims 1024 --batch_size 32 --test_batch_size 32 --epochs 100 --lr 0.001 --num_points 1024
```

### 3. DARK with the pretrained PCARD encoder

```bash
python dark/main.py --exp_name dark_pcard --dataset catmaus_csv --root path/to/private_csv_subject_folders --emb_nn pcard --pcard_checkpoint weights/pcard_pretrained.pth --pointer transformer --head mlp --emb_dims 128 --ff_dims 256 --batch_size 16 --test_batch_size 32 --epochs 150 --lr 0.0001 --num_points 1024
```

By default the pretrained encoder is fine-tuned at one tenth of the main
learning rate. Add `--freeze_encoder` to train only the Transformer and
registration head. Multi-angle training and PCARD integration are post-paper
experimental variants and are not the configuration behind the published
33.8° result.

## Train PCARD

PCARD training is included in full. Copy `configs/pcard.yaml` to the ignored
local file `config.yaml`, set `data.root` to a private CSV directory, then run:

```bash
python train_pcard.py --config config.yaml
```

The training losses do not use anatomical class labels. The current CSV adapter
infers bone names only for a diagnostic graph-purity metric and for naming
filtered output files. The split is deterministic and subject-level: all
positions and stochastic views from one subject remain in one partition. The
exact split is recorded in `outputs/split_manifest.json`.

W&B is disabled by default. To enable it, set `wandb.mode` to `offline` or
`online` and authenticate with `wandb login` or the `WANDB_API_KEY` environment
variable.

## Filter point clouds with PCARD

Input CSVs should be organised as `root/<subject>/*.csv`. The first three
columns are x, y, and z; an `x,y,z` header is optional. Bone substrings such as
`fem`, `pat`, and `tib` in filenames are used only to separate output files.

```bash
python filter_with_pcard.py --input path/to/private_csv_subject_folders --output path/to/generated_filtered_data --checkpoint weights/pcard_pretrained.pth
```

Large groups are deterministically subsampled to 16,384 points by default to
bound dynamic-kNN memory. Change `--max-points` deliberately and document it in
experiments.

## Pretrained PCARD checkpoint

`weights/pcard_pretrained.pth` is a tensor-only state dictionary for
`pcard.model.DGCNNWithKNN(k=20, feature_dim=128)`.

- Size: 717,226 bytes
- SHA-256: `9530818dfd77c754d84ad7425648d0fb4a3cef662b097a3f300dffce80efde40`

See `weights/MODEL_CARD.md` for intended use and limitations. A model trained
from clinical data should be released only after the relevant institutional,
data-governance, and intellectual-property checks have been completed.

## Data privacy and expected layout

The clinical freehand-ultrasound point clouds used in the study are not
redistributed. Users must supply appropriately governed data.

For HDF5 DARK input, use one subject per file whenever possible. Each file must
contain `data` with shape `(samples, points, 3)` and `label` with one integer per
sample. The release splits HDF5 files rather than individual samples. If only a
single HDF5 file is supplied, the code warns before falling back to a
sample-level split.

For CSV input, keep every subject in a separate directory. This is how both
PCARD and DARK prevent the same subject from appearing in training and test
sets.

## Evaluation

Evaluate a DARK checkpoint with the architecture arguments used for training:

```bash
python dark/main.py --eval --dataset catmaus --root path/to/private_h5 --emb_nn dgcnn --pointer transformer --head mlp --emb_dims 512 --model_path path/to/model.best.t7
```

Only the best DARK checkpoint is retained by default. Add
`--save_every_epoch` only when every epoch snapshot is genuinely needed.

## Scope and limitations

This is research software for coarse rigid registration (i.e., estimating coarse alignment matrices) and un-supervised
point-cloud filtering. It is not a medical device and must not be used for
diagnosis, treatment, or patient care without independent validation and all
applicable institutional and regulatory approvals. The published registration
accuracy is not sufficient evidence for autonomous clinical deployment.

## Citation

```bibtex
@inproceedings{hwang2026dark,
  author    = {Hwang, Injune and Mellon, Stephen and Tu, S. Jack},
  title     = {DARK: Dynamic Graphs Based Angle-Aware Registration of Knee Ultrasound Point Clouds},
  booktitle = {Simplifying Medical Ultrasound: ASMUS 2025},
  series    = {Lecture Notes in Computer Science},
  volume    = {16165},
  pages     = {87--97},
  publisher = {Springer},
  year      = {2026},
  doi       = {10.1007/978-3-032-06329-8_9}
}
```

The work was delivered through the NIHR Oxford Biomedical Research Centre. The
views expressed are those of the authors and not necessarily those of the NIHR.

## License and acknowledgements

Released under the MIT License. See `LICENSE` and `THIRD_PARTY_NOTICES.md`.
