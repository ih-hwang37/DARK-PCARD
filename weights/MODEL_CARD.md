# PCARD pretrained encoder

## Model

The file `pcard_pretrained.pth` is a tensor-only PyTorch state dictionary for
`DGCNNWithKNN(k=20, feature_dim=128)`. It contains the encoder weights only and
does not contain an optimiser, patient point clouds, filenames, or W&B state.

- Size: 717,226 bytes
- SHA-256: `9530818dfd77c754d84ad7425648d0fb4a3cef662b097a3f300dffce80efde40`

## Intended use

The checkpoint is intended for non-clinical research on 3D point-cloud feature
learning, experimental filtering, and initialisation of the DARK registration
pipeline. Inputs are centred and scaled to the unit sphere before inference.

## Limitations

The training data are not public, so external users cannot independently audit
the complete data distribution or exactly reproduce this checkpoint. The model
may fail under different anatomy, acquisition systems, reconstruction methods,
noise levels, point densities, or preprocessing. Its output must not be treated
as a diagnosis or a validated clinical measurement.

Release of weights derived from clinical data remains subject to the relevant
institutional, data-governance, consent, and intellectual-property approvals.
