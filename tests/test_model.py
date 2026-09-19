import torch
import pytest
from torch_geometric.data import Data

from pcard.model import DGCNNWithKNN


def _load_weights(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def test_pretrained_checkpoint_loads_strictly():
    model = DGCNNWithKNN(k=20, feature_dim=128)
    state = _load_weights("weights/pcard_pretrained.pth")
    model.load_state_dict(state, strict=True)


def test_pcard_forward_shape():
    pytest.importorskip("torch_cluster", reason="PyG kNN extension is platform-specific")
    model = DGCNNWithKNN(k=4, feature_dim=16).eval()
    data = Data(
        pos=torch.randn(32, 3),
        batch=torch.zeros(32, dtype=torch.long),
    )
    with torch.no_grad():
        features, graphs = model(data)
    assert features.shape == (32, 16)
    assert len(graphs) == 4
