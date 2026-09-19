import torch
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, knn_graph


class DGCNNWithKNN(torch.nn.Module):
    def __init__(self, k=20, feature_dim=128):
        super(DGCNNWithKNN, self).__init__()
        self.k = k

        # Define EdgeConv layers
        self.conv1 = EdgeConv(self.mlp(6, 64), aggr='max')
        self.conv2 = EdgeConv(self.mlp(128, 64), aggr='max')
        self.conv3 = EdgeConv(self.mlp(128, 64), aggr='max')
        self.conv4 = EdgeConv(self.mlp(128, 128), aggr='max')

        # MLP layers for feature projection.
        # mlp1 keeps its ReLU — it's an intermediate non-linearity that
        # gives the network its expressivity.
        # mlp2 has NO final activation. The output features are used as
        # embeddings compared via cosine similarity in train.py
        # (loss_nce, loss_cov, loss_scatter). A trailing ReLU would
        # restrict features to the non-negative orthant of R^feature_dim,
        # bounding cosine similarity to [0, 1] and making it impossible
        # for the geometric losses to push pairs apart (which requires
        # negative similarity). The vanilla DGCNN paper has a ReLU here
        # because it feeds a learned classifier on top; we feed
        # cosine-similarity losses, so the activation must be removed.
        self.mlp1 = torch.nn.Sequential(
            # Concatenated feature size: 320 (64+64+64+128)
            torch.nn.Linear(320, 256),
            torch.nn.ReLU(),
        )
        self.mlp2 = torch.nn.Linear(256, feature_dim)


    def mlp(self, input_dim, output_dim):
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(output_dim, output_dim)
        )

    def forward(self, data):
        pos, batch = data.pos, data.batch

        # Store k-NN graphs from each layer
        knn_graphs = []

        # First EdgeConv layer with k-NN graph
        edge_index1 = knn_graph(pos, self.k, batch=batch)
        knn_graphs.append(edge_index1)
        x1 = self.conv1(pos, edge_index1)

        # Second EdgeConv layer with k-NN graph
        edge_index2 = knn_graph(x1, self.k, batch=batch)
        knn_graphs.append(edge_index2)
        x2 = self.conv2(x1, edge_index2)

        # Third EdgeConv layer with k-NN graph
        edge_index3 = knn_graph(x2, self.k, batch=batch)
        knn_graphs.append(edge_index3)
        x3 = self.conv3(x2, edge_index3)

        # Fourth EdgeConv layer with k-NN graph
        edge_index4 = knn_graph(x3, self.k, batch=batch)
        knn_graphs.append(edge_index4)
        x4 = self.conv4(x3, edge_index4)

        # Concatenate learned features from different layers
        # Total feature size: 320 (64+64+64+128)
        x = torch.cat([x1, x2, x3, x4], dim=1)

         # Project to final pointwise feature embedding
        x = self.mlp1(x)
        features = self.mlp2(x)

        return features, knn_graphs
