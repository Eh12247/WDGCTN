import torch
from utils import load_adjacency_matrix, load_node_features_with_normalization, DynamicGraphDataset, GCN_AutoEncoder, train_autoencoder, export_embeddings_per_node
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data_folder = r"./data"


adj_file = os.path.join(data_folder, "adj_matrices.npy")
feature_folder = data_folder


adj_matrices = load_adjacency_matrix(adj_file)
node_features, timestamps, station_names = load_node_features_with_normalization(feature_folder)


dataset = DynamicGraphDataset(adj_matrices, node_features, device)


in_channels = node_features.shape[2]
hidden_dim = 64
out_dim = 32
epochs = 50
lr = 1e-3

model = GCN_AutoEncoder(in_channels, hidden_dim, out_dim)


train_autoencoder(model, dataset, device, epochs=epochs, lr=lr)


output_folder = "GCN_embeddings"


export_embeddings_per_node(model, dataset, device, save_folder=output_folder)