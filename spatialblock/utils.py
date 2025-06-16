import numpy as np
import torch
import os
import pandas as pd
from torch_geometric.data import Data, Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from tqdm import tqdm

def load_adjacency_matrix(file_path):

    adj_matrices = np.load(file_path)
    print(f"Loaded adjacency matrices with shape: {adj_matrices.shape}")
    return adj_matrices

def load_node_features_with_normalization(folder_path):

    file_list = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    file_list.sort()

    station_names = [os.path.splitext(f)[0] for f in file_list]
    data_frames = []

    for file in file_list:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)


        if "date" in df.columns[0].lower():
            df = df.set_index(df.columns[0])

        data_frames.append(df)

    
    timestamps = data_frames[0].index.tolist()
    for df in data_frames:
        assert df.index.tolist() == timestamps, 


    node_features = np.stack([df.values for df in data_frames], axis=1)


    mean = node_features.mean(axis=(0, 1))  
    std = node_features.std(axis=(0, 1))  


    std[std == 0] = 1.0


    node_features = (node_features - mean) / std

    print(f"Loaded and normalized node features with shape: {node_features.shape}")
    print(f"Feature mean after normalization: {node_features.mean():.4f}, std: {node_features.std():.4f}")

    return node_features, timestamps, station_names

class DynamicGraphDataset(Dataset):
    def __init__(self, adj_matrices, node_features, device, transform=None):

        super().__init__(transform=transform)
        self.adj_matrices = torch.tensor(adj_matrices, dtype=torch.float32, device=device)
        self.node_features = torch.tensor(node_features, dtype=torch.float32, device=device)
        self.T = adj_matrices.shape[0]  # 时间步数
        self.N = adj_matrices.shape[1]  # 节点数
        self.F = node_features.shape[2]  # 特征维度

    def __len__(self):
        return self.T

    def __getitem__(self, idx):

        A_t = self.adj_matrices[idx]
        X_t = self.node_features[idx]

        rows, cols = torch.nonzero(A_t, as_tuple=True)
        edge_index = torch.stack([rows, cols], dim=0)
        edge_attr = A_t[rows, cols]
        data = Data(x=X_t, edge_index=edge_index, edge_attr=edge_attr)
        data.full_adj = A_t

        return data

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_dim):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x

class InnerProductDecoder(nn.Module):
    def __init__(self):
        super(InnerProductDecoder, self).__init__()

    def forward(self, Z):
        return torch.matmul(Z, Z.t())

class GCN_AutoEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_dim):
        super(GCN_AutoEncoder, self).__init__()
        self.encoder = GCNEncoder(in_channels, hidden_dim, out_dim)
        self.decoder = InnerProductDecoder()

    def forward(self, x, edge_index, edge_weight):
        Z = self.encoder(x, edge_index, edge_weight)
        A_hat = self.decoder(Z)
        return A_hat, Z

def train_autoencoder(model, dataset, device, epochs=10, lr=1e-3, batch_size=32):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        print(f"Epoch {epoch + 1}/{epochs}")


        for batch_data in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}"):
            batch_data = batch_data.to(device)  


            batch_loss = 0


            for data in batch_data.to_data_list():
                x = data.x  # 节点特征 [N, F]
                edge_index = data.edge_index  
                edge_attr = data.edge_attr  
                full_adj = data.full_adj  


                A_hat, Z = model(x, edge_index, edge_weight=edge_attr)


                loss = F.mse_loss(A_hat, full_adj)
                batch_loss += loss


            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()


        avg_loss = total_loss / len(dataloader)
        print(f"Average Loss: {avg_loss:.4f}")

def get_output_filename(input_filename):

    base_name, ext = os.path.splitext(input_filename)
    return f"{base_name}_E{ext}"

def export_embeddings_per_node(model, dataset, device, save_folder="GCN_embeddings"):

    import shutil

    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)  
    os.makedirs(save_folder, exist_ok=True)

    model = model.to(device)
    model.eval()

    total_time_steps = len(dataset)  
    print(f"Total time steps: {total_time_steps}")

    for t in range(total_time_steps):
        data_t = dataset.__getitem__(t).to(device)
        x = data_t.x
        edge_index = data_t.edge_index
        edge_attr = data_t.edge_attr

        with torch.no_grad():
            Z_t = model.encoder(x, edge_index, edge_weight=edge_attr)  

        Z_t_np = Z_t.cpu().numpy()

        
        for i in range(Z_t_np.shape[0]):
            out_file = os.path.join(save_folder, get_output_filename(f"node_{i}.csv"))

            
            with open(out_file, "a") as f:
                row_str = ",".join([f"{val:.6f}" for val in Z_t_np[i]]) + "\n"
                f.write(row_str)

        # 打印进度日志
        if (t + 1) % 100 == 0:
            print(f"Processed {t + 1}/{total_time_steps} time steps")

    print(f"All node embeddings have been saved to {save_folder}")
