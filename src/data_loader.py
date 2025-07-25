import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from rdkit import Chem
import selfies as sf
import glob
from torch_geometric.data import Data, DataLoader

class ZINC_Dataset(Dataset):
    def __init__(self, root_dir, max_length=128):
        self.processed_files = glob.glob(os.path.join(root_dir, '*.pt'))
        self.max_length = max_length
        self.data = []
        for file_path in self.processed_files:
            self.data.extend(torch.load(file_path, weights_only=False))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mol = self.data[idx]
        if mol is None:
            return None

        # Node features
        features = []
        for atom in mol.GetAtoms():
            feature = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                atom.GetHybridization(),
                atom.GetIsAromatic(),
                atom.GetNumRadicalElectrons()
            ]
            features.append(feature)
        
        x = torch.tensor(features, dtype=torch.float)

        # Adjacency matrix to edge_index
        adj = Chem.GetAdjacencyMatrix(mol)
        edge_index = torch.tensor(adj).nonzero().t().contiguous()

        data = Data(x=x, edge_index=edge_index)
        
        # Add smiles and selfies for reference
        data.smiles = Chem.MolToSmiles(mol)
        data.selfies = sf.encoder(data.smiles)
        
        return data

def collate_fn(batch):
    # Filter out None values
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    # Use PyG's DataLoader to handle batching of Data objects
    return DataLoader(batch, batch_size=len(batch))
