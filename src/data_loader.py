import os
import torch
import pandas as pd
from torch.utils.data import Dataset, IterableDataset
from rdkit import Chem
import selfies as sf
import glob
from torch_geometric.data import Data, DataLoader
import random

class ZINC_Dataset(Dataset):
    def __init__(self, root_dir, max_length=128, cache_in_memory=False):
        self.processed_files = glob.glob(os.path.join(root_dir, '*.pt'))
        self.max_length = max_length
        self.cache_in_memory = cache_in_memory
        
        if cache_in_memory:
            self.data = []
            for file_path in self.processed_files:
                self.data.extend(torch.load(file_path, weights_only=False))
        else:
            # Lazy loading - store file indices and sizes
            self.file_indices = []
            self.file_sizes = []
            total_samples = 0
            
            for file_path in self.processed_files:
                data = torch.load(file_path, weights_only=False)
                self.file_indices.append(file_path)
                self.file_sizes.append(len(data))
                total_samples += len(data)
            
            self.total_samples = total_samples

    def __len__(self):
        if self.cache_in_memory:
            return len(self.data)
        return self.total_samples

    def __getitem__(self, idx):
        if self.cache_in_memory:
            mol = self.data[idx]
        else:
            # Find which file contains this index
            file_idx = 0
            cumulative_size = 0
            for i, size in enumerate(self.file_sizes):
                if idx < cumulative_size + size:
                    file_idx = i
                    local_idx = idx - cumulative_size
                    break
                cumulative_size += size
            
            # Load only the required file
            data = torch.load(self.file_indices[file_idx], weights_only=False)
            mol = data[local_idx]
            
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

class StreamingZINCDataset(IterableDataset):
    """Memory-efficient streaming dataset that loads data on-the-fly."""
    
    def __init__(self, root_dir, max_length=128, shuffle_files=True):
        self.processed_files = glob.glob(os.path.join(root_dir, '*.pt'))
        self.max_length = max_length
        self.shuffle_files = shuffle_files
        
    def __iter__(self):
        if self.shuffle_files:
            random.shuffle(self.processed_files)
            
        for file_path in self.processed_files:
            data = torch.load(file_path, weights_only=False)
            if self.shuffle_files:
                random.shuffle(data)
                
            for mol in data:
                if mol is None:
                    continue
                    
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

                data_obj = Data(x=x, edge_index=edge_index)
                
                # Add smiles and selfies for reference
                data_obj.smiles = Chem.MolToSmiles(mol)
                data_obj.selfies = sf.encoder(data_obj.smiles)
                
                yield data_obj

def collate_fn(batch):
    # Filter out None values
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    # Use PyG's DataLoader to handle batching of Data objects
    return DataLoader(batch, batch_size=len(batch))
