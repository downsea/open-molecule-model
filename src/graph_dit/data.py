import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
import os
import pickle
from pathlib import Path
from tqdm import tqdm


class MolecularGraphFeaturizer:
    """
    Featurizer for converting RDKit molecules to PyTorch Geometric graphs.
    """
    
    def __init__(self):
        # Define atom types (common in organic molecules)
        self.atom_types = [
            'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'H',
            'Si', 'B', 'Se', 'As', 'Ge', 'Sn', 'Te', 'Pb', 'Bi', 'Po'
        ]
        self.atom_to_idx = {atom: idx for idx, atom in enumerate(self.atom_types)}
        
        # Define bond types
        self.bond_types = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC
        ]
        self.bond_to_idx = {bond: idx for idx, bond in enumerate(self.bond_types)}
        
        # Define atom features
        self.atom_features = [
            'atomic_num', 'degree', 'formal_charge', 'num_hydrogens',
            'chiral_tag', 'hybridization', 'is_aromatic', 'is_in_ring'
        ]
    
    def mol_to_graph(self, mol: Chem.Mol) -> Optional[Data]:
        """
        Convert RDKit molecule to PyTorch Geometric graph.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            PyTorch Geometric Data object or None if invalid
        """
        if mol is None:
            return None
            
        try:
            Chem.SanitizeMol(mol)
        except:
            return None
            
        # Get atom features
        atom_features = []
        atom_types = []
        
        for atom in mol.GetAtoms():
            # Atom type
            symbol = atom.GetSymbol()
            atom_type_idx = self.atom_to_idx.get(symbol, len(self.atom_types))  # Unknown atoms -> last index
            atom_types.append(atom_type_idx)
            
            # Additional features
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                atom.GetTotalNumHs(),
                int(atom.GetChiralTag()),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                int(atom.IsInRing())
            ]
            atom_features.append(features)
        
        if not atom_features:  # Empty molecule
            return None
            
        atom_features = torch.tensor(atom_features, dtype=torch.float)
        atom_types = torch.tensor(atom_types, dtype=torch.long)
        
        # Get bond features
        edge_indices = []
        edge_features = []
        bond_types = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # Add both directions for undirected graph
            edge_indices.extend([[i, j], [j, i]])
            
            # Bond type
            bond_type_idx = self.bond_to_idx.get(bond.GetBondType(), len(self.bond_types))
            bond_types.extend([bond_type_idx, bond_type_idx])
            
            # Additional bond features
            features = [
                int(bond.GetBondTypeAsDouble()),
                int(bond.GetIsAromatic()),
                int(bond.IsInRing()),
                int(bond.GetIsConjugated())
            ]
            edge_features.extend([features, features])
        
        if not edge_indices:  # Single atom
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 4), dtype=torch.float)
            bond_types = torch.empty(0, dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
            bond_types = torch.tensor(bond_types, dtype=torch.long)
        
        # Create graph data
        data = Data(
            x=atom_types,  # Categorical atom types
            edge_index=edge_index,
            edge_attr=bond_types,  # Categorical bond types
            atom_features=atom_features,
            edge_features=edge_attr,
            num_nodes=len(atom_types)
        )
        
        return data
    
    def smiles_to_graph(self, smiles: str) -> Optional[Data]:
        """Convert SMILES string to graph."""
        mol = Chem.MolFromSmiles(smiles)
        return self.mol_to_graph(mol)


class ZINCDataset(Dataset):
    """
    PyTorch Geometric dataset for ZINC molecules.
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        transform=None,
        pre_transform=None,
        pre_filter=None,
        max_num_atoms: int = 100,
        min_num_atoms: int = 3
    ):
        self.data_path = data_path
        self.split = split
        self.max_num_atoms = max_num_atoms
        self.min_num_atoms = min_num_atoms
        
        super().__init__(data_path, transform, pre_transform, pre_filter)
    
    @property
    def raw_file_names(self):
        """Return raw file names."""
        return ['zinc250k.csv']
    
    @property
    def processed_file_names(self):
        """Return processed file names."""
        return [f'{self.split}.pt']
    
    def download(self):
        """Download dataset if not available."""
        # Note: In practice, you would download the ZINC dataset here
        # For now, we'll assume it's already available
        pass
    
    def process(self):
        """Process raw data into PyTorch Geometric format."""
        featurizer = MolecularGraphFeaturizer()
        
        # Load raw data
        data_file = os.path.join(self.raw_dir, 'zinc250k.csv')
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"ZINC dataset not found at {data_file}")
            
        df = pd.read_csv(data_file)
        
        # Split data (simplified - in practice use proper splits)
        total_size = len(df)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        
        if self.split == 'train':
            data_df = df[:train_size]
        elif self.split == 'val':
            data_df = df[train_size:train_size + val_size]
        else:  # test
            data_df = df[train_size + val_size:]
        
        data_list = []
        
        print(f"Processing {len(data_df)} molecules for {self.split} split...")
        
        for idx, row in tqdm(data_df.iterrows(), total=len(data_df), desc=f"Processing {self.split}"):
            smiles = row['smiles']
            
            # Convert to graph
            graph = featurizer.smiles_to_graph(smiles)
            if graph is None:
                continue
                
            # Filter by size
            if graph.num_nodes < self.min_num_atoms or graph.num_nodes > self.max_num_atoms:
                continue
            
            # Add molecular properties as targets
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                graph.y = torch.tensor([
                    Descriptors.MolWt(mol),
                    Descriptors.MolLogP(mol),
                    Descriptors.TPSA(mol),
                    Chem.QED.qed(mol)
                ], dtype=torch.float)
            else:
                graph.y = torch.zeros(4, dtype=torch.float)
            
            # Add SMILES for reference
            graph.smiles = smiles
            
            data_list.append(graph)
        
        # Save processed data
        torch.save(data_list, os.path.join(self.processed_dir, f'{self.split}.pt'))
        print(f"Saved {len(data_list)} molecules for {self.split} split")
    
    def len(self):
        """Return number of graphs."""
        return len(self.processed_file_names)
    
    def get(self, idx):
        """Get graph by index."""
        data = torch.load(os.path.join(self.processed_dir, f'{self.split}.pt'))
        return data[idx]


class GraphDiffusionDataset(Dataset):
    """
    Dataset wrapper for graph diffusion training.
    Adds noise to clean graphs on-the-fly.
    """
    
    def __init__(self, base_dataset, noise_scheduler, return_clean=False):
        self.base_dataset = base_dataset
        self.noise_scheduler = noise_scheduler
        self.return_clean = return_clean
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        data = self.base_dataset[idx]
        
        # Sample random timestep
        t = self.noise_scheduler.sample_timesteps(1, data.x.device)[0]
        
        # Add noise to node types
        x_noisy = self.noise_scheduler.q_sample(data.x, t)
        
        # Add noise to edge types
        edge_attr_noisy = self.noise_scheduler.q_sample(data.edge_attr, t)
        
        # Create new data object with noisy features
        noisy_data = Data(
            x=x_noisy,
            edge_index=data.edge_index,
            edge_attr=edge_attr_noisy,
            x_clean=data.x,
            edge_attr_clean=data.edge_attr,
            t=t,
            y=data.y if hasattr(data, 'y') else None,
            smiles=data.smiles if hasattr(data, 'smiles') else None,
            num_nodes=data.num_nodes
        )
        
        # Store noisy versions for trainer access
        noisy_data.x_noisy = x_noisy
        noisy_data.edge_attr_noisy = edge_attr_noisy
        
        if self.return_clean:
            return noisy_data, data
        
        return noisy_data


def collate_fn(batch):
    """
    Custom collate function for PyTorch Geometric Data objects.
    """
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)


def get_dataloaders(
    data_path: str,
    noise_scheduler,
    batch_size: int = 32,
    num_workers: int = 4,
    max_num_atoms: int = 100,
    **kwargs
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    Supports both unified data pipeline and legacy structures.
    
    Args:
        data_path: Path to dataset (supports unified pipeline)
        noise_scheduler: Noise scheduler instance
        batch_size: Batch size
        num_workers: Number of workers for data loading
        max_num_atoms: Maximum number of atoms in molecules
        **kwargs: Additional arguments
        
    Returns:
        Dictionary of data loaders
    """
    # Use original data_loader.py for GraphDiT
    from ..data_loader import StreamingZINCDataset, create_optimized_dataloader
    
    loaders = {}
    data_path = Path(data_path)
    
    for split in ['train', 'val', 'test']:
        # Load data directly from structure
        split_path = data_path / split
        data_file = split_path / f"{split}_molecules.pt"
        
        if data_file.exists():
            try:
                data_list = torch.load(data_file, weights_only=False)
                print(f"Loaded {len(data_list)} molecules from {data_file}")
                
                # Convert RDKit Mol objects to PyTorch Geometric Data objects
                from rdkit import Chem
                from torch_geometric.data import Data
                
                def mol_to_data(mol):
                    """Convert RDKit Mol to PyTorch Geometric Data."""
                    if mol is None:
                        return None
                    
                    try:
                        # Get atom types as categorical indices
                        atom_types = []
                        for atom in mol.GetAtoms():
                            # Map atomic numbers to categorical indices
                            atomic_num = atom.GetAtomicNum()
# Map to GraphDiT node dimensions (0-19)
                            atom_type_map = {
                                6: 0,   # Carbon
                                7: 1,   # Nitrogen  
                                8: 2,   # Oxygen
                                9: 3,   # Fluorine
                                15: 4,  # Phosphorus
                                16: 5,  # Sulfur
                                17: 6,  # Chlorine
                                35: 7,  # Bromine
                                53: 8,  # Iodine
                                1: 9,   # Hydrogen
                                5: 10,  # Boron
                                14: 11, # Silicon
                                33: 12, # Arsenic
                                34: 13, # Selenium
                                52: 14, # Tellurium
                                82: 15, # Lead
                                83: 16, # Bismuth
                                84: 17, # Polonium
                                51: 18, # Antimony
                                50: 19   # Tin
                            }
                            atom_type = atom_type_map.get(atomic_num, 0)  # Default to Carbon for unknown
                            # Ensure atom type is within valid range [0, 19]
                            atom_type = max(0, min(19, atom_type))
                            atom_types.append(atom_type)
                        
                        x = torch.tensor(atom_types, dtype=torch.long)
                        
                        # Get edge types (bond types)
                        edge_indices = []
                        edge_types = []
                        
                        for bond in mol.GetBonds():
                            i = bond.GetBeginAtomIdx()
                            j = bond.GetEndAtomIdx()
                            
                            # Map bond types to categorical indices
                            bond_type_map = {
                                Chem.rdchem.BondType.SINGLE: 0,
                                Chem.rdchem.BondType.DOUBLE: 1,
                                Chem.rdchem.BondType.TRIPLE: 2,
                                Chem.rdchem.BondType.AROMATIC: 3
                            }
                            bond_type = bond_type_map.get(bond.GetBondType(), 0)
                            # Ensure bond type is within valid range [0, 3]
                            bond_type = max(0, min(3, bond_type))
                            
                            # Add both directions for undirected graph
                            edge_indices.extend([[i, j], [j, i]])
                            edge_types.extend([bond_type, bond_type])
                        
                        if not edge_indices:  # Single atom
                            edge_index = torch.empty((2, 0), dtype=torch.long)
                            edge_attr = torch.empty(0, dtype=torch.long)
                        else:
                            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                            edge_attr = torch.tensor(edge_types, dtype=torch.long)
                        
                        # Create PyG Data object with categorical features
                        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                        data.mol = mol  # Store original mol
                        data.num_nodes = len(x)
                        
                        return data
                    except Exception as e:
                        print(f"Error converting molecule: {e}")
                        return None
                
                # Convert all molecules
                graph_data = []
                for mol in tqdm(data_list, desc=f"Converting {split} molecules"):
                    graph = mol_to_data(mol)
                    if graph is not None:
                        graph_data.append(graph)
                
                base_dataset = SimpleDataset(graph_data)
                diffusion_dataset = GraphDiffusionDataset(
                    base_dataset=base_dataset,
                    noise_scheduler=noise_scheduler
                )
                
                from torch_geometric.loader import DataLoader
                loaders[split] = DataLoader(
                    dataset=diffusion_dataset,
                    batch_size=batch_size,
                    shuffle=(split == 'train'),
                    num_workers=0,  # Disable multiprocessing for Windows compatibility
                    collate_fn=lambda x: x,
                    pin_memory=True
                )
            except Exception as e:
                print(f"Error loading {data_file}: {e}")
                # Fallback to dummy data
                base_dataset = create_dummy_dataset(None, split_path, split)
                diffusion_dataset = GraphDiffusionDataset(
                    base_dataset=base_dataset,
                    noise_scheduler=noise_scheduler
                )
                from torch_geometric.loader import DataLoader
                loaders[split] = DataLoader(
                    dataset=diffusion_dataset,
                    batch_size=batch_size,
                    shuffle=(split == 'train'),
                    num_workers=0,  # Disable multiprocessing for Windows compatibility
                    collate_fn=lambda x: x,
                    pin_memory=True
                )
        else:
            print(f"Warning: {data_file} not found, creating dummy data")
            base_dataset = create_dummy_dataset(None, split_path, split)
            diffusion_dataset = GraphDiffusionDataset(
                base_dataset=base_dataset,
                noise_scheduler=noise_scheduler
            )
            from torch_geometric.loader import DataLoader
            loaders[split] = DataLoader(
                dataset=diffusion_dataset,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=num_workers,
                collate_fn=lambda x: x,
                pin_memory=True
            )
    
    return loaders


class SimpleDataset(Dataset):
    """Simple dataset for direct .pt file loading."""
    
    def __init__(self, data_list):
        self.data_list = [d for d in data_list if d is not None]
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]


def create_dummy_dataset(config, split_dir: Path, split: str):
    """Create dummy dataset for testing when real data is not available."""
    print(f"Creating dummy {split} dataset...")
    
    # Create minimal dummy graphs
    from torch_geometric.data import Data
    import torch
    
    # Create 100 dummy molecules for testing
    dummy_data = []
    for i in range(100):
        # Simple molecule with 5-15 atoms
        num_nodes = torch.randint(5, 16, (1,)).item()
        
        # Atom types (categorical indices 0-19)
        x = torch.randint(0, 20, (num_nodes,))  # 20 possible atom types
        
        # Simple connectivity (star graph)
        edge_index = torch.tensor([[0] * (num_nodes-1), list(range(1, num_nodes))], dtype=torch.long)
        
        # Bond types (categorical indices 0-3)
        edge_attr = torch.randint(0, 4, (num_nodes-1,))  # 4 possible bond types
        
        # Create data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.randn(4)  # Dummy properties
        )
        
        data.num_nodes = num_nodes
        dummy_data.append(data)
    
    # Save dummy data
    split_dir.mkdir(parents=True, exist_ok=True)
    torch.save(dummy_data, split_dir / f"{split}_molecules.pt")
    
    from torch.utils.data import Dataset
    class SimpleDataset(Dataset):
        def __init__(self, data_list):
            self.data_list = data_list
        
        def __len__(self):
            return len(self.data_list)
        
        def __getitem__(self, idx):
            return self.data_list[idx]
    
    return SimpleDataset(dummy_data)


def get_unified_dataloaders(
    config,
    noise_scheduler,
    data_root: str = 'data'
) -> Dict[str, DataLoader]:
    """
    Create data loaders using the unified data pipeline structure.
    
    Args:
        config: GraphDiT configuration
        noise_scheduler: Noise scheduler instance
        data_root: Root data directory (default: 'data')
        
    Returns:
        Dictionary of data loaders for train/val/test
    """
    # Load data directly from unified structure
    standard_path = Path(data_root) / 'standard'
    
    loaders = {}
    for split in ['train', 'val', 'test']:
        split_dir = standard_path / split
        
        # Load data directly from unified structure
        data_file = split_dir / f"{split}_molecules.pt"
        if data_file.exists():
            try:
                data_list = torch.load(data_file, weights_only=False)
                print(f"Loaded {len(data_list)} molecules from {data_file}")
                
                # Convert RDKit Mol objects to PyTorch Geometric Data objects
                from rdkit import Chem
                from torch_geometric.data import Data
                
                def mol_to_data(mol):
                    """Convert RDKit Mol to PyTorch Geometric Data."""
                    if mol is None:
                        return None
                    
                    try:
                        # Get atom types as categorical indices
                        atom_types = []
                        for atom in mol.GetAtoms():
                            # Map atomic numbers to categorical indices
                            atomic_num = atom.GetAtomicNum()
# Map to GraphDiT node dimensions (0-19)
                            atom_type_map = {
                                6: 0,   # Carbon
                                7: 1,   # Nitrogen  
                                8: 2,   # Oxygen
                                9: 3,   # Fluorine
                                15: 4,  # Phosphorus
                                16: 5,  # Sulfur
                                17: 6,  # Chlorine
                                35: 7,  # Bromine
                                53: 8,  # Iodine
                                1: 9,   # Hydrogen
                                5: 10,  # Boron
                                14: 11, # Silicon
                                33: 12, # Arsenic
                                34: 13, # Selenium
                                52: 14, # Tellurium
                                82: 15, # Lead
                                83: 16, # Bismuth
                                84: 17, # Polonium
                                51: 18, # Antimony
                                50: 19   # Tin
                            }
                            atom_type = atom_type_map.get(atomic_num, 0)  # Default to Carbon for unknown
                            # Ensure atom type is within valid range [0, 19]
                            atom_type = max(0, min(19, atom_type))
                            atom_types.append(atom_type)
                        
                        x = torch.tensor(atom_types, dtype=torch.long)
                        
                        # Get edge types (bond types)
                        edge_indices = []
                        edge_types = []
                        
                        for bond in mol.GetBonds():
                            i = bond.GetBeginAtomIdx()
                            j = bond.GetEndAtomIdx()
                            
                            # Map bond types to categorical indices
                            bond_type_map = {
                                Chem.rdchem.BondType.SINGLE: 0,
                                Chem.rdchem.BondType.DOUBLE: 1,
                                Chem.rdchem.BondType.TRIPLE: 2,
                                Chem.rdchem.BondType.AROMATIC: 3
                            }
                            bond_type = bond_type_map.get(bond.GetBondType(), 0)
                            # Ensure bond type is within valid range [0, 3]
                            bond_type = max(0, min(3, bond_type))
                            
                            # Add both directions for undirected graph
                            edge_indices.extend([[i, j], [j, i]])
                            edge_types.extend([bond_type, bond_type])
                        
                        if not edge_indices:  # Single atom
                            edge_index = torch.empty((2, 0), dtype=torch.long)
                            edge_attr = torch.empty(0, dtype=torch.long)
                        else:
                            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                            edge_attr = torch.tensor(edge_types, dtype=torch.long)
                        
                        # Create PyG Data object with categorical features
                        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                        data.mol = mol  # Store original mol
                        data.num_nodes = len(x)
                        
                        return data
                    except Exception as e:
                        print(f"Error converting molecule: {e}")
                        return None
                
                # Convert all molecules
                graph_data = []
                for mol in tqdm(data_list, desc=f"Converting {split} molecules"):
                    graph = mol_to_data(mol)
                    if graph is not None:
                        graph_data.append(graph)
                
                base_dataset = SimpleDataset(graph_data)
                diffusion_dataset = GraphDiffusionDataset(
                    base_dataset=base_dataset,
                    noise_scheduler=noise_scheduler
                )
                
                from torch_geometric.loader import DataLoader
                loaders[split] = DataLoader(
                    dataset=diffusion_dataset,
                    batch_size=config.data.batch_size,
                    shuffle=(split == 'train'),
                    num_workers=0,  # Disable multiprocessing for Windows compatibility
                    collate_fn=lambda x: x,
                    pin_memory=True
                )
            except Exception as e:
                print(f"Error loading {data_file}: {e}")
                # Fallback to dummy data
                base_dataset = create_dummy_dataset(config, split_dir, split)
                diffusion_dataset = GraphDiffusionDataset(
                    base_dataset=base_dataset,
                    noise_scheduler=noise_scheduler
                )
                from torch_geometric.loader import DataLoader
                loaders[split] = DataLoader(
                    dataset=diffusion_dataset,
                    batch_size=config.data.batch_size,
                    shuffle=(split == 'train'),
                    num_workers=0,  # Disable multiprocessing for Windows compatibility
                    collate_fn=lambda x: x,
                    pin_memory=True
                )
        else:
            print(f"Warning: {data_file} not found, creating dummy data")
            base_dataset = create_dummy_dataset(config, split_dir, split)
            diffusion_dataset = GraphDiffusionDataset(
                base_dataset=base_dataset,
                noise_scheduler=noise_scheduler
            )
            from torch_geometric.loader import DataLoader
            loaders[split] = DataLoader(
                dataset=diffusion_dataset,
                batch_size=config.data.batch_size,
                shuffle=(split == 'train'),
                num_workers=config.data.num_workers,
                collate_fn=lambda x: x,
                pin_memory=True
            )
    
    return loaders


def create_featurizer():
    """Create a molecular graph featurizer."""
    return MolecularGraphFeaturizer()


def smiles_to_graph(smiles: str) -> Optional[Data]:
    """
    Convert a SMILES string to a PyTorch Geometric graph.
    
    Args:
        smiles: SMILES string
        
    Returns:
        PyTorch Geometric Data object or None if invalid
    """
    featurizer = create_featurizer()
    return featurizer.smiles_to_graph(smiles)