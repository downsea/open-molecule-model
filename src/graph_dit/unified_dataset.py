import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import os
import json
from typing import List, Optional, Dict
from pathlib import Path


class UnifiedMolecularDataset(Dataset):
    """
    PyTorch Geometric dataset for unified molecular data pipeline.
    
    This dataset works with the standardized data structure from the PanGu pipeline:
    - data/standard/train/train_molecules.pt
    - data/standard/val/val_molecules.pt  
    - data/standard/test/test_molecules.pt
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
        
        # Determine the actual file path based on the split
        if split in ['train', 'val', 'test']:
            self.file_path = os.path.join(data_path, f"{split}_molecules.pt")
        else:
            # Fallback for compatibility
            self.file_path = os.path.join(data_path, f"{split}.pt")
        
        super().__init__(data_path, transform, pre_transform, pre_filter)
    
    @property
    def raw_file_names(self):
        """Return raw file names."""
        return [f"{self.split}_molecules.pt"]
    
    @property
    def processed_file_names(self):
        """Return processed file names."""
        return [f"{self.split}_molecules.pt"]
    
    def download(self):
        """Download dataset if not available."""
        # No download needed - uses processed data
        pass
    
    def process(self):
        """Process raw data into PyTorch Geometric format."""
        # Data is already processed in unified format
        pass
    
    def len(self):
        """Return number of graphs."""
        if not os.path.exists(self.file_path):
            return 0
        data_list = torch.load(self.file_path, weights_only=False)
        return len(data_list)
    
    def get(self, idx):
        """Get graph by index."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        
        data_list = torch.load(self.file_path, weights_only=False)
        data = data_list[idx]
        
        # Ensure data has required attributes for GraphDiT
        if not hasattr(data, 'x'):
            raise ValueError(f"Data item {idx} missing required 'x' attribute")
        
        return data
    
    def get_data_info(self) -> Dict[str, any]:
        """Get information about the dataset."""
        if not os.path.exists(self.file_path):
            return {"num_graphs": 0, "error": f"File not found: {self.file_path}"}
        
        try:
            data_list = torch.load(self.file_path, weights_only=False)
            if not data_list:
                return {"num_graphs": 0, "error": "Empty dataset"}
            
            # Get info from first sample
            sample = data_list[0]
            
            info = {
                "num_graphs": len(data_list),
                "node_features_dim": sample.x.size(-1) if hasattr(sample, 'x') else None,
                "edge_features_dim": sample.edge_attr.size(-1) if hasattr(sample, 'edge_attr') else None,
                "avg_nodes_per_graph": sum(d.num_nodes for d in data_list) / len(data_list),
                "avg_edges_per_graph": sum(d.edge_index.size(1) for d in data_list) / len(data_list) / 2
            }
            
            return info
            
        except Exception as e:
            return {"num_graphs": 0, "error": str(e)}


def get_unified_molecular_dataloaders(
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
    loaders = {}
    
    # Use unified data structure
    standard_path = Path(data_root) / 'standard'
    
    for split in ['train', 'val', 'test']:
        split_dir = standard_path / split
        
        # Check if split directory exists and has processed files
        data_file = split_dir / f"{split}_molecules.pt"
        
        if data_file.exists():
            print(f"Found {split} data: {data_file}")
            
            base_dataset = UnifiedMolecularDataset(
                data_path=str(split_dir),
                split=split,
                max_num_atoms=config.data.max_num_atoms,
                min_num_atoms=config.data.min_num_atoms
            )
            
            # Check if dataset loaded successfully
            info = base_dataset.get_data_info()
            print(f"  {split} dataset: {info}")
            
            if info.get('num_graphs', 0) == 0:
                print(f"Warning: {split} dataset is empty, creating dummy data...")
                # Create dummy data for testing
                base_dataset = create_dummy_dataset(config, split_dir, split)
            
            from .data import GraphDiffusionDataset
            diffusion_dataset = GraphDiffusionDataset(
                base_dataset=base_dataset,
                noise_scheduler=noise_scheduler
            )
            
            loaders[split] = DataLoader(
                diffusion_dataset,
                batch_size=config.data.batch_size,
                shuffle=(split == 'train'),
                num_workers=config.data.num_workers,
                collate_fn=lambda x: x,  # Simple collate for now
                pin_memory=True
            )
        else:
            print(f"Warning: {split} data not found at {data_file}, creating dummy data...")
            base_dataset = create_dummy_dataset(config, split_dir, split)
            
            from .data import GraphDiffusionDataset
            diffusion_dataset = GraphDiffusionDataset(
                base_dataset=base_dataset,
                noise_scheduler=noise_scheduler
            )
            
            loaders[split] = DataLoader(
                diffusion_dataset,
                batch_size=config.data.batch_size,
                shuffle=(split == 'train'),
                num_workers=config.data.num_workers,
                collate_fn=lambda x: x,
                pin_memory=True
            )
    
    return loaders


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
        
        # Atom types (simple: C, N, O)
        x = torch.randint(0, 3, (num_nodes,))  # 3 atom types
        
        # Simple connectivity (star graph)
        edge_index = torch.tensor([[0] * (num_nodes-1), list(range(1, num_nodes))], dtype=torch.long)
        
        # Bond types (single bonds)
        edge_attr = torch.zeros(num_nodes-1, dtype=torch.long)
        
        # Create data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            atom_features=torch.randn(num_nodes, 8),  # Dummy features
            edge_features=torch.randn(num_nodes-1, 4),  # Dummy features
            y=torch.randn(4)  # Dummy properties
        )
        
        data.num_nodes = num_nodes
        dummy_data.append(data)
    
    # Save dummy data
    split_dir.mkdir(parents=True, exist_ok=True)
    torch.save(dummy_data, split_dir / f"{split}_molecules.pt")
    
    return UnifiedMolecularDataset(
        data_path=str(split_dir),
        split=split,
        max_num_atoms=config.data.max_num_atoms,
        min_num_atoms=config.data.min_num_atoms
    )