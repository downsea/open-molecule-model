import os
import torch
import pandas as pd
from torch.utils.data import Dataset, IterableDataset, DataLoader
from rdkit import Chem
import selfies as sf
import glob
from torch_geometric.data import Data, Batch
import random
from tqdm import tqdm

class ZINC_Dataset(Dataset):
    def __init__(self, root_dir, max_length=128, cache_in_memory=False,
                 precompute_features=True, use_memory_mapping=False):
        """
        Optimized ZINC dataset with better memory management.
        
        Args:
            root_dir: Directory containing processed .pt files
            max_length: Maximum sequence length
            cache_in_memory: Whether to cache all data in memory
            precompute_features: Whether to precompute molecular features
            use_memory_mapping: Whether to use memory mapping for large files
        """
        self.processed_files = glob.glob(os.path.join(root_dir, '**', '*.pt'), recursive=True)
        self.max_length = max_length
        self.cache_in_memory = cache_in_memory
        self.precompute_features = precompute_features
        self.use_memory_mapping = use_memory_mapping
        
        # Feature extraction cache
        self._feature_cache = {}
        
        if cache_in_memory:
            print("📥 Loading all data into memory...")
            self.data = []
            for file_path in tqdm(self.processed_files, desc="Loading files"):
                file_data = torch.load(file_path, weights_only=False)
                self.data.extend(file_data)
            print(f"Loaded {len(self.data)} molecules into memory")
        else:
            # Optimized lazy loading with better indexing
            self._build_file_index()
            
        # Pre-compute features if requested
        if precompute_features and not cache_in_memory:
            self._precompute_features()

    def _build_file_index(self):
        """Build efficient file index for lazy loading."""
        print("Building file index...")
        self.file_indices = []
        self.file_sizes = []
        self.cumulative_sizes = [0]
        total_samples = 0
        
        for file_path in tqdm(self.processed_files, desc="Indexing files"):
            try:
                # Quick size check without loading full data
                data = torch.load(file_path, weights_only=False)
                size = len(data)
                
                self.file_indices.append(file_path)
                self.file_sizes.append(size)
                total_samples += size
                self.cumulative_sizes.append(total_samples)
                
                # Clear data to save memory
                del data
                
            except Exception as e:
                print(f"Warning: Error loading {file_path}: {e}")
                continue
        
        self.total_samples = total_samples
        print(f"Indexed {len(self.file_indices)} files with {total_samples:,} molecules")

    def _precompute_features(self):
        """Pre-compute molecular features for faster access."""
        print("Pre-computing molecular features...")
        # This could be implemented to cache features on disk
        # For now, we'll compute them on-the-fly with caching
        pass

    def __len__(self):
        if self.cache_in_memory:
            return len(self.data)
        return self.total_samples

    def _find_file_and_index(self, idx):
        """Efficiently find file and local index using binary search."""
        # Binary search for file containing this index
        left, right = 0, len(self.cumulative_sizes) - 1
        
        while left < right:
            mid = (left + right) // 2
            if idx < self.cumulative_sizes[mid + 1]:
                right = mid
            else:
                left = mid + 1
        
        file_idx = left
        local_idx = idx - self.cumulative_sizes[file_idx]
        
        return file_idx, local_idx

    def __getitem__(self, idx):
        if self.cache_in_memory:
            mol = self.data[idx]
        else:
            # Efficient file lookup
            file_idx, local_idx = self._find_file_and_index(idx)
            
            # Load only the required file with caching
            file_path = self.file_indices[file_idx]
            if file_path not in self._feature_cache:
                data = torch.load(file_path, weights_only=False)
                # Cache small files, but not large ones
                if len(data) < 1000:
                    self._feature_cache[file_path] = data
                else:
                    mol = data[local_idx]
                    del data  # Free memory immediately
                    return self._process_molecule(mol)
            else:
                data = self._feature_cache[file_path]
            
            mol = data[local_idx]
            
        return self._process_molecule(mol)
    
    def _process_molecule(self, mol):
        """Process molecule into PyG Data object with optimizations."""
        if mol is None:
            return None

        try:
            # Optimized feature extraction
            atom_features = []
            for atom in mol.GetAtoms():
                # Pre-compute common values
                atomic_num = atom.GetAtomicNum()
                degree = atom.GetDegree()
                formal_charge = atom.GetFormalCharge()
                hybridization = int(atom.GetHybridization())
                is_aromatic = int(atom.GetIsAromatic())
                num_radical = atom.GetNumRadicalElectrons()
                
                feature = [atomic_num, degree, formal_charge, hybridization, is_aromatic, num_radical]
                atom_features.append(feature)
            
            x = torch.tensor(atom_features, dtype=torch.float)

            # More efficient edge index computation
            adj = Chem.GetAdjacencyMatrix(mol)
            edge_indices = torch.nonzero(torch.from_numpy(adj), as_tuple=False).t().contiguous()

            # Create PyG Data object
            data = Data(x=x, edge_index=edge_indices)
            
            # Add molecular information
            smiles = Chem.MolToSmiles(mol)
            data.smiles = smiles
            
            # Cache SELFIES conversion to avoid repeated computation
            try:
                data.selfies = sf.encoder(smiles)
            except:
                data.selfies = None
            
            return data
            
        except Exception as e:
            print(f"Warning: Error processing molecule: {e}")
            return None
    
    def clear_cache(self):
        """Clear feature cache to free memory."""
        self._feature_cache.clear()
    
    def get_cache_stats(self):
        """Get cache statistics."""
        return {
            'cached_files': len(self._feature_cache),
            'cache_memory_mb': sum(len(data) for data in self._feature_cache.values()) * 0.001
        }

class StreamingZINCDataset(IterableDataset):
    """Optimized memory-efficient streaming dataset with better performance."""
    
    def __init__(self, root_dir, max_length=128, shuffle_files=True,
                 buffer_size=1000, prefetch_factor=2):
        """
        Initialize streaming dataset with optimizations.
        
        Args:
            root_dir: Directory containing processed files
            max_length: Maximum sequence length
            shuffle_files: Whether to shuffle file order
            buffer_size: Size of internal buffer for shuffling
            prefetch_factor: Number of files to prefetch
        """
        self.root_dir = root_dir
        # Look for .pt files in subdirectories (train, val, test) as well as root directory
        self.processed_files = []
        
        # Check root directory
        root_files = glob.glob(os.path.join(root_dir, '*.pt'))
        self.processed_files.extend(root_files)
        
        # Check subdirectories (train, val, test)
        for subdir in ['train', 'val', 'test']:
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.exists(subdir_path):
                subdir_files = glob.glob(os.path.join(subdir_path, '*.pt'))
                self.processed_files.extend(subdir_files)
        self.max_length = max_length
        self.shuffle_files = shuffle_files
        self.buffer_size = buffer_size
        self.prefetch_factor = prefetch_factor
        
        # Sort files by size for better memory management
        self.processed_files.sort(key=lambda x: os.path.getsize(x))
        
        # Calculate approximate dataset size for training step calculation
        self._calculate_dataset_size()
        
        print(f"🔄 Streaming dataset initialized with {len(self.processed_files)} files")
    
    def _calculate_dataset_size(self):
        """Calculate approximate dataset size by sampling files."""
        if not self.processed_files:
            self.estimated_size = 0
            return
        
        # Sample a few files to estimate total size
        sample_size = min(5, len(self.processed_files))
        total_samples = 0
        
        for i in range(sample_size):
            try:
                file_path = self.processed_files[i]
                data = torch.load(file_path, weights_only=False)
                total_samples += len(data)
                del data  # Free memory
            except Exception:
                continue
        
        if sample_size > 0:
            avg_samples_per_file = total_samples / sample_size
            self.estimated_size = int(avg_samples_per_file * len(self.processed_files))
        else:
            self.estimated_size = 0
        
        print(f"Estimated dataset size: {self.estimated_size:,} molecules")
    
    def __len__(self):
        """Return estimated dataset size for training step calculation."""
        return self.estimated_size
        
    def __iter__(self):
        """Optimized iterator with buffering and prefetching."""
        worker_info = torch.utils.data.get_worker_info()
        
        # Handle multi-worker data loading
        if worker_info is not None:
            # Split files among workers
            per_worker = len(self.processed_files) // worker_info.num_workers
            worker_id = worker_info.id
            start_idx = worker_id * per_worker
            end_idx = start_idx + per_worker if worker_id < worker_info.num_workers - 1 else len(self.processed_files)
            files_to_process = self.processed_files[start_idx:end_idx]
        else:
            files_to_process = self.processed_files
        
        if self.shuffle_files:
            random.shuffle(files_to_process)
        
        # Process files with buffering
        buffer = []
        
        for file_path in files_to_process:
            try:
                # Load file data
                file_data = torch.load(file_path, weights_only=False)
                
                if self.shuffle_files:
                    random.shuffle(file_data)
                
                # Process molecules and add to buffer
                for mol in file_data:
                    if mol is None:
                        continue
                    
                    processed_data = self._process_molecule_fast(mol)
                    if processed_data is not None:
                        buffer.append(processed_data)
                        
                        # Yield from buffer when it's full
                        if len(buffer) >= self.buffer_size:
                            if self.shuffle_files:
                                random.shuffle(buffer)
                            
                            for item in buffer:
                                yield item
                            buffer.clear()
                
                # Clear file data to save memory
                del file_data
                
            except Exception as e:
                print(f"Warning: Error processing file {file_path}: {e}")
                continue
        
        # Yield remaining items in buffer
        if buffer:
            if self.shuffle_files:
                random.shuffle(buffer)
            for item in buffer:
                yield item
    
    def _process_molecule_fast(self, mol):
        """Fast molecule processing with minimal memory allocation."""
        try:
            # Pre-allocate lists for better performance
            num_atoms = mol.GetNumAtoms()
            if num_atoms == 0:
                return None
            
            # Vectorized feature extraction
            atom_features = torch.zeros(num_atoms, 6, dtype=torch.float)
            
            for i, atom in enumerate(mol.GetAtoms()):
                atom_features[i] = torch.tensor([
                    atom.GetAtomicNum(),
                    atom.GetDegree(),
                    atom.GetFormalCharge(),
                    int(atom.GetHybridization()),
                    int(atom.GetIsAromatic()),
                    atom.GetNumRadicalElectrons()
                ], dtype=torch.float)

            # Efficient edge index computation
            adj = Chem.GetAdjacencyMatrix(mol)
            edge_indices = torch.nonzero(torch.from_numpy(adj), as_tuple=False).t().contiguous()

            # Create data object
            data_obj = Data(x=atom_features, edge_index=edge_indices)
            
            # Add molecular information
            smiles = Chem.MolToSmiles(mol)
            data_obj.smiles = smiles
            
            # Efficient SELFIES encoding with error handling
            try:
                data_obj.selfies = sf.encoder(smiles)
            except:
                data_obj.selfies = None
                
            return data_obj
            
        except Exception as e:
            return None

def optimized_collate_fn(batch):
    """
    Optimized collate function for molecular data with better memory management.
    
    Args:
        batch: List of Data objects
        
    Returns:
        Batched Data object or None if batch is empty
    """
    # Filter out None values efficiently
    valid_batch = [item for item in batch if item is not None]
    
    if not valid_batch:
        return None
    
    # Use PyG's built-in batching (more efficient than manual batching)
    try:
        batched_data = Batch.from_data_list(valid_batch)
        return batched_data
    except Exception as e:
        print(f"Warning: Error in collate function: {e}")
        return None

def create_optimized_dataloader(dataset, batch_size, num_workers=0, shuffle=False,
                              pin_memory=True, persistent_workers=False):
    """
    Create optimized DataLoader with best practices for molecular data.
    
    Args:
        dataset: Dataset object
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        pin_memory: Whether to pin memory for faster GPU transfer
        persistent_workers: Whether to keep workers alive between epochs
        
    Returns:
        Optimized DataLoader
    """
    # Disable multiprocessing on Windows for streaming datasets to avoid issues
    if isinstance(dataset, StreamingZINCDataset) and os.name == 'nt':
        num_workers = 0
        persistent_workers = False
    
    # Configure prefetch_factor only when using multiprocessing
    dataloader_kwargs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'collate_fn': optimized_collate_fn,
        'pin_memory': pin_memory and torch.cuda.is_available(),
        'persistent_workers': persistent_workers and num_workers > 0,
        'drop_last': True,  # Drop incomplete batches for consistent training
    }
    
    # Only set prefetch_factor when using multiprocessing
    if num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = 2
    
    return DataLoader(**dataloader_kwargs)

# Legacy collate function for backward compatibility
def collate_fn(batch):
    """Legacy collate function - use optimized_collate_fn instead."""
    return optimized_collate_fn(batch)
