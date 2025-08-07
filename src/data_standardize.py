#!/usr/bin/env python3
"""
Data Standardization Module for PanGu Drug Model

This module applies configuration-based filtering to processed SMILES data
and generates final standardized train/test/valid datasets for training.

Pipeline:
1. Load processed SMILES from data/processed 
2. Apply molecular weight, atom type, and other filters from config
3. Standardize molecules (remove salts, neutralize charges, canonicalize)
4. Split into train/val/test sets
5. Save to data/standard for training
"""

import os
import sys
import torch
import random
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize
import pandas as pd
from tqdm import tqdm
import logging
from datetime import datetime
from typing import List, Tuple, Set, Iterator, Optional, Dict, Any
from sklearn.model_selection import train_test_split
import gc
import yaml
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from collections import defaultdict
import time
import psutil

try:
    import lz4.frame
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False
    print("⚠️  LZ4 not available. Install with: pip install lz4")

# Global worker variables for persistent objects
_worker_normalizer = None
_worker_uncharger = None
_worker_remover = None
_worker_allowed_atoms_set = None
_worker_config = None

class DataStandardizer:
    """Standardize processed SMILES data according to configuration filters."""
    
    def __init__(self, config_path: str = "config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Use standardize config section for standardization-specific parameters
        self.standardize_config = self.config.get('standardize', {})
        self.processing_config = self.config.get('processing', {})  # Still use processing for paths
        
        self.processed_data_path = self.processing_config.get('processed_data_path', 'data/processed')
        self.standard_data_path = self.standardize_config.get('standard_data_path', 'data/standard')
        
        # Filtering parameters from standardize config
        self.min_mw = self.standardize_config.get('min_molecular_weight', 100.0)
        self.max_mw = self.standardize_config.get('max_molecular_weight', 600.0)
        self.allowed_atoms = set(self.standardize_config.get('allowed_atoms',
            ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H']))
        self.num_workers = self.standardize_config.get('num_workers', 8)
        self.use_multiprocessing = self.standardize_config.get('use_multiprocessing', True)
        
        # Standardization options
        self.remove_salts = self.standardize_config.get('remove_salts', True)
        self.neutralize_charges = self.standardize_config.get('neutralize_charges', True)
        self.canonicalize = self.standardize_config.get('canonicalize', True)
        
        # Split ratios
        self.train_ratio = self.standardize_config.get('train_ratio', 0.8)
        self.val_ratio = self.standardize_config.get('val_ratio', 0.1)
        self.test_ratio = self.standardize_config.get('test_ratio', 0.1)
        
        # Performance settings
        self.chunk_size = self.standardize_config.get('chunk_size', 50000)
        self.progress_update_interval = self.standardize_config.get('progress_update_interval', 1000)
        
        # Setup logging
        self.logger = logging.getLogger("data_standardizer")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Create directories
        os.makedirs(self.standard_data_path, exist_ok=True)
        os.makedirs(os.path.join(self.standard_data_path, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.standard_data_path, "val"), exist_ok=True)
        os.makedirs(os.path.join(self.standard_data_path, "test"), exist_ok=True)
        
        # Setup standardization tools
        if self.remove_salts:
            self.remover = SaltRemover.SaltRemover()
        else:
            self.remover = None
            
        if self.neutralize_charges:
            self.normalizer = rdMolStandardize.Normalizer()
            self.uncharger = rdMolStandardize.Uncharger()
        else:
            self.normalizer = None
            self.uncharger = None
        
        # Worker pool for persistent objects
        self.worker_pool = None
        self.shared_config = None
        
        # Processing statistics
        self.stats = {
            'total_input_smiles': 0,
            'valid_molecules': 0,
            'invalid_molecules': 0,
            'filtered_by_weight': 0,
            'filtered_by_atoms': 0,
            'standardization_errors': 0,
            'final_molecules': 0
        }
        
        self.logger.info(f"DataStandardizer initialized with config: {config_path}")
        self.logger.info(f"Filters: MW {self.min_mw}-{self.max_mw}, atoms {self.allowed_atoms}")
    
    def _initialize_worker_pool(self):
        """Initialize persistent worker pool with pre-created objects."""
        if self.worker_pool is not None:
            return  # Already initialized
        
        # Create shared configuration
        manager = mp.Manager()
        self.shared_config = manager.dict(self._get_worker_config())
        
        # Initialize worker pool
        self.worker_pool = mp.Pool(
            processes=self.num_workers,
            initializer=self._init_worker,
            initargs=(self.shared_config,)
        )
        self.logger.info(f"Initialized persistent worker pool with {self.num_workers} workers")
    
    def _get_worker_config(self) -> Dict[str, Any]:
        """Get configuration for worker processes."""
        return {
            'remove_salts': self.remove_salts,
            'neutralize_charges': self.neutralize_charges,
            'canonicalize': self.canonicalize,
            'allowed_atoms': list(self.allowed_atoms),
            'min_molecular_weight': self.min_mw,
            'max_molecular_weight': self.max_mw
        }
    
    @staticmethod
    def _init_worker(shared_config):
        """Initialize worker with persistent RDKit objects."""
        global _worker_normalizer, _worker_uncharger, _worker_remover
        global _worker_allowed_atoms_set, _worker_config
        
        _worker_config = dict(shared_config)
        
        if _worker_config.get('remove_salts', True):
            _worker_remover = SaltRemover.SaltRemover()
        
        if _worker_config.get('neutralize_charges', True):
            _worker_normalizer = rdMolStandardize.Normalizer()
            _worker_uncharger = rdMolStandardize.Uncharger()
        
        _worker_allowed_atoms_set = set(_worker_config.get('allowed_atoms', []))
    
    @staticmethod
    def standardize_and_filter_smiles(smiles: str, config: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Worker function to standardize and filter a single SMILES string."""
        global _worker_normalizer, _worker_uncharger, _worker_remover
        global _worker_allowed_atoms_set, _worker_config
        
        # Use global config if available, otherwise fall back to passed config
        if _worker_config is not None:
            config = _worker_config
        elif config is None:
            return {'result': None, 'status': 'error', 'smiles': smiles, 'error': 'No configuration available'}
        
        try:
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or mol.GetNumAtoms() == 0:
                return {'result': None, 'status': 'invalid_parse', 'smiles': smiles}
            
            # Filter by molecular weight
            min_mw = config.get('min_molecular_weight', 100.0)
            max_mw = config.get('max_molecular_weight', 600.0)
            mw = Descriptors.MolWt(mol)
            if not (min_mw <= mw <= max_mw):
                return {'result': None, 'status': 'filtered_weight', 'smiles': smiles, 'weight': mw}
            
            # Filter by atom types using pre-computed set
            if _worker_allowed_atoms_set is not None:
                if any(atom.GetSymbol() not in _worker_allowed_atoms_set for atom in mol.GetAtoms()):
                    return {'result': None, 'status': 'filtered_atoms', 'smiles': smiles}
            else:
                # Fallback for non-persistent mode
                allowed_atoms = set(config.get('allowed_atoms', ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H']))
                if any(atom.GetSymbol() not in allowed_atoms for atom in mol.GetAtoms()):
                    return {'result': None, 'status': 'filtered_atoms', 'smiles': smiles}
            
            # Standardize molecule using pre-initialized objects
            if config.get('remove_salts', True) and _worker_remover is not None:
                mol = _worker_remover.StripMol(mol)
            
            if config.get('neutralize_charges', True):
                if _worker_normalizer is not None and _worker_uncharger is not None:
                    mol = _worker_normalizer.normalize(mol)
                    mol = _worker_uncharger.uncharge(mol)
                else:
                    # Fallback for non-persistent mode
                    normalizer = rdMolStandardize.Normalizer()
                    uncharger = rdMolStandardize.Uncharger()
                    mol = normalizer.normalize(mol)
                    mol = uncharger.uncharge(mol)
            
            # Canonicalize
            if config.get('canonicalize', True):
                canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
                return {'result': canonical_smiles, 'status': 'valid', 'smiles': smiles, 'molecule': mol}
            else:
                return {'result': smiles, 'status': 'valid', 'smiles': smiles, 'molecule': mol}
                
        except Exception as e:
            return {'result': None, 'status': 'standardization_error', 'smiles': smiles, 'error': str(e)}
    
    def load_processed_smiles(self) -> List[str]:
        """Load processed SMILES from compressed or uncompressed files."""
        self.logger.info(f"Loading processed SMILES from {self.processed_data_path}")
        
        # Try to load from compressed file first
        compressed_file = os.path.join(self.processed_data_path, "processed_molecules.txt.lz4")
        uncompressed_file = os.path.join(self.processed_data_path, "processed_molecules.txt")
        
        smiles_list = []
        
        if HAS_LZ4 and os.path.exists(compressed_file):
            try:
                self.logger.info("Loading from compressed file...")
                with open(compressed_file, 'rb') as f:
                    compressed_data = f.read()
                decompressed_data = lz4.frame.decompress(compressed_data)
                content = decompressed_data.decode('utf-8')
                smiles_list = [line.strip() for line in content.split('\n') if line.strip()]
                self.logger.info(f"Loaded {len(smiles_list):,} SMILES from compressed file")
            except Exception as e:
                self.logger.warning(f"Failed to load compressed file: {e}")
                smiles_list = []
        
        if not smiles_list and os.path.exists(uncompressed_file):
            self.logger.info("Loading from uncompressed file...")
            with open(uncompressed_file, 'r', encoding='utf-8') as f:
                smiles_list = [line.strip() for line in f if line.strip()]
            self.logger.info(f"Loaded {len(smiles_list):,} SMILES from uncompressed file")
        
        if not smiles_list:
            # Try loading from train/val/test splits if available
            self.logger.info("Trying to load from existing splits...")
            for split in ['train', 'val', 'test']:
                split_file = os.path.join(self.processed_data_path, split, f"{split}_molecules.pt")
                if os.path.exists(split_file):
                    try:
                        data = torch.load(split_file, weights_only=False)
                        for mol in data:
                            if mol is not None:
                                smiles = Chem.MolToSmiles(mol)
                                if smiles:
                                    smiles_list.append(smiles)
                        self.logger.info(f"Loaded {len(data)} molecules from {split} split")
                    except Exception as e:
                        self.logger.warning(f"Error loading {split_file}: {e}")
        
        if not smiles_list:
            raise FileNotFoundError(f"No processed SMILES data found in {self.processed_data_path}")
        
        self.stats['total_input_smiles'] = len(smiles_list)
        self.logger.info(f"Total SMILES loaded: {len(smiles_list):,}")
        return smiles_list
    
    def standardize_molecules_parallel(self, smiles_list: List[str]) -> List[Tuple[str, Chem.Mol]]:
        """Standardize and filter molecules using parallel processing."""
        self.logger.info(f"Standardizing {len(smiles_list):,} molecules with {self.num_workers} workers...")
        
        if not self.use_multiprocessing or len(smiles_list) < 5000:
            return self._standardize_single_threaded(smiles_list)
        
        # Initialize persistent worker pool
        self._initialize_worker_pool()
        
        valid_molecules = []
        status_counts = {
            'valid': 0, 
            'invalid_parse': 0, 
            'filtered_weight': 0, 
            'filtered_atoms': 0, 
            'standardization_error': 0
        }
        
        try:
            # Process in batches for better memory management
            batch_size = min(10000, max(1000, len(smiles_list) // self.num_workers))
            update_interval = max(self.progress_update_interval, len(smiles_list) // 100)
            processed_count = 0
            
            with tqdm(total=len(smiles_list), desc="Standardizing molecules",
                     unit="mol", unit_scale=True) as pbar:
                
                for i in range(0, len(smiles_list), batch_size):
                    batch = smiles_list[i:i + batch_size]
                    
                    # Process batch using worker pool
                    results = self.worker_pool.map(self.standardize_and_filter_smiles, batch)
                    
                    # Collect results
                    for result in results:
                        status = result['status']
                        status_counts[status] += 1
                        
                        if status == 'valid' and result['result'] is not None:
                            valid_molecules.append((result['result'], result.get('molecule')))
                        
                        processed_count += 1
                        
                        # Update progress bar less frequently
                        if processed_count % update_interval == 0 or processed_count == len(smiles_list):
                            pbar.update(processed_count - pbar.n)
                            pbar.set_postfix({
                                'valid': f"{status_counts['valid']:,}",
                                'filtered': f"{status_counts['filtered_weight'] + status_counts['filtered_atoms']:,}",
                                'errors': f"{status_counts['invalid_parse'] + status_counts['standardization_error']:,}",
                                'rate': f"{processed_count/pbar.format_dict.get('elapsed', 1):.0f}/s"
                            })
                
                # Final update
                pbar.update(len(smiles_list) - pbar.n)
        
        except Exception as e:
            self.logger.error(f"Error in multiprocessing: {e}")
            return self._standardize_single_threaded(smiles_list)
        
        # Update statistics
        self.stats['valid_molecules'] = status_counts['valid']
        self.stats['invalid_molecules'] = status_counts['invalid_parse']
        self.stats['filtered_by_weight'] = status_counts['filtered_weight']
        self.stats['filtered_by_atoms'] = status_counts['filtered_atoms']
        self.stats['standardization_errors'] = status_counts['standardization_error']
        
        self.logger.info(f"Standardization complete: {status_counts}")
        return valid_molecules
    
    def _standardize_single_threaded(self, smiles_list: List[str]) -> List[Tuple[str, Chem.Mol]]:
        """Single-threaded standardization for small datasets or debugging."""
        self.logger.info("Using single-threaded standardization...")
        valid_molecules = []
        status_counts = {
            'valid': 0, 
            'invalid_parse': 0, 
            'filtered_weight': 0, 
            'filtered_atoms': 0, 
            'standardization_error': 0
        }
        
        config = self._get_worker_config()
        
        with tqdm(total=len(smiles_list), desc="Standardizing molecules") as pbar:
            for smiles in smiles_list:
                result = self.standardize_and_filter_smiles(smiles, config)
                
                status = result['status']
                status_counts[status] += 1
                
                if status == 'valid' and result['result'] is not None:
                    # Create molecule object for single-threaded mode
                    mol = Chem.MolFromSmiles(result['result'])
                    if mol is not None:
                        valid_molecules.append((result['result'], mol))
                
                pbar.update(1)
                if status_counts['valid'] > 0:
                    pbar.set_postfix({
                        'valid': status_counts['valid'],
                        'filtered': status_counts['filtered_weight'] + status_counts['filtered_atoms'],
                        'errors': status_counts['invalid_parse'] + status_counts['standardization_error']
                    })
        
        # Update statistics
        self.stats['valid_molecules'] = status_counts['valid']
        self.stats['invalid_molecules'] = status_counts['invalid_parse']
        self.stats['filtered_by_weight'] = status_counts['filtered_weight']
        self.stats['filtered_by_atoms'] = status_counts['filtered_atoms']
        self.stats['standardization_errors'] = status_counts['standardization_error']
        
        self.logger.info(f"Single-threaded standardization complete: {status_counts}")
        return valid_molecules
    
    def split_and_save_data(self, molecules: List[Tuple[str, Chem.Mol]]):
        """Split standardized data into train/val/test sets and save."""
        self.logger.info("Splitting and saving standardized data...")
        
        if len(molecules) == 0:
            self.logger.error("No valid molecules to split")
            return
        
        # Extract SMILES and molecules
        smiles_list = [smiles for smiles, mol in molecules]
        mol_list = [mol for smiles, mol in molecules]
        
        # Shuffle data
        combined = list(zip(smiles_list, mol_list))
        random.shuffle(combined)
        smiles_list, mol_list = zip(*combined)
        
        # Split data based on configuration
        train_ratio, val_ratio, test_ratio = self.train_ratio, self.val_ratio, self.test_ratio
        
        # First split: separate test set
        train_val_smiles, test_smiles, train_val_mols, test_mols = train_test_split(
            smiles_list, mol_list,
            test_size=test_ratio, 
            random_state=42
        )
        
        # Second split: separate train and val
        val_size_adjusted = val_ratio / (train_ratio + val_ratio)
        train_smiles, val_smiles, train_mols, val_mols = train_test_split(
            train_val_smiles, train_val_mols,
            test_size=val_size_adjusted,
            random_state=42
        )
        
        print(f"📈 Standardized dataset split:")
        print(f"   Training: {len(train_smiles):,} molecules ({len(train_smiles)/len(smiles_list):.1%})")
        print(f"   Validation: {len(val_smiles):,} molecules ({len(val_smiles)/len(smiles_list):.1%})")
        print(f"   Test: {len(test_smiles):,} molecules ({len(test_smiles)/len(smiles_list):.1%})")
        
        # Save splits
        splits = {
            'train': (train_smiles, train_mols),
            'val': (val_smiles, val_mols),
            'test': (test_smiles, test_mols)
        }
        
        for split_name, (split_smiles, split_mols) in splits.items():
            split_dir = os.path.join(self.standard_data_path, split_name)
            os.makedirs(split_dir, exist_ok=True)
            
            # Save molecules as PyTorch tensors
            output_path = os.path.join(split_dir, f"{split_name}_molecules.pt")
            torch.save(split_mols, output_path)
            
            # Also save SMILES as text file for reference
            smiles_path = os.path.join(split_dir, f"{split_name}_smiles.txt")
            with open(smiles_path, 'w') as f:
                for smiles in split_smiles:
                    f.write(smiles + '\n')
            
            self.logger.info(f"Saved {len(split_mols)} molecules to {output_path}")
            print(f"   ✅ Saved {len(split_mols)} molecules to {output_path}")
        
        # Save summary
        summary = {
            'total_standardized_molecules': len(molecules),
            'train_size': len(train_smiles),
            'val_size': len(val_smiles),
            'test_size': len(test_smiles),
            'split_ratios': [self.train_ratio, self.val_ratio, self.test_ratio],
            'filtering_config': {
                'min_molecular_weight': self.min_mw,
                'max_molecular_weight': self.max_mw,
                'allowed_atoms': list(self.allowed_atoms),
                'remove_salts': self.remove_salts,
                'neutralize_charges': self.neutralize_charges,
                'canonicalize': self.canonicalize
            },
            'statistics': self.stats
        }
        
        summary_path = os.path.join(self.standard_data_path, "dataset_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.stats['final_molecules'] = len(molecules)
        self.logger.info(f"Standardized data processing complete! Files saved to {self.standard_data_path}")
        print(f"✅ Standardized data processing complete! Files saved to {self.standard_data_path}")
    
    def run_standardization(self):
        """Run the complete standardization pipeline."""
        print("=" * 80)
        print("🧪 DATA STANDARDIZATION STARTED")
        print("=" * 80)
        print(f"📁 Input: {self.processed_data_path}")
        print(f"📁 Output: {self.standard_data_path}")
        print(f"⚙️  Filters: MW {self.min_mw}-{self.max_mw}, atoms {len(self.allowed_atoms)}")
        print(f"🔧 Standardization: salts={self.remove_salts}, charges={self.neutralize_charges}, canonical={self.canonicalize}")
        print("-" * 80)
        
        start_time = time.time()
        
        try:
            # Load processed SMILES
            smiles_list = self.load_processed_smiles()
            
            # Standardize and filter
            valid_molecules = self.standardize_molecules_parallel(smiles_list)
            
            if not valid_molecules:
                self.logger.error("No valid molecules after standardization")
                print("❌ No valid molecules after standardization")
                return
            
            # Split and save
            self.split_and_save_data(valid_molecules)
            
            # Print final summary
            processing_time = time.time() - start_time
            print("\n" + "=" * 80)
            print("📊 STANDARDIZATION SUMMARY")
            print("=" * 80)
            print(f"⏱️  Processing time: {processing_time:.1f} seconds")
            print(f"📥 Input SMILES: {self.stats['total_input_smiles']:,}")
            print(f"✅ Valid molecules: {self.stats['valid_molecules']:,}")
            print(f"❌ Invalid molecules: {self.stats['invalid_molecules']:,}")
            print(f"🔍 Filtered by weight: {self.stats['filtered_by_weight']:,}")
            print(f"🔍 Filtered by atoms: {self.stats['filtered_by_atoms']:,}")
            print(f"⚠️  Standardization errors: {self.stats['standardization_errors']:,}")
            print(f"📊 Final molecules: {self.stats['final_molecules']:,}")
            
            if self.stats['total_input_smiles'] > 0:
                success_rate = (self.stats['final_molecules'] / self.stats['total_input_smiles']) * 100
                print(f"📈 Success rate: {success_rate:.1f}%")
            
            print("=" * 80)
            print("✅ Standardization complete!")
            
        except Exception as e:
            self.logger.error(f"Standardization failed: {e}")
            print(f"❌ Standardization failed: {e}")
            raise
        finally:
            self._cleanup_resources()
    
    def _cleanup_resources(self):
        """Clean up worker pool resources."""
        try:
            if self.worker_pool is not None:
                self.worker_pool.close()
                self.worker_pool.join()
                self.worker_pool = None
                self.logger.info("Worker pool cleaned up")
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")


def standardize_data(config_path: str = "config.yaml"):
    """Main function to standardize processed molecular data."""
    standardizer = DataStandardizer(config_path)
    standardizer.run_standardization()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Standardize processed molecular data for PanGu Drug Model")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    
    args = parser.parse_args()
    
    standardize_data(args.config)