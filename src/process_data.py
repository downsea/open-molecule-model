import os
import torch
import random
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize
import pandas as pd
from tqdm import tqdm
import glob
import json
import tempfile
import sqlite3
from typing import List, Tuple, Set, Iterator, Optional
from sklearn.model_selection import train_test_split
import gc

# Allowed atoms for drug-like molecules
ALLOWED_ATOMS = {'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H'}

class DataProcessor:
    """Comprehensive data processor for molecular data following the data analysis plan."""
    
    def __init__(self, raw_data_path: str = "data/raw", processed_data_path: str = "data/processed", 
                 chunk_size: int = 10000, memory_limit_gb: float = 4.0):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.chunk_size = chunk_size
        self.memory_limit_gb = memory_limit_gb
        self.remover = SaltRemover.SaltRemover()
        self.normalizer = rdMolStandardize.Normalizer()
        
        # Create directories
        os.makedirs(processed_data_path, exist_ok=True)
        os.makedirs(os.path.join(processed_data_path, "train"), exist_ok=True)
        os.makedirs(os.path.join(processed_data_path, "val"), exist_ok=True)
        os.makedirs(os.path.join(processed_data_path, "test"), exist_ok=True)
        
        # Initialize disk-based deduplication
        self.temp_db_path = os.path.join(processed_data_path, "dedup_cache.db")
        self.init_deduplication_db()
    
    def init_deduplication_db(self):
        """Initialize SQLite database for disk-based deduplication."""
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS seen_smiles (
                smiles TEXT PRIMARY KEY,
                count INTEGER DEFAULT 1
            )
        ''')
        conn.commit()
        conn.close()
    
    def is_duplicate(self, smiles: str) -> bool:
        """Check if SMILES already exists using disk-based storage."""
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM seen_smiles WHERE smiles = ?', (smiles,))
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0
    
    def add_smiles(self, smiles: str):
        """Add SMILES to deduplication database."""
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR IGNORE INTO seen_smiles (smiles) VALUES (?)
        ''', (smiles,))
        conn.commit()
        conn.close()
    
    def load_and_parse_molecules_streaming(self, input_path: str) -> Iterator[str]:
        """Stream molecules from large files to avoid memory issues."""
        print(f"📂 Streaming data from {input_path}...")
        
        try:
            if input_path.endswith('.smi'):
                # Read .smi files line by line
                with open(input_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parts = line.split()
                            if parts:
                                yield parts[0]  # First part is SMILES
            
            elif input_path.endswith('.csv'):
                # Read CSV in chunks
                for chunk in pd.read_csv(input_path, chunksize=self.chunk_size):
                    if 'smiles' in chunk.columns:
                        for smiles in chunk['smiles'].dropna():
                            yield str(smiles)
                    elif 'SMILES' in chunk.columns:
                        for smiles in chunk['SMILES'].dropna():
                            yield str(smiles)
            
            else:
                # Generic text file processing
                with open(input_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            yield line
                            
        except Exception as e:
            print(f"⚠️  Error loading {input_path}: {e}")
            return
    
    def validate_and_parse_smiles(self, smiles: str) -> Chem.Mol:
        """Parse SMILES string and return RDKit molecule if valid."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Basic sanity checks
            if mol.GetNumAtoms() == 0:
                return None
                
            return mol
        except:
            return None
    
    def standardize_molecule(self, mol: Chem.Mol) -> Chem.Mol:
        """Standardize molecule: remove salts, neutralize charges, normalize."""
        try:
            # Remove salts
            mol = self.remover.StripMol(mol)
            
            # Normalize molecule
            mol = self.normalizer.normalize(mol)
            
            # Neutralize charges
            uncharger = rdMolStandardize.Uncharger()
            mol = uncharger.uncharge(mol)
            
            return mol
        except Exception as e:
            print(f"⚠️  Error standardizing molecule: {e}")
            return None
    
    def filter_by_atom_types(self, mol: Chem.Mol) -> bool:
        """Filter molecules by allowed atom types."""
        try:
            for atom in mol.GetAtoms():
                if atom.GetSymbol() not in ALLOWED_ATOMS:
                    return False
            return True
        except:
            return False
    
    def filter_by_molecular_weight(self, mol: Chem.Mol, min_mw: float = 100.0, max_mw: float = 600.0) -> bool:
        """Filter molecules by molecular weight range."""
        try:
            mw = Descriptors.MolWt(mol)
            return min_mw <= mw <= max_mw
        except:
            return False
    
    def canonicalize_smiles(self, mol: Chem.Mol) -> str:
        """Convert molecule to canonical SMILES."""
        try:
            return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        except:
            return None
    
    def process_molecules_streaming(self, smiles_iterator: Iterator[str], output_file: str) -> int:
        """Process molecules through the complete pipeline using streaming."""
        print("🔬 Processing molecules through cleaning pipeline (streaming)...")
        
        valid_molecules = 0
        skipped_invalid = 0
        skipped_atom_type = 0
        skipped_mw = 0
        skipped_duplicate = 0
        
        # Use temporary file for processed results
        temp_output = output_file + '.tmp'
        processed_molecules = []
        
        try:
            for smiles in tqdm(smiles_iterator, desc="Processing molecules"):
                smiles = smiles.strip()
                if not smiles:
                    continue
                
                # Parse and validate
                mol = self.validate_and_parse_smiles(smiles)
                if mol is None:
                    skipped_invalid += 1
                    continue
                
                # Standardize
                mol = self.standardize_molecule(mol)
                if mol is None:
                    skipped_invalid += 1
                    continue
                
                # Filter by atom types
                if not self.filter_by_atom_types(mol):
                    skipped_atom_type += 1
                    continue
                
                # Filter by molecular weight
                if not self.filter_by_molecular_weight(mol):
                    skipped_mw += 1
                    continue
                
                # Canonicalize
                canonical_smiles = self.canonicalize_smiles(mol)
                if canonical_smiles is None:
                    skipped_invalid += 1
                    continue
                
                # Check for duplicates using disk-based storage
                if self.is_duplicate(canonical_smiles):
                    skipped_duplicate += 1
                    continue
                
                self.add_smiles(canonical_smiles)
                processed_molecules.append(canonical_smiles)
                valid_molecules += 1
                
                # Save in chunks to avoid memory issues
                if len(processed_molecules) >= self.chunk_size:
                    self.save_chunk(processed_molecules, temp_output)
                    processed_molecules = []
                    gc.collect()  # Force garbage collection
            
            # Save remaining molecules
            if processed_molecules:
                self.save_chunk(processed_molecules, temp_output)
            
        except Exception as e:
            print(f"⚠️  Error during processing: {e}")
        
        print(f"✅ Processing complete!")
        print(f"   Valid molecules: {valid_molecules:,}")
        print(f"   Skipped invalid: {skipped_invalid:,}")
        print(f"   Skipped atom type: {skipped_atom_type:,}")
        print(f"   Skipped MW: {skipped_mw:,}")
        print(f"   Skipped duplicates: {skipped_duplicate:,}")
        
        return valid_molecules
    
    def save_chunk(self, molecules: List[str], output_file: str):
        """Save a chunk of processed molecules to file."""
        with open(output_file, 'a', encoding='utf-8') as f:
            for mol in molecules:
                f.write(mol + '\n')
    
    def load_processed_chunks(self, temp_file: str) -> List[str]:
        """Load processed molecules from temporary file."""
        molecules = []
        with open(temp_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    molecules.append(line)
        return molecules
    
    def split_and_save_data(self, smiles_list: List[str], split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)):
        """Split data into train/val/test sets and save."""
        print("📊 Splitting and saving data...")
        
        if len(smiles_list) == 0:
            print("❌ No valid molecules to process")
            return
        
        # Shuffle data
        random.shuffle(smiles_list)
        
        # Split data
        train_ratio, val_ratio, test_ratio = split_ratios
        
        # First split: separate test set
        train_val, test = train_test_split(
            smiles_list, 
            test_size=test_ratio, 
            random_state=42
        )
        
        # Second split: separate train and val
        val_size_adjusted = val_ratio / (train_ratio + val_ratio)
        train, val = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            random_state=42
        )
        
        print(f"📈 Dataset split:")
        print(f"   Training: {len(train):,} molecules ({len(train)/len(smiles_list):.1%})")
        print(f"   Validation: {len(val):,} molecules ({len(val)/len(smiles_list):.1%})")
        print(f"   Test: {len(test):,} molecules ({len(test)/len(smiles_list):.1%})")
        
        # Save splits
        splits = {
            'train': train,
            'val': val,
            'test': test
        }
        
        for split_name, molecules in splits.items():
            split_dir = os.path.join(self.processed_data_path, split_name)
            os.makedirs(split_dir, exist_ok=True)
            
            # Convert to RDKit molecules and save
            rdkit_mols = []
            for smiles in molecules:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    rdkit_mols.append(mol)
            
            output_path = os.path.join(split_dir, f"{split_name}_molecules.pt")
            torch.save(rdkit_mols, output_path)
            print(f"   Saved {len(rdkit_mols)} molecules to {output_path}")
        
        # Save summary
        summary = {
            'total_molecules': len(smiles_list),
            'train_size': len(train),
            'val_size': len(val),
            'test_size': len(test),
            'split_ratios': split_ratios
        }
        
        summary_path = os.path.join(self.processed_data_path, "dataset_summary.json")
        with open(summary_path, 'w') as f:
            import json
            json.dump(summary, f, indent=2)
        
        print(f"✅ Data processing complete! Files saved to {self.processed_data_path}")
    
    def process_all_files(self):
        """Process all raw data files using streaming approach."""
        # Find all raw files
        raw_files = glob.glob(os.path.join(self.raw_data_path, "**/*.smi"), recursive=True)
        raw_files.extend(glob.glob(os.path.join(self.raw_data_path, "**/*.csv"), recursive=True))
        
        if not raw_files:
            print(f"❌ No raw data files found in {self.raw_data_path}")
            return
        
        print(f"📁 Found {len(raw_files)} raw data files")
        print(f"📊 Using streaming approach with chunk size {self.chunk_size:,}")
        
        # Temporary file for all processed molecules
        temp_processed = os.path.join(self.processed_data_path, "processed_molecules.txt")
        
        # Clean up any existing temporary files
        if os.path.exists(temp_processed):
            os.remove(temp_processed)
        
        total_processed = 0
        
        # Process each file sequentially to avoid memory issues
        for file_path in raw_files:
            print(f"📂 Streaming {os.path.basename(file_path)}...")
            
            # Stream molecules from file
            smiles_iterator = self.load_and_parse_molecules_streaming(file_path)
            
            # Process in streaming fashion
            file_processed = self.process_molecules_streaming(smiles_iterator, temp_processed)
            total_processed += file_processed
            
            print(f"   ✅ Processed {file_processed:,} valid molecules")
            
            # Force garbage collection after each file
            gc.collect()
        
        print(f"📊 Total valid molecules across all files: {total_processed:,}")
        
        if total_processed == 0:
            print("❌ No valid molecules found")
            return
        
        # Load processed molecules and split
        valid_molecules = self.load_processed_chunks(temp_processed)
        
        # Shuffle and split
        random.shuffle(valid_molecules)
        self.split_and_save_data(valid_molecules)
        
        # Clean up temporary files
        if os.path.exists(temp_processed):
            os.remove(temp_processed)
        
        # Clean up deduplication database
        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)
        
        print("✅ All processing complete!")


def process_data(raw_path: str = "data/raw", processed_path: str = "data/processed", chunk_size: int = 10000):
    """Main function to process molecular data with memory-efficient streaming."""
    processor = DataProcessor(raw_path, processed_path, chunk_size=chunk_size)
    processor.process_all_files()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process molecular data for PanGu Drug Model (Memory-efficient)")
    parser.add_argument("--raw-path", default="data/raw", help="Path to raw data")
    parser.add_argument("--processed-path", default="data/processed", help="Path to save processed data")
    parser.add_argument("--chunk-size", type=int, default=10000, help="Number of molecules to process in each chunk")
    
    args = parser.parse_args()
    
    process_data(args.raw_path, args.processed_path, chunk_size=args.chunk_size)