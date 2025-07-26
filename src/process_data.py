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
import glob
import tempfile
import sqlite3
import logging
from datetime import datetime
from typing import List, Tuple, Set, Iterator, Optional, Dict, Any
from sklearn.model_selection import train_test_split
import gc
import yaml
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

class DataProcessor:
    """Comprehensive data processor for molecular data following the data analysis plan."""
    
    def __init__(self, config_path: str = "config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.processing_config = self.config.get('processing', {})
        self.raw_data_path = self.processing_config.get('raw_data_path', 'data/raw')
        self.processed_data_path = self.processing_config.get('processed_data_path', 'data/processed')
        self.chunk_size = self.processing_config.get('chunk_size', 10000)
        self.memory_limit_gb = self.processing_config.get('memory_limit_gb', 4.0)
        self.min_mw = self.processing_config.get('min_molecular_weight', 100.0)
        self.max_mw = self.processing_config.get('max_molecular_weight', 600.0)
        self.allowed_atoms = set(self.processing_config.get('allowed_atoms', 
            ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H']))
        self.num_workers = self.processing_config.get('num_workers', 4)
        self.use_multiprocessing = self.processing_config.get('use_multiprocessing', True)
        
        # Processing options
        self.remove_salts = self.processing_config.get('remove_salts', True)
        self.neutralize_charges = self.processing_config.get('neutralize_charges', True)
        self.canonicalize = self.processing_config.get('canonicalize', True)
        self.deduplicate = self.processing_config.get('deduplicate', True)
        
        # Split ratios
        self.train_ratio = self.processing_config.get('train_ratio', 0.8)
        self.val_ratio = self.processing_config.get('val_ratio', 0.1)
        self.test_ratio = self.processing_config.get('test_ratio', 0.1)
        
        # Setup centralized logging
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            from logger import setup_task_logger
            self.logger = setup_task_logger("processing")
        except ImportError:
            # Fallback to basic logging if logger module not found
            self.logger = logging.getLogger("processing")
            self.logger.setLevel(logging.INFO)
            
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            self.logger.addHandler(console_handler)
        
        # Setup processing tools
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
        
        # Create directories
        os.makedirs(self.processed_data_path, exist_ok=True)
        os.makedirs(os.path.join(self.processed_data_path, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.processed_data_path, "val"), exist_ok=True)
        os.makedirs(os.path.join(self.processed_data_path, "test"), exist_ok=True)
        
        # Initialize disk-based deduplication
        if self.deduplicate:
            self.temp_db_path = os.path.join(self.processed_data_path, "dedup_cache.db")
            self.init_deduplication_db()
        else:
            self.temp_db_path = None
            
        # Processing statistics
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'total_molecules': 0,
            'valid_molecules': 0,
            'invalid_molecules': 0,
            'duplicate_molecules': 0,
            'filtered_molecules': 0,
            'errors': {}
        }
        
        self.logger.info(f"DataProcessor initialized with config: {config_path}")
        self.logger.info(f"Processing configuration: {json.dumps(self.processing_config, indent=2)}")
    
    def setup_logging(self):
        """Setup comprehensive logging for data processing."""
        log_file = os.path.join(self.processed_data_path, 'logs', f'processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        # Create logger
        self.logger = logging.getLogger('DataProcessor')
        self.logger.setLevel(logging.INFO)
        
        # File handler for detailed logging
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler for progress
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatters
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_formatter = logging.Formatter('%(message)s')
        
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Data processing started with config: {self.config}")
        
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
        self.logger.info(f"Streaming data from {input_path}")
        
        try:
            molecule_count = 0
            if input_path.endswith('.smi'):
                # Read .smi files line by line
                with open(input_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parts = line.split()
                            if parts:
                                yield parts[0]  # First part is SMILES
                                molecule_count += 1
                        if line_num % 10000 == 0:
                            self.logger.debug(f"Read {line_num:,} lines from {os.path.basename(input_path)}")
            
            elif input_path.endswith('.csv'):
                # Read CSV in chunks
                chunk_num = 0
                for chunk in pd.read_csv(input_path, chunksize=self.chunk_size):
                    chunk_num += 1
                    if 'smiles' in chunk.columns:
                        for smiles in chunk['smiles'].dropna():
                            yield str(smiles)
                            molecule_count += 1
                    elif 'SMILES' in chunk.columns:
                        for smiles in chunk['SMILES'].dropna():
                            yield str(smiles)
                            molecule_count += 1
                    self.logger.debug(f"Processed chunk {chunk_num} from {os.path.basename(input_path)}")
            
            else:
                # Generic text file processing
                with open(input_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            yield line
                            molecule_count += 1
                        if line_num % 10000 == 0:
                            self.logger.debug(f"Read {line_num:,} lines from {os.path.basename(input_path)}")
            
            self.logger.info(f"Loaded {molecule_count:,} molecules from {os.path.basename(input_path)}")
                            
        except Exception as e:
            self.logger.error(f"Error loading {input_path}: {e}")
            self.stats['errors'][os.path.basename(input_path)] = str(e)
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
                if atom.GetSymbol() not in self.allowed_atoms:
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
    
    @staticmethod
    def process_molecule_worker(smiles: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Worker function for processing a single molecule."""
        try:
            # Parse and validate
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or mol.GetNumAtoms() == 0:
                return {'result': None, 'status': 'invalid_parse', 'smiles': smiles}
            
            # Standardize molecule
            if config.get('remove_salts', True):
                remover = SaltRemover.SaltRemover()
                mol = remover.StripMol(mol)
            
            if config.get('neutralize_charges', True):
                normalizer = rdMolStandardize.Normalizer()
                uncharger = rdMolStandardize.Uncharger()
                mol = normalizer.normalize(mol)
                mol = uncharger.uncharge(mol)
            
            # Filter by atom types
            allowed_atoms = set(config.get('allowed_atoms', ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H']))
            for atom in mol.GetAtoms():
                if atom.GetSymbol() not in allowed_atoms:
                    return {'result': None, 'status': 'invalid_atoms', 'smiles': smiles}
            
            # Filter by molecular weight
            min_mw = config.get('min_molecular_weight', 100.0)
            max_mw = config.get('max_molecular_weight', 600.0)
            mw = Descriptors.MolWt(mol)
            if not (min_mw <= mw <= max_mw):
                return {'result': None, 'status': 'invalid_weight', 'smiles': smiles}
            
            # Canonicalize
            if config.get('canonicalize', True):
                canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
                return {'result': canonical_smiles, 'status': 'valid', 'smiles': smiles}
            else:
                return {'result': smiles, 'status': 'valid', 'smiles': smiles}
                
        except Exception as e:
            return {'result': None, 'status': 'error', 'smiles': smiles, 'error': str(e)}
    
    def process_molecules_multiprocessing(self, smiles_list: List[str], file_name: str = "unknown") -> List[str]:
        """Process molecules using multiprocessing for acceleration."""
        self.logger.info(f"Processing {len(smiles_list):,} molecules from {file_name} with {self.num_workers} workers...")
        
        if not self.use_multiprocessing or len(smiles_list) < 1000:
            # Fallback to single-threaded processing for small datasets
            return self._process_single_threaded(smiles_list, file_name)
        
        # Prepare processing config for workers
        worker_config = {
            'remove_salts': self.remove_salts,
            'neutralize_charges': self.neutralize_charges,
            'canonicalize': self.canonicalize,
            'allowed_atoms': list(self.allowed_atoms),
            'min_molecular_weight': self.min_mw,
            'max_molecular_weight': self.max_mw
        }
        
        valid_molecules = []
        status_counts = {'valid': 0, 'invalid_parse': 0, 'invalid_atoms': 0, 'invalid_weight': 0, 'error': 0}
        
        # Use ProcessPoolExecutor for multiprocessing
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit tasks in chunks
            future_to_smiles = {
                executor.submit(self.process_molecule_worker, smiles, worker_config): smiles
                for smiles in smiles_list
            }
            
            # Collect results with progress bar
            with tqdm(total=len(smiles_list), desc=f"Processing {file_name}") as pbar:
                for future in as_completed(future_to_smiles):
                    result = future.result()
                    status = result['status']
                    status_counts[status] += 1
                    
                    if status == 'valid' and result['result'] is not None:
                        valid_molecules.append(result['result'])
                    
                    pbar.update(1)
                    if status_counts['valid'] % 1000 == 0:
                        pbar.set_postfix({
                            'valid': status_counts['valid'],
                            'invalid': status_counts['invalid_parse'] + status_counts['invalid_atoms'] + status_counts['invalid_weight'],
                            'errors': status_counts['error']
                        })
        
        self.logger.info(f"Processing complete for {file_name}: {status_counts}")
        self.stats['valid_molecules'] += status_counts['valid']
        self.stats['invalid_molecules'] += status_counts['invalid_parse'] + status_counts['invalid_atoms'] + status_counts['invalid_weight']
        
        return valid_molecules
    
    def _process_single_threaded(self, smiles_list: List[str], file_name: str = "unknown") -> List[str]:
        """Single-threaded processing for small datasets or debugging."""
        self.logger.info(f"Processing molecules single-threaded from {file_name}...")
        valid_molecules = []
        status_counts = {'valid': 0, 'invalid_parse': 0, 'invalid_atoms': 0, 'invalid_weight': 0, 'error': 0}
        
        with tqdm(total=len(smiles_list), desc=f"Processing {file_name}") as pbar:
            for smiles in smiles_list:
                result = self.process_molecule_worker(smiles, {
                    'remove_salts': self.remove_salts,
                    'neutralize_charges': self.neutralize_charges,
                    'canonicalize': self.canonicalize,
                    'allowed_atoms': list(self.allowed_atoms),
                    'min_molecular_weight': self.min_mw,
                    'max_molecular_weight': self.max_mw
                })
                
                status = result['status']
                status_counts[status] += 1
                
                if status == 'valid' and result['result'] is not None:
                    valid_molecules.append(result['result'])
                
                pbar.update(1)
                if status_counts['valid'] % 1000 == 0:
                    pbar.set_postfix({
                        'valid': status_counts['valid'],
                        'invalid': status_counts['invalid_parse'] + status_counts['invalid_atoms'] + status_counts['invalid_weight'],
                        'errors': status_counts['error']
                    })
        
        self.logger.info(f"Single-threaded processing complete for {file_name}: {status_counts}")
        self.stats['valid_molecules'] += status_counts['valid']
        self.stats['invalid_molecules'] += status_counts['invalid_parse'] + status_counts['invalid_atoms'] + status_counts['invalid_weight']
        
        return valid_molecules
    
    def deduplicate_molecules(self, molecules: List[str], file_name: str = "unknown") -> List[str]:
        """Deduplicate molecules using SQLite."""
        if not self.deduplicate:
            return molecules
        
        self.logger.info(f"Starting deduplication for {file_name}...")
        unique_molecules = []
        seen_count = 0
        
        with tqdm(total=len(molecules), desc=f"Deduplicating {file_name}") as pbar:
            for smiles in molecules:
                if not self.is_duplicate(smiles):
                    self.add_smiles(smiles)
                    unique_molecules.append(smiles)
                else:
                    seen_count += 1
                pbar.update(1)
                if seen_count % 1000 == 0:
                    pbar.set_postfix({'unique': len(unique_molecules), 'duplicates': seen_count})
        
        self.logger.info(f"Deduplication complete for {file_name}: {len(unique_molecules):,} unique, {seen_count:,} duplicates")
        print(f"   ✅ {len(unique_molecules):,} unique molecules, {seen_count:,} duplicates removed")
        return unique_molecules
    
    def process_molecules_streaming(self, smiles_iterator: Iterator[str], output_file: str, file_name: str = "unknown") -> int:
        """Process molecules through the complete pipeline using streaming."""
        self.logger.info(f"Starting pipeline processing for {file_name}")
        
        # Collect all molecules from iterator
        all_molecules = list(smiles_iterator)
        self.logger.info(f"Collected {len(all_molecules):,} molecules for processing from {file_name}")
        
        if len(all_molecules) == 0:
            self.logger.warning(f"No molecules found in {file_name}")
            return 0
        
        # Update stats
        self.stats['total_molecules'] += len(all_molecules)
        
        # Process in parallel
        processed_molecules = self.process_molecules_multiprocessing(all_molecules, file_name)
        
        # Deduplicate if enabled
        unique_molecules = self.deduplicate_molecules(processed_molecules, file_name)
        
        # Save results
        temp_output = output_file + '.tmp'
        if unique_molecules:
            self.save_chunk(unique_molecules, temp_output)
        
        # Update stats
        duplicate_count = len(processed_molecules) - len(unique_molecules)
        self.stats['duplicate_molecules'] += duplicate_count
        self.stats['filtered_molecules'] += (len(all_molecules) - len(processed_molecules))
        
        self.logger.info(f"Processing complete for {file_name}")
        self.logger.info(f"   Valid molecules: {len(unique_molecules):,}")
        self.logger.info(f"   Original molecules: {len(all_molecules):,}")
        self.logger.info(f"   Duplicates removed: {duplicate_count:,}")
        self.logger.info(f"   Invalid/Filtered: {len(all_molecules) - len(processed_molecules):,}")
        
        # Print summary for current file
        print(f"📊 File: {file_name}")
        print(f"   📥 Input: {len(all_molecules):,} molecules")
        print(f"   ✅ Valid: {len(unique_molecules):,} molecules")
        print(f"   🔄 Duplicates: {duplicate_count:,}")
        print(f"   ❌ Invalid: {len(all_molecules) - len(processed_molecules):,}")
        
        return len(unique_molecules)
    
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
    
    def split_and_save_data(self, smiles_list: List[str]):
        """Split data into train/val/test sets and save."""
        print("📊 Splitting and saving data...")
        
        if len(smiles_list) == 0:
            print("❌ No valid molecules to process")
            return
        
        # Shuffle data
        random.shuffle(smiles_list)
        
        # Split data based on configuration
        train_ratio, val_ratio, test_ratio = self.train_ratio, self.val_ratio, self.test_ratio
        
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
            'split_ratios': [self.train_ratio, self.val_ratio, self.test_ratio]
        }
        
        summary_path = os.path.join(self.processed_data_path, "dataset_summary.json")
        with open(summary_path, 'w') as f:
            import json
            json.dump(summary, f, indent=2)
        
        print(f"✅ Data processing complete! Files saved to {self.processed_data_path}")
    
    def process_all_files(self):
        """Process all raw data files using streaming approach with comprehensive progress visualization."""
        # Find all raw files
        raw_files = glob.glob(os.path.join(self.raw_data_path, "**/*.smi"), recursive=True)
        raw_files.extend(glob.glob(os.path.join(self.raw_data_path, "**/*.csv"), recursive=True))
        
        if not raw_files:
            self.logger.error(f"No raw data files found in {self.raw_data_path}")
            print(f"❌ No raw data files found in {self.raw_data_path}")
            return
        
        self.stats['total_files'] = len(raw_files)
        
        # Print processing overview
        print("=" * 80)
        print("🚀 DATA PROCESSING STARTED")
        print("=" * 80)
        print(f"📁 Found {len(raw_files)} raw data files")
        print(f"📊 Using streaming approach with chunk size {self.chunk_size:,}")
        print(f"⚙️  Configuration: {self.processing_config}")
        print("-" * 80)
        
        # Log processing start
        self.logger.info("=" * 50)
        self.logger.info("DATA PROCESSING SESSION STARTED")
        self.logger.info(f"Raw files found: {len(raw_files)}")
        self.logger.info(f"Processing config: {self.processing_config}")
        
        # Temporary file for all processed molecules
        temp_processed = os.path.join(self.processed_data_path, "processed_molecules.txt")
        
        # Clean up any existing temporary files
        if os.path.exists(temp_processed):
            os.remove(temp_processed)
        
        total_processed = 0
        file_stats = []
        
        # Process each file sequentially to avoid memory issues
        for file_idx, file_path in enumerate(raw_files, 1):
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            
            print(f"\n📂 Processing file {file_idx}/{len(raw_files)}: {file_name}")
            print(f"   📊 Size: {file_size:.1f} MB")
            print("   " + "─" * 50)
            
            self.logger.info(f"Processing file {file_idx}/{len(raw_files)}: {file_name} ({file_size:.1f} MB)")
            
            # Stream molecules from file
            smiles_iterator = self.load_and_parse_molecules_streaming(file_path)
            
            # Process in streaming fashion
            file_processed = self.process_molecules_streaming(smiles_iterator, temp_processed, file_name)
            total_processed += file_processed
            
            file_stats.append({
                'file': file_name,
                'size_mb': file_size,
                'valid_molecules': file_processed
            })
            
            self.stats['processed_files'] += 1
            
            # Force garbage collection after each file
            gc.collect()
        
        # Print final summary
        print("\n" + "=" * 80)
        print("📊 FINAL PROCESSING SUMMARY")
        print("=" * 80)
        
        # File processing summary
        print("\n📁 File Processing Summary:")
        for stat in file_stats:
            print(f"   📄 {stat['file']}: {stat['valid_molecules']:,} molecules ({stat['size_mb']:.1f} MB)")
        
        # Overall statistics
        print(f"\n📊 Overall Statistics:")
        print(f"   📥 Total files: {self.stats['total_files']}")
        print(f"   ✅ Files processed: {self.stats['processed_files']}")
        print(f"   📊 Total molecules: {self.stats['total_molecules']:,}")
        print(f"   ✅ Valid molecules: {self.stats['valid_molecules']:,}")
        print(f"   ❌ Invalid molecules: {self.stats['invalid_molecules']:,}")
        print(f"   🔄 Duplicates removed: {self.stats['duplicate_molecules']:,}")
        print(f"   🔍 Filtered out: {self.stats['filtered_molecules']:,}")
        
        # Processing efficiency
        if self.stats['total_molecules'] > 0:
            valid_rate = (self.stats['valid_molecules'] / self.stats['total_molecules']) * 100
            print(f"   📈 Success rate: {valid_rate:.1f}%")
        
        # Errors summary
        if self.stats['errors']:
            print(f"\n⚠️  Errors encountered:")
            for file_name, error in self.stats['errors'].items():
                print(f"   ❌ {file_name}: {error}")
        
        print("-" * 80)
        
        if total_processed == 0:
            self.logger.error("No valid molecules found across all files")
            print("❌ No valid molecules found - processing aborted")
            return
        
        # Load processed molecules and split
        print("\n📊 Loading final processed molecules...")
        valid_molecules = self.load_processed_chunks(temp_processed)
        
        # Shuffle and split
        print("🔄 Shuffling and splitting data...")
        random.shuffle(valid_molecules)
        self.split_and_save_data(valid_molecules)
        
        # Clean up temporary files
        if os.path.exists(temp_processed):
            os.remove(temp_processed)
        
        # Clean up deduplication database
        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)
        
        # Save processing report
        self.save_processing_report()
        
        print("\n✅ All processing complete!")
        print(f"📋 Processing report saved to: {os.path.join(self.processed_data_path, 'logs')}")
        self.logger.info("Data processing completed successfully")

    def save_processing_report(self):
        """Save comprehensive processing report using centralized logger."""
        from src.logger import log_processing_stats
        
        summary = {
            'total_input_molecules': self.stats['total_molecules'],
            'total_valid_molecules': self.stats['valid_molecules'],
            'total_invalid_molecules': self.stats['invalid_molecules'],
            'total_duplicates': self.stats['duplicate_molecules'],
            'success_rate': (self.stats['valid_molecules'] / max(self.stats['total_molecules'], 1)) * 100,
            'configuration': self.processing_config,
            'statistics': self.stats
        }
        
        # Save processing report
        report_path = os.path.join("logs", "processing", f"processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'configuration': self.processing_config,
                'statistics': self.stats,
                'summary': summary
            }, f, indent=2)
        
        self.logger.info(f"Processing report saved to {report_path}")


def process_data(config_path: str = "config.yaml"):
    """Main function to process molecular data with memory-efficient streaming."""
    processor = DataProcessor(config_path)
    processor.process_all_files()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process molecular data for PanGu Drug Model (Memory-efficient)")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    
    args = parser.parse_args()
    
    process_data(args.config)