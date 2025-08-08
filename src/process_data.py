#!/usr/bin/env python3
"""
Basic Data Processing Module for PanGu Drug Model

This module performs basic processing of raw molecular data:
1. Filter out non-SMILES items (basic validation)
2. Remove duplicates using high-performance deduplication
3. Save all valid, unique SMILES to data/processed

No molecular standardization or train/test splitting is performed here.
Those operations are handled by src/data_standardize.py.
"""

import os
import sys
import logging
from datetime import datetime
from typing import List, Iterator, Dict, Any
import gc
import yaml
import json
import glob
import mmap
import hashlib
import time
import sqlite3
from tqdm import tqdm
from rdkit import Chem
from collections import defaultdict

try:
    import lz4.frame
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False
    print("⚠️  LZ4 not available. Install with: pip install lz4")

try:
    from pybloom_live import BloomFilter
    HAS_BLOOM = True
except ImportError:
    HAS_BLOOM = False
    print("⚠️  Bloom filter not available. Install with: pip install pybloom-live")


class HighPerformanceDeduplicator:
    """High-performance deduplication using Bloom filter + SQLite hybrid approach."""
    
    def __init__(self, expected_items: int = 10_000_000, error_rate: float = 0.001):
        self.expected_items = expected_items
        self.error_rate = error_rate
        
        # Initialize Bloom filter if available
        if HAS_BLOOM:
            self.bloom_filter = BloomFilter(capacity=expected_items, error_rate=error_rate)
            self.use_bloom = True
        else:
            self.bloom_filter = None
            self.use_bloom = False
            print("⚠️  Using fallback deduplication without Bloom filter")
        
        # Use in-memory SQLite for definitive storage
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("CREATE TABLE seen_smiles (hash TEXT PRIMARY KEY)")
        self.conn.execute("CREATE INDEX idx_hash ON seen_smiles(hash)")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        
        self.stats = {'total_checked': 0, 'duplicates_found': 0, 'bloom_hits': 0}
    
    def _get_hash(self, smiles: str) -> str:
        """Get MD5 hash of SMILES string."""
        return hashlib.md5(smiles.encode('utf-8')).hexdigest()
    
    def is_duplicate(self, smiles: str) -> bool:
        """Fast duplicate check using Bloom filter + SQLite."""
        self.stats['total_checked'] += 1
        smiles_hash = self._get_hash(smiles)
        
        # Fast negative check with Bloom filter
        if self.use_bloom and smiles_hash not in self.bloom_filter:
            return False
        
        if self.use_bloom:
            self.stats['bloom_hits'] += 1
        
        # Definitive check with SQLite
        cursor = self.conn.execute("SELECT 1 FROM seen_smiles WHERE hash = ? LIMIT 1", (smiles_hash,))
        is_dup = cursor.fetchone() is not None
        
        if is_dup:
            self.stats['duplicates_found'] += 1
        
        return is_dup
    
    def add_smiles(self, smiles: str):
        """Add SMILES to deduplication store."""
        smiles_hash = self._get_hash(smiles)
        
        if self.use_bloom:
            self.bloom_filter.add(smiles_hash)
        
        self.conn.execute("INSERT OR IGNORE INTO seen_smiles (hash) VALUES (?)", (smiles_hash,))
    
    def get_stats(self) -> Dict[str, int]:
        """Get deduplication statistics."""
        return self.stats.copy()
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


class BasicDataProcessor:
    """Basic data processor that only filters invalid SMILES and removes duplicates."""
    
    def __init__(self, config_path: str = "config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.processing_config = self.config.get('processing', {})
        self.raw_data_path = self.processing_config.get('raw_data_path', 'data/raw')
        self.processed_data_path = self.processing_config.get('processed_data_path', 'data/processed')
        self.chunk_size = self.processing_config.get('chunk_size', 100000)
        self.use_compression = self.processing_config.get('use_compression', HAS_LZ4)
        self.use_mmap = self.processing_config.get('use_memory_mapping', True)
        self.progress_update_interval = self.processing_config.get('progress_update_interval', 1000)
        
        # Setup logging
        self.logger = logging.getLogger("basic_processing")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Create directories
        os.makedirs(self.processed_data_path, exist_ok=True)
        
        # Initialize deduplication
        expected_molecules = self.processing_config.get('expected_molecules', 10_000_000)
        self.deduplicator = HighPerformanceDeduplicator(
            expected_items=expected_molecules,
            error_rate=0.001
        )
        
        # Processing statistics
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'total_lines': 0,
            'valid_smiles': 0,
            'invalid_smiles': 0,
            'duplicate_smiles': 0,
            'final_unique_smiles': 0,
            'errors': {}
        }
        
        self.logger.info(f"BasicDataProcessor initialized with config: {config_path}")
    
    def is_valid_smiles(self, smiles: str) -> bool:
        """Basic SMILES validation using RDKit with sanitization check."""
        if not smiles or not smiles.strip():
            return False
        
        try:
            mol = Chem.MolFromSmiles(smiles.strip())
            if mol is None or mol.GetNumAtoms() == 0:
                return False
            
            # Add sanitization check for extra safety
            try:
                Chem.SanitizeMol(mol)
                return True
            except Chem.rdchem.MolSanitizeException:
                # This will catch issues like kekulization failure
                return False
        except Exception:
            return False
    
    def load_and_parse_molecules_mmap(self, input_path: str) -> Iterator[str]:
        """Memory-mapped file processing for large files."""
        self.logger.info(f"Memory-mapped streaming from {input_path}")
        
        try:
            with open(input_path, 'rb') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                    chunk_size = 64 * 1024  # 64KB chunks
                    buffer = b""
                    line_count = 0
                    
                    for i in range(0, len(mmapped_file), chunk_size):
                        chunk = mmapped_file[i:i + chunk_size]
                        buffer += chunk
                        
                        # Process complete lines
                        while b'\n' in buffer:
                            line, buffer = buffer.split(b'\n', 1)
                            line_str = line.decode('utf-8', errors='ignore').strip()
                            
                            if line_str and not line_str.startswith('#'):
                                if input_path.endswith('.smi'):
                                    parts = line_str.split()
                                    if parts:
                                        yield parts[0]  # First part is SMILES
                                        line_count += 1
                                else:
                                    yield line_str
                                    line_count += 1
                    
                    # Process remaining buffer
                    if buffer:
                        line_str = buffer.decode('utf-8', errors='ignore').strip()
                        if line_str:
                            if input_path.endswith('.smi'):
                                parts = line_str.split()
                                if parts:
                                    yield parts[0]
                                    line_count += 1
                            else:
                                yield line_str
                                line_count += 1
                    
                    self.logger.info(f"Memory-mapped loaded {line_count:,} lines from {os.path.basename(input_path)}")
                    
        except Exception as e:
            self.logger.error(f"Error in memory-mapped processing of {input_path}: {e}")
            # Fallback to regular streaming
            yield from self.load_and_parse_molecules_streaming(input_path)
    
    def load_and_parse_molecules_streaming(self, input_path: str) -> Iterator[str]:
        """Stream molecules from files."""
        self.logger.info(f"Streaming data from {input_path}")
        
        try:
            line_count = 0
            if input_path.endswith('.smi'):
                with open(input_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parts = line.split()
                            if parts:
                                yield parts[0]  # First part is SMILES
                                line_count += 1
                        if line_num % 50000 == 0:
                            self.logger.debug(f"Read {line_num:,} lines from {os.path.basename(input_path)}")
            else:
                # Generic text file processing
                with open(input_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            yield line
                            line_count += 1
                        if line_num % 50000 == 0:
                            self.logger.debug(f"Read {line_num:,} lines from {os.path.basename(input_path)}")
            
            self.logger.info(f"Loaded {line_count:,} lines from {os.path.basename(input_path)}")
                            
        except Exception as e:
            self.logger.error(f"Error loading {input_path}: {e}")
            self.stats['errors'][os.path.basename(input_path)] = str(e)
    
    def process_file_streaming(self, file_path: str) -> List[str]:
        """Process a single file and return valid, unique SMILES."""
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        
        self.logger.info(f"Processing {file_name} ({file_size:.1f} MB)")
        
        # Choose loading method based on file size
        if self.use_mmap and file_size > 10:  # Use mmap for files > 10MB
            smiles_iterator = self.load_and_parse_molecules_mmap(file_path)
        else:
            smiles_iterator = self.load_and_parse_molecules_streaming(file_path)
        
        valid_unique_smiles = []
        file_stats = {
            'total_lines': 0,
            'valid_smiles': 0,
            'invalid_smiles': 0,
            'duplicates': 0
        }
        
        # Process in chunks
        chunk = []
        for smiles in smiles_iterator:
            chunk.append(smiles)
            file_stats['total_lines'] += 1
            
            # Process chunk when it reaches chunk_size
            if len(chunk) >= self.chunk_size:
                chunk_results = self._process_chunk(chunk, file_name)
                valid_unique_smiles.extend(chunk_results['unique_smiles'])
                
                # Update stats
                file_stats['valid_smiles'] += chunk_results['valid_count']
                file_stats['invalid_smiles'] += chunk_results['invalid_count']
                file_stats['duplicates'] += chunk_results['duplicate_count']
                
                chunk = []  # Reset chunk
                gc.collect()  # Force garbage collection
        
        # Process remaining chunk
        if chunk:
            chunk_results = self._process_chunk(chunk, file_name)
            valid_unique_smiles.extend(chunk_results['unique_smiles'])
            
            file_stats['valid_smiles'] += chunk_results['valid_count']
            file_stats['invalid_smiles'] += chunk_results['invalid_count']
            file_stats['duplicates'] += chunk_results['duplicate_count']
        
        # Update global stats
        self.stats['total_lines'] += file_stats['total_lines']
        self.stats['valid_smiles'] += file_stats['valid_smiles']
        self.stats['invalid_smiles'] += file_stats['invalid_smiles']
        self.stats['duplicate_smiles'] += file_stats['duplicates']
        
        self.logger.info(f"File {file_name} processed: {len(valid_unique_smiles):,} unique valid SMILES")
        return valid_unique_smiles
    
    def _process_chunk(self, chunk: List[str], file_name: str) -> Dict[str, Any]:
        """Process a chunk of SMILES strings."""
        valid_count = 0
        invalid_count = 0
        duplicate_count = 0
        unique_smiles = []
        
        with tqdm(total=len(chunk), desc=f"Processing {file_name} chunk", 
                 unit="smiles", leave=False) as pbar:
            
            for smiles in chunk:
                # Basic SMILES validation
                if self.is_valid_smiles(smiles):
                    valid_count += 1
                    
                    # Check for duplicates
                    if not self.deduplicator.is_duplicate(smiles):
                        self.deduplicator.add_smiles(smiles)
                        unique_smiles.append(smiles.strip())
                    else:
                        duplicate_count += 1
                else:
                    invalid_count += 1
                
                pbar.update(1)
                if valid_count > 0:
                    pbar.set_postfix({
                        'valid': valid_count,
                        'invalid': invalid_count,
                        'duplicates': duplicate_count,
                        'unique': len(unique_smiles)
                    })
        
        return {
            'unique_smiles': unique_smiles,
            'valid_count': valid_count,
            'invalid_count': invalid_count,
            'duplicate_count': duplicate_count
        }
    
    def save_processed_data(self, all_smiles: List[str]):
        """Save all processed SMILES to file."""
        if not all_smiles:
            print("❌ No valid SMILES to save")
            return
        
        output_file = os.path.join(self.processed_data_path, "processed_molecules.txt")
        
        # Save with optional compression
        if self.use_compression and HAS_LZ4:
            try:
                compressed_file = output_file + '.lz4'
                compressed_data = lz4.frame.compress(
                    '\n'.join(all_smiles).encode('utf-8'),
                    compression_level=1  # Fast compression
                )
                with open(compressed_file, 'wb') as f:
                    f.write(compressed_data)
                print(f"   ✅ Saved {len(all_smiles):,} SMILES to {compressed_file} (compressed)")
                self.logger.info(f"Saved {len(all_smiles):,} SMILES to compressed file")
            except Exception as e:
                self.logger.warning(f"Compression failed, saving uncompressed: {e}")
                # Fallback to uncompressed
                with open(output_file, 'w', encoding='utf-8') as f:
                    for smiles in all_smiles:
                        f.write(smiles + '\n')
                print(f"   ✅ Saved {len(all_smiles):,} SMILES to {output_file} (uncompressed)")
        else:
            # Save uncompressed
            with open(output_file, 'w', encoding='utf-8') as f:
                for smiles in all_smiles:
                    f.write(smiles + '\n')
            print(f"   ✅ Saved {len(all_smiles):,} SMILES to {output_file}")
        
        # Save processing summary
        summary = {
            'total_processed_molecules': len(all_smiles),
            'processing_timestamp': datetime.now().isoformat(),
            'processing_config': self.processing_config,
            'statistics': self.stats
        }
        
        summary_path = os.path.join(self.processed_data_path, "processing_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.stats['final_unique_smiles'] = len(all_smiles)
        print(f"✅ Basic processing complete! {len(all_smiles):,} unique SMILES saved to {self.processed_data_path}")
        self.logger.info(f"Basic processing complete: {len(all_smiles):,} unique SMILES saved")
    
    def process_all_files(self):
        """Process all raw data files."""
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
        print("🔧 BASIC DATA PROCESSING STARTED")
        print("=" * 80)
        print(f"📁 Found {len(raw_files)} raw data files")
        print(f"🎯 Goal: Filter invalid SMILES and remove duplicates")
        print(f"📊 Chunk size: {self.chunk_size:,}")
        print("-" * 80)
        
        all_unique_smiles = []
        
        # Process each file
        for file_idx, file_path in enumerate(raw_files, 1):
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            
            print(f"\n📂 Processing file {file_idx}/{len(raw_files)}: {file_name}")
            print(f"   📊 Size: {file_size:.1f} MB")
            print("   " + "─" * 50)
            
            start_time = time.time()
            file_smiles = self.process_file_streaming(file_path)
            processing_time = time.time() - start_time
            
            if file_smiles:
                all_unique_smiles.extend(file_smiles)
                molecules_per_second = len(file_smiles) / max(processing_time, 0.001)
                print(f"   ⚡ Performance: {molecules_per_second:.0f} SMILES/sec")
            
            self.stats['processed_files'] += 1
            gc.collect()
        
        # Print final summary
        print("\n" + "=" * 80)
        print("📊 BASIC PROCESSING SUMMARY")
        print("=" * 80)
        print(f"📥 Total files processed: {self.stats['processed_files']}")
        print(f"📊 Total lines read: {self.stats['total_lines']:,}")
        print(f"✅ Valid SMILES: {self.stats['valid_smiles']:,}")
        print(f"❌ Invalid SMILES: {self.stats['invalid_smiles']:,}")
        print(f"🔄 Duplicates removed: {self.stats['duplicate_smiles']:,}")
        print(f"🎯 Final unique SMILES: {len(all_unique_smiles):,}")
        
        if self.stats['total_lines'] > 0:
            valid_rate = (self.stats['valid_smiles'] / self.stats['total_lines']) * 100
            print(f"📈 Validity rate: {valid_rate:.1f}%")
        
        # Get deduplication stats
        dedup_stats = self.deduplicator.get_stats()
        if dedup_stats['bloom_hits'] > 0:
            print(f"🚀 Bloom filter accelerated {dedup_stats['bloom_hits']:,} duplicate checks")
        
        print("=" * 80)
        
        # Save all processed data
        self.save_processed_data(all_unique_smiles)
        
        # Cleanup
        self.deduplicator.close()
        
        print("\n✅ Basic processing complete!")
        print("📋 Next steps:")
        print("   1. Run analysis: ./bootstrap.sh --analyze")
        print("   2. Run standardization: ./bootstrap.sh --standardize config_optimized.yaml")
        print("   3. Run training: ./bootstrap.sh --train --config config_optimized.yaml")


def process_data(config_path: str = "config.yaml"):
    """Main function to process molecular data with basic filtering and deduplication."""
    processor = BasicDataProcessor(config_path)
    processor.process_all_files()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Basic processing: filter invalid SMILES and remove duplicates")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    
    args = parser.parse_args()
    
    process_data(args.config)