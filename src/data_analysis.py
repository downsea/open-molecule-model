#!/usr/bin/env python3
"""
Comprehensive SMILES data analysis for PanGu Drug Model - Optimized for Performance

This module provides detailed analysis of processed molecular data including:
- Parallel molecular property distributions
- Optimized vocabulary analysis
- Streaming sequence length statistics
- Chemical diversity metrics with caching
- Data quality assessment
- Training configuration recommendations
"""

import os
import json
import glob
import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import selfies as sf
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm
import time
import gc
import psutil
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for better performance
import matplotlib
matplotlib.use('Agg')

def calculate_properties_chunk(smiles_chunk: List[str]) -> Dict[str, List[float]]:
    """Calculate properties for a chunk of SMILES strings in parallel."""
    properties = defaultdict(list)
    
    for smiles in smiles_chunk:
        if not smiles:
            continue
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
                
            properties['molecular_weight'].append(Descriptors.MolWt(mol))
            properties['logp'].append(Descriptors.MolLogP(mol))
            properties['num_atoms'].append(mol.GetNumAtoms())
            properties['num_rings'].append(int(Chem.GetSSSR(mol)))
            properties['num_aromatic_rings'].append(rdMolDescriptors.CalcNumAromaticRings(mol))
            properties['num_donors'].append(rdMolDescriptors.CalcNumHBD(mol))
            properties['num_acceptors'].append(rdMolDescriptors.CalcNumHBA(mol))
            properties['tpsa'].append(Descriptors.TPSA(mol))
            properties['qed'].append(Descriptors.qed(mol))
            properties['num_rotatable_bonds'].append(rdMolDescriptors.CalcNumRotatableBonds(mol))
        except Exception:
            continue
    
    return dict(properties)

def calculate_diversity_chunk(smiles_chunk: List[str]) -> Dict[str, Any]:
    """Calculate diversity metrics for a chunk of SMILES strings."""
    scaffolds = []
    element_counts = Counter()
    bond_counts = Counter()
    ring_counts = []
    aromatic_ring_counts = []
    
    for smiles in smiles_chunk:
        if not smiles:
            continue
            
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
                
            # Element analysis
            for atom in mol.GetAtoms():
                element_counts[atom.GetSymbol()] += 1
                
            # Bond analysis
            for bond in mol.GetBonds():
                bond_type = str(bond.GetBondType())
                bond_counts[bond_type] += 1
                
            # Ring analysis
            ring_counts.append(int(Chem.GetSSSR(mol)))
            aromatic_ring_counts.append(rdMolDescriptors.CalcNumAromaticRings(mol))
            
            # Scaffold analysis (Murcko scaffold)
            scaffold = Chem.MurckoDecompose(mol)
            if scaffold:
                scaffold_smiles = Chem.MolToSmiles(scaffold)
                scaffolds.append(scaffold_smiles)
        except Exception:
            continue
    
    return {
        'scaffolds': scaffolds,
        'element_counts': dict(element_counts),
        'bond_counts': dict(bond_counts),
        'ring_counts': ring_counts,
        'aromatic_ring_counts': aromatic_ring_counts
    }

class SMILESDataAnalyzer:
    """Optimized analyzer for processed molecular data with parallel processing."""
    
    def __init__(self, data_path: str = "data/processed", output_path: str = "data/data_report"):
        self.data_path = data_path
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        
        # Cache directory for intermediate results
        self.cache_dir = os.path.join(output_path, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Analysis results storage
        self.results = {}
        self.molecules = []
        self.smiles_list = []
        self.selfies_list = []
        
        # Chemical properties
        self.properties = defaultdict(list)
        
        # Performance settings
        self.num_workers = min(mp.cpu_count(), 8)  # Limit to 8 workers max
        self.chunk_size = 1000  # Molecules per chunk
        self.use_streaming = True  # Use streaming for large datasets
        
        # Checkpoint tracking
        self.checkpoint_file = os.path.join(self.cache_dir, "analysis_checkpoint.json")
        self.progress_file = os.path.join(self.cache_dir, "progress.json")
        
        # Load existing checkpoint if available
        self.completed_steps = self._load_checkpoint()
        
        print(f"🚀 Initialized optimized analyzer with {self.num_workers} workers")
        if self.completed_steps:
            print(f"📋 Resuming from checkpoint: {len(self.completed_steps)} steps completed")
    
    def load_data(self) -> bool:
        """Load all processed data from processed molecules file."""
        print("📊 Loading all processed data...")
        
        # Try to load from compressed file first
        compressed_file = os.path.join(self.data_path, "processed_molecules.txt.lz4")
        uncompressed_file = os.path.join(self.data_path, "processed_molecules.txt")
        
        smiles_list = []
        
        # Load compressed file if available
        if os.path.exists(compressed_file):
            try:
                print("📁 Loading from compressed file...")
                import lz4.frame
                with open(compressed_file, 'rb') as f:
                    compressed_data = f.read()
                decompressed_data = lz4.frame.decompress(compressed_data)
                content = decompressed_data.decode('utf-8')
                smiles_list = [line.strip() for line in content.split('\n') if line.strip()]
                print(f"✅ Loaded {len(smiles_list):,} SMILES from compressed file")
            except Exception as e:
                print(f"⚠️  Error loading compressed file: {e}")
                smiles_list = []
        
        # Fallback to uncompressed file
        if not smiles_list and os.path.exists(uncompressed_file):
            print("📁 Loading from uncompressed file...")
            with open(uncompressed_file, 'r', encoding='utf-8') as f:
                smiles_list = [line.strip() for line in f if line.strip()]
            print(f"✅ Loaded {len(smiles_list):,} SMILES from uncompressed file")
        
        if not smiles_list:
            print("❌ No processed SMILES data found")
            return False
        
        # Convert SMILES to molecules and generate SELFIES
        total_molecules = 0
        print("🔄 Converting SMILES to molecules and generating SELFIES...")
        
        for smiles in tqdm(smiles_list, desc="Processing SMILES"):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    self.molecules.append(mol)
                    self.smiles_list.append(smiles)
                    
                    # Generate SELFIES for vocabulary analysis
                    try:
                        selfies = sf.encoder(smiles)
                        self.selfies_list.append(selfies)
                    except:
                        self.selfies_list.append(None)
                    
                    total_molecules += 1
            except Exception:
                continue
        
        if total_molecules == 0:
            print("❌ No valid molecules loaded")
            return False
        
        print(f"✅ Loaded {total_molecules:,} valid molecules from processed data")
        
        # Load processing summary if available
        summary_path = os.path.join(self.data_path, "processing_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                self.dataset_summary = json.load(f)
                print(f"📊 Processing summary loaded:")
                print(f"   Total processed: {self.dataset_summary.get('total_processed_molecules', 0):,}")
                print(f"   Processing time: {self.dataset_summary.get('processing_timestamp', 'Unknown')}")
        else:
            self.dataset_summary = {}
        
        return True
    
    def _load_checkpoint(self) -> set:
        """Load checkpoint data to resume analysis from interruption."""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                    return set(checkpoint.get('completed_steps', []))
            except Exception as e:
                print(f"⚠️  Error loading checkpoint: {e}")
        return set()
    
    def _save_checkpoint(self, step: str, data: Any = None) -> None:
        """Save checkpoint after completing each analysis step."""
        self.completed_steps.add(step)
        checkpoint = {
            'completed_steps': list(self.completed_steps),
            'timestamp': time.time(),
            'step_data': data or {}
        }
        
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️  Error saving checkpoint: {e}")
    
    def _save_intermediate_results(self, step_name: str, data: Any) -> None:
        """Save intermediate results immediately to disk."""
        step_file = os.path.join(self.cache_dir, f"{step_name}_results.json")
        try:
            with open(step_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️  Error saving intermediate results for {step_name}: {e}")
    
    def _load_intermediate_results(self, step_name: str) -> Any:
        """Load intermediate results from cache."""
        step_file = os.path.join(self.cache_dir, f"{step_name}_results.json")
        if os.path.exists(step_file):
            try:
                with open(step_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Error loading intermediate results for {step_name}: {e}")
        return None
    
    def _cache_data(self, data_name: str, data: Any) -> None:
        """Cache data to disk for memory efficiency."""
        cache_file = os.path.join(self.cache_dir, f"{data_name}_cache.pkl")
        try:
            torch.save(data, cache_file)
        except Exception as e:
            print(f"⚠️  Error caching data {data_name}: {e}")
    
    def _load_cached_data(self, data_name: str) -> Any:
        """Load cached data from disk."""
        cache_file = os.path.join(self.cache_dir, f"{data_name}_cache.pkl")
        if os.path.exists(cache_file):
            try:
                return torch.load(cache_file)
            except Exception as e:
                print(f"⚠️  Error loading cached data {data_name}: {e}")
        return None
    
    def analyze_molecular_properties(self) -> Dict:
        """Analyze molecular properties using parallel processing with caching."""
        step_name = "molecular_properties"
        
        # Check if already completed
        if step_name in self.completed_steps:
            print("📋 Molecular properties analysis already completed, loading from cache...")
            cached_results = self._load_intermediate_results(step_name)
            if cached_results:
                self.results['molecular_properties'] = cached_results
                return cached_results
        
        print("🔬 Analyzing molecular properties with parallel processing...")
        start_time = time.time()
        
        if not self.molecules:
            print("❌ No molecules to analyze")
            return {}
        
        # Convert molecules to SMILES for parallel processing (avoid pickling issues)
        smiles_chunks = [
            self.smiles_list[i:i + self.chunk_size]
            for i in range(0, len(self.smiles_list), self.chunk_size)
        ]
        
        print(f"📊 Processing {len(self.smiles_list):,} molecules in {len(smiles_chunks)} chunks using {self.num_workers} workers")
        
        # Process chunks in parallel with progress checkpointing
        all_properties = defaultdict(list)
        valid_molecules = 0
        
        # Resume from partial progress if available
        progress_file = os.path.join(self.cache_dir, f"{step_name}_progress.json")
        processed_chunks = set()
        
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                    processed_chunks = set(progress.get('processed_chunks', []))
                    all_properties.update(progress.get('partial_results', {}))
                    valid_molecules = progress.get('valid_molecules', 0)
                    print(f"📋 Resuming molecular properties analysis from {len(processed_chunks)} processed chunks")
            except Exception as e:
                print(f"⚠️  Error loading progress: {e}")
        
        remaining_chunks = [
            (i, chunk) for i, chunk in enumerate(smiles_chunks)
            if i not in processed_chunks
        ]
        
        if remaining_chunks:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit remaining chunks
                futures = [
                    executor.submit(calculate_properties_chunk, chunk)
                    for _, chunk in remaining_chunks
                ]
                
                # Collect results with progress checkpointing
                with tqdm(total=len(remaining_chunks), desc="Property analysis", unit="chunk") as pbar:
                    for chunk_idx, (original_idx, _) in enumerate(remaining_chunks):
                        try:
                            chunk_properties = futures[chunk_idx].result()
                            
                            # Merge results
                            for prop_name, values in chunk_properties.items():
                                all_properties[prop_name].extend(values)
                                if prop_name == 'molecular_weight':  # Count valid molecules once
                                    valid_molecules += len(values)
                            
                            # Save progress after each chunk
                            processed_chunks.add(original_idx)
                            progress_data = {
                                'processed_chunks': list(processed_chunks),
                                'partial_results': dict(all_properties),
                                'valid_molecules': valid_molecules
                            }
                            
                            try:
                                with open(progress_file, 'w') as f:
                                    json.dump(progress_data, f, indent=2, default=str)
                            except Exception as e:
                                print(f"⚠️  Error saving progress: {e}")
                            
                            pbar.update(1)
                            pbar.set_postfix({'valid_molecules': f"{valid_molecules:,}"})
                            
                        except Exception as e:
                            print(f"⚠️  Error processing chunk: {e}")
                            pbar.update(1)
        
        # Clean up progress file
        if os.path.exists(progress_file):
            try:
                os.remove(progress_file)
            except Exception:
                pass
        
        # Calculate statistics using vectorized operations
        stats = {}
        for prop_name, values in all_properties.items():
            if values:
                try:
                    values_array = np.array(values)
                    stats[prop_name] = {
                        'mean': float(np.mean(values_array)),
                        'median': float(np.median(values_array)),
                        'std': float(np.std(values_array)),
                        'min': float(np.min(values_array)),
                        'max': float(np.max(values_array)),
                        'percentiles': {
                            '5th': float(np.percentile(values_array, 5)),
                            '25th': float(np.percentile(values_array, 25)),
                            '75th': float(np.percentile(values_array, 75)),
                            '95th': float(np.percentile(values_array, 95))
                        }
                    }
                except Exception as e:
                    print(f"⚠️  Error calculating statistics for {prop_name}: {e}")
        
        processing_time = time.time() - start_time
        molecules_per_second = valid_molecules / max(processing_time, 0.001)
        
        # Save results immediately
        self.results['molecular_properties'] = stats
        self._save_intermediate_results(step_name, stats)
        self._save_checkpoint(step_name, {
            'valid_molecules': valid_molecules,
            'processing_time': processing_time,
            'molecules_per_second': molecules_per_second
        })
        
        # Store properties for visualization
        self.properties = dict(all_properties)
        
        print(f"✅ Analyzed properties for {valid_molecules:,} molecules in {processing_time:.1f}s")
        print(f"⚡ Performance: {molecules_per_second:.0f} molecules/second")
        
        return stats
    
    def analyze_vocabulary(self) -> Dict:
        """Analyze SELFIES vocabulary with caching."""
        step_name = "vocabulary"
        
        # Check if already completed
        if step_name in self.completed_steps:
            print("📋 Vocabulary analysis already completed, loading from cache...")
            cached_results = self._load_intermediate_results(step_name)
            if cached_results:
                self.results['vocabulary'] = cached_results
                return cached_results
        
        print("📝 Analyzing SELFIES vocabulary...")
        
        # Filter valid SELFIES
        valid_selfies = [s for s in self.selfies_list if s is not None]
        
        if not valid_selfies:
            print("❌ No valid SELFIES strings found")
            return {}
            
        # Token analysis
        all_tokens = []
        for selfies in valid_selfies:
            tokens = sf.split_selfies(selfies)
            all_tokens.extend(tokens)
            
        token_counts = Counter(all_tokens)
        vocab_size = len(token_counts)
        
        # Sequence length analysis
        sequence_lengths = [len(list(sf.split_selfies(s))) for s in valid_selfies]
        
        vocab_analysis = {
            'vocab_size': vocab_size,
            'total_tokens': len(all_tokens),
            'unique_tokens': vocab_size,
            'most_common_tokens': token_counts.most_common(20),
            'sequence_length_stats': {
                'mean': float(np.mean(sequence_lengths)),
                'median': float(np.median(sequence_lengths)),
                'min': int(np.min(sequence_lengths)),
                'max': int(np.max(sequence_lengths)),
                'percentiles': {
                    '5th': float(np.percentile(sequence_lengths, 5)),
                    '95th': float(np.percentile(sequence_lengths, 95)),
                    '99th': float(np.percentile(sequence_lengths, 99))
                }
            },
            'token_frequency_distribution': dict(token_counts)
        }
        
        # Save results immediately
        self.results['vocabulary'] = vocab_analysis
        self._save_intermediate_results(step_name, vocab_analysis)
        self._save_checkpoint(step_name)
        
        print(f"✅ Analyzed vocabulary: {vocab_size:,} unique tokens")
        return vocab_analysis
    
    def analyze_chemical_diversity(self) -> Dict:
        """Analyze chemical diversity using parallel processing with caching."""
        step_name = "chemical_diversity"
        
        # Check if already completed
        if step_name in self.completed_steps:
            print("📋 Chemical diversity analysis already completed, loading from cache...")
            cached_results = self._load_intermediate_results(step_name)
            if cached_results:
                self.results['chemical_diversity'] = cached_results
                return cached_results
        
        print("🧪 Analyzing chemical diversity with parallel processing...")
        start_time = time.time()
        
        if not self.molecules:
            print("❌ No molecules to analyze")
            return {}
        
        # Convert molecules to SMILES for parallel processing (avoid pickling issues)
        smiles_chunks = [
            self.smiles_list[i:i + self.chunk_size]
            for i in range(0, len(self.smiles_list), self.chunk_size)
        ]
        
        print(f"📊 Processing {len(self.smiles_list):,} molecules in {len(smiles_chunks)} chunks")
        
        # Process chunks in parallel with progress checkpointing
        all_scaffolds = []
        combined_element_counts = Counter()
        combined_bond_counts = Counter()
        all_ring_counts = []
        all_aromatic_ring_counts = []
        
        # Resume from partial progress if available
        progress_file = os.path.join(self.cache_dir, f"{step_name}_progress.json")
        processed_chunks = set()
        
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                    processed_chunks = set(progress.get('processed_chunks', []))
                    all_scaffolds.extend(progress.get('all_scaffolds', []))
                    combined_element_counts.update(progress.get('element_counts', {}))
                    combined_bond_counts.update(progress.get('bond_counts', {}))
                    all_ring_counts.extend(progress.get('ring_counts', []))
                    all_aromatic_ring_counts.extend(progress.get('aromatic_ring_counts', []))
                    print(f"📋 Resuming chemical diversity analysis from {len(processed_chunks)} processed chunks")
            except Exception as e:
                print(f"⚠️  Error loading progress: {e}")
        
        remaining_chunks = [
            (i, chunk) for i, chunk in enumerate(smiles_chunks)
            if i not in processed_chunks
        ]
        
        if remaining_chunks:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit remaining chunks
                futures = [
                    executor.submit(calculate_diversity_chunk, chunk)
                    for _, chunk in remaining_chunks
                ]
                
                # Collect results with progress checkpointing
                with tqdm(total=len(remaining_chunks), desc="Diversity analysis", unit="chunk") as pbar:
                    for chunk_idx, (original_idx, _) in enumerate(remaining_chunks):
                        try:
                            chunk_result = futures[chunk_idx].result()
                            
                            # Merge results
                            all_scaffolds.extend(chunk_result['scaffolds'])
                            
                            # Combine counters
                            for element, count in chunk_result['element_counts'].items():
                                combined_element_counts[element] += count
                            
                            for bond_type, count in chunk_result['bond_counts'].items():
                                combined_bond_counts[bond_type] += count
                            
                            all_ring_counts.extend(chunk_result['ring_counts'])
                            all_aromatic_ring_counts.extend(chunk_result['aromatic_ring_counts'])
                            
                            # Save progress after each chunk
                            processed_chunks.add(original_idx)
                            progress_data = {
                                'processed_chunks': list(processed_chunks),
                                'all_scaffolds': all_scaffolds,
                                'element_counts': dict(combined_element_counts),
                                'bond_counts': dict(combined_bond_counts),
                                'ring_counts': all_ring_counts,
                                'aromatic_ring_counts': all_aromatic_ring_counts
                            }
                            
                            try:
                                with open(progress_file, 'w') as f:
                                    json.dump(progress_data, f, indent=2, default=str)
                            except Exception as e:
                                print(f"⚠️  Error saving progress: {e}")
                            
                            pbar.update(1)
                            pbar.set_postfix({'scaffolds': f"{len(all_scaffolds):,}"})
                            
                        except Exception as e:
                            print(f"⚠️  Error processing diversity chunk: {e}")
                            pbar.update(1)
        
        # Clean up progress file
        if os.path.exists(progress_file):
            try:
                os.remove(progress_file)
            except Exception:
                pass
        
        # Calculate final statistics
        scaffold_counts = Counter(all_scaffolds)
        ring_distribution = Counter(all_ring_counts)
        aromatic_ring_distribution = Counter(all_aromatic_ring_counts)
        
        diversity_analysis = {
            'unique_scaffolds': len(scaffold_counts),
            'most_common_scaffolds': scaffold_counts.most_common(15),
            'element_distribution': dict(combined_element_counts),
            'bond_distribution': dict(combined_bond_counts),
            'ring_distribution': dict(ring_distribution),
            'aromatic_ring_distribution': dict(aromatic_ring_distribution),
            'scaffold_diversity': len(scaffold_counts) / len(self.molecules) if self.molecules else 0
        }
        
        processing_time = time.time() - start_time
        molecules_per_second = len(self.molecules) / max(processing_time, 0.001)
        
        # Save results immediately
        self.results['chemical_diversity'] = diversity_analysis
        self._save_intermediate_results(step_name, diversity_analysis)
        self._save_checkpoint(step_name, {
            'unique_scaffolds': len(scaffold_counts),
            'processing_time': processing_time,
            'molecules_per_second': molecules_per_second
        })
        
        print(f"✅ Analyzed chemical diversity: {len(scaffold_counts):,} unique scaffolds in {processing_time:.1f}s")
        print(f"⚡ Performance: {molecules_per_second:.0f} molecules/second")
        
        return diversity_analysis
    
    def analyze_data_quality(self) -> Dict:
        """Analyze data quality and validity with caching."""
        step_name = "data_quality"
        
        # Check if already completed
        if step_name in self.completed_steps:
            print("📋 Data quality analysis already completed, loading from cache...")
            cached_results = self._load_intermediate_results(step_name)
            if cached_results:
                self.results['data_quality'] = cached_results
                return cached_results
        
        print("🔍 Analyzing data quality...")
        
        total_molecules = len(self.molecules)
        valid_molecules = 0
        unique_smiles = set()
        invalid_molecules = 0
        
        # Additional quality metrics
        valid_selfies_count = 0
        
        # Use progress tracking for large datasets
        progress_file = os.path.join(self.cache_dir, f"{step_name}_progress.json")
        processed_count = 0
        
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                    processed_count = progress.get('processed_count', 0)
                    valid_molecules = progress.get('valid_molecules', 0)
                    unique_smiles = set(progress.get('unique_smiles', []))
                    invalid_molecules = progress.get('invalid_molecules', 0)
                    valid_selfies_count = progress.get('valid_selfies_count', 0)
                    print(f"📋 Resuming data quality analysis from {processed_count} processed molecules")
            except Exception as e:
                print(f"⚠️  Error loading progress: {e}")
        
        # Resume from where we left off
        start_idx = processed_count
        
        with tqdm(total=len(self.molecules), initial=start_idx, desc="Quality analysis") as pbar:
            for i in range(start_idx, len(self.molecules)):
                mol = self.molecules[i]
                if mol is None:
                    invalid_molecules += 1
                    continue
                    
                try:
                    smiles = Chem.MolToSmiles(mol)
                    if smiles:
                        valid_molecules += 1
                        unique_smiles.add(smiles)
                        
                        # Check SELFIES validity
                        if self.selfies_list[i] is not None:
                            valid_selfies_count += 1
                    else:
                        invalid_molecules += 1
                except:
                    invalid_molecules += 1
                
                # Save progress periodically
                if (i + 1) % 1000 == 0:
                    progress_data = {
                        'processed_count': i + 1,
                        'valid_molecules': valid_molecules,
                        'unique_smiles': list(unique_smiles),
                        'invalid_molecules': invalid_molecules,
                        'valid_selfies_count': valid_selfies_count
                    }
                    try:
                        with open(progress_file, 'w') as f:
                            json.dump(progress_data, f, indent=2, default=str)
                    except Exception as e:
                        print(f"⚠️  Error saving progress: {e}")
                
                pbar.update(1)
        
        # Clean up progress file
        if os.path.exists(progress_file):
            try:
                os.remove(progress_file)
            except Exception:
                pass
        
        quality_analysis = {
            'total_molecules': total_molecules,
            'valid_molecules': valid_molecules,
            'invalid_molecules': invalid_molecules,
            'validity_rate': valid_molecules / total_molecules if total_molecules > 0 else 0,
            'unique_molecules': len(unique_smiles),
            'duplicate_rate': 1 - (len(unique_smiles) / valid_molecules) if valid_molecules > 0 else 0,
            'valid_selfies_rate': valid_selfies_count / valid_molecules if valid_molecules > 0 else 0,
            'processing_summary': self.dataset_summary
        }
        
        # Save results immediately
        self.results['data_quality'] = quality_analysis
        self._save_intermediate_results(step_name, quality_analysis)
        self._save_checkpoint(step_name)
        
        print(f"✅ Analyzed data quality: {valid_molecules:,} valid molecules ({quality_analysis['validity_rate']:.2%})")
        return quality_analysis
    
    def generate_recommendations(self) -> Dict:
        """Generate training recommendations based on analysis with immediate saving."""
        step_name = "recommendations"
        
        # Check if already completed
        if step_name in self.completed_steps:
            print("📋 Recommendations already generated, loading from cache...")
            cached_results = self._load_intermediate_results(step_name)
            if cached_results:
                self.results['recommendations'] = cached_results
                return cached_results
        
        print("💡 Generating training recommendations...")
        
        recommendations = {}
        
        # Vocabulary-based recommendations
        vocab_analysis = self.results.get('vocabulary', {})
        if vocab_analysis:
            max_length = int(vocab_analysis['sequence_length_stats']['percentiles']['99th'] * 1.2)
            recommendations['max_sequence_length'] = max(128, min(max_length, 512))
            recommendations['vocab_size'] = vocab_analysis['vocab_size']
            recommendations['max_length_for_padding'] = int(vocab_analysis['sequence_length_stats']['max'] * 1.1)
            
        # Batch size recommendations based on molecule complexity
        property_stats = self.results.get('molecular_properties', {})
        if property_stats:
            avg_atoms = property_stats.get('num_atoms', {}).get('mean', 20)
            avg_mw = property_stats.get('molecular_weight', {}).get('mean', 300)
            
            if avg_atoms < 20 and avg_mw < 300:
                recommendations['suggested_batch_size'] = 128
            elif avg_atoms < 40 and avg_mw < 500:
                recommendations['suggested_batch_size'] = 64
            else:
                recommendations['suggested_batch_size'] = 32
                
        # Memory recommendations based on dataset size
        total_molecules = len(self.molecules)
        if total_molecules < 50000:
            recommendations['memory_mode'] = 'in_memory'
            recommendations['gradient_accumulation_steps'] = 1
        elif total_molecules < 200000:
            recommendations['memory_mode'] = 'streaming'
            recommendations['gradient_accumulation_steps'] = 2
        else:
            recommendations['memory_mode'] = 'streaming'
            recommendations['gradient_accumulation_steps'] = 4
            
        # Model architecture recommendations
        if property_stats:
            avg_rings = property_stats.get('num_rings', {}).get('mean', 2)
            if avg_rings < 2:
                recommendations['suggested_encoder_layers'] = 6
                recommendations['suggested_decoder_layers'] = 4
            elif avg_rings < 4:
                recommendations['suggested_encoder_layers'] = 8
                recommendations['suggested_decoder_layers'] = 6
            else:
                recommendations['suggested_encoder_layers'] = 10
                recommendations['suggested_decoder_layers'] = 8
        
        # Training hyperparameters
        recommendations['learning_rate'] = 1e-4
        recommendations['weight_decay'] = 1e-4
        recommendations['beta_vae'] = 0.001
        recommendations['gradient_clip'] = 1.0
        recommendations['early_stopping_patience'] = 10
        recommendations['min_epochs'] = 50
        recommendations['max_epochs'] = 200
        
        # Save results immediately
        self.results['recommendations'] = recommendations
        self._save_intermediate_results(step_name, recommendations)
        self._save_checkpoint(step_name)
        
        # Save recommendations as separate file for immediate access
        recommendations_file = os.path.join(self.output_path, 'training_recommendations.json')
        try:
            with open(recommendations_file, 'w') as f:
                json.dump(recommendations, f, indent=2, default=str)
            print(f"✅ Saved training recommendations to {recommendations_file}")
        except Exception as e:
            print(f"⚠️  Error saving recommendations: {e}")
        
        return recommendations
    
    def generate_visualizations(self) -> None:
        """Generate visualization plots with immediate saving."""
        step_name = "visualizations"
        
        # Check if already completed
        if step_name in self.completed_steps:
            print("📋 Visualizations already generated, skipping...")
            return
        
        print("📊 Generating visualizations...")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        
        # Generate individual plots for immediate feedback
        plots_generated = []
        
        # 1. Molecular weight distribution
        if self.properties.get('molecular_weight'):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(self.properties['molecular_weight'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title('Molecular Weight Distribution')
            ax.set_xlabel('Molecular Weight (Da)')
            ax.set_ylabel('Frequency')
            ax.axvline(x=100, color='red', linestyle='--', alpha=0.5, label='Min filter')
            ax.axvline(x=600, color='red', linestyle='--', alpha=0.5, label='Max filter')
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, 'molecular_weight_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
            plots_generated.append('molecular_weight_distribution.png')
            print("✅ Generated molecular weight distribution plot")
        
        # 2. LogP distribution
        if self.properties.get('logp'):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(self.properties['logp'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
            ax.set_title('LogP Distribution')
            ax.set_xlabel('LogP')
            ax.set_ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, 'logp_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
            plots_generated.append('logp_distribution.png')
            print("✅ Generated LogP distribution plot")
        
        # 3. Sequence length distribution
        vocab_analysis = self.results.get('vocabulary', {})
        if vocab_analysis.get('sequence_length_stats'):
            lengths = [len(list(sf.split_selfies(s))) for s in self.selfies_list if s is not None]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(lengths, bins=50, alpha=0.7, color='salmon', edgecolor='black')
            ax.set_title('SELFIES Sequence Length Distribution')
            ax.set_xlabel('Sequence Length')
            ax.set_ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, 'sequence_length_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
            plots_generated.append('sequence_length_distribution.png')
            print("✅ Generated sequence length distribution plot")
        
        # 4. Element distribution
        diversity_analysis = self.results.get('chemical_diversity', {})
        if diversity_analysis.get('element_distribution'):
            elements = list(diversity_analysis['element_distribution'].keys())
            counts = list(diversity_analysis['element_distribution'].values())
            top_elements = sorted(zip(elements, counts), key=lambda x: x[1], reverse=True)[:10]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar([e[0] for e in top_elements], [e[1] for e in top_elements], color='teal')
            ax.set_title('Element Distribution (Top 10)')
            ax.set_xlabel('Element')
            ax.set_ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, 'element_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
            plots_generated.append('element_distribution.png')
            print("✅ Generated element distribution plot")
        
        # 5. Comprehensive overview
        fig, axes = plt.subplots(3, 3, figsize=(20, 18))
        fig.suptitle('Comprehensive SMILES Data Analysis Report', fontsize=18, fontweight='bold')
        
        # Fill the comprehensive plot with available data
        # 1. Molecular weight distribution
        if self.properties.get('molecular_weight'):
            axes[0, 0].hist(self.properties['molecular_weight'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Molecular Weight Distribution')
            axes[0, 0].set_xlabel('Molecular Weight (Da)')
            axes[0, 0].set_ylabel('Frequency')
        
        # 2. LogP distribution
        if self.properties.get('logp'):
            axes[0, 1].hist(self.properties['logp'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[0, 1].set_title('LogP Distribution')
            axes[0, 1].set_xlabel('LogP')
            axes[0, 1].set_ylabel('Frequency')
        
        # 3. Sequence length distribution
        if vocab_analysis.get('sequence_length_stats'):
            lengths = [len(list(sf.split_selfies(s))) for s in self.selfies_list if s is not None]
            axes[0, 2].hist(lengths, bins=50, alpha=0.7, color='salmon', edgecolor='black')
            axes[0, 2].set_title('SELFIES Sequence Length Distribution')
            axes[0, 2].set_xlabel('Sequence Length')
            axes[0, 2].set_ylabel('Frequency')
        
        # 4. Number of atoms distribution
        if self.properties.get('num_atoms'):
            axes[1, 0].hist(self.properties['num_atoms'], bins=30, alpha=0.7, color='gold', edgecolor='black')
            axes[1, 0].set_title('Number of Atoms Distribution')
            axes[1, 0].set_xlabel('Number of Atoms')
            axes[1, 0].set_ylabel('Frequency')
        
        # 5. QED distribution
        if self.properties.get('qed'):
            axes[1, 1].hist(self.properties['qed'], bins=50, alpha=0.7, color='plum', edgecolor='black')
            axes[1, 1].set_title('QED Distribution')
            axes[1, 1].set_xlabel('QED Score')
            axes[1, 1].set_ylabel('Frequency')
        
        # 6. Element distribution
        if diversity_analysis.get('element_distribution'):
            elements = list(diversity_analysis['element_distribution'].keys())
            counts = list(diversity_analysis['element_distribution'].values())
            top_elements = sorted(zip(elements, counts), key=lambda x: x[1], reverse=True)[:10]
            axes[1, 2].bar([e[0] for e in top_elements], [e[1] for e in top_elements], color='teal')
            axes[1, 2].set_title('Element Distribution (Top 10)')
            axes[1, 2].set_xlabel('Element')
            axes[1, 2].set_ylabel('Count')
            axes[1, 2].tick_params(axis='x', rotation=45)
        
        # 7. Ring distribution
        if diversity_analysis.get('ring_distribution'):
            ring_data = list(diversity_analysis['ring_distribution'].items())
            ring_sizes, ring_counts = zip(*sorted(ring_data))
            axes[2, 0].bar(ring_sizes, ring_counts, color='orange')
            axes[2, 0].set_title('Ring Count Distribution')
            axes[2, 0].set_xlabel('Number of Rings')
            axes[2, 0].set_ylabel('Frequency')
        
        # 8. TPSA distribution
        if self.properties.get('tpsa'):
            axes[2, 1].hist(self.properties['tpsa'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[2, 1].set_title('Topological Polar Surface Area (TPSA)')
            axes[2, 1].set_xlabel('TPSA (Å²)')
            axes[2, 1].set_ylabel('Frequency')
        
        # 9. Rotatable bonds distribution
        if self.properties.get('num_rotatable_bonds'):
            axes[2, 2].hist(self.properties['num_rotatable_bonds'], bins=30, alpha=0.7, color='lightsteelblue', edgecolor='black')
            axes[2, 2].set_title('Number of Rotatable Bonds')
            axes[2, 2].set_xlabel('Rotatable Bonds')
            axes[2, 2].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'comprehensive_data_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        plots_generated.append('comprehensive_data_analysis.png')
        
        # Save checkpoint
        self._save_checkpoint(step_name, {'plots_generated': plots_generated})
        print(f"✅ Generated {len(plots_generated)} visualization plots")
    
    def generate_detailed_report(self) -> None:
        """Generate comprehensive analysis report."""
        print("📋 Generating comprehensive report...")
        
        report_path = os.path.join(self.output_path, 'data_analysis_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Comprehensive SMILES Data Analysis Report\n\n")
            f.write("Generated by PanGu Drug Model Data Analyzer - Updated for Processed Data\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            quality = self.results.get('data_quality', {})
            f.write(f"- **Total Molecules**: {quality.get('total_molecules', 0):,}\n")
            f.write(f"- **Valid Molecules**: {quality.get('valid_molecules', 0):,}\n")
            f.write(f"- **Validity Rate**: {quality.get('validity_rate', 0):.2%}\n")
            f.write(f"- **Unique Molecules**: {quality.get('unique_molecules', 0):,}\n")
            
            vocab = self.results.get('vocabulary', {})
            f.write(f"- **Vocabulary Size**: {vocab.get('vocab_size', 0):,}\n")
            f.write(f"- **Max Sequence Length**: {vocab.get('sequence_length_stats', {}).get('max', 0)}\n")
            
            # Processing summary
            if 'dataset_splits' in quality:
                splits = quality['dataset_splits']
                f.write(f"- **Total Processed**: {splits.get('total_processed_molecules', 0):,} molecules\n")
                f.write(f"- **Processing Timestamp**: {splits.get('processing_timestamp', 'Unknown')}\n")
            
            # Molecular Properties
            f.write("\n## Molecular Properties\n\n")
            props = self.results.get('molecular_properties', {})
            for prop_name, stats in props.items():
                f.write(f"### {prop_name.replace('_', ' ').title()}\n")
                f.write(f"- **Mean**: {stats['mean']:.2f}\n")
                f.write(f"- **Median**: {stats['median']:.2f}\n")
                f.write(f"- **Range**: {stats['min']:.2f} - {stats['max']:.2f}\n")
                f.write(f"- **Standard Deviation**: {stats['std']:.2f}\n")
                f.write(f"- **5th percentile**: {stats['percentiles']['5th']:.2f}\n")
                f.write(f"- **95th percentile**: {stats['percentiles']['95th']:.2f}\n\n")
            
            # Chemical Diversity
            f.write("## Chemical Diversity\n\n")
            diversity = self.results.get('chemical_diversity', {})
            f.write(f"- **Unique Scaffolds**: {diversity.get('unique_scaffolds', 0):,}\n")
            f.write(f"- **Scaffold Diversity**: {diversity.get('scaffold_diversity', 0):.3f}\n")
            
            f.write("\n### Top 10 Most Common Scaffolds\n\n")
            for scaffold, count in diversity.get('most_common_scaffolds', [])[:10]:
                f.write(f"- `{scaffold}`: {count:,} occurrences\n")
            
            f.write("\n### Element Distribution\n\n")
            elements = diversity.get('element_distribution', {})
            for element, count in sorted(elements.items(), key=lambda x: x[1], reverse=True):
                f.write(f"- **{element}**: {count:,} atoms\n")
            
            # Training Recommendations
            f.write("\n## Training Recommendations\n\n")
            recs = self.results.get('recommendations', {})
            for key, value in recs.items():
                f.write(f"- **{key.replace('_', ' ').title()}**: `{value}`\n")
            
            # Data Quality Issues
            f.write("\n## Data Quality Issues\n\n")
            if quality.get('invalid_molecules', 0) > 0:
                f.write(f"- **Invalid Molecules**: {quality['invalid_molecules']:,}\n")
            if quality.get('duplicate_rate', 0) > 0:
                f.write(f"- **Duplicate Rate**: {quality['duplicate_rate']:.2%}\n")
            if quality.get('valid_selfies_rate', 0) < 1.0:
                f.write(f"- **SELFIES Validity Rate**: {quality['valid_selfies_rate']:.2%}\n")
            
        print("✅ Generated comprehensive report")
    
    def save_analysis(self) -> None:
        """Save analysis results to JSON file."""
        results_path = os.path.join(self.output_path, 'analysis_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print("✅ Saved analysis results")
    
    def run_analysis(self) -> Dict:
        """Run complete data analysis pipeline."""
        print("🚀 Starting comprehensive SMILES data analysis...")
        
        if not self.load_data():
            print("❌ Failed to load data")
            return {}
            
        # Run all analysis components
        self.analyze_data_quality()
        self.analyze_molecular_properties()
        self.analyze_vocabulary()
        self.analyze_chemical_diversity()
        self.generate_recommendations()
        self.generate_visualizations()
        self.generate_detailed_report()
        self.save_analysis()
        
        print("✅ Analysis complete!")
        print(f"📁 Results saved to: {self.output_path}")
        
        return self.results


def analyze_data(data_path: str = "data/processed", output_path: str = "data/data_report") -> Dict:
    """Main function to run data analysis."""
    analyzer = SMILESDataAnalyzer(data_path, output_path)
    return analyzer.run_analysis()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze processed SMILES data for PanGu Drug Model")
    parser.add_argument("--data-path", default="data/processed", help="Path to processed data directory")
    parser.add_argument("--output-path", default="data/data_report", help="Output directory for reports")
    
    args = parser.parse_args()
    
    results = analyze_data(args.data_path, args.output_path)
    print(f"📊 Analysis complete! Check {args.output_path} for detailed reports.")