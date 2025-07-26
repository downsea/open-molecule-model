#!/usr/bin/env python3
"""
Comprehensive SMILES data analysis for PanGu Drug Model

This module provides detailed analysis of SMILES data including:
- Molecular property distributions
- Vocabulary analysis
- Sequence length statistics
- Chemical diversity metrics
- Data quality assessment
- Training configuration recommendations
"""

import os
import json
import pickle
import glob
import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import selfies as sf
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

class SMILESDataAnalyzer:
    """Comprehensive analyzer for SMILES molecular data."""
    
    def __init__(self, data_path: str = "data/processed", output_path: str = "data/data_report"):
        self.data_path = data_path
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        
        # Analysis results storage
        self.results = {}
        self.molecules = []
        self.smiles_list = []
        self.selfies_list = []
        
        # Chemical properties
        self.properties = defaultdict(list)
        
    def load_data(self) -> None:
        """Load processed data files."""
        print("📊 Loading processed data...")
        
        processed_files = glob.glob(os.path.join(self.data_path, "*.pt"))
        if not processed_files:
            print(f"❌ No processed files found in {self.data_path}")
            return False
            
        total_files = len(processed_files)
        total_molecules = 0
        
        for i, file_path in enumerate(processed_files):
            try:
                print(f"📁 Loading {i+1}/{total_files}: {os.path.basename(file_path)}")
                data = torch.load(file_path, weights_only=False)
                
                for mol in data:
                    if mol is not None:
                        self.molecules.append(mol)
                        smiles = Chem.MolToSmiles(mol)
                        self.smiles_list.append(smiles)
                        
                        # Generate SELFIES for vocabulary analysis
                        try:
                            selfies = sf.encoder(smiles)
                            self.selfies_list.append(selfies)
                        except:
                            self.selfies_list.append(None)
                            
                        total_molecules += 1
                            
            except Exception as e:
                print(f"⚠️  Error loading {file_path}: {e}")
                continue
                
        print(f"✅ Loaded {total_molecules:,} molecules from {total_files} files")
        return True
        
    def analyze_molecular_properties(self) -> Dict:
        """Analyze molecular properties and distributions."""
        print("🔬 Analyzing molecular properties...")
        
        properties = {
            'molecular_weight': [],
            'logp': [],
            'num_atoms': [],
            'num_rings': [],
            'num_aromatic_rings': [],
            'num_donors': [],
            'num_acceptors': [],
            'tpsa': [],
            'qed': []
        }
        
        valid_molecules = 0
        for mol in self.molecules:
            try:
                if mol is None:
                    continue
                    
                # Basic molecular descriptors
                properties['molecular_weight'].append(Descriptors.MolWt(mol))
                properties['logp'].append(Descriptors.MolLogP(mol))
                properties['num_atoms'].append(mol.GetNumAtoms())
                properties['num_rings'].append(Chem.GetSSSR(mol))
                properties['num_aromatic_rings'].append(rdMolDescriptors.CalcNumAromaticRings(mol))
                properties['num_donors'].append(rdMolDescriptors.CalcNumHBD(mol))
                properties['num_acceptors'].append(rdMolDescriptors.CalcNumHBA(mol))
                properties['tpsa'].append(Descriptors.TPSA(mol))
                properties['qed'].append(Descriptors.qed(mol))
                
                valid_molecules += 1
                
            except Exception as e:
                continue
                
        self.properties = properties
        
        # Statistical summaries
        stats = {}
        for prop_name, values in properties.items():
            if values:
                stats[prop_name] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'percentiles': {
                        '5th': np.percentile(values, 5),
                        '25th': np.percentile(values, 25),
                        '75th': np.percentile(values, 75),
                        '95th': np.percentile(values, 95)
                    }
                }
        
        self.results['molecular_properties'] = stats
        print(f"✅ Analyzed properties for {valid_molecules:,} valid molecules")
        return stats
        
    def analyze_vocabulary(self) -> Dict:
        """Analyze SELFIES vocabulary."""
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
        sequence_lengths = [len(sf.split_selfies(s)) for s in valid_selfies]
        
        vocab_analysis = {
            'vocab_size': vocab_size,
            'total_tokens': len(all_tokens),
            'unique_tokens': vocab_size,
            'most_common_tokens': token_counts.most_common(20),
            'sequence_length_stats': {
                'mean': np.mean(sequence_lengths),
                'median': np.median(sequence_lengths),
                'min': np.min(sequence_lengths),
                'max': np.max(sequence_lengths),
                'percentiles': {
                    '5th': np.percentile(sequence_lengths, 5),
                    '95th': np.percentile(sequence_lengths, 95),
                    '99th': np.percentile(sequence_lengths, 99)
                }
            },
            'token_frequency_distribution': dict(token_counts)
        }
        
        self.results['vocabulary'] = vocab_analysis
        print(f"✅ Analyzed vocabulary: {vocab_size:,} unique tokens")
        return vocab_analysis
        
    def analyze_chemical_diversity(self) -> Dict:
        """Analyze chemical diversity and scaffolds."""
        print("🧪 Analyzing chemical diversity...")
        
        scaffolds = []
        element_counts = Counter()
        bond_counts = Counter()
        
        for mol in self.molecules:
            if mol is None:
                continue
                
            # Element analysis
            for atom in mol.GetAtoms():
                element_counts[atom.GetSymbol()] += 1
                
            # Bond analysis
            for bond in mol.GetBonds():
                bond_type = str(bond.GetBondType())
                bond_counts[bond_type] += 1
                
            # Scaffold analysis (Murcko scaffold)
            try:
                scaffold = Chem.MurckoDecompose(mol)
                if scaffold:
                    scaffold_smiles = Chem.MolToSmiles(scaffold)
                    scaffolds.append(scaffold_smiles)
            except:
                continue
                
        scaffold_counts = Counter(scaffolds)
        
        diversity_analysis = {
            'unique_scaffolds': len(scaffold_counts),
            'most_common_scaffolds': scaffold_counts.most_common(10),
            'element_distribution': dict(element_counts),
            'bond_distribution': dict(bond_counts),
            'scaffold_diversity': len(scaffold_counts) / len(self.molecules) if self.molecules else 0
        }
        
        self.results['chemical_diversity'] = diversity_analysis
        print(f"✅ Analyzed chemical diversity: {len(scaffold_counts):,} unique scaffolds")
        return diversity_analysis
        
    def analyze_data_quality(self) -> Dict:
        """Analyze data quality and validity."""
        print("🔍 Analyzing data quality...")
        
        total_molecules = len(self.molecules)
        valid_molecules = 0
        unique_smiles = set()
        invalid_molecules = 0
        
        for mol in self.molecules:
            if mol is None:
                invalid_molecules += 1
                continue
                
            try:
                smiles = Chem.MolToSmiles(mol)
                if smiles:
                    valid_molecules += 1
                    unique_smiles.add(smiles)
            except:
                invalid_molecules += 1
                
        quality_analysis = {
            'total_molecules': total_molecules,
            'valid_molecules': valid_molecules,
            'invalid_molecules': invalid_molecules,
            'validity_rate': valid_molecules / total_molecules if total_molecules > 0 else 0,
            'unique_molecules': len(unique_smiles),
            'duplicate_rate': 1 - (len(unique_smiles) / valid_molecules) if valid_molecules > 0 else 0
        }
        
        self.results['data_quality'] = quality_analysis
        print(f"✅ Analyzed data quality: {valid_molecules:,} valid molecules ({quality_analysis['validity_rate']:.2%})")
        return quality_analysis
        
    def generate_recommendations(self) -> Dict:
        """Generate training recommendations based on analysis."""
        print("💡 Generating training recommendations...")
        
        recommendations = {}
        
        # Vocabulary-based recommendations
        vocab_analysis = self.results.get('vocabulary', {})
        if vocab_analysis:
            max_length = int(vocab_analysis['sequence_length_stats']['percentiles']['99th'] * 1.2)
            recommendations['max_sequence_length'] = max(128, min(max_length, 512))
            recommendations['vocab_size'] = vocab_analysis['vocab_size']
            
        # Batch size recommendations based on molecule complexity
        property_stats = self.results.get('molecular_properties', {})
        if property_stats:
            avg_atoms = property_stats.get('num_atoms', {}).get('mean', 20)
            if avg_atoms < 15:
                recommendations['suggested_batch_size'] = 64
            elif avg_atoms < 30:
                recommendations['suggested_batch_size'] = 32
            else:
                recommendations['suggested_batch_size'] = 16
                
        # Memory recommendations
        total_molecules = len(self.molecules)
        if total_molecules < 10000:
            recommendations['memory_mode'] = 'in_memory'
        elif total_molecules < 100000:
            recommendations['memory_mode'] = 'streaming'
        else:
            recommendations['memory_mode'] = 'streaming'
            recommendations['gradient_accumulation_steps'] = 4
            
        self.results['recommendations'] = recommendations
        return recommendations
        
    def generate_visualizations(self) -> None:
        """Generate visualization plots."""
        print("📊 Generating visualizations...")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('SMILES Data Analysis Report', fontsize=16, fontweight='bold')
        
        # 1. Molecular weight distribution
        if self.properties.get('molecular_weight'):
            axes[0, 0].hist(self.properties['molecular_weight'], bins=50, alpha=0.7, color='skyblue')
            axes[0, 0].set_title('Molecular Weight Distribution')
            axes[0, 0].set_xlabel('Molecular Weight')
            axes[0, 0].set_ylabel('Frequency')
            
        # 2. LogP distribution
        if self.properties.get('logp'):
            axes[0, 1].hist(self.properties['logp'], bins=50, alpha=0.7, color='lightgreen')
            axes[0, 1].set_title('LogP Distribution')
            axes[0, 1].set_xlabel('LogP')
            axes[0, 1].set_ylabel('Frequency')
            
        # 3. Sequence length distribution
        vocab_analysis = self.results.get('vocabulary', {})
        if vocab_analysis.get('sequence_length_stats'):
            lengths = [len(sf.split_selfies(s)) for s in self.selfies_list if s is not None]
            axes[0, 2].hist(lengths, bins=50, alpha=0.7, color='salmon')
            axes[0, 2].set_title('SELFIES Sequence Length Distribution')
            axes[0, 2].set_xlabel('Sequence Length')
            axes[0, 2].set_ylabel('Frequency')
            
        # 4. Number of atoms distribution
        if self.properties.get('num_atoms'):
            axes[1, 0].hist(self.properties['num_atoms'], bins=30, alpha=0.7, color='gold')
            axes[1, 0].set_title('Number of Atoms Distribution')
            axes[1, 0].set_xlabel('Number of Atoms')
            axes[1, 0].set_ylabel('Frequency')
            
        # 5. QED distribution
        if self.properties.get('qed'):
            axes[1, 1].hist(self.properties['qed'], bins=50, alpha=0.7, color='plum')
            axes[1, 1].set_title('QED Distribution')
            axes[1, 1].set_xlabel('QED Score')
            axes[1, 1].set_ylabel('Frequency')
            
        # 6. Element distribution
        diversity_analysis = self.results.get('chemical_diversity', {})
        if diversity_analysis.get('element_distribution'):
            elements = list(diversity_analysis['element_distribution'].keys())
            counts = list(diversity_analysis['element_distribution'].values())
            top_elements = sorted(zip(elements, counts), key=lambda x: x[1], reverse=True)[:10]
            axes[1, 2].bar([e[0] for e in top_elements], [e[1] for e in top_elements])
            axes[1, 2].set_title('Element Distribution (Top 10)')
            axes[1, 2].set_xlabel('Element')
            axes[1, 2].set_ylabel('Count')
            axes[1, 2].tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'data_analysis_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Generated visualizations")
        
    def generate_report(self) -> None:
        """Generate comprehensive analysis report."""
        print("📋 Generating comprehensive report...")
        
        report_path = os.path.join(self.output_path, 'data_analysis_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# SMILES Data Analysis Report\n\n")
            f.write("Generated by PanGu Drug Model Data Analyzer\n\n")
            
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
            
            # Molecular Properties
            f.write("\n## Molecular Properties\n\n")
            props = self.results.get('molecular_properties', {})
            for prop_name, stats in props.items():
                f.write(f"### {prop_name.replace('_', ' ').title()}\n")
                f.write(f"- **Mean**: {stats['mean']:.2f}\n")
                f.write(f"- **Median**: {stats['median']:.2f}\n")
                f.write(f"- **Range**: {stats['min']:.2f} - {stats['max']:.2f}\n")
                f.write(f"- **Standard Deviation**: {stats['std']:.2f}\n\n")
                
            # Training Recommendations
            f.write("## Training Recommendations\n\n")
            recs = self.results.get('recommendations', {})
            for key, value in recs.items():
                f.write(f"- **{key.replace('_', ' ').title()}**: {value}\n")
                
            # Data Quality Issues
            f.write("\n## Data Quality Issues\n\n")
            if quality.get('invalid_molecules', 0) > 0:
                f.write(f"- **Invalid Molecules**: {quality['invalid_molecules']:,}\n")
            if quality.get('duplicate_rate', 0) > 0:
                f.write(f"- **Duplicate Rate**: {quality['duplicate_rate']:.2%}\n")
                
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
        self.generate_report()
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
    
    parser = argparse.ArgumentParser(description="Analyze SMILES data for PanGu Drug Model")
    parser.add_argument("--data-path", default="data/processed", help="Path to processed data")
    parser.add_argument("--output-path", default="data/data_report", help="Output directory for reports")
    
    args = parser.parse_args()
    
    results = analyze_data(args.data_path, args.output_path)
    print(f"📊 Analysis complete! Check {args.output_path} for detailed reports.")