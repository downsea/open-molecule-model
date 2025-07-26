#!/usr/bin/env python3
"""
Configuration updater based on data analysis results

This module automatically updates training configurations based on
comprehensive SMILES data analysis.
"""

import os
import json
import yaml
from typing import Dict, Any

class ConfigUpdater:
    """Updates training configurations based on data analysis results."""
    
    def __init__(self, analysis_path: str = "data/data_report/analysis_results.json", 
                 config_path: str = "config.yaml"):
        self.analysis_path = analysis_path
        self.config_path = config_path
        
    def load_analysis_results(self) -> Dict[str, Any]:
        """Load analysis results from JSON file."""
        if not os.path.exists(self.analysis_path):
            print(f"❌ Analysis file not found: {self.analysis_path}")
            print("Run './bootstrap.sh --analyze' first to generate analysis results")
            return {}
            
        with open(self.analysis_path, 'r') as f:
            return json.load(f)
    
    def load_config(self) -> Dict[str, Any]:
        """Load current configuration."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def save_config(self, config: Dict[str, Any], output_path: str) -> None:
        """Save updated configuration."""
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    
    def update_config_from_analysis(self, config: Dict[str, Any], 
                                  analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration based on analysis results."""
        
        # Get recommendations from analysis
        recommendations = analysis.get('recommendations', {})
        vocabulary = analysis.get('vocabulary', {})
        properties = analysis.get('molecular_properties', {})
        quality = analysis.get('data_quality', {})
        
        updated_config = config.copy()
        
        # Update data configuration
        if vocabulary.get('sequence_length_stats'):
            max_len = int(vocabulary['sequence_length_stats']['percentiles']['99th'] * 1.2)
            updated_config['data']['max_length'] = max(max_len, 128)
            
        # Update model configuration
        if vocabulary.get('vocab_size'):
            vocab_size = vocabulary['vocab_size']
            updated_config['model']['output_dim'] = vocab_size
            
        # Update training configuration based on data size
        total_molecules = quality.get('total_molecules', 0)
        if total_molecules < 50000:
            updated_config['data']['batch_size'] = 32
            updated_config['training']['learning_rate'] = 5e-4
        elif total_molecules < 200000:
            updated_config['data']['batch_size'] = 64
            updated_config['training']['learning_rate'] = 1e-4
        else:
            updated_config['data']['batch_size'] = 128
            updated_config['training']['learning_rate'] = 5e-5
            
        # Adjust based on molecule complexity
        if properties.get('num_atoms', {}).get('mean'):
            avg_atoms = properties['num_atoms']['mean']
            if avg_atoms > 30:
                # Large molecules need smaller batches
                updated_config['data']['batch_size'] = max(8, updated_config['data']['batch_size'] // 2)
                updated_config['model']['hidden_dim'] = min(512, updated_config['model']['hidden_dim'] * 1.5)
            elif avg_atoms < 15:
                # Small molecules can use larger batches
                updated_config['data']['batch_size'] = min(256, updated_config['data']['batch_size'] * 2)
                
        # Memory recommendations based on data size
        if total_molecules > 100000:
            updated_config['data']['use_streaming'] = True
            updated_config['data']['cache_in_memory'] = False
            updated_config['system']['num_workers'] = 0
            updated_config['training']['gradient_accumulation_steps'] = 4
        else:
            updated_config['data']['use_streaming'] = False
            updated_config['data']['cache_in_memory'] = True
            updated_config['system']['num_workers'] = 4
            
        # Update number of epochs based on dataset size
        if total_molecules < 10000:
            updated_config['training']['num_epochs'] = 20
        elif total_molecules < 50000:
            updated_config['training']['num_epochs'] = 15
        else:
            updated_config['training']['num_epochs'] = 10
            
        return updated_config
    
    def create_memory_efficient_config(self, config: Dict[str, Any], 
                                     analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create memory-efficient configuration."""
        
        memory_config = config.copy()
        
        # Reduce model size for memory efficiency
        memory_config['model']['hidden_dim'] = 128
        memory_config['model']['num_encoder_layers'] = 8
        memory_config['model']['num_encoder_heads'] = 4
        memory_config['model']['num_decoder_layers'] = 4
        memory_config['model']['num_decoder_heads'] = 4
        memory_config['model']['latent_dim'] = 64
        
        # Reduce batch size and use accumulation
        memory_config['data']['batch_size'] = 16
        memory_config['training']['gradient_accumulation_steps'] = 4
        
        # Always use streaming for memory efficiency
        memory_config['data']['use_streaming'] = True
        memory_config['data']['cache_in_memory'] = False
        memory_config['system']['num_workers'] = 0
        
        # Aggressive memory optimization
        memory_config['system']['mixed_precision'] = True
        memory_config['training']['learning_rate'] = 1e-3
        
        return memory_config
    
    def update_configs(self) -> None:
        """Update all configurations based on analysis."""
        print("🔄 Updating configurations based on data analysis...")
        
        analysis = self.load_analysis_results()
        if not analysis:
            return
            
        config = self.load_config()
        
        # Update main configuration
        updated_config = self.update_config_from_analysis(config, analysis)
        self.save_config(updated_config, "config_updated.yaml")
        print("✅ Updated config_updated.yaml")
        
        # Create memory-efficient configuration
        memory_config = self.create_memory_efficient_config(config, analysis)
        self.save_config(memory_config, "config_memory_efficient_updated.yaml")
        print("✅ Updated config_memory_efficient_updated.yaml")
        
        # Print summary of changes
        print("\n📊 Configuration Updates Summary:")
        print("=" * 50)
        
        original_batch = config['data']['batch_size']
        updated_batch = updated_config['data']['batch_size']
        print(f"Batch size: {original_batch} → {updated_batch}")
        
        original_max_len = config['data']['max_length']
        updated_max_len = updated_config['data']['max_length']
        print(f"Max length: {original_max_len} → {updated_max_len}")
        
        original_epochs = config['training']['num_epochs']
        updated_epochs = updated_config['training']['num_epochs']
        print(f"Epochs: {original_epochs} → {updated_epochs}")
        
        print(f"Streaming mode: {updated_config['data']['use_streaming']}")
        print(f"Memory efficient: {True}")

if __name__ == "__main__":
    updater = ConfigUpdater()
    updater.update_configs()