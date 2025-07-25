#!/usr/bin/env python3
"""
Hyperparameter search script for PanGu Drug Model.

Usage:
    python hyperparameter_search.py --config config.yaml --trials 20
    python hyperparameter_search.py --learning-rate 1e-4 1e-3 5e-4 --hidden-dim 256 512 --batch-size 32 64
"""

import yaml
import argparse
import itertools
import os
import json
from datetime import datetime
from typing import List, Dict, Any
import torch
import numpy as np

from src.config import Config
from src.train import train

def generate_random_search_space(base_config: Dict[str, Any], n_trials: int) -> List[Dict[str, Any]]:
    """Generate random hyperparameter search space."""
    import random
    
    search_space = []
    
    for i in range(n_trials):
        config = Config.from_dict(base_config)
        
        # Random hyperparameters
        config.training.learning_rate = 10 ** random.uniform(-5, -3)  # 1e-5 to 1e-3
        config.model.hidden_dim = random.choice([128, 256, 512])
        config.model.latent_dim = random.choice([64, 128, 256])
        config.data.batch_size = random.choice([16, 32, 64])
        config.training.beta = random.choice([0.0001, 0.001, 0.01])
        config.optimization.weight_decay = random.choice([0.0, 1e-5, 1e-4])
        config.training.gradient_clip = random.choice([0.0, 0.5, 1.0])
        
        search_space.append(config)
    
    return search_space

def generate_grid_search_space(base_config: Dict[str, Any], param_grid: Dict[str, List]) -> List[Dict[str, Any]]:
    """Generate grid search space from parameter grid."""
    
    # Convert parameter names to nested dict structure
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    
    search_space = []
    
    for combination in itertools.product(*param_values):
        config = Config.from_dict(base_config)
        
        # Map flat parameters to nested structure
        for name, value in zip(param_names, combination):
            if name == 'lr':
                config.training.learning_rate = value
            elif name == 'hidden_dim':
                config.model.hidden_dim = value
            elif name == 'latent_dim':
                config.model.latent_dim = value
            elif name == 'batch_size':
                config.data.batch_size = value
            elif name == 'beta':
                config.training.beta = value
            elif name == 'weight_decay':
                config.optimization.weight_decay = value
            elif name == 'encoder_layers':
                config.model.num_encoder_layers = value
            elif name == 'decoder_layers':
                config.model.num_decoder_layers = value
            elif name == 'encoder_heads':
                config.model.num_encoder_heads = value
            elif name == 'decoder_heads':
                config.model.num_decoder_heads = value
        
        search_space.append(config)
    
    return search_space

def run_hyperparameter_search(args):
    """Run hyperparameter search."""
    
    # Load base configuration
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Create search space
    if args.random_search:
        search_space = generate_random_search_space(base_config, args.trials)
        search_type = "random"
    else:
        param_grid = {}
        
        if args.learning_rate:
            param_grid['lr'] = [float(x) for x in args.learning_rate]
        if args.hidden_dim:
            param_grid['hidden_dim'] = [int(x) for x in args.hidden_dim]
        if args.latent_dim:
            param_grid['latent_dim'] = [int(x) for x in args.latent_dim]
        if args.batch_size:
            param_grid['batch_size'] = [int(x) for x in args.batch_size]
        if args.beta:
            param_grid['beta'] = [float(x) for x in args.beta]
        if args.weight_decay:
            param_grid['weight_decay'] = [float(x) for x in args.weight_decay]
        if args.encoder_layers:
            param_grid['encoder_layers'] = [int(x) for x in args.encoder_layers]
        if args.decoder_layers:
            param_grid['decoder_layers'] = [int(x) for x in args.decoder_layers]
        if args.encoder_heads:
            param_grid['encoder_heads'] = [int(x) for x in args.encoder_heads]
        if args.decoder_heads:
            param_grid['decoder_heads'] = [int(x) for x in args.decoder_heads]
        
        if not param_grid:
            # Default grid search
            param_grid = {
                'lr': [1e-4, 5e-4, 1e-3],
                'hidden_dim': [256, 512],
                'batch_size': [16, 32, 64],
                'beta': [0.0001, 0.001, 0.01]
            }
        
        search_space = generate_grid_search_space(base_config, param_grid)
        search_type = "grid"
    
    print(f"Running {search_type} search with {len(search_space)} trials")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"hyperparameter_search_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    results = []
    
    for i, config in enumerate(search_space):
        print(f"\nTrial {i+1}/{len(search_space)}")
        print(f"Configuration: {config}")
        
        # Create trial-specific config
        trial_config = Config.from_dict(config.to_dict())
        trial_config.paths.log_dir = f"{results_dir}/trial_{i+1}"
        trial_config.paths.checkpoint_path = f"{results_dir}/trial_{i+1}/model.pt"
        trial_config.training.num_epochs = min(trial_config.training.num_epochs, args.epochs)
        
        try:
            # Run training
            final_config = train(trial_config)
            
            # Load final results
            checkpoint_path = final_config.paths.checkpoint_path
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                final_loss = checkpoint.get('loss', float('inf'))
            else:
                final_loss = float('inf')
            
            # Record results
            result = {
                'trial': i+1,
                'config': final_config.to_dict(),
                'final_loss': final_loss,
                'search_type': search_type
            }
            
            results.append(result)
            
            # Save intermediate results
            with open(f"{results_dir}/results.json", 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"Trial {i+1} completed with loss: {final_loss:.4f}")
            
        except Exception as e:
            print(f"Trial {i+1} failed: {str(e)}")
            result = {
                'trial': i+1,
                'config': trial_config.to_dict(),
                'error': str(e),
                'final_loss': float('inf')
            }
            results.append(result)
    
    # Sort results by loss
    results.sort(key=lambda x: x['final_loss'])
    
    # Save final results
    with open(f"{results_dir}/final_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print top 5 configurations
    print("\n" + "="*50)
    print("TOP 5 CONFIGURATIONS:")
    print("="*50)
    
    for i, result in enumerate(results[:5]):
        print(f"\n#{i+1} - Loss: {result['final_loss']:.4f}")
        config = result['config']
        print(f"  LR: {config['training']['learning_rate']}")
        print(f"  Hidden: {config['model']['hidden_dim']}")
        print(f"  Latent: {config['model']['latent_dim']}")
        print(f"  Batch: {config['data']['batch_size']}")
        print(f"  Beta: {config['training']['beta']}")
    
    print(f"\nResults saved to {results_dir}/")
    return results

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter search for PanGu Drug Model')
    
    # Search configuration
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Base configuration file')
    parser.add_argument('--random-search', action='store_true',
                        help='Use random search instead of grid search')
    parser.add_argument('--trials', type=int, default=10,
                        help='Number of random trials (for random search)')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Maximum epochs per trial (for speed)')
    
    # Grid search parameters
    parser.add_argument('--learning-rate', nargs='+', type=float,
                        help='Learning rates to try')
    parser.add_argument('--hidden-dim', nargs='+', type=int,
                        help='Hidden dimensions to try')
    parser.add_argument('--latent-dim', nargs='+', type=int,
                        help='Latent dimensions to try')
    parser.add_argument('--batch-size', nargs='+', type=int,
                        help='Batch sizes to try')
    parser.add_argument('--beta', nargs='+', type=float,
                        help='KL divergence weights to try')
    parser.add_argument('--weight-decay', nargs='+', type=float,
                        help='Weight decay values to try')
    parser.add_argument('--encoder-layers', nargs='+', type=int,
                        help='Number of encoder layers to try')
    parser.add_argument('--decoder-layers', nargs='+', type=int,
                        help='Number of decoder layers to try')
    parser.add_argument('--encoder-heads', nargs='+', type=int,
                        help='Number of encoder heads to try')
    parser.add_argument('--decoder-heads', nargs='+', type=int,
                        help='Number of decoder heads to try')
    
    args = parser.parse_args()
    
    run_hyperparameter_search(args)

if __name__ == "__main__":
    main()