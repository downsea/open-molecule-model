#!/usr/bin/env python3
"""
Molecule optimization script using guided diffusion.

This script demonstrates how to use a trained GraphDiT model
with a critic model to optimize molecules for specific properties.
"""

import argparse
import json
import os
from src.graph_dit import (
    load_generator,
    CriticModel,
    GuidedDiffusionOptimizer,
    PropertyFunctions,
    GraphDiTConfig,
    ConfigManager
)


def main():
    parser = argparse.ArgumentParser(description='Optimize molecules with GraphDiT')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to GraphDiT model checkpoint')
    parser.add_argument('--critic-checkpoint', type=str, required=True,
                        help='Path to critic model checkpoint')
    parser.add_argument('--config', type=str, default='configs/graph_dit_default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--input-smiles', type=str, required=True,
                        help='Input SMILES string to optimize')
    parser.add_argument('--property', type=str, choices=['logp', 'qed', 'molecular_weight', 'tpsa'],
                        required=True, help='Property to optimize')
    parser.add_argument('--target', type=float, required=True,
                        help='Target property value')
    parser.add_argument('--num-steps', type=int, default=50,
                        help='Number of optimization steps')
    parser.add_argument('--guidance-scale', type=float, default=1.0,
                        help='Guidance scale for optimization')
    parser.add_argument('--similarity-constraint', type=float, default=0.7,
                        help='Minimum similarity to original molecule')
    parser.add_argument('--output', type=str, default='optimization_results.json',
                        help='Output file for optimization results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigManager()
    if os.path.exists(args.config):
        config = config_manager.load_config(args.config)
    else:
        config = GraphDiTConfig()
    
    # Map property names to functions
    property_functions = {
        'logp': PropertyFunctions.logp,
        'qed': PropertyFunctions.qed,
        'molecular_weight': PropertyFunctions.molecular_weight,
        'tpsa': PropertyFunctions.tpsa
    }
    
    print("Loading models...")
    
    # Load generator
    generator = load_generator(
        checkpoint_path=args.checkpoint,
        config=config,
        device=args.device
    )
    
    # Load critic model
    critic_model = CriticModel(
        node_dim=config.model.node_dim,
        edge_dim=config.model.edge_dim,
        hidden_dim=config.model.hidden_dim // 2,
        num_layers=3
    )
    
    if os.path.exists(args.critic_checkpoint):
        checkpoint = torch.load(args.critic_checkpoint, map_location=args.device)
        critic_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create optimizer
    optimizer = GuidedDiffusionOptimizer(
        generator=generator,
        critic_model=critic_model,
        device=args.device
    )
    
    print(f"Optimizing molecule: {args.input_smiles}")
    print(f"Target property: {args.property} = {args.target}")
    
    # Run optimization
    result = optimizer.optimize_molecule(
        smiles=args.input_smiles,
        property_function=property_functions[args.property],
        property_target=args.target,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        similarity_constraint=args.similarity_constraint
    )
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nOptimization Results:")
    print(f"  Original: {result['original']['smiles']}")
    print(f"    {args.property}: {result['original']['property_value']:.3f}")
    
    if result.get('optimized'):
        opt = result['optimized']
        print(f"  Optimized: {opt['smiles']}")
        print(f"    {args.property}: {opt['property_value']:.3f}")
        print(f"    Similarity: {opt['similarity']:.3f}")
        print(f"    Improvement: {opt['property_value'] - result['original']['property_value']:.3f}")
    else:
        print("  Optimization failed")
    
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    import torch
    main()