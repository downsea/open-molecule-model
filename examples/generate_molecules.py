#!/usr/bin/env python3
"""
Molecule generation script using trained GraphDiT model.

This script demonstrates how to use a trained GraphDiT model
to generate novel molecules.
"""

import argparse
import json
import os
from src.graph_dit import (
    load_generator,
    GraphDiTConfig,
    ConfigManager,
    MoleculeEvaluator
)


def main():
    parser = argparse.ArgumentParser(description='Generate molecules with GraphDiT')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/graph_dit_default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of molecules to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--output', type=str, default='generated_molecules.json',
                        help='Output file for generated molecules')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate generated molecules')
    
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigManager()
    if os.path.exists(args.config):
        config = config_manager.load_config(args.config)
    else:
        config = GraphDiTConfig()
    
    print("Loading model...")
    
    # Load generator
    generator = load_generator(
        checkpoint_path=args.checkpoint,
        config=config,
        device=args.device
    )
    
    print(f"Generating {args.num_samples} molecules...")
    
    # Generate molecules
    molecules = generator.generate_batch(
        num_samples=args.num_samples,
        temperature=args.temperature,
        batch_size=32
    )
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(molecules, f, indent=2)
    
    print(f"Generated {len(molecules)} molecules saved to {args.output}")
    
    # Evaluate if requested
    if args.evaluate:
        print("Evaluating generated molecules...")
        evaluator = MoleculeEvaluator()
        
        metrics = evaluator.evaluate_generation(
            molecules,
            save_path=args.output.replace('.json', '_metrics.json')
        )
        
        print("\nEvaluation Metrics:")
        print(f"  Validity: {metrics['validity']:.3f}")
        print(f"  Chemical Validity: {metrics['chemical_validity']:.3f}")
        print(f"  Uniqueness: {metrics['uniqueness']:.3f}")
        print(f"  Novelty: {metrics['novelty']:.3f}")
        print(f"  Valid Molecules: {metrics['valid_molecules']}")
        
        if 'molecular_weight' in metrics:
            print(f"  Avg Molecular Weight: {metrics['molecular_weight']['mean']:.2f} ± {metrics['molecular_weight']['std']:.2f}")
            print(f"  Avg LogP: {metrics['logp']['mean']:.2f} ± {metrics['logp']['std']:.2f}")
            print(f"  Avg QED: {metrics['qed']['mean']:.3f} ± {metrics['qed']['std']:.3f}")


if __name__ == '__main__':
    main()