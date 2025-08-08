#!/usr/bin/env python3
"""
Training script for GraphDiT pre-training.

This script demonstrates how to train a Graph Diffusion Transformer
on molecular datasets for generation tasks.
"""

import argparse
import os
import torch
from src.graph_dit import (
    GraphDiT, 
    CategoricalNoiseScheduler, 
    GraphDiTTrainer,
    ConfigManager,
    get_pretraining_config
)


def main():
    parser = argparse.ArgumentParser(description='Train GraphDiT model')
    parser.add_argument('--config', type=str, default='configs/graph_dit_default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data-path', type=str, default='data/zinc250k',
                        help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory to save logs')
    parser.add_argument('--use-wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    
    args = parser.parse_args()
    
    # Load configuration
    from src.graph_dit.config import UnifiedConfigManager, ConfigManager
    
    if os.path.exists(args.config):
        # Try unified config first, then fall back to direct config
        try:
            config = UnifiedConfigManager.load_unified_config(args.config)
            print("Loaded configuration from unified config file")
        except:
            config_manager = ConfigManager()
            config = config_manager.load_config(args.config)
            print("Loaded configuration from direct config file")
    else:
        config_manager = ConfigManager()
        config = get_pretraining_config()
        config_manager.save_config(config, args.config)
    
    # Update configuration with command line arguments
    config.data.data_path = args.data_path
    config.training.num_epochs = args.epochs
    config.data.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.device = args.device
    config.training.checkpoint_dir = args.save_dir
    config.training.log_dir = args.log_dir
    config.training.use_wandb = args.use_wandb
    
    print("Configuration:")
    print(f"  Data path: {config.data.data_path}")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Batch size: {config.data.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Device: {config.device}")
    
    # Create model and scheduler
    model = GraphDiT(
        node_dim=config.model.node_dim,
        edge_dim=config.model.edge_dim,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        max_time_steps=config.scheduler.num_timesteps
    )
    
    scheduler = CategoricalNoiseScheduler(
        num_node_classes=config.model.node_dim,
        num_edge_classes=config.model.edge_dim,
        num_timesteps=config.scheduler.num_timesteps,
        beta_start=config.scheduler.beta_start,
        beta_end=config.scheduler.beta_end,
        schedule=config.scheduler.schedule
    )
    
    # Create dataloaders first to get dataset info
    from src.graph_dit.data import get_unified_dataloaders
    
    loaders = get_unified_dataloaders(
        config=config,
        noise_scheduler=scheduler,
        data_root='data'
    )
    
    # Get dataset statistics
    train_info = loaders['train'].dataset.base_dataset.get_data_info() if hasattr(loaders['train'].dataset, 'base_dataset') else {}
    val_info = loaders['val'].dataset.base_dataset.get_data_info() if hasattr(loaders['val'].dataset, 'base_dataset') else {}
    test_info = loaders['test'].dataset.base_dataset.get_data_info() if hasattr(loaders['test'].dataset, 'base_dataset') else {}
    
    # Calculate model architecture details
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Model memory estimation (approximate)
    model_memory_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    
    # Print comprehensive model and data information
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE & DATA SUMMARY")
    print("="*80)
    
    print("\nMODEL ARCHITECTURE:")
    print(f"  Model Type:        Graph Diffusion Transformer (GraphDiT)")
    print(f"  Total Parameters:  {total_params:,}")
    print(f"  Trainable Params:  {trainable_params:,}")
    print(f"  Model Memory:      {model_memory_mb:.1f} MB")
    print()
    print(f"  Node Dimension:    {config.model.node_dim}")
    print(f"  Edge Dimension:    {config.model.edge_dim}")
    print(f"  Hidden Dimension:  {config.model.hidden_dim}")
    print(f"  Number of Layers:  {config.model.num_layers}")
    print(f"  Attention Heads:   {config.model.num_heads}")
    print(f"  Max Time Steps:    {config.scheduler.num_timesteps}")
    
    print("\nDATASET STATISTICS:")
    print(f"  Training Set:      {train_info.get('num_graphs', 0):,} molecules")
    print(f"  Validation Set:    {val_info.get('num_graphs', 0):,} molecules")
    print(f"  Test Set:          {test_info.get('num_graphs', 0):,} molecules")
    
    if train_info.get('num_graphs', 0) > 0:
        print(f"  Avg Nodes/Mol:     {train_info.get('avg_nodes_per_graph', 0):.1f}")
        print(f"  Avg Edges/Mol:     {train_info.get('avg_edges_per_graph', 0):.1f}")
        print(f"  Node Features:     {train_info.get('node_features_dim', 0)}")
        print(f"  Edge Features:     {train_info.get('edge_features_dim', 0)}")
    
    print("\nTRAINING CONFIGURATION:")
    print(f"  Batch Size:        {config.data.batch_size}")
    print(f"  Learning Rate:     {config.training.learning_rate}")
    print(f"  Epochs:            {config.training.num_epochs}")
    print(f"  Device:            {config.device}")
    print(f"  Optimizer:         AdamW")
    print(f"  Scheduler:         Cosine Annealing")
    print(f"  Weight Decay:      {config.training.weight_decay}")
    
    # Calculate training statistics
    total_steps = (train_info.get('num_graphs', 0) // config.data.batch_size) * config.training.num_epochs
    print(f"  Total Steps:       ~{total_steps:,}")
    
    print("\n" + "="*80)
    
    trainer = GraphDiTTrainer(
        model=model,
        noise_scheduler=scheduler,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        optimizer=torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        ),
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
            torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate),
            T_max=config.training.num_epochs
        ),
        device=config.device,
        log_dir=config.training.log_dir,
        checkpoint_dir=config.training.checkpoint_dir,
        use_wandb=config.training.use_wandb,
        wandb_project=config.training.wandb_project
    )
    
    # Start training
    print("\nStarting training...")
    trainer.train(
        num_epochs=config.training.num_epochs,
        save_every=config.training.save_every,
        eval_every=config.training.eval_every
    )
    
    print("\nTraining completed!")
    print(f"Best model saved to: {config.training.checkpoint_dir}/best_model.pt")


if __name__ == '__main__':
    main()