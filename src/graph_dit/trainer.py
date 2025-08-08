import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
import os
import time
import json
from typing import Dict, Any, Optional
import numpy as np
from tqdm import tqdm
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .model import GraphDiT, PropertyPredictionHead
from .scheduler import CategoricalNoiseScheduler, DiscreteDiffusionLoss
from ..data_loader import ZINC_Dataset, StreamingZINCDataset, optimized_collate_fn, create_optimized_dataloader


class GraphDiTTrainer:
    """
    Trainer class for Graph Diffusion Transformer models.
    """
    
    def __init__(
        self,
        model: GraphDiT,
        noise_scheduler: CategoricalNoiseScheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        log_dir: str = 'logs',
        checkpoint_dir: str = 'checkpoints',
        use_wandb: bool = False,
        wandb_project: str = 'graph-dit',
        **kwargs
    ):
        self.model = model.to(device)
        self.noise_scheduler = noise_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.use_wandb = use_wandb
        
        # Loss function
        self.node_loss_fn = DiscreteDiffusionLoss()
        self.edge_loss_fn = DiscreteDiffusionLoss()
        
        # Logging
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir)
        
        if use_wandb and WANDB_AVAILABLE:
            wandb.init(project=wandb_project, config=kwargs)
            wandb.watch(model)
        
        # Training state
        self.epoch = 0
        self.step = 0
        self.best_val_loss = float('inf')
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_node_loss = 0
        total_edge_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Training Epoch {self.epoch}")
        
        for batch in pbar:
            batch = batch.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Ensure correct data types for model inputs
            x_noisy = batch.x_noisy.long()
            edge_attr_noisy = batch.edge_attr_noisy.long()
            timesteps = batch.t.long()
            
            # Forward pass
            node_logits, edge_logits = self.model(
                x=x_noisy,
                edge_index=batch.edge_index,
                edge_attr=edge_attr_noisy,
                batch=batch.batch,
                timesteps=timesteps
            )
            
            # Compute losses - ensure targets are long tensors for categorical loss
            node_clean = batch.x_clean.long()
            edge_clean = batch.edge_attr_clean.long()
            
            node_loss = self.node_loss_fn(node_logits, node_clean)
            edge_loss = self.edge_loss_fn(edge_logits, edge_clean)
            loss = node_loss + edge_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_node_loss += node_loss.item()
            total_edge_loss += edge_loss.item()
            num_batches += 1
            
            # Logging
            if num_batches % 100 == 0:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'node_loss': f"{node_loss.item():.4f}",
                    'edge_loss': f"{edge_loss.item():.4f}"
                })
                
                # Log to tensorboard
                self.writer.add_scalar('Train/Loss', loss.item(), self.step)
                self.writer.add_scalar('Train/Node_Loss', node_loss.item(), self.step)
                self.writer.add_scalar('Train/Edge_Loss', edge_loss.item(), self.step)
                
                if self.use_wandb:
                    wandb.log({
                        'train_loss': loss.item(),
                        'train_node_loss': node_loss.item(),
                        'train_edge_loss': edge_loss.item(),
                        'step': self.step
                    })
            
            self.step += 1
        
        if self.scheduler:
            self.scheduler.step()
        
        avg_loss = total_loss / num_batches
        avg_node_loss = total_node_loss / num_batches
        avg_edge_loss = total_edge_loss / num_batches
        
        return {
            'loss': avg_loss,
            'node_loss': avg_node_loss,
            'edge_loss': avg_edge_loss
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        total_node_loss = 0
        total_edge_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                batch = batch.to(self.device)
                
                # Forward pass
                node_logits, edge_logits = self.model(
                    x=batch.x_noisy.float(),
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr_noisy.float(),
                    batch=batch.batch,
                    timesteps=batch.t
                )
                
                # Compute losses
                node_loss = self.node_loss_fn(node_logits, batch.x_clean)
                edge_loss = self.edge_loss_fn(edge_logits, batch.edge_attr_clean)
                loss = node_loss + edge_loss
                
                # Update metrics
                total_loss += loss.item()
                total_node_loss += node_loss.item()
                total_edge_loss += edge_loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_node_loss = total_node_loss / num_batches
        avg_edge_loss = total_edge_loss / num_batches
        
        return {
            'loss': avg_loss,
            'node_loss': avg_node_loss,
            'edge_loss': avg_edge_loss
        }
    
    def train(self, num_epochs: int, save_every: int = 1, eval_every: int = 1) -> None:
        """Train the model for specified number of epochs."""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            if epoch % eval_every == 0:
                val_metrics = self.validate()
                
                # Log validation metrics
                self.writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
                self.writer.add_scalar('Val/Node_Loss', val_metrics['node_loss'], epoch)
                self.writer.add_scalar('Val/Edge_Loss', val_metrics['edge_loss'], epoch)
                
                if self.use_wandb:
                    wandb.log({
                        'val_loss': val_metrics['loss'],
                        'val_node_loss': val_metrics['node_loss'],
                        'val_edge_loss': val_metrics['edge_loss'],
                        'epoch': epoch
                    })
                
                print(f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}, "
                      f"Val Loss: {val_metrics['loss']:.4f}")
                
                # Save best model
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint('best_model.pt')
            else:
                print(f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}")
            
            # Save checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
        
        print("Training completed!")
        self.writer.close()
        
        if self.use_wandb:
            wandb.finish()
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': {
                'num_node_classes': self.noise_scheduler.num_node_classes,
                'num_edge_classes': self.noise_scheduler.num_edge_classes,
                'num_timesteps': self.noise_scheduler.num_timesteps
            }
        }
        
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, filename))
        print(f"Checkpoint saved: {filename}")
    
    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, filename), map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Checkpoint loaded: {filename}")
    
    def generate_samples(self, num_samples: int = 10) -> None:
        """Generate sample molecules."""
        print("Generating sample molecules...")
        samples = self.noise_scheduler.sample(self.model, num_samples, device=self.device)
        
        # Save samples
        samples_file = os.path.join(self.log_dir, f'samples_epoch_{self.epoch}.json')
        with open(samples_file, 'w') as f:
            json.dump(samples, f, indent=2, default=lambda x: x.tolist())
        
        print(f"Generated {len(samples)} samples saved to {samples_file}")


def create_trainer(
    config: Dict[str, Any],
    device: str = 'cuda'
) -> GraphDiTTrainer:
    """
    Create trainer from configuration.
    
    Args:
        config: Configuration dictionary
        device: Device to use
        
    Returns:
        GraphDiTTrainer instance
    """
    # Create model
    model = GraphDiT(
        node_dim=config['model']['node_dim'],
        edge_dim=config['model']['edge_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        max_time_steps=config['scheduler']['num_timesteps']
    )
    
    # Create noise scheduler
    noise_scheduler = CategoricalNoiseScheduler(
        num_node_classes=config['model']['node_dim'],
        num_edge_classes=config['model']['edge_dim'],
        num_timesteps=config['scheduler']['num_timesteps'],
        beta_start=config['scheduler']['beta_start'],
        beta_end=config['scheduler']['beta_end'],
        schedule=config['scheduler']['schedule']
    )
    
    # Create data loaders using unified pipeline
    data_path = config['data']['data_path']
    
    # Use get_unified_dataloaders which includes GraphDiffusionDataset wrapper
    from .data import get_unified_dataloaders
    loaders = get_unified_dataloaders(config, noise_scheduler)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs']
    )
    
    # Create trainer
    trainer = GraphDiTTrainer(
        model=model,
        noise_scheduler=noise_scheduler,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        log_dir=config['training']['log_dir'],
        checkpoint_dir=config['training']['checkpoint_dir'],
        use_wandb=config['training'].get('use_wandb', False),
        **config
    )
    
    return trainer


class PropertyPredictionTrainer:
    """
    Trainer for property prediction using pre-trained GraphDiT as feature extractor.
    """
    
    def __init__(
        self,
        graph_dit_model: GraphDiT,
        prediction_head: PropertyPredictionHead,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = 'cuda',
        **kwargs
    ):
        self.graph_dit = graph_dit_model.to(device)
        self.prediction_head = prediction_head.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # Freeze GraphDiT weights
        for param in self.graph_dit.parameters():
            param.requires_grad = False
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.prediction_head.train()
        total_loss = 0
        
        for batch in self.train_loader:
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Extract features using frozen GraphDiT
            with torch.no_grad():
                graph_embeddings = self.graph_dit.get_graph_embedding(
                    batch.x.float(),
                    batch.edge_index,
                    batch.edge_attr.float(),
                    batch.batch,
                    torch.zeros(batch.num_graphs, device=self.device)
                )
            
            # Predict properties
            predictions = self.prediction_head(graph_embeddings)
            
            # Compute loss
            loss = self.criterion(predictions, batch.y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self) -> float:
        """Validate the model."""
        self.prediction_head.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                
                # Extract features
                graph_embeddings = self.graph_dit.get_graph_embedding(
                    batch.x.float(),
                    batch.edge_index,
                    batch.edge_attr.float(),
                    batch.batch,
                    torch.zeros(batch.num_graphs, device=self.device)
                )
                
                # Predict properties
                predictions = self.prediction_head(graph_embeddings)
                
                # Compute loss
                loss = self.criterion(predictions, batch.y)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)