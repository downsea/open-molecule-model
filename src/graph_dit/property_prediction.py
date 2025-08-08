import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, accuracy_score
from typing import Dict, List, Any, Optional, Tuple
import json
import os
import pandas as pd
from tqdm import tqdm

from .model import GraphDiT, PropertyPredictionHead
from .data import get_dataloaders, collate_fn


class PropertyPredictionTask:
    """Base class for property prediction tasks."""
    
    def __init__(self, task_name: str, task_type: str, num_classes: int = 1):
        self.task_name = task_name
        self.task_type = task_type  # 'regression' or 'classification'
        self.num_classes = num_classes


class MolecularPropertyPredictor:
    """
    Property predictor using pre-trained GraphDiT as feature extractor.
    """
    
    def __init__(
        self,
        graph_dit_model: GraphDiT,
        task: PropertyPredictionTask,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.1,
        device: str = 'cuda'
    ):
        self.graph_dit = graph_dit_model.to(device)
        self.task = task
        self.device = device
        
        # Freeze GraphDiT parameters
        for param in self.graph_dit.parameters():
            param.requires_grad = False
        
        # Create prediction head
        input_dim = self.graph_dit.hidden_dim  # From GraphDiT hidden dimension
        self.prediction_head = PropertyPredictionHead(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=task.num_classes,
            dropout=dropout
        ).to(device)
        
        # Loss function
        if task.task_type == 'regression':
            self.criterion = nn.MSELoss()
        else:
            if task.num_classes == 1:
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                self.criterion = nn.CrossEntropyLoss()
    
    def extract_features(self, data) -> torch.Tensor:
        """Extract features from molecular graph using frozen GraphDiT."""
        with torch.no_grad():
            features = self.graph_dit.get_graph_embedding(
                x=data.x.float(),
                edge_index=data.edge_index,
                edge_attr=data.edge_attr.float(),
                batch=data.batch,
                timesteps=torch.zeros(data.num_graphs, device=self.device)
            )
        return features
    
    def forward(self, data) -> torch.Tensor:
        """Forward pass through the model."""
        features = self.extract_features(data)
        predictions = self.prediction_head(features)
        return predictions
    
    def predict(self, data) -> np.ndarray:
        """Make predictions on data."""
        self.graph_dit.eval()
        self.prediction_head.eval()
        
        with torch.no_grad():
            predictions = self.forward(data)
            
            if self.task.task_type == 'regression':
                return predictions.cpu().numpy()
            else:
                if self.task.num_classes == 1:
                    return torch.sigmoid(predictions).cpu().numpy()
                else:
                    return F.softmax(predictions, dim=-1).cpu().numpy()


class PropertyPredictionTrainer:
    """
    Trainer for molecular property prediction tasks.
    """
    
    def __init__(
        self,
        predictor: MolecularPropertyPredictor,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda'
    ):
        self.predictor = predictor
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.predictor.prediction_head.train()
        total_loss = 0
        num_batches = 0
        
        for batch in self.train_loader:
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.predictor.forward(batch)
            
            # Compute loss
            loss = self.predictor.criterion(predictions.squeeze(), batch.y)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        if self.scheduler:
            self.scheduler.step()
        
        return {'loss': total_loss / num_batches}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.predictor.prediction_head.eval()
        total_loss = 0
        predictions_list = []
        targets_list = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                
                # Forward pass
                predictions = self.predictor.forward(batch)
                
                # Compute loss
                loss = self.predictor.criterion(predictions.squeeze(), batch.y)
                
                total_loss += loss.item()
                predictions_list.extend(predictions.squeeze().cpu().numpy())
                targets_list.extend(batch.y.cpu().numpy())
                num_batches += 1
        
        # Calculate metrics
        predictions = np.array(predictions_list)
        targets = np.array(targets_list)
        
        metrics = {'loss': total_loss / num_batches}
        
        if self.predictor.task.task_type == 'regression':
            metrics.update({
                'mse': mean_squared_error(targets, predictions),
                'rmse': np.sqrt(mean_squared_error(targets, predictions)),
                'r2': r2_score(targets, predictions)
            })
        else:
            if self.predictor.task.num_classes == 1:
                # Binary classification
                predicted_labels = (predictions > 0.5).astype(int)
                metrics.update({
                    'auc': roc_auc_score(targets, predictions),
                    'accuracy': accuracy_score(targets, predicted_labels)
                })
            else:
                # Multi-class classification
                predicted_labels = np.argmax(predictions, axis=1)
                metrics.update({
                    'accuracy': accuracy_score(targets, predicted_labels)
                })
        
        return metrics
    
    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        """Train the model for specified number of epochs."""
        history = {'train_loss': [], 'val_loss': []}
        
        if self.predictor.task.task_type == 'regression':
            history.update({'val_mse': [], 'val_rmse': [], 'val_r2': []})
        else:
            if self.predictor.task.num_classes == 1:
                history.update({'val_auc': [], 'val_accuracy': []})
            else:
                history.update({'val_accuracy': []})
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            
            for key in val_metrics:
                if key != 'loss':
                    history[f'val_{key}'].append(val_metrics[key])
            
            print(f"Epoch {epoch + 1}/{num_epochs} - "
                  f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Val Loss: {val_metrics['loss']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save(self.predictor.prediction_head.state_dict(), 'best_property_model.pt')
        
        return history


class MultiTaskPropertyPredictor(nn.Module):
    """
    Multi-task property predictor using shared GraphDiT features.
    """
    
    def __init__(
        self,
        graph_dit_model: GraphDiT,
        tasks: List[PropertyPredictionTask],
        shared_hidden_dim: int = 256,
        task_specific_dims: Optional[Dict[str, List[int]]] = None
    ):
        super().__init__()
        self.graph_dit = graph_dit_model
        self.tasks = tasks
        
        # Freeze GraphDiT
        for param in self.graph_dit.parameters():
            param.requires_grad = False
        
        # Shared feature extraction
        self.shared_layer = nn.Sequential(
            nn.Linear(self.graph_dit.hidden_dim, shared_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for task in tasks:
            if task_specific_dims and task.task_name in task_specific_dims:
                dims = task_specific_dims[task.task_name]
            else:
                dims = [128, 64]
            
            self.task_heads[task.task_name] = PropertyPredictionHead(
                input_dim=shared_hidden_dim,
                hidden_dims=dims,
                output_dim=task.num_classes
            )
    
    def forward(self, data, task_name: str) -> torch.Tensor:
        """Forward pass for specific task."""
        with torch.no_grad():
            features = self.graph_dit.get_graph_embedding(
                data.x.float(),
                data.edge_index,
                data.edge_attr.float(),
                data.batch,
                torch.zeros(data.num_graphs, device=data.x.device)
            )
        
        shared_features = self.shared_layer(features)
        return self.task_heads[task_name](shared_features)
    
    def forward_all(self, data) -> Dict[str, torch.Tensor]:
        """Forward pass for all tasks."""
        with torch.no_grad():
            features = self.graph_dit.get_graph_embedding(
                data.x.float(),
                data.edge_index,
                data.edge_attr.float(),
                data.batch,
                torch.zeros(data.num_graphs, device=data.x.device)
            )
        
        shared_features = self.shared_layer(features)
        
        results = {}
        for task in self.tasks:
            results[task.task_name] = self.task_heads[task.task_name](shared_features)
        
        return results


class PropertyPredictionEvaluator:
    """
    Evaluator for property prediction models.
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
    
    def evaluate_model(
        self,
        predictor: MolecularPropertyPredictor,
        test_loader: DataLoader,
        save_path: Optional[str] = None
    ) -> Dict[str, float]:
        """Comprehensive model evaluation."""
        predictor.predictor.eval()
        predictions_list = []
        targets_list = []
        smiles_list = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                
                predictions = predictor.predict(batch)
                predictions_list.extend(predictions)
                targets_list.extend(batch.y.cpu().numpy())
                
                if hasattr(batch, 'smiles'):
                    smiles_list.extend(batch.smiles)
        
        predictions = np.array(predictions_list)
        targets = np.array(targets_list)
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions, targets, predictor.task)
        
        # Save detailed results
        if save_path:
            results_df = pd.DataFrame({
                'smiles': smiles_list,
                'predictions': predictions.squeeze(),
                'targets': targets.squeeze()
            })
            results_df.to_csv(save_path, index=False)
        
        return metrics
    
    def _calculate_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        task: PropertyPredictionTask
    ) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        if task.task_type == 'regression':
            return {
                'mse': float(mean_squared_error(targets, predictions)),
                'rmse': float(np.sqrt(mean_squared_error(targets, predictions))),
                'mae': float(np.mean(np.abs(targets - predictions))),
                'r2': float(r2_score(targets, predictions)),
                'pearson_r': float(np.corrcoef(targets, predictions)[0, 1])
            }
        else:
            if task.num_classes == 1:
                # Binary classification
                predicted_labels = (predictions > 0.5).astype(int)
                return {
                    'accuracy': float(accuracy_score(targets, predicted_labels)),
                    'auc': float(roc_auc_score(targets, predictions)),
                    'f1': float(2 * (predictions * targets).sum() / (predictions.sum() + targets.sum())),
                    'precision': float((predictions * targets).sum() / predictions.sum() if predictions.sum() > 0 else 0),
                    'recall': float((predictions * targets).sum() / targets.sum() if targets.sum() > 0 else 0)
                }
            else:
                # Multi-class classification
                predicted_labels = np.argmax(predictions, axis=1)
                return {
                    'accuracy': float(accuracy_score(targets, predicted_labels))
                }


def create_property_predictor(
    graph_dit_model: GraphDiT,
    task: PropertyPredictionTask,
    device: str = 'cuda',
    **kwargs
) -> MolecularPropertyPredictor:
    """Create property predictor from configuration."""
    return MolecularPropertyPredictor(
        graph_dit_model=graph_dit_model,
        task=task,
        device=device,
        **kwargs
    )


def create_multitask_predictor(
    graph_dit_model: GraphDiT,
    tasks: List[PropertyPredictionTask],
    device: str = 'cuda',
    **kwargs
) -> MultiTaskPropertyPredictor:
    """Create multi-task predictor from configuration."""
    return MultiTaskPropertyPredictor(
        graph_dit_model=graph_dit_model,
        tasks=tasks,
        **kwargs
    )