"""
Graph Diffusion Transformer (GraphDiT) for Molecular Generation and Optimization.

This package provides a complete implementation of Graph DiT for:
1. Pre-training on molecular datasets
2. Molecule generation via diffusion
3. Property prediction using frozen embeddings
4. Guided molecule optimization

Modules:
- model: Core GraphDiT architecture
- scheduler: Noise scheduling for diffusion
- data: Data loading and preprocessing
- trainer: Training loops for pre-training and property prediction
- generator: Molecule generation pipeline
- property_prediction: Property prediction framework
- guided_optimization: Molecule optimization with guidance
- config: Configuration management
- evaluation: Comprehensive evaluation and benchmarking
"""

from .model import GraphDiT, PropertyPredictionHead, CriticModel
from .scheduler import CategoricalNoiseScheduler, DiscreteDiffusionLoss
from .data import get_dataloaders, create_featurizer, smiles_to_graph
from .trainer import GraphDiTTrainer, PropertyPredictionTrainer, create_trainer
from .generator import GraphDiTGenerator, load_generator
from .config import GraphDiTConfig, ConfigManager, get_pretraining_config
from .evaluation import MoleculeEvaluator, PropertyPredictionEvaluator, BenchmarkSuite
from .guided_optimization import GuidedDiffusionOptimizer, PropertyFunctions

__version__ = "1.0.0"
__author__ = "GraphDiT Team"
__description__ = "Graph Diffusion Transformer for Molecular Design"

__all__ = [
    # Core components
    'GraphDiT',
    'CategoricalNoiseScheduler',
    'GraphDiTGenerator',
    'GraphDiTTrainer',
    'GraphDiTConfig',
    
    # Utilities
    'create_trainer',
    'load_generator',
    'create_featurizer',
    'smiles_to_graph',
    
    # Evaluation
    'MoleculeEvaluator',
    'PropertyPredictionEvaluator',
    'BenchmarkSuite',
    
    # Optimization
    'GuidedDiffusionOptimizer',
    'PropertyFunctions',
    
    # Configuration
    'ConfigManager',
    'get_pretraining_config'
]