import yaml
import argparse
from typing import Dict, Any
from dataclasses import dataclass, asdict, field
from pathlib import Path

@dataclass
class DataConfig:
    dataset_path: str = "data/processed"
    max_length: int = 128
    batch_size: int = 32

@dataclass
class ModelConfig:
    num_node_features: int = 6
    hidden_dim: int = 256
    num_encoder_layers: int = 10
    num_encoder_heads: int = 8
    num_selected_layers: int = 8
    num_decoder_layers: int = 6
    num_decoder_heads: int = 8
    latent_dim: int = 128
    output_dim: int = None

@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    num_epochs: int = 10
    beta: float = 0.001
    gradient_clip: float = 1.0

@dataclass
class PathsConfig:
    log_dir: str = "runs/pangu_drug_model"
    checkpoint_path: str = "checkpoints/pangu_drug_model.pt"

@dataclass
class EvaluationConfig:
    num_samples: int = 100
    device: str = "cuda"

@dataclass
class OptimizationConfig:
    warmup_steps: int = 1000
    scheduler: str = "cosine"
    weight_decay: float = 1e-4

@dataclass
class SystemConfig:
    device: str = "cuda"
    mixed_precision: bool = True
    num_workers: int = 4

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = cls()
        
        # Update nested configs
        if config_dict and 'data' in config_dict:
            for key, value in config_dict['data'].items():
                setattr(config.data, key, value)
        if config_dict and 'model' in config_dict:
            for key, value in config_dict['model'].items():
                setattr(config.model, key, value)
        if config_dict and 'training' in config_dict:
            for key, value in config_dict['training'].items():
                setattr(config.training, key, value)
        if config_dict and 'paths' in config_dict:
            for key, value in config_dict['paths'].items():
                setattr(config.paths, key, value)
        if config_dict and 'evaluation' in config_dict:
            for key, value in config_dict['evaluation'].items():
                setattr(config.evaluation, key, value)
        if config_dict and 'optimization' in config_dict:
            for key, value in config_dict['optimization'].items():
                setattr(config.optimization, key, value)
        if config_dict and 'system' in config_dict:
            for key, value in config_dict['system'].items():
                setattr(config.system, key, value)
            
        return config
    
    @classmethod
    def from_args(cls) -> 'Config':
        """Load configuration from command line arguments and YAML file."""
        parser = argparse.ArgumentParser(description='PanGu Drug Model Configuration')
        parser.add_argument('--config', type=str, default='config.yaml',
                          help='Path to configuration file')
        parser.add_argument('--batch-size', type=int, help='Batch size')
        parser.add_argument('--learning-rate', type=float, help='Learning rate')
        parser.add_argument('--num-epochs', type=int, help='Number of epochs')
        parser.add_argument('--hidden-dim', type=int, help='Hidden dimension')
        parser.add_argument('--latent-dim', type=int, help='Latent dimension')
        parser.add_argument('--beta', type=float, help='KL divergence weight')
        parser.add_argument('--device', type=str, help='Device (cuda/cpu)')
        
        args = parser.parse_args()
        
        # Load from YAML first
        config = cls.from_yaml(args.config)
        
        # Override with command line arguments
        if args.batch_size is not None:
            config.data.batch_size = args.batch_size
        if args.learning_rate is not None:
            config.training.learning_rate = args.learning_rate
        if args.num_epochs is not None:
            config.training.num_epochs = args.num_epochs
        if args.hidden_dim is not None:
            config.model.hidden_dim = args.hidden_dim
        if args.latent_dim is not None:
            config.model.latent_dim = args.latent_dim
        if args.beta is not None:
            config.training.beta = args.beta
        if args.device is not None:
            config.system.device = args.device
            config.evaluation.device = args.device
            
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def save(self, yaml_path: str):
        """Save configuration to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def __str__(self):
        """String representation for logging."""
        return yaml.dump(self.to_dict(), default_flow_style=False)

# Example usage:
if __name__ == "__main__":
    # Load default config
    config = Config()
    print("Default config:")
    print(config)
    
    # Load from YAML
    config_yaml = Config.from_yaml("config.yaml")
    print("\nConfig from YAML:")
    print(config_yaml)
    
    # Save config
    config.save("test_config.yaml")