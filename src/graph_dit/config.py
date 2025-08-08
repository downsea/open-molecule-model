import yaml
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for GraphDiT model."""
    node_dim: int = 20  # Number of atom types
    edge_dim: int = 4   # Number of bond types
    hidden_dim: int = 256
    num_layers: int = 8
    num_heads: int = 8
    dropout: float = 0.1
    max_time_steps: int = 1000


@dataclass
class SchedulerConfig:
    """Configuration for noise scheduler."""
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule: str = 'cosine'  # 'linear', 'cosine', 'quadratic'


@dataclass
class DataConfig:
    """Configuration for data processing."""
    data_path: str = 'data/zinc250k'
    max_num_atoms: int = 50
    min_num_atoms: int = 3
    batch_size: int = 32
    num_workers: int = 4
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1


@dataclass
class TrainingConfig:
    """Configuration for training."""
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    log_dir: str = 'logs/graph_dit'
    checkpoint_dir: str = 'checkpoints/graph_dit'
    save_every: int = 5
    eval_every: int = 1
    use_wandb: bool = False
    wandb_project: str = 'graph-dit'


@dataclass
class GenerationConfig:
    """Configuration for molecule generation."""
    num_samples: int = 100
    temperature: float = 1.0
    guidance_scale: float = 0.0
    num_nodes_range: tuple = (3, 50)
    batch_size: int = 32


@dataclass
class OptimizationConfig:
    """Configuration for guided optimization."""
    num_steps: int = 50
    guidance_scale: float = 1.0
    noise_start: float = 0.8
    noise_end: float = 0.0
    temperature: float = 1.0
    similarity_constraint: float = 0.7
    max_attempts: int = 10


@dataclass
class PropertyPredictionConfig:
    """Configuration for property prediction tasks."""
    task_name: str = 'molecular_weight'
    task_type: str = 'regression'  # 'regression' or 'classification'
    num_classes: int = 1
    hidden_dims: list = None
    dropout: float = 0.1
    learning_rate: float = 1e-3
    num_epochs: int = 50


@dataclass
class GraphDiTConfig:
    """Complete configuration for GraphDiT system."""
    model: ModelConfig = field(default_factory=ModelConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    property_prediction: PropertyPredictionConfig = field(default_factory=PropertyPredictionConfig)
    
    # Device configuration
    device: str = 'cuda'
    seed: int = 42
    
    # Directories
    data_dir: str = 'data'
    output_dir: str = 'outputs'


class ConfigManager:
    """Manager for loading and saving configurations."""
    
    def __init__(self, config_dir: str = 'configs'):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
    
    def load_config(self, config_path: str) -> GraphDiTConfig:
        """Load configuration from YAML or JSON file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                config_dict = yaml.safe_load(f)
            else:
                config_dict = json.load(f)
        
        return self._dict_to_config(config_dict)
    
    def save_config(self, config: GraphDiTConfig, config_path: str) -> None:
        """Save configuration to file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self._config_to_dict(config)
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                yaml.dump(config_dict, f, indent=2)
            else:
                json.dump(config_dict, f, indent=2)
    
    def create_default_config(self, config_path: str) -> GraphDiTConfig:
        """Create and save default configuration."""
        config = GraphDiTConfig()
        self.save_config(config, config_path)
        return config
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> GraphDiTConfig:
        """Convert dictionary to configuration dataclass."""
        config = GraphDiTConfig()
        
        # Update model config
        if 'model' in config_dict:
            for key, value in config_dict['model'].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)
        
        # Update scheduler config
        if 'scheduler' in config_dict:
            for key, value in config_dict['scheduler'].items():
                if hasattr(config.scheduler, key):
                    setattr(config.scheduler, key, value)
        
        # Update data config
        if 'data' in config_dict:
            for key, value in config_dict['data'].items():
                if hasattr(config.data, key):
                    setattr(config.data, key, value)
        
        # Update training config
        if 'training' in config_dict:
            for key, value in config_dict['training'].items():
                if hasattr(config.training, key):
                    setattr(config.training, key, value)
        
        # Update generation config
        if 'generation' in config_dict:
            for key, value in config_dict['generation'].items():
                if hasattr(config.generation, key):
                    setattr(config.generation, key, value)
        
        # Update optimization config
        if 'optimization' in config_dict:
            for key, value in config_dict['optimization'].items():
                if hasattr(config.optimization, key):
                    setattr(config.optimization, key, value)
        
        # Update property prediction config
        if 'property_prediction' in config_dict:
            for key, value in config_dict['property_prediction'].items():
                if hasattr(config.property_prediction, key):
                    setattr(config.property_prediction, key, value)
        
        return config
    
    def _config_to_dict(self, config: GraphDiTConfig) -> Dict[str, Any]:
        """Convert configuration dataclass to dictionary."""
        return {
            'model': asdict(config.model),
            'scheduler': asdict(config.scheduler),
            'data': asdict(config.data),
            'training': asdict(config.training),
            'generation': asdict(config.generation),
            'optimization': asdict(config.optimization),
            'property_prediction': asdict(config.property_prediction),
            'device': config.device,
            'seed': config.seed,
            'data_dir': config.data_dir,
            'output_dir': config.output_dir
        }
    
    def create_config_for_task(
        self,
        task: str,
        overrides: Optional[Dict[str, Any]] = None
    ) -> GraphDiTConfig:
        """Create configuration for specific task."""
        base_config = GraphDiTConfig()
        
        task_configs = {
            'pretraining': {
                'model': {'num_layers': 8, 'hidden_dim': 256},
                'training': {'num_epochs': 100, 'batch_size': 32},
                'scheduler': {'num_timesteps': 1000}
            },
            'generation': {
                'generation': {'num_samples': 1000, 'temperature': 1.0},
                'optimization': {'num_steps': 50, 'guidance_scale': 1.0}
            },
            'property_prediction': {
                'training': {'num_epochs': 50, 'batch_size': 64},
                'property_prediction': {'hidden_dims': [512, 256]}
            },
            'optimization': {
                'optimization': {'num_steps': 100, 'guidance_scale': 2.0},
                'generation': {'num_samples': 100}
            }
        }
        
        if task in task_configs:
            config_dict = task_configs[task]
            if overrides:
                config_dict.update(overrides)
            
            return self._dict_to_config(config_dict)
        
        return base_config


# Predefined configurations for common tasks
def get_pretraining_config() -> GraphDiTConfig:
    """Get configuration for pre-training."""
    return GraphDiTConfig(
        model=ModelConfig(
            node_dim=20,
            edge_dim=4,
            hidden_dim=256,
            num_layers=8,
            num_heads=8,
            max_time_steps=1000
        ),
        training=TrainingConfig(
            num_epochs=100,
            batch_size=32,
            learning_rate=1e-4,
            use_wandb=True
        ),
        data=DataConfig(
            data_path='data/zinc250k',
            max_num_atoms=50
        )
    )


def get_generation_config() -> GraphDiTConfig:
    """Get configuration for generation."""
    return GraphDiTConfig(
        generation=GenerationConfig(
            num_samples=1000,
            temperature=1.0,
            num_nodes_range=(3, 50)
        )
    )


def get_optimization_config() -> GraphDiTConfig:
    """Get configuration for optimization."""
    return GraphDiTConfig(
        optimization=OptimizationConfig(
            num_steps=100,
            guidance_scale=2.0,
            similarity_constraint=0.7
        )
    )


def get_property_prediction_config(task_name: str) -> GraphDiTConfig:
    """Get configuration for property prediction."""
    return GraphDiTConfig(
        property_prediction=PropertyPredictionConfig(
            task_name=task_name,
            task_type='regression',
            num_epochs=50,
            learning_rate=1e-3
        )
    )


class UnifiedConfigManager:
    """Manager for unified configuration files supporting both PanGu and GraphDiT."""
    
    @staticmethod
    def load_unified_config(config_path: str) -> GraphDiTConfig:
        """Load GraphDiT configuration from unified config file."""
        import yaml
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return UnifiedConfigManager._extract_graphdit_config(config_dict)
    
    @staticmethod
    def _extract_graphdit_config(config_dict: Dict[str, Any]) -> GraphDiTConfig:
        """Extract GraphDiT configuration from unified config dictionary."""
        config = GraphDiTConfig()
        
        # Handle both direct graphdit config and unified config
        if 'graphdit' in config_dict:
            # Unified config format
            gd_config = config_dict['graphdit']
        else:
            # Direct graphdit config format
            gd_config = config_dict
        
        # Update model config
        if 'model' in gd_config:
            model_config = gd_config['model']
            for key, value in model_config.items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)
        
        # Update scheduler config
        if 'scheduler' in gd_config:
            scheduler_config = gd_config['scheduler']
            for key, value in scheduler_config.items():
                if hasattr(config.scheduler, key):
                    setattr(config.scheduler, key, value)
        
        # Update data config
        if 'data' in config_dict:
            data_config = config_dict['data']
            for key, value in data_config.items():
                if hasattr(config.data, key):
                    setattr(config.data, key, value)
        elif 'data' in gd_config:
            data_config = gd_config['data']
            for key, value in data_config.items():
                if hasattr(config.data, key):
                    setattr(config.data, key, value)
        
        # Update training config
        if 'training' in gd_config:
            training_config = gd_config['training']
            for key, value in training_config.items():
                if hasattr(config.training, key):
                    setattr(config.training, key, value)
        
        # Update generation config
        if 'generation' in gd_config:
            gen_config = gd_config['generation']
            for key, value in gen_config.items():
                if hasattr(config.generation, key):
                    setattr(config.generation, key, value)
        
        # Update optimization config
        if 'optimization' in gd_config:
            opt_config = gd_config['optimization']
            for key, value in opt_config.items():
                if hasattr(config.optimization, key):
                    setattr(config.optimization, key, value)
        
        # Update system and path configurations
        if 'system' in config_dict:
            system_config = config_dict['system']
            if 'device' in system_config:
                config.device = system_config['device']
        
        if 'paths' in config_dict:
            paths_config = config_dict['paths']
            if 'graphdit_log_dir' in paths_config:
                config.training.log_dir = paths_config['graphdit_log_dir']
            if 'graphdit_checkpoint_path' in paths_config:
                checkpoint_path = paths_config['graphdit_checkpoint_path']
                config.training.checkpoint_dir = str(Path(checkpoint_path).parent)
        
        if 'logging' in config_dict:
            logging_config = config_dict['logging']
            if 'use_wandb' in logging_config:
                config.training.use_wandb = logging_config['use_wandb']
            if 'wandb_project' in logging_config:
                config.training.wandb_project = logging_config['wandb_project']
        
        return config
    
    @staticmethod
    def get_data_paths(config_dict: Dict[str, Any]) -> Dict[str, str]:
        """Get data paths from unified configuration."""
        paths = {}
        
        if 'data' in config_dict:
            data_config = config_dict['data']
            paths['dataset_path'] = data_config.get('dataset_path', 'data/standard')
            paths['train_path'] = data_config.get('train_dataset_path', 'data/standard/train')
            paths['val_path'] = data_config.get('val_dataset_path', 'data/standard/val')
            paths['test_path'] = data_config.get('test_dataset_path', 'data/standard/test')
            paths['raw_path'] = data_config.get('raw_data_path', 'data/raw')
            paths['processed_path'] = data_config.get('processed_data_path', 'data/processed')
            paths['standard_path'] = data_config.get('standard_data_path', 'data/standard')
        
        return paths