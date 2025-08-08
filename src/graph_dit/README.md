# GraphDiT: Graph Diffusion Transformer for Molecular Design

A comprehensive implementation of Graph Diffusion Transformer (GraphDiT) for molecular generation, property prediction, and optimization.

## Overview

GraphDiT is a state-of-the-art generative model for chemistry that uses diffusion processes on molecular graphs. It consists of:

1. **Pre-trained Graph DiT**: A transformer-based denoising model trained on large molecular datasets
2. **Property Prediction**: Fine-tuned models for predicting molecular properties
3. **Guided Optimization**: Molecule optimization using gradient-based guidance

## Architecture

```
Clean Molecule (Graph) → Add Noise → Noisy Graph → GraphDiT → Predict Clean Graph
```

### Core Components

- **Graph Transformer**: GATv2-based transformer blocks with edge features
- **Noise Scheduler**: Discrete diffusion for categorical graph features
- **Guidance System**: Critic models for property-guided generation

## Installation

```bash
pip install torch torch-geometric rdkit-pypi
pip install -e .
```

## Quick Start

### 1. Pre-training

```bash
python examples/train_graph_dit.py \
    --config configs/graph_dit_default.yaml \
    --data-path data/zinc250k \
    --epochs 100 \
    --batch-size 32
```

### 2. Generation

```bash
python examples/generate_molecules.py \
    --checkpoint checkpoints/best_model.pt \
    --num-samples 100 \
    --output generated_molecules.json
```

### 3. Optimization

```bash
python examples/optimize_molecules.py \
    --checkpoint checkpoints/best_model.pt \
    --critic-checkpoint checkpoints/critic_logp.pt \
    --input-smiles "CCO" \
    --property logp \
    --target 3.0 \
    --output optimization_results.json
```

## Usage Examples

### Basic Usage

```python
from src.graph_dit import GraphDiT, CategoricalNoiseScheduler

# Create model
model = GraphDiT(
    node_dim=20,  # atom types
    edge_dim=4,   # bond types
    hidden_dim=256,
    num_layers=8,
    num_heads=8
)

# Create scheduler
scheduler = CategoricalNoiseScheduler(
    num_node_classes=20,
    num_edge_classes=4,
    num_timesteps=1000
)
```

### Training

```python
from src.graph_dit import GraphDiTTrainer

trainer = GraphDiTTrainer(
    model=model,
    noise_scheduler=scheduler,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4)
)

trainer.train(num_epochs=100)
```

### Generation

```python
from src.graph_dit import GraphDiTGenerator

generator = GraphDiTGenerator(model, scheduler, device='cuda')
molecules = generator.generate_batch(
    num_samples=100,
    temperature=1.0
)
```

### Property Prediction

```python
from src.graph_dit import MolecularPropertyPredictor

predictor = MolecularPropertyPredictor(
    graph_dit_model=model,
    task=PropertyPredictionTask('logp', 'regression'),
    hidden_dims=[512, 256]
)

predictions = predictor.predict(data)
```

### Guided Optimization

```python
from src.graph_dit import GuidedDiffusionOptimizer

optimizer = GuidedDiffusionOptimizer(
    generator=generator,
    critic_model=critic_model,
    device='cuda'
)

result = optimizer.optimize_molecule(
    smiles="CCO",
    property_function=lambda mol: Descriptors.MolLogP(mol),
    property_target=3.0
)
```

## Configuration

All configurations are managed via YAML files. See `configs/graph_dit_default.yaml` for a complete example.

### Key Configuration Sections

- `model`: Model architecture parameters
- `scheduler`: Noise scheduling parameters
- `data`: Data processing settings
- `training`: Training hyperparameters
- `generation`: Generation settings
- `optimization`: Optimization parameters

## Data Preparation

The system expects molecular data in SMILES format. You can use any dataset (ZINC, ChEMBL, etc.):

1. **ZINC250k**: Default dataset for pre-training
2. **Custom datasets**: Provide SMILES strings with optional property labels

### Data Format

```csv
smiles,logp,qed,tpsa
CCO,0.5,0.8,20.2
CC(=O)O,-0.2,0.9,37.3
...
```

## Evaluation

Comprehensive evaluation metrics are provided:

### Generation Metrics

- **Validity**: Chemical validity rate
- **Uniqueness**: Unique molecule ratio
- **Novelty**: Novel structure ratio
- **Property distributions**: MW, LogP, QED, TPSA statistics

### Property Prediction Metrics

- **Regression**: RMSE, R², MAE
- **Classification**: Accuracy, AUC, F1-score

### Optimization Metrics

- **Success rate**: Molecules meeting criteria
- **Improvement**: Property change magnitude
- **Similarity**: Tanimoto similarity to original

## Advanced Features

### Multi-objective Optimization

```python
from src.graph_dit import MultiObjectiveOptimizer

optimizer = MultiObjectiveOptimizer(generator, critic_model)
result = optimizer.optimize_multiple_properties(
    smiles="CCO",
    property_functions=[logp, qed, tpsa],
    property_targets=[3.0, 0.8, 50.0],
    property_weights=[1.0, 0.5, 0.3]
)
```

### Custom Property Functions

```python
def custom_property(mol):
    # Your property calculation here
    return Descriptors.MolLogP(mol) + Descriptors.TPSA(mol) / 100

result = optimizer.optimize_molecule(
    smiles="CCO",
    property_function=custom_property,
    property_target=5.0
)
```

## Benchmarks

Run comprehensive benchmarks:

```python
from src.graph_dit import BenchmarkSuite

suite = BenchmarkSuite()
suite.run_generation_benchmark(generator)
suite.run_property_prediction_benchmark(predictor, test_loader)
suite.run_optimization_benchmark(optimizer, test_molecules, logp, 3.0)
```

## File Structure

```
src/graph_dit/
├── model.py              # Core architecture
├── scheduler.py          # Noise scheduling
├── data.py               # Data processing
├── trainer.py            # Training loops
├── generator.py          # Generation pipeline
├── property_prediction.py # Property prediction
├── guided_optimization.py # Molecule optimization
├── config.py             # Configuration management
├── evaluation.py         # Evaluation metrics
├── __init__.py           # Package initialization

examples/
├── train_graph_dit.py    # Pre-training script
├── generate_molecules.py # Generation script
├── optimize_molecules.py # Optimization script

configs/
├── graph_dit_default.yaml # Default configuration
```

## Performance Tips

1. **GPU Usage**: Use CUDA for best performance
2. **Batch Size**: Adjust based on GPU memory (16-64 for 8GB VRAM)
3. **Checkpointing**: Save checkpoints every 5-10 epochs
4. **Evaluation**: Use validation sets for hyperparameter tuning

## Troubleshooting

### Common Issues

1. **CUDA Memory**: Reduce batch size or use gradient accumulation
2. **Invalid Molecules**: Check chemical validity filters
3. **Poor Generation**: Increase training epochs or model capacity
4. **Slow Optimization**: Reduce guidance steps or use smaller molecules

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## Citation

If you use GraphDiT in your research, please cite:

```bibtex
@misc{graphdit2024,
  title={GraphDiT: Graph Diffusion Transformer for Molecular Design},
  author={GraphDiT Team},
  year={2024},
  howpublished={\url{https://github.com/your-repo/graph-dit}}
}
```

## License

MIT License - see LICENSE file for details.