# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project reproduces the PanGu Drug Model from "PanGu Drug Model: Learn a Molecule Like a Human" - a conditional variational autoencoder (cVAE) that translates between 2D molecular graphs and SELFIES string representations for drug discovery applications.

## Architecture

- **Encoder**: Graph transformer with 10 layers, 256-dim hidden units, 8 attention heads
- **Decoder**: Transformer-based sequence model with 6 layers (layer 1 absolute + layers 2-6 relative positional encoding)
- **Latent Space**: 8×256 matrix from concatenated encoder layer outputs (layers 1,2,3,4,5,6,8,10)
- **Training**: CVAE with ELBO loss (cross-entropy + KL divergence with β=0.001)

## Key Commands

### Development Workflow
```bash
# Setup environment (Windows)
.\bootstrap.ps1 -Install

# Setup environment (Linux/macOS)
./bootstrap.sh --install

# Download ZINC dataset
./bootstrap.sh --download    # or .\bootstrap.ps1 -Download

# Process data
./bootstrap.sh --process     # or .\bootstrap.ps1 -Process

# Train model
./bootstrap.sh --train       # or .\bootstrap.ps1 -Train

# Launch TensorBoard
./bootstrap.sh --board       # or .\bootstrap.ps1 -Board

# Run hyperparameter search
./bootstrap.sh --search --random --trials 50 --epochs 5

# Manual training with custom parameters
python -m src.train --learning-rate 1e-3 --batch-size 64 --hidden-dim 512
```

### Advanced Usage
```bash
# Training with custom config
./bootstrap.sh --train --config custom_config.yaml --epochs 20

# Evaluation modes
./bootstrap.sh --evaluate --mode reconstruction
./bootstrap.sh --evaluate --mode generation --num-samples 100
./bootstrap.sh --evaluate --mode latent_space

# Hyperparameter grid search
./bootstrap.sh --search --learning-rate 1e-4 5e-4 1e-3 --batch-size 32 64
```

### Direct Commands
```bash
# Activate virtual environment
source .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1

# Install dependencies
uv pip install -r requirements.txt

# Process raw SMILES data
python src/process_data.py

# Train model with full configuration
python -m src.train --config config.yaml

# View training progress
tensorboard --logdir=runs

# Run evaluation
python -m src.evaluate --mode generation --num-samples 50
```

## Configuration System

### Configuration File (config.yaml)
All tunable parameters are centralized in `config.yaml`:

```yaml
data:
  dataset_path: "data/processed"
  max_length: 128
  batch_size: 32

model:
  num_node_features: 6
  hidden_dim: 256
  num_encoder_layers: 10
  num_encoder_heads: 8
  num_selected_layers: 8
  num_decoder_layers: 6
  num_decoder_heads: 8
  latent_dim: 128

training:
  learning_rate: 1e-4
  num_epochs: 10
  beta: 0.001
  gradient_clip: 1.0
```

### Hyperparameter Search
Use `hyperparameter_search.py` for systematic optimization:

```bash
# Random search
python hyperparameter_search.py --random-search --trials 50 --epochs 3

# Grid search
python hyperparameter_search.py --learning-rate 1e-4 5e-4 1e-3 --batch-size 32 64
```

## Data Pipeline

1. **Raw Data**: SMILES files in `data/raw/` (AA/*.smi format)
2. **Processed Data**: PyTorch tensors in `data/processed/` (*.pt files)
3. **Features**: Atom features (atomic number, degree, charge, hybridization, aromaticity, radicals)
4. **Training**: Uses ZINC_Dataset class in src/data_loader.py

## Model Structure

- **src/model.py**: Main PanGuDrugModel class combining encoder/decoder with VAE loss
- **src/encoder.py**: GraphTransformerEncoder with 8-layer concatenation for latent space
- **src/decoder.py**: TransformerDecoder with relative positional encoding
- **src/train.py**: Training loop with checkpointing, TensorBoard logging, and config integration
- **src/evaluate.py**: Comprehensive evaluation (reconstruction, generation, latent space analysis)
- **src/config.py**: Configuration management with YAML support and hyperparameter search
- **src/utils.py**: SELFIESProcessor for tokenization and encoding/decoding
- **src/data_loader.py**: ZINC dataset loading and processing
- **checkpoints/**: Model checkpoints saved as pangu_drug_model.pt

## File Structure

```
src/
├── model.py              # Main cVAE model
├── encoder.py            # Graph transformer encoder
├── decoder.py            # Transformer sequence decoder
├── data_loader.py        # ZINC dataset loading
├── process_data.py       # SMILES to graph conversion
├── train.py             # Training script with config integration
├── evaluate.py          # Comprehensive evaluation
├── config.py            # Configuration management
└── utils.py             # SELFIES processing utilities

data/
├── raw/                 # Original SMILES files
├── processed/           # PyTorch tensor files
└── *.pdf                # Research paper

runs/                    # TensorBoard logs
checkpoints/             # Model checkpoints
```

## Dependencies

PyTorch ecosystem: torch, torchvision, torchaudio, torch_geometric
Chemistry: rdkit, selfies
Utilities: pandas, tqdm, tensorboard, uv, pyyaml

## Current Status

✅ **Fully Functional System**
- Complete architecture with proper 8×256 latent matrix
- Relative positional encoding implemented in decoder
- Real SELFIES data pipeline working
- Comprehensive evaluation capabilities
- Configuration system with hyperparameter search
- All runtime issues resolved

## Key Improvements

- **Configuration**: Centralized YAML-based parameter management
- **Hyperparameter Search**: Grid and random search capabilities
- **Evaluation**: Multi-mode evaluation (reconstruction, generation, latent space)
- **Flexibility**: Command-line parameter overrides
- **Scalability**: Configurable architecture parameters