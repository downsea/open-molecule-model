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
# Setup environment (Windows with Git Bash/WSL)
./bootstrap.sh --install

# Download ZINC dataset (1900+ SMI files)
./bootstrap.sh --download                    # Default URI file
./bootstrap.sh --download data/custom.uri    # Custom URI file

# Process data
./bootstrap.sh --process     

# Train model
./bootstrap.sh --train       

# Launch TensorBoard
./bootstrap.sh --board      

# Run hyperparameter search
./bootstrap.sh --search --random --trials 50 --epochs 5
./bootstrap.sh --search --learning-rate 1e-4 5e-4 1e-3 --batch-size 32 64

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

# Direct Python commands
python -m src.evaluate --mode generation --num-samples 50
python hyperparameter_search.py --random-search --trials 50 --epochs 3
```

### Direct Commands
```bash
# Activate virtual environment
source .venv/Scripts/activate  

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

## Data Pipeline

### Download System
- **Source**: 1900+ SMI files from ZINC database via HTTP URLs
- **URI File**: `data/ZINC-downloader-2D-smi.uri` contains all SMI file URLs
- **Download Tool**: aria2 (multi-threaded downloader)
- **Storage**: `data/raw/` directory for SMI files
- **Failed Downloads**: Logged to timestamped fail files in `data/` directory

### Data Processing
1. **Raw Data**: SMILES files in `data/raw/` (AA/*.smi format)
2. **Processed Data**: PyTorch tensors in `data/processed/` (*.pt files)
3. **Features**: Atom features (atomic number, degree, charge, hybridization, aromaticity, radicals)
4. **Training**: Uses ZINC_Dataset class in src/data_loader.py

### Download Process
```bash
# Multi-threaded download with aria2
aria2c --input-file="data/ZINC-downloader-2D-smi.uri" \
       --dir="data/raw" \
       --max-concurrent-downloads=10 \
       --max-connection-per-server=4 \
       --continue=true \
       --max-tries=3 \
       --retry-wait=30 \
       --timeout=60 \
       --log="data/download.log" \
       --save-session="data/timestamp_fail.uri"
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

optimization:
  warmup_steps: 1000
  scheduler: "cosine"
  weight_decay: 1e-4

evaluation:
  num_samples: 100
  device: "cuda"
```

### Configuration Classes
- **src/config.py**: Complete configuration management system
- **DataConfig**: Dataset parameters and batch settings
- **ModelConfig**: Architecture dimensions and layer counts
- **TrainingConfig**: Learning rate, epochs, loss parameters
- **OptimizationConfig**: Scheduler and optimizer settings
- **EvaluationConfig**: Evaluation parameters
- **PathsConfig**: File paths for logs and checkpoints

### Hyperparameter Search
Use `hyperparameter_search.py` for systematic optimization:

```bash
# Random search
python hyperparameter_search.py --random-search --trials 50 --epochs 3

# Grid search with specific parameters
python hyperparameter_search.py \
  --learning-rate 1e-4 5e-4 1e-3 \
  --batch-size 32 64 \
  --hidden-dim 256 512 \
  --beta 0.0001 0.001 0.01

# Via bootstrap script
./bootstrap.sh --search --random --trials 20 --epochs 5
```

## Model Structure

### Core Components
- **src/model.py**: Main PanGuDrugModel class combining encoder/decoder with VAE loss
- **src/encoder.py**: GraphTransformerEncoder with 8-layer concatenation for latent space
- **src/decoder.py**: TransformerDecoder with relative positional encoding
- **src/train.py**: Training loop with checkpointing, TensorBoard logging, and config integration
- **src/evaluate.py**: Comprehensive evaluation (reconstruction, generation, latent space analysis)
- **src/config.py**: Configuration management with YAML support and hyperparameter search
- **src/utils.py**: SELFIESProcessor for tokenization and encoding/decoding
- **src/data_loader.py**: ZINC dataset loading and processing

### Key Features
- **8×256 latent matrix** construction from selected encoder layers
- **Relative positional encoding** in decoder layers 2-6
- **Real SELFIES processing** with vocabulary management
- **Cross-entropy loss** for sequence generation
- **KL divergence regularization** with configurable β parameter

## File Structure

```
src/
├── model.py              # Main cVAE model with VAE loss
├── encoder.py            # Graph transformer encoder (10 layers, 8 heads)
├── decoder.py            # Transformer sequence decoder (6 layers, relative encoding)
├── data_loader.py        # ZINC dataset loading and processing
├── process_data.py       # SMILES to graph conversion
├── train.py             # Training script with config integration
├── evaluate.py          # Comprehensive evaluation system
├── config.py            # Configuration management system
└── utils.py             # SELFIES processing utilities

data/
├── raw/                 # Original SMI files from ZINC (1900+ files)
├── processed/           # PyTorch tensor files
└── ZINC-downloader-2D-smi.uri  # List of SMI file URLs

runs/                    # TensorBoard logs
checkpoints/             # Model checkpoints
hyperparameter_search/   # Hyperparameter search results
```

## Dependencies

### Core Dependencies
- **PyTorch**: torch, torchvision, torchaudio
- **PyTorch Geometric**: torch_geometric, torch_scatter, torch_sparse
- **Chemistry**: rdkit, selfies
- **Utilities**: pandas, tqdm, tensorboard, uv, pyyaml
- **Machine Learning**: numpy, scikit-learn

### Installation
```bash
# Using uv (recommended)
uv pip install -r requirements.txt

# Using pip
pip install -r requirements.txt
```

## Evaluation Capabilities

### Evaluation Modes
```bash
# Reconstruction accuracy
python -m src.evaluate --mode reconstruction

# Generate new molecules
python -m src.evaluate --mode generation --num-samples 100

# Latent space analysis
python -m src.evaluate --mode latent_space
```

### Metrics
- **Reconstruction**: Exact match rate, valid molecule rate
- **Generation**: Valid molecule rate, molecular properties (MW, LogP, QED)
- **Latent Space**: Property distributions, vector analysis

## Current Status

✅ **Fully Functional Production System**
- Complete architecture with proper 8×256 latent matrix
- Relative positional encoding implemented in decoder
- Real SELFIES data pipeline working
- Comprehensive evaluation capabilities
- Configuration system with hyperparameter search
- aria2-based multi-threaded data downloading
- All runtime issues resolved
- Git Bash/WSL support for Windows

## Key Improvements

### System Enhancements
- **Configuration Management**: Centralized YAML-based parameter system
- **Hyperparameter Search**: Grid and random search with automatic tracking
- **Data Pipeline**: aria2 multi-threaded downloader for 1900+ SMI files
- **Evaluation System**: Multi-mode evaluation with comprehensive metrics
- **Cross-Platform**: Windows (Git Bash/WSL) and Unix bash scripts
- **Error Handling**: Failed download tracking and retry mechanisms

### Technical Features
- **8×256 Latent Matrix**: Proper concatenation from encoder layers 1,2,3,4,5,6,8,10
- **Relative Positional Encoding**: Layer 1 absolute, layers 2-6 relative encoding
- **Real Data Pipeline**: SELFIES tokenization with 150+ token vocabulary
- **Configurable Architecture**: All parameters adjustable via YAML/config
- **Comprehensive Logging**: TensorBoard integration with detailed metrics
- **Checkpoint Management**: Automatic saving and loading of model states
- **Download System**: Multi-threaded aria2-based data downloading with failure handling
- **Cross-Platform**: Windows (Git Bash/WSL) and Unix shell support

## Download System

The updated download system uses aria2 for multi-threaded downloads:

```bash
# Download using default URI file (data/ZINC-downloader-2D-smi.uri)
./bootstrap.sh --download

# Download using custom URI file
./bootstrap.sh --download custom.uri

# Manual download with aria2
aria2c --input-file=data/ZINC-downloader-2D-smi.uri \
       --dir=data/raw \
       --max-concurrent-downloads=10 \
       --max-connection-per-server=4 \
       --continue=true \
       --max-tries=3 \
       --retry-wait=30 \
       --timeout=60
```

Failed downloads are logged to timestamped files in the format `YYYYMMDD_HHMMSS_fail.uri`.