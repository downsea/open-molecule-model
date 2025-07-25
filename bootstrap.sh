#!/usr/bin/env bash

# Exit on error
set -e

# --- Configuration ---
VENV_DIR=".venv"
ZINC_DATASET_URL="https://zinc.docking.org/20/substances/subsets/for-sale.csv.gz"
DATA_DIR="data"
CONFIG_FILE="config.yaml"

# --- Functions ---

# Function to display help message
usage() {
  echo "Usage: $0 [ -i | --install | -d | --download | -p | --process | -t | --train | -e | --evaluate | -b | --board | -s | --search | -h | --help ]"
  echo "Options:"
  echo "  -i, --install     Install dependencies."
  echo "  -d, --download    Download the ZINC dataset."
  echo "  -p, --process     Process the downloaded data."
  echo "  -t, --train       Run the training script."
  echo "  -e, --evaluate    Run the evaluation script."
  echo "  -b, --board       Launch TensorBoard."
  echo "  -s, --search      Run hyperparameter search."
  echo "  -h, --help        Display this help message."
  echo ""
  echo "Training options:"
  echo "  --config FILE     Use specific config file (default: config.yaml)"
  echo "  --lr LR           Learning rate"
  echo "  --batch-size BS   Batch size"
  echo "  --hidden-dim HD   Hidden dimension"
  echo "  --epochs N        Number of epochs"
  echo "  --device DEVICE   Device (cuda/cpu)"
}

# Function to install dependencies
install_deps() {
  if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    uv venv $VENV_DIR
  fi
  echo "Activating virtual environment..."
  source .venv/Scripts/activate
  echo "Installing dependencies from requirements.txt..."
  uv pip install -r requirements.txt
  echo "Installation complete."
}

# Function to download the ZINC dataset
download_data() {
  echo "Downloading ZINC dataset..."
  mkdir -p $DATA_DIR
  wget -t 100   -P $DATA_DIR $ZINC_DATASET_URL
  echo "Decompressing dataset..."
  gunzip $DATA_DIR/for-sale.csv.gz
  echo "Download complete."
}

# Function to process the data
process_data() {
  echo "Processing data..."
  source $VENV_DIR/Scripts/activate
  python src/process_data.py
}

# Function to run training
run_train() {
  echo "Running training..."
  source $VENV_DIR/Scripts/activate
  
  # Parse training arguments
  TRAIN_ARGS=""
  while [[ $# -gt 0 ]]; do
    case $1 in
      --config)
        TRAIN_ARGS="$TRAIN_ARGS --config $2"
        shift 2
        ;;
      --lr)
        TRAIN_ARGS="$TRAIN_ARGS --learning-rate $2"
        shift 2
        ;;
      --batch-size)
        TRAIN_ARGS="$TRAIN_ARGS --batch-size $2"
        shift 2
        ;;
      --hidden-dim)
        TRAIN_ARGS="$TRAIN_ARGS --hidden-dim $2"
        shift 2
        ;;
      --epochs)
        TRAIN_ARGS="$TRAIN_ARGS --num-epochs $2"
        shift 2
        ;;
      --device)
        TRAIN_ARGS="$TRAIN_ARGS --device $2"
        shift 2
        ;;
      *)
        shift
        ;;
    esac
  done
  
  python -m src.train $TRAIN_ARGS
}

# Function to run evaluation
run_evaluate() {
  echo "Running evaluation..."
  source $VENV_DIR/Scripts/activate
  
  # Parse evaluation arguments
  EVAL_ARGS=""
  while [[ $# -gt 0 ]]; do
    case $1 in
      --config)
        EVAL_ARGS="$EVAL_ARGS --config $2"
        shift 2
        ;;
      --mode)
        EVAL_ARGS="$EVAL_ARGS --mode $2"
        shift 2
        ;;
      --num-samples)
        EVAL_ARGS="$EVAL_ARGS --num-samples $2"
        shift 2
        ;;
      --device)
        EVAL_ARGS="$EVAL_ARGS --device $2"
        shift 2
        ;;
      *)
        shift
        ;;
    esac
  done
  
  python -m src.evaluate $EVAL_ARGS
}

# Function to launch TensorBoard
run_tensorboard() {
  echo "Launching TensorBoard..."
  source $VENV_DIR/Scripts/activate
  tensorboard --logdir=runs
}

# Function to run hyperparameter search
run_search() {
  echo "Running hyperparameter search..."
  source $VENV_DIR/Scripts/activate
  
  # Parse search arguments
  SEARCH_ARGS=""
  while [[ $# -gt 0 ]]; do
    case $1 in
      --config)
        SEARCH_ARGS="$SEARCH_ARGS --config $2"
        shift 2
        ;;
      --random)
        SEARCH_ARGS="$SEARCH_ARGS --random-search"
        shift
        ;;
      --trials)
        SEARCH_ARGS="$SEARCH_ARGS --trials $2"
        shift 2
        ;;
      --epochs)
        SEARCH_ARGS="$SEARCH_ARGS --epochs $2"
        shift 2
        ;;
      *)
        shift
        ;;
    esac
  done
  
  python hyperparameter_search.py $SEARCH_ARGS
}

# --- Main Script ---

# Parse command-line arguments
if [ "$#" -eq 0 ]; then
  usage
  exit 1
fi

while [[ "$#" -gt 0 ]]; do
  case $1 in
    -i|--install) install_deps; shift ;;
    -d|--download) download_data; shift ;;
    -p|--process) process_data; shift ;;
    -t|--train) 
      shift
      run_train "$@"
      break
      ;;
    -e|--evaluate) 
      shift
      run_evaluate "$@"
      break
      ;;
    -s|--search) 
      shift
      run_search "$@"
      break
      ;;
    -b|--board) run_tensorboard; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown parameter passed: $1"; usage; exit 1 ;;
  esac
done
