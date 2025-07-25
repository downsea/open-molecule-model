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
  echo "Usage: $0 [ -i | --install | -d | --download [URI_FILE] | -p | --process | -t | --train | -e | --evaluate | -b | --board | -s | --search | -h | --help ]"
  echo "Options:"
  echo "  -i, --install     Install dependencies."
  echo "  -d, --download    Download the ZINC dataset using aria2."
  echo "                    Optionally specify URI file (default: data/ZINC-downloader-2D-smi.uri)"
  echo "  -p, --process     Process the downloaded data."
  echo "  -t, --train       Run the training script."
  echo "  -e, --evaluate    Run the evaluation script."
  echo "  -b, --board       Launch TensorBoard."
  echo "  -s, --search      Run hyperparameter search."
  echo "  -h, --help        Display this help message."
  echo ""
  echo "Examples:"
  echo "  $0 --download                    # Download using default URI file"
  echo "  $0 --download custom.uri         # Download using custom URI file"
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
  
  # Default URI file
  URI_FILE="${1:-data/ZINC-downloader-2D-smi.uri}"
  
  # Create directories
  mkdir -p "$DATA_DIR/raw"
  mkdir -p "$DATA_DIR"
  
  # Check if URI file exists
  if [ ! -f "$URI_FILE" ]; then
    echo "URI file not found: $URI_FILE"
    echo "Please provide a valid URI file containing SMI file URLs"
    exit 1
  fi
  
  # Check if aria2c is installed
  if ! command -v aria2c &> /dev/null; then
    echo "aria2c not found. Installing aria2..."
    
    # Try different installation methods
    if command -v apt-get &> /dev/null; then
      sudo apt-get update && sudo apt-get install -y aria2
    elif command -v yum &> /dev/null; then
      sudo yum install -y aria2
    elif command -v brew &> /dev/null; then
      brew install aria2
    else
      echo "Please install aria2 manually: https://aria2.github.io/"
      exit 1
    fi
  fi
  
  # Create timestamp for failed downloads log
  DATE_TIME=$(date +"%Y%m%d_%H%M%S")
  FAIL_FILE="$DATA_DIR/${DATE_TIME}_fail.uri"
  
  # Create temporary URI file for files that need downloading
  TEMP_URI_FILE="$DATA_DIR/temp_download.uri"
  
  # Filter out already downloaded files
  echo "Checking for existing files..."
  > "$TEMP_URI_FILE"
  while IFS= read -r url; do
    if [[ -n "$url" && ! "$url" =~ ^# ]]; then
      # Extract filename from URL
      filename=$(basename "$url")
      if [ ! -f "$DATA_DIR/raw/$filename" ]; then
        echo "$url" >> "$TEMP_URI_FILE"
      fi
    fi
  done < "$URI_FILE"
  
  # Count files to download
  FILES_TO_DOWNLOAD=$(wc -l < "$TEMP_URI_FILE")
  TOTAL_FILES=$(grep -v '^#' "$URI_FILE" | grep -v '^$' | wc -l)
  EXISTING_FILES=$((TOTAL_FILES - FILES_TO_DOWNLOAD))
  
  echo "Found $EXISTING_FILES existing files, $FILES_TO_DOWNLOAD files need to be downloaded."
  
  if [ "$FILES_TO_DOWNLOAD" -eq 0 ]; then
    echo "All files already exist. Skipping download."
    rm -f "$TEMP_URI_FILE"
    return 0
  fi
  
  echo "Downloading $FILES_TO_DOWNLOAD SMI files..."
  echo "Saving to: $DATA_DIR/raw/"
  echo "Failed downloads will be logged to: $FAIL_FILE"
  
  # Use aria2 for multi-threaded download with filtered list
  aria2c --input-file="$TEMP_URI_FILE" \
         --dir="$DATA_DIR/raw" \
         --max-concurrent-downloads=10 \
         --max-connection-per-server=4 \
         --continue=true \
         --max-tries=3 \
         --retry-wait=30 \
         --timeout=60 \
         --log-level=info \
         --log="$DATA_DIR/download.log" \
         --out="%f" \
         --save-session="$FAIL_FILE" \
         --save-session-interval=60 \
         --console-log-level=info
  
  # Clean up temporary file
  rm -f "$TEMP_URI_FILE"
  
  echo "Download completed!"
  
  # Check if any downloads failed
  if [ -f "$FAIL_FILE" ] && [ -s "$FAIL_FILE" ]; then
    echo "Some downloads failed. Check $FAIL_FILE for failed URLs."
    echo "Failed URLs:"
    cat "$FAIL_FILE"
  else
    echo "All downloads completed successfully!"
    # Clean up empty fail file
    rm -f "$FAIL_FILE"
  fi
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
    -d|--download) 
      shift
      if [[ "$#" -gt 0 ]] && [[ "$1" != -* ]]; then
        download_data "$1"
        shift
      else
        download_data
      fi
      ;;
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
