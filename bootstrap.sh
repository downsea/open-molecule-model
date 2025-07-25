#!D:\app\Scoop\shims\bash.exe

# Exit on error
set -e

# --- Configuration ---
VENV_DIR=".venv"
ZINC_DATASET_URL="https://zinc.docking.org/20/substances/subsets/for-sale.csv.gz"
DATA_DIR="data"

# --- Functions ---

# Function to display help message
usage() {
  echo "Usage: $0 [ -i | --install | -d | --download | -p | --process | -t | --train | -e | --evaluate | -b | --board | -h | --help ]"
  echo "Options:"
  echo "  -i, --install     Install dependencies."
  echo "  -d, --download    Download the ZINC dataset."
  echo "  -p, --process     Process the downloaded data."
  echo "  -t, --train       Run the training script."
  echo "  -e, --evaluate    Run the evaluation script."
  echo "  -b, --board       Launch TensorBoard."
  echo "  -h, --help        Display this help message."
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
  python -m src.train
}

# Function to run evaluation
run_evaluate() {
  echo "Running evaluation..."
  source $VENV_DIR/Scripts/activate
  python -m src.evaluate
}

# Function to launch TensorBoard
run_tensorboard() {
  echo "Launching TensorBoard..."
  source $VENV_DIR/Scripts/activate
  tensorboard --logdir=runs
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
    -t|--train) run_train; shift ;;
    -e|--evaluate) run_evaluate; shift ;;
    -b|--board) run_tensorboard; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown parameter passed: $1"; usage; exit 1 ;;
  esac
done
