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
  echo "Usage: $0 [ -i | --install | -d | --download [URI_FILE] | -p | --process | --standardize | -t | --train | -e | --evaluate | --analyze | --benchmark | -b | --board | -s | --search | -h | --help ]"
  echo "Options:"
  echo "  -i, --install     Install dependencies."
  echo "  -d, --download    Download the ZINC dataset using aria2 with optimized settings."
  echo "                    Optionally specify URI file (default: data/ZINC-downloader-2D-smi.uri)"
  echo "  -p, --process     Basic processing: filter invalid SMILES and remove duplicates."
  echo "                    Saves all valid unique SMILES to data/processed/"
  echo "  --analyze         Analyze ALL processed data and generate comprehensive reports."
  echo "                    Loads from data/processed/ and saves reports to data/data_report/"
  echo "  --standardize     Apply config-based filters and create train/val/test splits."
  echo "                    Loads from data/processed/ and saves to data/standard/"
  echo "                    Optionally specify config file (default: config_optimized.yaml)"
  echo "  -t, --train       Run the training script using standardized data."
  echo "  -e, --evaluate    Run the evaluation script."
  echo "  --benchmark       Run performance benchmark on processing pipeline."
  echo "  -b, --board       Launch TensorBoard."
  echo "  -s, --search      Run hyperparameter search."
  echo "  -h, --help        Display this help message."
  echo ""
  echo "🔄 New Data Pipeline (Restructured):"
  echo "  1. ./bootstrap.sh --download     # Download raw ZINC data → data/raw/"
  echo "  2. ./bootstrap.sh --process      # Basic filter + dedup → data/processed/"
  echo "  3. ./bootstrap.sh --analyze      # Analyze all data → data/data_report/"
  echo "  4. ./bootstrap.sh --standardize  # Apply filters + split → data/standard/"
  echo "  5. ./bootstrap.sh --train        # Train using train data from data/standard/"
  echo "  6. ./bootstrap.sh --evaluate     # Evaluate using test data from data/standard/"
  echo ""
  echo "Examples:"
  echo "  $0 --download                    # Download using default URI file with optimizations"
  echo "  $0 --download custom.uri         # Download using custom URI file"
  echo "  $0 --process                     # Basic processing (filter + dedup only)"
  echo "  $0 --analyze                     # Analyze all processed molecules"
  echo "  $0 --standardize config_optimized.yaml  # Create training data with filters"
  echo "  $0 --benchmark                   # Run performance benchmark"
  echo ""
  echo "🔄 GraphDiT Pipeline:"
  echo "  $0 --train-graphdit              # Train GraphDiT model"
  echo "  $0 --generate-graphdit           # Generate molecules with GraphDiT"
  echo "  $0 --optimize-graphdit           # Optimize molecules with GraphDiT"
  echo ""
  echo "Training options:"
  echo "  --config FILE     Use specific config file (default: config.yaml)"
  echo "  --lr LR           Learning rate"
  echo "  --batch-size BS   Batch size"
  echo "  --hidden-dim HD   Hidden dimension"
  echo "  --epochs N        Number of epochs"
  echo "  --device DEVICE   Device (cuda/cpu)"
  echo ""
  echo "Performance features:"
  echo "  🚀 Optimized aria2 downloads with system capability detection"
  echo "  📊 Resource monitoring during processing and downloads"
  echo "  ⚡ Performance benchmarking tools"
  echo "  🔧 Automatic system optimization based on available resources"
  echo "  🔄 Clean separation: basic processing → analysis → standardization → training"
}

# Function to install dependencies
install_deps() {
  # Check if uv is available, if not try common Windows paths
  if ! command -v uv &> /dev/null; then
    echo "uv not found in PATH, trying common locations..."
    
    # Common Windows uv locations
    if [ -f "/c/Users/$USER/AppData/Roaming/uv/uv.exe" ]; then
      export PATH="/c/Users/$USER/AppData/Roaming/uv:$PATH"
    elif [ -f "/d/app/Scoop/shims/uv.exe" ]; then
      export PATH="/d/app/Scoop/shims:$PATH"
    elif [ -f "/c/Program Files/uv/uv.exe" ]; then
      export PATH="/c/Program Files/uv:$PATH"
    elif command -v python &> /dev/null; then
      echo "uv not found, using pip instead..."
      if [ ! -d "$VENV_DIR" ]; then
        echo "Creating virtual environment..."
        python -m venv $VENV_DIR
      fi
      source $VENV_DIR/bin/activate 2>/dev/null || source $VENV_DIR/Scripts/activate
      pip install -r requirements.txt
      python test_cuda.py
      echo "Installation complete with pip."
      return 0
    else
      echo "Error: uv not found and python not available"
      echo "Please install uv first: https://github.com/astral-sh/uv"
      exit 1
    fi
  fi
  
  if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    uv venv $VENV_DIR --python 3.11
  fi
  
  echo "Installing PyTorch with CUDA 12.8..."
  uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
  
  echo "Installing PyTorch Geometric dependencies..."
  # Install compatible versions that work with CUDA 12.8
  uv pip install torch-geometric
  uv pip install torch-scatter torch-sparse || echo "Warning: torch-scatter/torch-sparse installation failed, continuing..."
  
  echo "Installing remaining dependencies..."
  uv pip install -r requirements.txt
  
  echo "Testing CUDA setup..."
  source $VENV_DIR/bin/activate 2>/dev/null || source $VENV_DIR/Scripts/activate
  python test_cuda.py
  
  echo "Installation complete."
}

# Function to detect system capabilities
detect_system_capabilities() {
  # Detect CPU cores
  if command -v nproc &> /dev/null; then
    CPU_CORES=$(nproc)
  elif [ -f /proc/cpuinfo ]; then
    CPU_CORES=$(grep -c ^processor /proc/cpuinfo)
  else
    CPU_CORES=4
  fi
  
  # Detect memory
  if command -v free &> /dev/null; then
    MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
  else
    MEMORY_GB=8
  fi
  
  # Calculate optimal settings
  MAX_CONCURRENT=$((CPU_CORES * 2))
  MAX_CONNECTIONS=$((CPU_CORES))
  
  # Limit based on memory
  if [ "$MEMORY_GB" -lt 4 ]; then
    MAX_CONCURRENT=$((MAX_CONCURRENT / 2))
  fi
  
  echo "🖥️  System: ${CPU_CORES} cores, ${MEMORY_GB}GB RAM"
  echo "⚙️  Optimized: ${MAX_CONCURRENT} concurrent downloads, ${MAX_CONNECTIONS} connections per server"
}

# Function to monitor system resources
monitor_system_resources() {
  local process_name="$1"
  local log_file="$2"
  local interval="${3:-30}"
  
  echo "📊 Starting resource monitoring (interval: ${interval}s)"
  
  while pgrep -f "$process_name" > /dev/null; do
    {
      echo "$(date '+%Y-%m-%d %H:%M:%S'): $(ps -o pid,pcpu,pmem,cmd -C "$process_name" --no-headers | head -1)"
      if command -v free &> /dev/null; then
        echo "$(date '+%Y-%m-%d %H:%M:%S'): Memory: $(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')"
      fi
      if command -v iostat &> /dev/null; then
        echo "$(date '+%Y-%m-%d %H:%M:%S'): I/O: $(iostat -d 1 1 | tail -n +4 | awk '{print $4 " " $5}' | head -1)"
      fi
    } >> "$log_file"
    sleep "$interval"
  done
}

# Enhanced download function with optimized aria2 settings
download_data() {
  echo "🚀 Downloading ZINC dataset with optimized settings..."
  
  # Detect system capabilities
  detect_system_capabilities
  
  # Default URI file
  URI_FILE="${1:-data/ZINC-downloader-2D-smi.uri}"
  
  # Create directories
  mkdir -p "$DATA_DIR/raw"
  mkdir -p "$DATA_DIR"
  
  # Check if URI file exists
  if [ ! -f "$URI_FILE" ]; then
    echo "❌ URI file not found: $URI_FILE"
    echo "Please provide a valid URI file containing SMI file URLs"
    exit 1
  fi
  
  # Check if aria2c is installed
  if ! command -v aria2c &> /dev/null; then
    echo "📦 aria2c not found. Installing aria2..."
    
    # Try different installation methods
    if command -v apt-get &> /dev/null; then
      sudo apt-get update && sudo apt-get install -y aria2
    elif command -v yum &> /dev/null; then
      sudo yum install -y aria2
    elif command -v brew &> /dev/null; then
      brew install aria2
    elif command -v scoop &> /dev/null; then
      scoop install aria2
    else
      echo "❌ Please install aria2 manually: https://aria2.github.io/"
      exit 1
    fi
  fi
  
  # Create timestamp for logs
  DATE_TIME=$(date +"%Y%m%d_%H%M%S")
  FAIL_FILE="$DATA_DIR/${DATE_TIME}_failed.uri"
  RESOURCE_LOG="$DATA_DIR/${DATE_TIME}_resources.log"
  
  # Create temporary URI file for files that need downloading
  TEMP_URI_FILE="$DATA_DIR/temp_download.uri"
  
  # Filter out already downloaded files
  echo "🔍 Checking for existing files..."
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
  FILES_TO_DOWNLOAD=$(wc -l < "$TEMP_URI_FILE" 2>/dev/null || echo "0")
  TOTAL_FILES=$(grep -v '^#' "$URI_FILE" | grep -v '^$' | wc -l 2>/dev/null || echo "0")
  EXISTING_FILES=$((TOTAL_FILES - FILES_TO_DOWNLOAD))
  
  echo "📊 Found $EXISTING_FILES existing files, $FILES_TO_DOWNLOAD files need to be downloaded."
  
  if [ "$FILES_TO_DOWNLOAD" -eq 0 ]; then
    echo "✅ All files already exist. Skipping download."
    rm -f "$TEMP_URI_FILE"
    return 0
  fi
  
  echo "⬇️  Downloading $FILES_TO_DOWNLOAD SMI files..."
  echo "📁 Saving to: $DATA_DIR/raw/"
  echo "📋 Failed downloads will be logged to: $FAIL_FILE"
  echo "📊 Resource usage will be logged to: $RESOURCE_LOG"
  
  # Start resource monitoring in background
  monitor_system_resources "aria2c" "$RESOURCE_LOG" 30 &
  MONITOR_PID=$!
  
  # Use optimized aria2 settings
  aria2c --input-file="$TEMP_URI_FILE" \
         --dir="$DATA_DIR/raw" \
         --max-concurrent-downloads="$MAX_CONCURRENT" \
         --max-connection-per-server="$MAX_CONNECTIONS" \
         --min-split-size=1M \
         --split=4 \
         --continue=true \
         --max-tries=5 \
         --retry-wait=10 \
         --timeout=30 \
         --connect-timeout=10 \
         --lowest-speed-limit=1K \
         --max-overall-download-limit=0 \
         --disk-cache=64M \
         --file-allocation=falloc \
         --log-level=notice \
         --summary-interval=10 \
         --download-result=full \
         --log="$DATA_DIR/download_${DATE_TIME}.log" \
         --save-session="$FAIL_FILE" \
         --save-session-interval=30 \
         --console-log-level=info
  
  # Stop resource monitoring
  kill $MONITOR_PID 2>/dev/null || true
  
  # Clean up temporary file
  rm -f "$TEMP_URI_FILE"
  
  echo "✅ Download completed!"
  
  # Check if any downloads failed
  if [ -f "$FAIL_FILE" ] && [ -s "$FAIL_FILE" ]; then
    echo "⚠️  Some downloads failed. Check $FAIL_FILE for failed URLs."
    echo "Failed URLs:"
    cat "$FAIL_FILE"
  else
    echo "🎉 All downloads completed successfully!"
    # Clean up empty fail file
    rm -f "$FAIL_FILE"
  fi
  
  # Show resource usage summary
  if [ -f "$RESOURCE_LOG" ]; then
    echo "📊 Resource usage summary saved to: $RESOURCE_LOG"
  fi
}

# Function to benchmark processing performance
benchmark_processing() {
  echo "🏃 Running performance benchmark..."
  
  local start_time=$(date +%s)
  local test_file="$DATA_DIR/benchmark_test.smi"
  local benchmark_size=10000
  
  # Create test dataset if it doesn't exist
  if [ ! -f "$test_file" ]; then
    echo "📝 Creating benchmark test file with $benchmark_size molecules..."
    
    # Find the first available SMI file
    local source_file=$(find "$DATA_DIR/raw" -name "*.smi" -type f | head -1)
    
    if [ -z "$source_file" ]; then
      echo "❌ No SMI files found for benchmarking"
      return 1
    fi
    
    head -n "$benchmark_size" "$source_file" > "$test_file" 2>/dev/null || {
      echo "❌ Failed to create benchmark file"
      return 1
    }
    
    echo "✅ Created benchmark file with $(wc -l < "$test_file") molecules"
  fi
  
  # Activate environment
  source $VENV_DIR/Scripts/activate
  
  # Run benchmark
  echo "⏱️  Running benchmark on $benchmark_size molecules..."
  
  python -c "
import time
import sys
import os
sys.path.append('src')

try:
    from process_data import DataProcessor
    
    print('🚀 Starting benchmark...')
    start = time.time()
    
    # Create processor with benchmark config
    processor = DataProcessor('$CONFIG_FILE')
    
    # Process test file
    smiles_iterator = processor.load_and_parse_molecules_streaming('$test_file')
    smiles_list = list(smiles_iterator)
    
    if smiles_list:
        processed = processor.process_molecules_multiprocessing(smiles_list, 'benchmark')
        end = time.time()
        
        duration = end - start
        molecules_per_second = len(processed) / duration if duration > 0 else 0
        
        print(f'📊 Benchmark Results:')
        print(f'   Processed: {len(processed):,} molecules')
        print(f'   Duration: {duration:.1f} seconds')
        print(f'   Speed: {molecules_per_second:.0f} molecules/second')
        print(f'   Estimated time for 1M molecules: {1000000/molecules_per_second/60:.1f} minutes')
        
        # Save benchmark results
        with open('$DATA_DIR/benchmark_results.txt', 'w') as f:
            f.write(f'Benchmark Results ({time.strftime(\"%Y-%m-%d %H:%M:%S\")})\n')
            f.write(f'Processed: {len(processed):,} molecules\n')
            f.write(f'Duration: {duration:.1f} seconds\n')
            f.write(f'Speed: {molecules_per_second:.0f} molecules/second\n')
            f.write(f'Estimated 1M molecules: {1000000/molecules_per_second/60:.1f} minutes\n')
    else:
        print('❌ No molecules processed in benchmark')
        
except Exception as e:
    print(f'❌ Benchmark failed: {e}')
    sys.exit(1)
"
  
  local end_time=$(date +%s)
  local duration=$((end_time - start_time))
  echo "✅ Benchmark completed in ${duration} seconds"
  
  if [ -f "$DATA_DIR/benchmark_results.txt" ]; then
    echo "📋 Results saved to: $DATA_DIR/benchmark_results.txt"
    cat "$DATA_DIR/benchmark_results.txt"
  fi
}

# Enhanced processing function with resource management
process_data() {
  echo "🔄 Processing data with resource monitoring..."
  
  # Check available resources
  detect_system_capabilities
  
  # Start resource monitoring
  local start_time=$(date +%s)
  local resource_log="$DATA_DIR/processing_resources_$(date +%Y%m%d_%H%M%S).log"
  
  # Activate virtual environment
  source $VENV_DIR/Scripts/activate
  
  # Start resource monitoring in background
  monitor_system_resources "python" "$resource_log" 60 &
  local monitor_pid=$!
  
  # Run processing with error handling
  if python src/process_data.py --config "$CONFIG_FILE"; then
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo "✅ Processing completed successfully in ${duration} seconds"
    
    # Stop monitoring
    kill $monitor_pid 2>/dev/null || true
    
    # Show resource summary
    if [ -f "$resource_log" ]; then
      echo "📊 Resource usage logged to: $resource_log"
      echo "📈 Processing summary:"
      echo "   Duration: ${duration} seconds"
      echo "   Resource log: $resource_log"
    fi
    
  else
    echo "❌ Processing failed"
    kill $monitor_pid 2>/dev/null || true
    return 1
  fi
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

# Function to train GraphDiT model
run_train_graphdit() {
  echo "🧬 Training GraphDiT model..."
  source $VENV_DIR/Scripts/activate
  
  # Ensure Python path includes the project root
  export PYTHONPATH="${PWD}:$PYTHONPATH"
  
  # Parse training arguments
  TRAIN_ARGS=""
  while [[ $# -gt 0 ]]; do
    case $1 in
      --config)
        TRAIN_ARGS="$TRAIN_ARGS --config $2"
        shift 2
        ;;
      --data-path)
        TRAIN_ARGS="$TRAIN_ARGS --data-path $2"
        shift 2
        ;;
      --epochs)
        TRAIN_ARGS="$TRAIN_ARGS --epochs $2"
        shift 2
        ;;
      --batch-size)
        TRAIN_ARGS="$TRAIN_ARGS --batch-size $2"
        shift 2
        ;;
      --learning-rate)
        TRAIN_ARGS="$TRAIN_ARGS --learning-rate $2"
        shift 2
        ;;
      --device)
        TRAIN_ARGS="$TRAIN_ARGS --device $2"
        shift 2
        ;;
      --save-dir)
        TRAIN_ARGS="$TRAIN_ARGS --save-dir $2"
        shift 2
        ;;
      --log-dir)
        TRAIN_ARGS="$TRAIN_ARGS --log-dir $2"
        shift 2
        ;;
      --use-wandb)
        TRAIN_ARGS="$TRAIN_ARGS --use-wandb"
        shift
        ;;
      *)
        shift
        ;;
    esac
  done
  
  python -m src.graph_dit.train_graph_dit $TRAIN_ARGS
}

# Function to generate molecules with GraphDiT
run_generate_graphdit() {
  echo "⚗️  Generating molecules with GraphDiT..."
  source $VENV_DIR/Scripts/activate
  
  # Ensure Python path includes the project root
  export PYTHONPATH="${PWD}:$PYTHONPATH"
  
  # Parse generation arguments
  GEN_ARGS=""
  while [[ $# -gt 0 ]]; do
    case $1 in
      --checkpoint)
        GEN_ARGS="$GEN_ARGS --checkpoint $2"
        shift 2
        ;;
      --config)
        GEN_ARGS="$GEN_ARGS --config $2"
        shift 2
        ;;
      --num-samples)
        GEN_ARGS="$GEN_ARGS --num-samples $2"
        shift 2
        ;;
      --temperature)
        GEN_ARGS="$GEN_ARGS --temperature $2"
        shift 2
        ;;
      --output)
        GEN_ARGS="$GEN_ARGS --output $2"
        shift 2
        ;;
      --device)
        GEN_ARGS="$GEN_ARGS --device $2"
        shift 2
        ;;
      --evaluate)
        GEN_ARGS="$GEN_ARGS --evaluate"
        shift
        ;;
      *)
        shift
        ;;
    esac
  done
  
  python -m src.graph_dit.generate_molecules $GEN_ARGS
}

# Function to optimize molecules with GraphDiT
run_optimize_graphdit() {
  echo "🔬 Optimizing molecules with GraphDiT..."
  source $VENV_DIR/Scripts/activate
  
  # Ensure Python path includes the project root
  export PYTHONPATH="${PWD}:$PYTHONPATH"
  
  # Parse optimization arguments
  OPT_ARGS=""
  while [[ $# -gt 0 ]]; do
    case $1 in
      --checkpoint)
        OPT_ARGS="$OPT_ARGS --checkpoint $2"
        shift 2
        ;;
      --critic-checkpoint)
        OPT_ARGS="$OPT_ARGS --critic-checkpoint $2"
        shift 2
        ;;
      --config)
        OPT_ARGS="$OPT_ARGS --config $2"
        shift 2
        ;;
      --input-smiles)
        OPT_ARGS="$OPT_ARGS --input-smiles $2"
        shift 2
        ;;
      --property)
        OPT_ARGS="$OPT_ARGS --property $2"
        shift 2
        ;;
      --target)
        OPT_ARGS="$OPT_ARGS --target $2"
        shift 2
        ;;
      --num-steps)
        OPT_ARGS="$OPT_ARGS --num-steps $2"
        shift 2
        ;;
      --guidance-scale)
        OPT_ARGS="$OPT_ARGS --guidance-scale $2"
        shift 2
        ;;
      --similarity-constraint)
        OPT_ARGS="$OPT_ARGS --similarity-constraint $2"
        shift 2
        ;;
      --output)
        OPT_ARGS="$OPT_ARGS --output $2"
        shift 2
        ;;
      --device)
        OPT_ARGS="$OPT_ARGS --device $2"
        shift 2
        ;;
      *)
        shift
        ;;
    esac
  done
  
  python -m src.graph_dit.optimize_molecules $OPT_ARGS
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
    --analyze)
      echo "🔬 Running comprehensive data analysis..."
      source $VENV_DIR/Scripts/activate
      python -m src.data_analysis --data-path "data/processed" --output-path "data/data_report"
      exit $?
      ;;
    --standardize)
      shift
      CONFIG_FILE="${1:-config_optimized.yaml}"
      echo "🧪 Running data standardization with config: $CONFIG_FILE"
      source $VENV_DIR/Scripts/activate
      python -m src.data_standardize --config "$CONFIG_FILE"
      exit $?
      ;;
    --benchmark)
      benchmark_processing
      exit $?
      ;;
    -s|--search) 
      shift
      run_search "$@"
      break
      ;;
    -b|--board) run_tensorboard; shift ;;
    --analyze-update)
      echo "🔬 Running data analysis and updating configurations..."
      source $VENV_DIR/Scripts/activate
      python -m src.data_analysis --data-path "data/processed" --output-path "data/data_report"
      python -m src.config_updater
      exit $?
      ;;
    --train-graphdit)
      shift
      run_train_graphdit "$@"
      break
      ;;
    --generate-graphdit)
      shift
      run_generate_graphdit "$@"
      break
      ;;
    --optimize-graphdit)
      shift
      run_optimize_graphdit "$@"
      break
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown parameter passed: $1"; usage; exit 1 ;;
  esac
done
