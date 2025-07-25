# bootstrap.ps1

# Exit on error
$ErrorActionPreference = 'Stop'

# --- Configuration ---
$VenvDir = ".venv"
$ZincDatasetUrl = "https://zinc.docking.org/20/substances/subsets/for-sale.csv.gz"
$DataDir = "data"
$CompressedDataFile = Join-Path $DataDir "for-sale.csv.gz"
$DecompressedDataFile = Join-Path $DataDir "for-sale.csv"

# --- Parameter Definition ---
param(
    [switch]$Install,
    [switch]$Download,
    [switch]$Process,
    [switch]$Train,
    [switch]$Evaluate,
    [switch]$Help
)

# --- Functions ---

# Function to display help message
function Show-Usage {
    Write-Host "Usage: .\bootstrap.ps1 [ -Install | -Download | -Process | -Train | -Evaluate | -Help ]"
    Write-Host "Options:"
    Write-Host "  -Install      Install dependencies."
    Write-Host "  -Download     Download the ZINC dataset."
    Write-Host "  -Process      Process the downloaded data."
    Write-Host "  -Train        Run the training script."
    Write-Host "  -Evaluate     Run the evaluation script."
    Write-Host "  -Help         Display this help message."
}

# Function to install dependencies
function Install-Dependencies {
    if (-not (Test-Path $VenvDir)) {
        Write-Host "Creating virtual environment in $VenvDir..."
        uv venv $VenvDir
    }
    Write-Host "Installing dependencies..."
    $UvPath = Join-Path $VenvDir "Scripts" "uv.exe"
    & $UvPath pip install -r requirements.txt
    Write-Host "Installation complete."
    Write-Host "To activate the environment in your shell, run: .\.venv\Scripts\Activate.ps1"
}

# Function to download the ZINC dataset
function Download-Data {
    Write-Host "Starting ZINC dataset download..."
    if (-not (Test-Path $DataDir)) {
        Write-Host "Creating data directory: $DataDir"
        New-Item -ItemType Directory -Path $DataDir
    }
    Write-Host "Downloading from $ZincDatasetUrl to $CompressedDataFile"
    Invoke-WebRequest -Uri $ZincDatasetUrl -OutFile $CompressedDataFile -Verbose
    Write-Host "Download command finished."

    if (Test-Path $CompressedDataFile) {
        Write-Host "Decompressing dataset: $CompressedDataFile"
        tar -xzf $CompressedDataFile -C $DataDir
        Write-Host "Decompression command finished."
        Remove-Item $CompressedDataFile
        Write-Host "Removed compressed file."
        Write-Host "Download and decompression complete. Dataset at $DecompressedDataFile"
    } else {
        Write-Host "Error: Download failed. Compressed file not found at $CompressedDataFile"
    }
}

# Function to process the data
function Process-Data {
    Write-Host "Processing data..."
    if (-not (Test-Path (Join-Path $VenvDir "Scripts" "python.exe"))) {
        Write-Host "Virtual environment not found or dependencies not installed. Please run with -Install first."
        exit 1
    }
    $PythonPath = Join-Path $VenvDir "Scripts" "python.exe"
    & $PythonPath src/process_data.py
}

# Function to run training
function Run-Train {
    Write-Host "Running training..."
    if (-not (Test-Path (Join-Path $VenvDir "Scripts" "python.exe"))) {
        Write-Host "Virtual environment not found or dependencies not installed. Please run with -Install first."
        exit 1
    }
    $PythonPath = Join-Path $VenvDir "Scripts" "python.exe"
    & $PythonPath -m src.train
}

# Function to run evaluation
function Run-Evaluate {
    Write-Host "Running evaluation..."
    if (-not (Test-Path (Join-Path $VenvDir "Scripts" "python.exe"))) {
        Write-Host "Virtual environment not found or dependencies not installed. Please run with -Install first."
        exit 1
    }
    $PythonPath = Join-Path $VenvDir "Scripts" "python.exe"
    & $PythonPath -m src.evaluate
}


# --- Main Script ---

if ($Help -or -not ($Install -or $Download -or $Process -or $Train -or $Evaluate)) {
    Show-Usage
    exit 0
}

if ($Install) {
    Install-Dependencies
}

if ($Download) {
    Download-Data
}

if ($Process) {
    Process-Data
}

if ($Train) {
    Run-Train
}

if ($Evaluate) {
    Run-Evaluate
}