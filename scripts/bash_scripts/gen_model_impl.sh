#!/bin/bash

# Generator Model Training Script
# This script runs the generator model training with specified hyperparameters and logs output

# Configuration
EPOCHS=${1:-50}  # Default to 50 epochs if not specified
BATCH_SIZE=${2:-32}  # Default batch size
LR=${3:-0.002}  # Default learning rate
DATA_FRAC=${4:-0.25}  # Default to 2% of dataset
MODEL=${5:-gan}  # Default model type: gan, vae, or diffusion

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/gen_model_${MODEL}_${TIMESTAMP}.log"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Print minimal information to console
echo "Starting Generator Model Training (${MODEL})..."
echo "Log file: $LOG_FILE"

# Change to scripts directory
cd "$SCRIPT_DIR"

# Write header to log file FIRST
{
    echo "Starting Generator Model Training at $(date)"
    echo "Model: $MODEL"
    echo "Epochs: $EPOCHS"
    echo "Batch Size: $BATCH_SIZE"
    echo "Learning Rate: $LR"
    echo "Data Fraction: $DATA_FRAC"
    echo "Process ID: $$"
    echo "======================================================"
} > "$LOG_FILE"

# Run Python in background with nohup to ensure it survives terminal exit
nohup python -u gen_model_impl.py --epochs $EPOCHS --batch-size $BATCH_SIZE --lr $LR --data-frac $DATA_FRAC --model $MODEL >> "$LOG_FILE" 2>&1 &

# Get the process ID 
PID=$!

# Print to console only (avoid file race condition)
echo "Process ID: $PID"

# Append PID to log file
echo "Python Process PID: $PID" >> "$LOG_FILE"
