#!/bin/bash

# Script to train ImageNet CNN model in the background
# This script runs imagenet_impl.py with specified training parameters

# Navigate to the scripts directory
cd "$(dirname "$0")/.." || exit 1

# Define training parameters
EPOCHS=10
TRAIN_RATIO=0.008
VAL_RATIO=0.004

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate log filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/imagenet_training_${TIMESTAMP}.log"

echo "Starting ImageNet CNN training in the background..."
echo "Parameters:"
echo "  Epochs: $EPOCHS"
echo "  Train Ratio: $TRAIN_RATIO"
echo "  Validation Ratio: $VAL_RATIO"
echo "  Log file: $LOG_FILE"
echo ""

# Run the training in the background with nohup
nohup python -u imagenet_impl.py \
    --epochs $EPOCHS \
    --train_ratio $TRAIN_RATIO \
    --val_ratio $VAL_RATIO \
    > "$LOG_FILE" 2>&1 &

# Capture the process ID
PID=$!

echo "Training process started with PID: $PID"
echo "To monitor progress, run: tail -f $LOG_FILE"
echo "To stop training, run: kill $PID"