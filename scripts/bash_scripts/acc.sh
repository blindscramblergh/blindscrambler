#!/bin/bash

# ACC Classifier Training Script
# This script runs the ACC classifier with specified number of epochs and logs output

# Configuration
EPOCHS=${1:-30}  # Default to 50 epochs if not specified
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/acc_classifier_${TIMESTAMP}.log"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Print minimal information to console
echo "Starting ACC Classifier Training..."
echo "Log file: $LOG_FILE"

# Change to scripts directory
cd "$SCRIPT_DIR"

# Write header to log file FIRST
{
    echo "Starting ACC Classifier Training at $(date)"
    echo "Epochs: $EPOCHS"
    echo "Process ID: $$"
    echo "Script PID: $$"
    echo "======================================================"
} > "$LOG_FILE"

# Run Python in background with nohup to ensure it survives terminal exit
# DO NOT write to log from shell script after this - let Python handle all appends
nohup python -u acc_classifier_impl.py $EPOCHS >> "$LOG_FILE" 2>&1 &

# Get the process ID 
PID=$!

# Print to console only (avoid file race condition)
echo "Process ID: $PID"

# Append PID to log file
echo "Python Process PID: $PID" >> "$LOG_FILE"

