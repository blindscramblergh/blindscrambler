#!/bin/bash

# Move to project root (important if script is run from elsewhere)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1

# Run python script in background
python3 binaryclassification_animate_impl.py \
    > logs/weights_animation.out \
    2>&1 &

# Print process ID
echo "Started weights animation with process ID: $!"
