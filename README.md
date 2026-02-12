# BlindScrambler

A Python library for machine learning and deep learning

## Installation

### Prerequisites

- Python >= 3.12
- Rust (for building from source)
- FFmpeg (required for Manim video rendering)

### Install using PyPi

- pip install blindscrambler==0.1.15

### Install from Source

```bash
pip install -e .
```

This will install all required dependencies including:
- PyTorch
- Manim (animation library)
- NumPy, Pandas, Matplotlib
- scikit-learn, SciPy

## Running the Weight Animation

The weight animation visualizes how neural network weights evolve during training on a binary classification task.

### Quick Start

#### Option 1: Using the Bash Script (Recommended)

```bash
cd scripts
bash bash_scripts/weight_animation.sh
```

This will:
- Start the training and animation process in the background
- Log output to `scripts/logs/weights_animation.out`
- Display the process ID for monitoring

#### Option 2: Run Python Script Directly

```bash
cd scripts
python binaryclassification_animate_impl.py
```

### What the Animation Does

1. **Training Phase**: Trains a 2-layer neural network for binary classification
   - Input features: 40,000 dimensions
   - Training samples: 200
   - Training epochs: 5,000
   - Learning rate: 0.01
   - Uses GPU if available (CUDA), otherwise CPU

2. **Animation Phase**: Creates 4 separate video animations showing the evolution of:
   - Weight matrix 1
   - Weight matrix 2
   - Weight matrix 3
   - Weight matrix 4

3. **Output**: Videos are saved in the current directory as:
   - `weight_1_evolution.mp4`
   - `weight_2_evolution.mp4`
   - `weight_3_evolution.mp4`
   - `weight_4_evolution.mp4`

### Configuration

You can modify the training parameters in `scripts/binaryclassification_animate_impl.py`:

```python
# Adjust dataset size
n = 200      # Number of samples
d = 40000    # Number of features

# Adjust training parameters
result = binary_classification(d, n, epochs=5000, lr=0.01)
```

To enable loss plotting, set:
```python
plot = True
```

### Animation Parameters

Adjust animation settings by modifying the `animate_large_heatmap` call:

```python
animation.animate_large_heatmap(
    weights[i], 
    dt=0.04,              # Time per frame (seconds)
    file_name=f"weight_{i + 1}_evolution",
    title_str="Weight Evolution"
)
```

### Monitoring Progress

- Check the log file: `cat scripts/logs/weights_animation.out`
- Monitor the process: `ps aux | grep binaryclassification`
- Training progress and rendering status are printed to the log

## Project Structure

```
blindscrambler/
├── src/blindscrambler/          # Python package
│   ├── animation/               # Animation modules
│   ├── deepl/                   # Deep learning models
│   ├── model/                   # ML models (logit, regression)
│   └── matrix/                  # Matrix operations
├── scripts/                     # Executable scripts
│   ├── binaryclassification_animate_impl.py
│   └── bash_scripts/            # Automation scripts
└── target/                      # Rust build artifacts
```

## Authors

- blindscramblergh (blindscrambler@gmail.com)
