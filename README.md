# BlindScrambler

A deep learning project implementing CNN architectures for image classification and accelerometer-based classification tasks.

## CNN Architecture

The **ImageNetCNN** is a 5-layer convolutional neural network designed for image classification on ImageNet-scale datasets. The architecture features:

- **5 Convolutional Blocks**: Progressive feature extraction with channels: 3 → 64 → 128 → 256 → 512 → 512
- **Batch Normalization**: Applied after each convolutional layer for training stability
- **Max Pooling**: Spatial dimensionality reduction after each block
- **Global Average Pooling**: Converts spatial features to a fixed-size vector
- **2 Fully Connected Layers**: 512 → 1024 → 1000 (for ImageNet classification)
- **Total Trainable Parameters**: ~5.46 million

The network uses ReLU activations and 3×3 kernels throughout, with Dropout (p=0.2) applied before the final classification layer.

## Loading and Using ONNX Models

The trained models are exported as ONNX format for efficient inference. Use `onnxruntime` to load and run predictions:

### Installation

```bash
uv add onnxruntime
```

### CNN Model Inference

Load and use the trained CNN model (`models_CNN_epoch_40.onnx`):

```python
import onnxruntime as rt
import numpy as np

# Load the model
sess = rt.InferenceSession("models/models_CNN_epoch_40.onnx")

# Get input/output names
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# Prepare input (example: 224x224 RGB image, normalized)
# Shape: (1, 3, 224, 224) - batch size 1, 3 channels, 224x224 pixels
image = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Run inference
predictions = sess.run([output_name], {input_name: image})[0]

# Get predicted class
predicted_class = np.argmax(predictions[0])
print(f"Predicted class: {predicted_class}")
```

### ACC Classifier Model Inference

Load and use the accelerometer classifier model (`models_ACC_epoch_30.onnx`):

```python
import onnxruntime as rt
import numpy as np

# Load the model
sess = rt.InferenceSession("models/models_ACC_epoch_30.onnx")

# Get input/output names
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# Prepare input (accelerometer features with time lags)
# Adjust shape based on your feature dimension
features = np.random.randn(1, feature_dim).astype(np.float32)

# Run inference
predictions = sess.run([output_name], {input_name: features})[0]

# Get predicted class
predicted_class = np.argmax(predictions[0])
print(f"Predicted class: {predicted_class}")
```

### Batch Inference

For processing multiple samples at once:

```python
import onnxruntime as rt
import numpy as np

sess = rt.InferenceSession("models/models_CNN_epoch_40.onnx")
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# Batch of images (shape: 32, 3, 224, 224)
batch = np.random.randn(32, 3, 224, 224).astype(np.float32)

# Run inference on batch
predictions = sess.run([output_name], {input_name: batch})[0]

# Get predictions for each sample
predicted_classes = np.argmax(predictions, axis=1)
print(f"Batch predictions: {predicted_classes}")
```

## Key Notes

- ONNX Runtime provides fast, cross-platform inference without requiring PyTorch at runtime
- Ensure input data is normalized appropriately for your model
- Models expect `float32` inputs
- Batch processing significantly speeds up throughput
