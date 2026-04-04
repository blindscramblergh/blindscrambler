import pandas as pd
import numpy as np
from torch import nn 
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from blindscrambler.deepl import CNNTrainer, ImageNetCNN
from datasets import load_dataset, load_from_disk
import argparse
from datetime import datetime
import os
import sys
from torch.utils.data import DataLoader
from torchvision import transforms
import onnxruntime as ort

if __name__ == "__main__":

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    print("Loading ImageNet dataset...")
    dataset = load_from_disk("/data/CPE_487-587/imagenet-1k-arrow")

    val_dataset = dataset["validation"]
    num_classes = len(val_dataset.features["label"].names)
    class_names = val_dataset.features["label"].names
    print(f"Number of classes: {num_classes}")

    # Select 10 test images
    print("Selecting 10 test images...")
    test_size = 10
    test_dataset_subset = val_dataset.select(range(test_size))

    # Create transform for test data
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Preprocess test data
    def preprocess_test(example):
        images = [test_transform(img.convert("RGB")) for img in example["image"]]
        labels = example["label"]

        return {
            "pixel_values": images,
            "labels": labels
        }

    test_dataset_transformed = test_dataset_subset.with_transform(preprocess_test)

    # Create collate function
    def collate_fn(batch):
        """
        To extract pixel values and labels from each item.
        Returns a tuple of (images, labels) for compatibility with CNNTrainer
        """
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = torch.tensor([item["labels"] for item in batch])

        return pixel_values, labels

    # Create test data loader
    pin_memory = torch.cuda.is_available()
    test_loader = DataLoader(
        test_dataset_transformed,
        batch_size=1,
        shuffle=False,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )

    # Load ONNX model using ONNX Runtime
    print("Loading trained ONNX model from models/models_CNN_epoch_40.onnx...")
    onnx_model_path = "models/models_CNN_epoch_40.onnx"
    
    if not os.path.exists(onnx_model_path):
        print(f"Error: Model file not found at {onnx_model_path}")
        sys.exit(1)
    
    session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    print(f"Model loaded successfully. Using provider: {session.get_providers()}")

    # Get input and output names from the model
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"Input name: {input_name}, Output name: {output_name}")

    # Run inference on 10 test images
    print("\n" + "=" * 80)
    print("Running inference on 10 test images using ONNX Runtime...")
    print("=" * 80)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for idx, (image, label) in enumerate(test_loader):
            # Convert image to numpy for ONNX Runtime
            image_np = image.numpy()
            
            # Run inference
            outputs = session.run([output_name], {input_name: image_np})[0]
            
            # Get prediction
            predicted = np.argmax(outputs, axis=1)[0]
            true_label = label.item()
            
            # Update accuracy
            if predicted == true_label:
                correct += 1
            total += 1
            
            # Print results
            pred_class_name = class_names[predicted]
            true_class_name = class_names[true_label]
            match = "✓" if predicted == true_label else "✗"
            
            print(f"Image {idx+1}:")
            print(f"  Predicted: {pred_class_name}")
            print(f"  True:      {true_class_name}")
            print(f"  Match: {match}")
    
    # Calculate accuracy
    accuracy = correct / total
    print("\n" + "=" * 80)
    print(f"Test Accuracy: {accuracy:.4f} ({correct}/{total})")
    print("=" * 80)

