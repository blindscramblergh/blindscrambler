import pandas as pd
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
import subprocess
from torch.utils.data import DataLoader
from torchvision import transforms

# make a global function about choosing the best GPU available
def get_best_gpu(strategy="utilization"):
    """
    Select best GPU by utilization or memory
    """
    if strategy == "memory":
        # Use PyTorch directly for free memory
        free_mem = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.mem_get_info(i) # (free, total)
            free_mem.append(props[0])
    
        return free_mem.index(max(free_mem))

    elif strategy == "utilization":
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
    
        utilizations = [int(x.strip()) for x in result.stdout.strip().split("\n")]
        return utilizations.index(min(utilizations))


# main function for all the tasks:
if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train CNNTrainer on ImageNet dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--train_ratio", type=float, default=0.005, help="Training data ratio (default: 0.005)")
    parser.add_argument("--val_ratio", type=float, default=0.0008, help="Validation data ratio (default: 0.0008)")
    args = parser.parse_args()

    # implementing the multiclass thing
    device_id = get_best_gpu(strategy="utilization")
    device = torch.device(f"cuda:{device_id}")
    print(f"Selected GPU: {device_id}")

    # Load dataset
    dataset = load_from_disk("/data/CPE_487-587/imagenet-1k-arrow")

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    num_classes = len(train_dataset.features["label"].names)
    print(f"Number of classes: {num_classes}")

    # select subset for training and testing using command line arguments
    train_size = int(len(dataset["train"]) * args.train_ratio)      
    val_size = int(len(dataset["validation"]) * args.val_ratio)

    print(f"Training with {args.epochs} epochs, train_ratio={args.train_ratio}, val_ratio={args.val_ratio}")
    print(f"Train size: {train_size}, Validation size: {val_size}")

    train_dataset = dataset["train"].select(range(train_size))
    val_dataset = dataset["validation"].select(range(val_size))

    class_names = train_dataset.features["label"].names

    # making the transforms, for train and val images
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # apply transforms
    def preprocess_train(example):
        images = [train_transform(img.convert("RGB")) for img in example["image"]]
        labels = example["label"]

        return {
            "pixel_values": images,
            "labels": labels
        }

    def preprocess_val(example):
        images = [val_transform(img.convert("RGB")) for img in example["image"]]
        labels = example["label"]

        return {
            "pixel_values": images,
            "labels": labels
        }

    train_dataset = train_dataset.with_transform(preprocess_train)
    val_dataset = val_dataset.with_transform(preprocess_val)

    # the next step would be to create Data loaders
    def collate_fn(batch):
        """
        To extract pixel values and labels from each item.
        Returns a tuple of (images, labels) for compatibility with CNNTrainer
        """

        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = torch.tensor([item["labels"] for item in batch])

        return pixel_values, labels

    # create data loaders with pin_memory only if GPU is available
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        pin_memory=pin_memory,    # for faster GPU transfer
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )

    # Initialize model 
    model = ImageNetCNN()

    # make the optimizer and the scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    trainer = CNNTrainer(
        train_dataloader=train_loader,
        eta=0.01,
        epoch=args.epochs,
        loss_function=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        loss_vector=[],
        accuracy_vector=[],
        model=model,
        device=device,
        val_dataloader=val_loader,
        scheduler=scheduler
    )
     
    # Train the model
    trainer.train()
     
    # Evaluate on validation set
    val_loss, val_accuracy = trainer.test()

    # make plots for (1) loss and (2) accuracy
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(trainer.loss_vector, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()    

    plt.subplot(1, 2, 2)
    plt.plot(trainer.accuracy_vector, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.savefig("training_plots.png")

    print(f"Validation loss: {val_loss}")

    # save the model
    trainer.save_onnx('CNN_model.onnx')