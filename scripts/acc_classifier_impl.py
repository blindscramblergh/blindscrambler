import pandas as pd
import polars as pl
from torch import nn 
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from datasets import load_dataset
from datasets import load_from_disk
from torchvision import transforms
from torchvision.ops import sigmoid_focal_loss
import sys
import argparse
from torch.utils.data import DataLoader
import subprocess
import glob
from sklearn.model_selection import train_test_split
from blindscrambler.deepl import Dataset, Resblock1d, ResNet, ACCTrainer, dice_loss

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

if __name__ == "__main__":

    # implementing the multiclass thing
    device_id = get_best_gpu(strategy="utilization")
    device = torch.device(f"cuda:{device_id}")
    print(f"Selected GPU: {device_id}")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train ACC Classifier")
    parser.add_argument("epochs", type=int, nargs='?', default=50, help="Number of training epochs (default: 50)")
    args = parser.parse_args()
    
    print("Oh, look at me I am implementing acc classifier")
    print(f"Training with {args.epochs} epochs")
    
    preprocessed_data_saved = True
    # get the data and store it somewhere as a .csv file as "ACC_processed_data"
    ACC = Dataset("/data/CPE_487-587/ACCDataset", "decoded_wheel_speed_fl.csv", "acc_status.csv")
    preprocessed_data = ACC.read_data(preprocessed_data_saved, "/home/sar0033/blindscrambler/scripts/data/ACC_processed_data.csv")
    
    # Convert to pandas for easier preprocessing
    data_pd = preprocessed_data.to_pandas()
    
    # Separate features and labels
    X = data_pd.drop(columns=['time', 'acc_label']).values.astype(np.float32)
    y = data_pd['acc_label'].values.astype(np.float32)
    
    print(f"Dataset shape: {X.shape}, Label shape: {y.shape}")
    print(f"Class distribution: {np.bincount(y.astype(int))}")
    
    # Split data: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # Normalize features using training data statistics
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    print(f"Features normalized - Train mean: {X_train.mean():.4f}, std: {X_train.std():.4f}")
    
    # Convert to PyTorch tensors
    # Add channel dimension for Conv1d: (batch_size, sequence_length) -> (batch_size, 1, sequence_length)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    # Create simple datasets
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, features, labels):
            self.features = features
            self.labels = labels
        
        def __len__(self):
            return len(self.features)
        
        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]
    
    train_dataset = SimpleDataset(X_train_tensor, y_train_tensor)
    val_dataset = SimpleDataset(X_val_tensor, y_val_tensor)
    test_dataset = SimpleDataset(X_test_tensor, y_test_tensor)
    
    # Create data loaders
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    # Initialize model
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ResNet()
    
    # Initialize loss function (Dice Loss for imbalanced binary classification)
    loss_function = dice_loss
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Initialize learning rate scheduler (optional)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Initialize vectors to track metrics
    loss_vector = []
    accuracy_vector = []
    
    # Initialize trainer with proper parameters
    trainer = ACCTrainer(
        eta=0.001,
        epoch=args.epochs,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        loss_vector=loss_vector,
        accuracy_vector=accuracy_vector,
        model=model,
        device=device,
        scheduler=scheduler
    )
    
    # Train the model
    print("\nStarting training...")
    trainer.train()
    
    # Save the model in ONNX format
    print("\nSaving model to ONNX format...")
    trainer.save_onnx(file_path='/home/sar0033/blindscrambler/scripts/models')
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_test_batch, y_test_batch in test_loader:
            X_test_batch = X_test_batch.to(device)
            y_test_batch = y_test_batch.to(device).float().view(-1, 1)
            
            test_outputs = model(X_test_batch)
            test_pred = (test_outputs >= 0.5).float()
            
            test_correct += (test_pred == y_test_batch).sum().item()
            test_total += y_test_batch.size(0)
            
            all_preds.extend(test_pred.cpu().numpy().flatten())
            all_labels.extend(y_test_batch.cpu().numpy().flatten())
    
    test_accuracy = test_correct / test_total
    test_precision = precision_score(all_labels, all_preds, zero_division=0)
    test_recall = recall_score(all_labels, all_preds, zero_division=0)
    test_f1 = f1_score(all_labels, all_preds, zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    print(f"\nTest Metrics:")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    print(f"  Confusion Matrix:\n{conf_matrix}")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(loss_vector, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracy_vector, label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('acc_training_metrics.png', dpi=150, bbox_inches='tight')
    print("\nTraining metrics saved to acc_training_metrics.png")
    plt.show()
