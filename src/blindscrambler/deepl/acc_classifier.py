import pandas as pd
from torch import nn
import torch
import glob
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn import functional as F
import subprocess

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

class ResidualBlock(nn.Module):
    """
    Simple Residual Block for ResNet.
    
    If input and output dimensions differ, uses 1x1 convolution to match dimensions.
    """
    def __init__(self, in_features, out_features, hidden_features=None):
        super(ResidualBlock, self).__init__()
        if hidden_features is None:
            hidden_features = out_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.relu = nn.ReLU()
        
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        
        # Shortcut connection
        self.shortcut = nn.Identity()
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        # Residual path
        residual = self.shortcut(x)
        
        # Main path
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        
        # Add residual connection
        out = out + residual
        out = self.relu(out)
        
        return out

class ACCNet(nn.Module):
    """
    Simple ResNet-based neural network for ACC (Adaptive Cruise Control) state prediction.
    
    Architecture:
    - Input: 11 features (v_t, v_{t-1}, ..., v_{t-10})
    - ResNet blocks with residual connections
    - Output: 2 classes (ACC enabled or not)
    
    Input shape: (batch_size, 11)
    Output shape: (batch_size, 2)
    """
    
    def __init__(self, input_features=11, hidden_sizes=[64, 128, 64], num_classes=2, dropout_rate=0.3):
        """
        Initialize ACCNet.
        
        Args:
            input_features (int): Number of input features (default: 11 for historical speeds)
            hidden_sizes (list): List of hidden layer sizes for ResNet blocks
            num_classes (int): Number of output classes (default: 2 for binary classification)
            dropout_rate (float): Dropout rate for regularization
        """
        super(ACCNet, self).__init__()
        
        self.input_features = input_features
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        
        # Initial layer
        self.initial = nn.Linear(input_features, hidden_sizes[0])
        self.initial_bn = nn.BatchNorm1d(hidden_sizes[0])
        self.initial_relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # ResNet blocks
        self.res_blocks = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.res_blocks.append(
                ResidualBlock(hidden_sizes[i], hidden_sizes[i + 1], hidden_sizes[i])
            )
        
        # Output layer
        self.output = nn.Linear(hidden_sizes[-1], num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_features)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Initial layer
        x = self.initial(x)
        x = self.initial_bn(x)
        x = self.initial_relu(x)
        x = self.dropout(x)
        
        # ResNet blocks
        for block in self.res_blocks:
            x = block(x)
            x = self.dropout(x)
        
        # Output layer
        x = self.output(x)
        
        return x


class ACCNetTrainer(nn.Module):
    """
    Training class for ACCNet with support for imbalanced classification losses.
    
    Supports:
    - Dice Loss for imbalanced data
    - Focal Loss for hard examples
    - CrossEntropyLoss for baseline
    """
    
    def __init__(self, model, train_dataloader, val_dataloader,
                 loss_type='dice', eta=0.001, epochs=50, device=None):
        """
        Initialize ACCNetTrainer.
        
        Args:
            model (nn.Module): The neural network model
            train_dataloader (DataLoader): Training data loader
            val_dataloader (DataLoader): Validation data loader
            loss_type (str): Type of loss function ('dice', 'focal', or 'cross_entropy')
            eta (float): Learning rate
            epochs (int): Number of training epochs
            device (torch.device): Device to use for training (default: CPU)
        """
        super(ACCNetTrainer, self).__init__()
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_type = loss_type
        self.eta = eta
        self.epochs = epochs
        self.device = device if device is not None else torch.device('cpu')
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=eta)
        
        # Initialize loss function
        if loss_type == 'dice':
            self.loss_function = self._dice_loss
        elif loss_type == 'focal':
            self.loss_function = self._focal_loss
        else:
            self.loss_function = nn.CrossEntropyLoss()
        
        # Tracking metrics
        self.loss_vector = []
        self.accuracy_vector = []
        self.val_loss_vector = []
        self.val_accuracy_vector = []
        
        # Normalization coefficients (for saving with ONNX)
        self.speed_mean = None
        self.speed_std = None
        
    @staticmethod
    def _dice_loss(predictions, targets, smooth=1.0):
        """
        Dice Loss for imbalanced classification - more numerically stable.
        
        Args:
            predictions (torch.Tensor): Model output logits
            targets (torch.Tensor): Ground truth labels
            smooth (float): Smoothing constant to avoid division by zero
            
        Returns:
            torch.Tensor: Dice loss value
        """
        # Convert logits to probabilities
        probs = F.softmax(predictions, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=2).float()
        
        # Calculate Dice loss per class
        intersection = (probs * targets_one_hot).sum(dim=0)
        cardinality = probs.sum(dim=0) + targets_one_hot.sum(dim=0)
        
        # Dice coefficient per class
        dice_scores = 1 - (2.0 * intersection + smooth) / (cardinality + smooth)
        
        # Return mean over classes
        return dice_scores.mean()
    
    @staticmethod
    def _focal_loss(predictions, targets, alpha=1.0, gamma=2.0):
        """
        Focal Loss for hard examples.
        
        Args:
            predictions (torch.Tensor): Model output logits
            targets (torch.Tensor): Ground truth labels
            alpha (float): Weighting factor
            gamma (float): Focusing parameter
            
        Returns:
            torch.Tensor: Focal loss value
        """
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        
        # Get probabilities
        probs = F.softmax(predictions, dim=1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Calculate focal loss
        focal_loss = alpha * (1 - p_t) ** gamma * ce_loss
        return focal_loss.mean()
    
    def train(self):
        """Train the model for specified epochs."""
        self.model.train()
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (features, labels) in enumerate(self.train_dataloader):
                # Move data to device
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.loss_function(outputs, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Metrics
                _, predicted = torch.max(outputs.data, 1)
                epoch_correct += (predicted == labels).sum().item()
                epoch_total += labels.size(0)
                epoch_loss += loss.item()
                
                if (batch_idx + 1) % 10 == 0:
                    batch_acc = (predicted == labels).sum().item() / labels.size(0)
                    print(f'Epoch [{epoch + 1}/{self.epochs}], Batch [{batch_idx + 1}/{len(self.train_dataloader)}], Loss: {loss.item():.4f}, Acc: {batch_acc:.4f}')
            
            # Calculate epoch metrics
            avg_loss = epoch_loss / len(self.train_dataloader)
            avg_accuracy = epoch_correct / epoch_total
            
            self.loss_vector.append(avg_loss)
            self.accuracy_vector.append(avg_accuracy)
            
            # Validation
            val_loss, val_acc = self.validate()
            self.val_loss_vector.append(val_loss)
            self.val_accuracy_vector.append(val_acc)
            
            print(f'Epoch [{epoch + 1}/{self.epochs}] - Train Loss: {avg_loss:.4f}, Train Acc: {avg_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    def validate(self):
        """Validate the model on validation dataset."""
        self.model.eval()
        
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in self.val_dataloader:
                # Move data to device
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                loss = self.loss_function(outputs, labels)
                
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        self.model.train()
        
        avg_val_loss = val_loss / len(self.val_dataloader)
        val_accuracy = correct / total
        
        return avg_val_loss, val_accuracy
    
    def save_onnx(self, file_path):
        """
        Save model in ONNX format with normalization coefficients.
        
        Args:
            file_path (str): Path to save the ONNX model
        """
        self.model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, self.model.input_features)
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            file_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        print(f'Model saved to {file_path}')


class ACCNetDataset(Dataset):
    """
    PyTorch Dataset class for Adaptive Cruise Control (ACC) state prediction.
    
    Reads wheel speed data (decoded_wheel_speed_f1.csv) and ACC status data (acc_status.csv),
    aligns them using Zero-Order Hold technique, and creates features from historical speed values.
    
    ACC Status Values:
    - 0: off (System Inactive)
    - 2: disabled (Standby)
    - 5: faulted (System Error)
    - 6: enabled (Active cruising) <- TARGET CLASS (label=1)
    - 10: hold_waiting_user_cmd
    - 11: hold
    
    Binary Classification Task: Predict if ACC is enabled (6) or not (0 = other states)
    
    Features created: v_t, v_{t-1}, v_{t-2}, ..., v_{t-k}
    where k=10 (number of historical values to look back)
    """
    
    def __init__(self, data_dir="/data/CPE_487-587/ACCDataset", k=10):
        """
        Initialize the ACCNetDataset.
        
        Args:
            data_dir (str): Directory containing CSV files
            k (int): Number of historical values to look back (default: 10)
        """
        self.k = k
        self.data_dir = data_dir
        
        # Load wheel speed files
        speed_pattern = f"{data_dir}/*decoded_wheel_speed_fl.csv"
        speed_files = glob.glob(speed_pattern)
        
        # Load ACC status files
        status_pattern = f"{data_dir}/*acc_status.csv"
        status_files = glob.glob(status_pattern)
        
        if len(speed_files) == 0:
            raise FileNotFoundError(f"No speed files found matching pattern: {speed_pattern}")
        if len(status_files) == 0:
            raise FileNotFoundError(f"No status files found matching pattern: {status_pattern}")
        
        print(f"Found {len(speed_files)} speed files and {len(status_files)} status files")
        
        # Read all speed files and align with status files
        all_samples = []
        
        # Match speed and status files by timestamp prefix
        for speed_file in speed_files:
            # Extract timestamp from filename (e.g., "1234567890_decoded_wheel_speed_f1.csv")
            timestamp = speed_file.split('/')[-1].split('_')[0]
            
            # Find corresponding status file
            status_file = None
            for sf in status_files:
                if timestamp in sf:
                    status_file = sf
                    break
            
            if status_file is None:
                print(f"  Warning: No matching status file for {speed_file}, skipping...")
                continue
            
            try:
                # Load speed data
                speed_df = pd.read_csv(speed_file)
                status_df = pd.read_csv(status_file)
                
                print(f"  Processing pair:")
                print(f"    Speed:  {speed_file.split('/')[-1]}")
                print(f"    Status: {status_file.split('/')[-1]}")
                
                # Extract Time and Message columns
                if 'Time' not in speed_df.columns or 'Message' not in speed_df.columns:
                    print(f"    Skipping: Missing Time or Message column in speed file")
                    continue
                if 'Time' not in status_df.columns or 'Message' not in status_df.columns:
                    print(f"    Skipping: Missing Time or Message column in status file")
                    continue
                
                # Get speed times and values (convert km/h to m/s)
                speed_times = speed_df['Time'].values.astype(np.float32)
                speed_values = (speed_df['Message'].values.astype(np.float32)) / 3.6  # km/h to m/s
                
                # Get status times and values
                status_times = status_df['Time'].values.astype(np.float32)
                status_values = status_df['Message'].values.astype(np.int64)
                
                # Align using Zero-Order Hold: for each speed sample, use the latest status value
                # Create a synchronized dataset
                for i, (t, v) in enumerate(zip(speed_times, speed_values)):
                    # Find the latest status value at or before this time
                    status_mask = status_times <= t
                    if status_mask.any():
                        latest_status_idx = np.where(status_mask)[0][-1]
                        acc_status = status_values[latest_status_idx]
                        
                        # Binary label: 1 if enabled (6), 0 otherwise
                        label = 1 if acc_status == 6 else 0
                        
                        all_samples.append({
                            'time': t,
                            'speed': v,
                            'label': label,
                            'acc_status': acc_status
                        })
                
                print(f"    Processed {len(all_samples)} total samples so far")
                
            except Exception as e:
                print(f"    Error processing files: {e}")
                continue
        
        if not all_samples:
            raise ValueError("No samples could be created from the data files")
        
        # Create DataFrame from all samples
        self.df = pd.DataFrame(all_samples)
        self.speeds = self.df['speed'].values.astype(np.float32)
        self.labels = self.df['label'].values.astype(int)
        
        print(f"\nTotal samples created: {len(self.speeds)}")
        print(f"Label distribution:")
        unique, counts = np.unique(self.labels, return_counts=True)
        for label_val, count in zip(unique, counts):
            state = "enabled (6)" if label_val == 1 else "not enabled (other)"
            print(f"  Label {label_val} ({state}): {count} samples ({100*count/len(self.labels):.1f}%)")
        
        # Calculate valid indices (we need at least k previous values)
        self.valid_indices = np.arange(self.k, len(self.speeds))
        print(f"Valid samples after windowing (k={self.k}): {len(self.valid_indices)}")
        
    def __len__(self):
        """Return the number of valid samples."""
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (features, label) where features is a tensor of historical speeds [v_t, v_{t-1}, ..., v_{t-k}]
        """
        # Get the actual index in the data
        actual_idx = self.valid_indices[idx]
        
        # Create feature vector: [v_t, v_{t-1}, v_{t-2}, ..., v_{t-k}]
        # Note: speeds are already in m/s
        feature_indices = np.arange(actual_idx - self.k, actual_idx + 1)
        features = self.speeds[feature_indices].astype(np.float32)
        
        # Get the label for this time step (binary: 1 if ACC enabled, 0 otherwise)
        label = self.labels[actual_idx]
        
        # Convert to tensors
        features_tensor = torch.from_numpy(features)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return features_tensor, label_tensor

if __name__ == "__main__":
    # get the best gpu based on utilization:

    # implementing the multiclass thing
    device_id = get_best_gpu(strategy="utilization")
    device = torch.device(f"cuda:{device_id}")
    print(f"Selected GPU: {device_id}")

    # Create dataset instance
    print("Creating ACCNetDataset...")
    dataset = ACCNetDataset(data_dir="/data/CPE_487-587/ACCDataset", k=10)
    
    print(f"\nDataset created with {len(dataset)} samples")
    
    # Split dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create DataLoaders for batch training
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nDataLoaders created with batch size: {batch_size}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Display a sample batch
    print("\nSample batch:")
    for features, labels in train_loader:
        print(f"  Features shape: {features.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Features (first sample): {features[0]}")
        print(f"  Labels (first sample): {labels[0]}")
        print(f"  Label distribution in batch: {np.unique(labels.numpy(), return_counts=True)}")
        break

    print("\nACCNetDataset is ready for training!")

    # Create model
    model = ACCNet(input_features=11, hidden_sizes=[64, 128, 64], num_classes=2)
    model.to(device)

    # Create trainer with CrossEntropyLoss
    trainer = ACCNetTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        loss_type='cross_entropy',  # or 'dice', 'focal'
        eta=0.001,
        epochs=50,
        device=device
    )

    # Train
    trainer.train()

    # Save
    trainer.save_onnx('acc_model.onnx')