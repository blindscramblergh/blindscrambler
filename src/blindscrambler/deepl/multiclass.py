import pandas as pd
from torch import nn 
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, ConfusionMatrixDisplay
from datasets import load_dataset
from datasets import load_from_disk
from torchvision import transforms
import sys
from torch.utils.data import DataLoader
import subprocess

class SimpleNN(nn.Module):
    def __init__(self, in_features, num_classes):
        super(SimpleNN, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.fc1 = nn.Linear(self.in_features, 3)
        self.fc2 = nn.Linear(3, 4)
        self.fc3 = nn.Linear(4, 5)
        self.fc4 = nn.Linear(5, self.num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
class ClassTrainer(nn.Module):
    def __init__(self, X_train, y_train, eta, epoch, loss_function, 
                 optimizer, loss_vector, accuracy_vector, model, device):
        super().__init__()
        # the class variables are:
        self.X_train = X_train
        self.y_train = y_train
        self.eta = eta
        self.epoch = epoch
        self.loss = loss_function 
        self.optimizer = optimizer
        self.loss_vector = loss_vector 
        self.accuracy_vector = accuracy_vector
        self.model = model
        self.device = device

    # now for the function this class is going to have are
    def train(self):
        """
        Trains the model using the training data stored in class attributes.
        Updates loss_vector and accuracy_vector during training.
        """
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Training loop
        for epoch in range(self.epoch):
            self.model.train()
            
            # Move data to device
            X_batch = self.X_train.to(self.device)
            y_batch = self.y_train.to(self.device)
            
            # Zero the gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(X_batch)
            
            # Compute loss
            loss = self.loss(outputs, y_batch)
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y_batch).sum().item()
            accuracy = correct / y_batch.size(0)
            
            # Store loss and accuracy
            self.loss_vector.append(loss.item())
            self.accuracy_vector.append(accuracy)
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f'Epoch [{epoch + 1}/{self.epoch}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')
    
    def test(self, X_test, y_test):
        """
        Tests the model on the test dataset.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            tuple: (test_loss, test_accuracy, predictions)
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Move test data to device
        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)
        
        # Disable gradient computation
        with torch.no_grad():
            # Forward pass
            outputs = self.model(X_test)
            
            # Compute loss
            test_loss = self.loss(outputs, y_test)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y_test).sum().item()
            test_accuracy = correct / y_test.size(0)
            
            print(f'Test Loss: {test_loss.item():.4f}, Test Accuracy: {test_accuracy:.4f}')
            
        return test_loss.item(), test_accuracy, predicted
    
    def predict(self, X):
        """
        Makes predictions on new data using the trained model.
        
        Args:
            X: Input features
            
        Returns:
            predicted: Predicted class labels
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Move data to device
        X = X.to(self.device)
        
        # Disable gradient computation
        with torch.no_grad():
            # Forward pass
            outputs = self.model(X)
            
            # Get predicted class
            _, predicted = torch.max(outputs.data, 1)
            
        return predicted

    def save(self, file_path):
        """
        Saves the trained model in ONNX format.
        
        Args:
            file_path: Path where the ONNX model will be saved
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Create a dummy input with the correct shape (batch_size=1, input_features)
        dummy_input = torch.randn(1, self.model.in_features, device=self.device)
        
        # Export the model to ONNX format
        torch.onnx.export(
            self.model,                          # Model to export
            dummy_input,                         # Model input
            file_path,                           # Output file path
            export_params=True,                  # Store trained parameters
            opset_version=11,                    # ONNX version
            do_constant_folding=True,            # Optimize constant folding
            input_names=['input'],               # Input tensor name
            output_names=['output'],             # Output tensor name
            dynamic_axes={                       # Allow dynamic batch size
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f'Model saved to {file_path}')
    
    def evaluation(self, X_test, y_test):
        """
        Evaluates model performance on training and test sets.
        Creates plots for loss/accuracy curves and confusion matrices.
        Displays F1 score, precision, recall, and accuracy metrics.
        
        Args:
            X_test: Test features
            y_test: Test labels
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Get predictions for training set
        with torch.no_grad():
            train_outputs = self.model(self.X_train.to(self.device))
            _, train_predictions = torch.max(train_outputs.data, 1)
            train_predictions = train_predictions.cpu().numpy()
            y_train_np = self.y_train.cpu().numpy()
        
        # Get predictions for test set
        with torch.no_grad():
            test_outputs = self.model(X_test.to(self.device))
            _, test_predictions = torch.max(test_outputs.data, 1)
            test_predictions = test_predictions.cpu().numpy()
            y_test_np = y_test.cpu().numpy()
        
        # Calculate metrics for training set
        train_accuracy = accuracy_score(y_train_np, train_predictions)
        train_precision = precision_score(y_train_np, train_predictions, average='weighted', zero_division=0)
        train_recall = recall_score(y_train_np, train_predictions, average='weighted', zero_division=0)
        train_f1 = f1_score(y_train_np, train_predictions, average='weighted', zero_division=0)
        
        # Calculate metrics for test set
        test_accuracy = accuracy_score(y_test_np, test_predictions)
        test_precision = precision_score(y_test_np, test_predictions, average='weighted', zero_division=0)
        test_recall = recall_score(y_test_np, test_predictions, average='weighted', zero_division=0)
        test_f1 = f1_score(y_test_np, test_predictions, average='weighted', zero_division=0)
        
        # Print metrics
        print("\n" + "="*50)
        print("TRAINING SET METRICS")
        print("="*50)
        print(f"Accuracy:  {train_accuracy:.4f}")
        print(f"Precision: {train_precision:.4f}")
        print(f"Recall:    {train_recall:.4f}")
        print(f"F1 Score:  {train_f1:.4f}")
        
        print("\n" + "="*50)
        print("TEST SET METRICS")
        print("="*50)
        print(f"Accuracy:  {test_accuracy:.4f}")
        print(f"Precision: {test_precision:.4f}")
        print(f"Recall:    {test_recall:.4f}")
        print(f"F1 Score:  {test_f1:.4f}")
        print("="*50 + "\n")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Training Loss
        axes[0, 0].plot(self.loss_vector, label='Training Loss', color='blue')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss Over Epochs')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot 2: Training Accuracy
        axes[0, 1].plot(self.accuracy_vector, label='Training Accuracy', color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training Accuracy Over Epochs')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot 3: Confusion Matrix for Training Set
        cm_train = confusion_matrix(y_train_np, train_predictions)
        disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train)
        disp_train.plot(ax=axes[1, 0], cmap='Blues')
        axes[1, 0].set_title('Confusion Matrix - Training Set')
        
        # Plot 4: Confusion Matrix for Test Set
        cm_test = confusion_matrix(y_test_np, test_predictions)
        disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test)
        disp_test.plot(ax=axes[1, 1], cmap='Greens')
        axes[1, 1].set_title('Confusion Matrix - Test Set')
        
        plt.tight_layout()
        plt.show()


#####################################################################################################################

# THINGS FOR HW# START HERE 

# the custom composite layer called ConvLayer
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True):
        super(Conv2d, self).__init__()
        
        # Store hyperparameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        # Initialize learnable parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights properly
        self.reset_parameters()
    
    def reset_parameters(self):
        # Kaiming initialization
        nn.init.kaiming_uniform_(self.weight, a=torch.nn.init.calculate_gain('relu'))
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x):
        # Use PyTorch's built-in conv2d function
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding)


# CNN ARCHITECTURE
class ImageNetCNN(nn.Module):
    def __init__(self):
        super(ImageNetCNN, self).__init__()

        # get the ReLU, and pooling here since it will be the same for all the Blocks 
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #---------------------------------- Block 1 ----------------------------------#

        # first convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3)
        # batch normalizatiopn and RELU activation after first convolutional layer
        self.bn1 = nn.BatchNorm2d(64)

        #---------------------------------- Block 2 ----------------------------------#

        # second convolutional layer 
        self.conv2 = nn.Conv2d(64, 128, kernel_size = 3)
        # batch normalization and RELU again
        self.bn2 = nn.BatchNorm2d(128)

        #---------------------------------- Block 3 ----------------------------------#

        # third convolutional layer
        self.conv3 = nn.Conv2d(128, 256, kernel_size = 3)
        # batch normalization again
        self.bn3 = nn.BatchNorm2d(256)

        #---------------------------------- Block 4 ----------------------------------#

        # fourth convolutional layer    
        self.conv4 = nn.Conv2d(256, 512, kernel_size = 3)
        # batch normalization again
        self.bn4 = nn.BatchNorm2d(512)

        #---------------------------------- Block 5 ----------------------------------#

        # fifth convolutional layer
        self.conv5 = nn.Conv2d(512, 512, kernel_size = 3)
        # batch normalization again
        self.bn5 = nn.BatchNorm2d(512)
        
        #-------------------------- Global Pool and Flatten --------------------------#

        # global average pooling layer and flatten layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        #----------------------------- Fully connected 1 -----------------------------#

        # fully connected layer 1
        self.fc1 = nn.Linear(512, 1024)
        # relu and dropout
        self.dropout = nn.Dropout(0.5)

        #----------------------------- Fully connected 2 -----------------------------#

        # fully connected layer 2, the output layer with 1000 classes, and softmax activation
        self.fc2 = nn.Linear(1024, 1000)
        self.softmax = nn.Softmax(dim=1)
        

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))                # apply block 1: conv1 -> BN -> relu -> maxpool
        x = self.maxpool(self.relu(self.bn2(self.conv2(x))))                # apply block 2: conv2 -> BN -> relu -> maxpool
        x = self.maxpool(self.relu(self.bn3(self.conv3(x))))                # apply block 3: conv3 -> BN -> relu -> maxpool
        x = self.maxpool(self.relu(self.bn4(self.conv4(x))))                # apply block 4: conv4 -> BN -> relu -> maxpool
        x = self.maxpool(self.relu(self.bn5(self.conv5(x))))                # apply block 3: conv4 -> BN -> relu -> maxpool
        x = self.flatten(self.global_avg_pool(x))                           # apply the flobal pool and flatten
        x = self.dropout(self.relu(self.fc1(x)))                            # apply fully connected layer 1
        x = self.softmax(self.fc2(x))                                       # apply fully connected layer 2

        return x

# next step is how to train the CNN architecture:
class CNNTrainer(nn.Module):

    # the initializer function
    def __init__(self, train_dataloader, val_dataloader, eta, epoch, loss_function, 
                optimizer, loss_vector, accuracy_vector, model, device, scheduler):
        super().__init__()
        # the class variables or attributes
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.eta = eta 
        self.epoch = epoch
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.loss_vector = loss_vector 
        self.accuracy_vector = accuracy_vector
        self.model = model
        self.device = device 
        self.scheduler = scheduler

    # the training function
    def train(self):
        """
        This class is to train the CNN architecture. This is cool because you can train many CNN
        architectures based on this training class.
        """

        # move the model to device 
        self.model = self.model.to(self.device)

        # Training loop
        for epoch in range(self.epoch):
            self.model.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            # Iterate through batches from the DataLoader
            for batch_idx, (X_batch, y_batch) in enumerate(self.train_dataloader):
                # move the data to device 
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # zero the gradients 
                self.optimizer.zero_grad()

                # forward pass
                outputs = self.model(X_batch)

                # compute the loss
                loss = self.loss_function(outputs, y_batch)

                # backwards pass and optimization
                loss.backward()
                self.optimizer.step()

                # calculate the accuracy
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == y_batch).sum().item()
                
                # accumulate epoch statistics
                epoch_loss += loss.item()
                epoch_correct += correct
                epoch_total += y_batch.size(0)
                
                # Print progress every 10 batches with validation accuracy
                if (batch_idx + 1) % 10 == 0:
                    # Calculate training accuracy for this batch
                    batch_train_accuracy = correct / y_batch.size(0)
                    
                    # Calculate validation accuracy
                    self.model.eval()
                    val_correct = 0
                    val_total = 0
                    with torch.no_grad():
                        for val_X, val_y in self.val_dataloader:
                            val_X = val_X.to(self.device)
                            val_y = val_y.to(self.device)
                            val_outputs = self.model(val_X)
                            _, val_pred = torch.max(val_outputs.data, 1)
                            val_correct += (val_pred == val_y).sum().item()
                            val_total += val_y.size(0)
                    val_accuracy = val_correct / val_total
                    self.model.train()
                    
                    # Print loss, training accuracy, and validation accuracy
                    print(f'Epoch [{epoch + 1}/{self.epoch}], Batch [{batch_idx + 1}/{len(self.train_dataloader)}], Loss: {loss.item():.4f}, Train Acc: {batch_train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}')

            # Calculate average loss and accuracy for the epoch
            avg_loss = epoch_loss / len(self.train_dataloader)
            avg_accuracy = epoch_correct / epoch_total
            
            # Store loss and accuracy
            self.loss_vector.append(avg_loss)
            self.accuracy_vector.append(avg_accuracy)

            # make the scheduler step
            self.scheduler.step()

    def test(self):
        """
        Evaluate the model on validation data
        
        Returns:
            tuple: (test_loss, test_accuracy)
        """

        # set model to evaluation mode
        self.model.eval()
        
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in self.val_dataloader:
                # move test data to the device 
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Forward pass
                outputs = self.model(X_batch)
                
                # compute loss
                loss = self.loss_function(outputs, y_batch)
                test_loss += loss.item()

                # compute accuracy
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)
        
        avg_test_loss = test_loss / len(self.val_dataloader)
        test_accuracy = correct / total
        
        print(f'Validation Loss: {avg_test_loss:.4f}, Validation Accuracy: {test_accuracy:.4f}')
        
        return avg_test_loss, test_accuracy

    def predict(self, X):
        """
        Makes predictions on new data using the trained model
        
        Args:
            X: Input tensor or DataLoader
            
        Returns:
            predictions: Predicted class labels
        """
        # set model to evaluation mode
        self.model.eval()

        if isinstance(X, DataLoader):
            # If X is a DataLoader, iterate through batches
            all_predictions = []
            with torch.no_grad():
                for X_batch, _ in X:
                    X_batch = X_batch.to(self.device)
                    outputs = self.model(X_batch)
                    _, predicted = torch.max(outputs.data, 1)
                    all_predictions.append(predicted.cpu())
            return torch.cat(all_predictions, dim=0)
        else:
            # If X is a tensor
            X = X.to(self.device)
            with torch.no_grad():
                outputs = self.model(X)
                _, predicted = torch.max(outputs.data, 1)
            return predicted

    def save_onnx(self, file_path):
        """
        Saves the trained model in ONNX format.
        
        Args:
            file_path (str): Path where the ONNX model will be saved
            
        Raises:
            RuntimeError: If the model has not been trained yet
        """
        # Check if model has been trained
        if not self.loss_vector or len(self.loss_vector) == 0:
            raise RuntimeError("Model has not been trained yet. Please train the model before saving.")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Create a dummy input with the correct shape (batch_size=1, channels=3, height=224, width=224)
        dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
        
        # Export the model to ONNX format
        torch.onnx.export(
            self.model,                          # Model to export
            dummy_input,                         # Model input
            file_path,                           # Output file path
            export_params=True,                  # Store trained parameters
            opset_version=11,                    # ONNX opset version
            do_constant_folding=True,            # Optimize constant folding
            input_names=['input'],               # Input tensor name
            output_names=['output'],             # Output tensor name
            dynamic_axes={                       # Allow dynamic batch size
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f'Model saved to {file_path}')


if __name__ == "__main__":

    # pick the device to be used, use GPU if available, otherwise use CPU
    device_id = get_best_gpu(strategy="utilization")
    device = torch.device(f"cuda:{device_id}")
    print(f"Selected GPU: {device_id}")

    # Load dataset
    dataset = load_from_disk("/data/CPE_487-587/imagenet-1k-arrow")

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    num_classes = len(train_dataset.features["label"].names)
    print(f"Number of classes: {num_classes}")

    # select subset for training and testing
    train_size = int(len(dataset["train"]) * 0.005)      
    val_size = int(len(dataset["validation"]) * 0.0008)
    
    train_dataset = dataset["train"].select(range(train_size))
    val_dataset = dataset["validation"].select(range(val_size))

    class_names = train_dataset.features["label"].names

    # in order to display a sample image - uncomment this code snippet to look at an example figure
    """
    first_example = train_dataset[55]
    image = first_example["image"]
    label_id = first_example["label"]

    full_label = class_names[label_id]
    primary_name = full_label.split(',')[0].strip()

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title(f"ID {label_id}: {primary_name}\n({full_label})", fontsize=10)
    plt.axis("off")
    plt.savefig("sample_image.png")
    """

    # Transforming train and val images 
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    val_transform = transforms.Compose({
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    })

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

    # Initialize model and trainer
    model = ImageNetCNN()
    trainer = CNNTrainer(
        train_dataloader=train_loader,
        eta=0.001,
        epoch=50,
        loss_function=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        loss_vector=[],
        accuracy_vector=[],
        model=model,
        device=device,
        val_dataloader=val_loader
    )
     
    # Train the model
    trainer.train()
     
    # Evaluate on validation set
    val_loss, val_accuracy = trainer.test()

    print(f"Validation loss: {val_loss}")