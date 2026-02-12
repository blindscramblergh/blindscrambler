import pandas as pd
from torch import nn 
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, ConfusionMatrixDisplay

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

if __name__ == "__main__":
    # get the data path
    data_path = "../../../scripts/data/Android_Malware.csv"

    # read it using polars library:
    df = pd.read_csv(data_path, low_memory=False)

    # make all the columns have no empty spaces ahead or back
    df.columns = df.columns.str.strip()


    # delete the columns that are note important
    df = df.drop(columns=["Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port", "Protocol", "Timestamp"])

    # print statement
    print("Columns afterclaening the data: ", df.columns)