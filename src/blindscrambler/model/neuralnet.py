# the class TorchNet will be made for 

# Author metadata
__Name__ = "Syed Raza"
__email__ = "sar0033@uah.edu"

import torch
import polars as pl
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# make the neural network class
class TorchNet(torch.nn.Module):
    """
    A neural network class using PyTorch. This class will have the following 
    qualities:
    1. Inherits from torch.nn.Module
    2. Has an __init__ method to initialize layers, activation function, 
       input dimensions, output dimensions.
    """

    # initialization function
    def __init__(self, input_dim, output_dim, lr=0.01):
        """
        Initializes the neural network with two hidden layers. 
        - First hidden layer: 4 neurons
        - Second hidden layer: 3 neurons
        - Output layer: output_dim neurons

        ReLU activation function is used for the hidden layers.
        """
        super(TorchNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr

        self.scaler = None
        
        # first hidden layer: 
        self.W1 = torch.nn.Parameter(torch.randn(input_dim, 4) * 0.01)
        self.b1 = torch.nn.Parameter(torch.zeros(4))
        
        # second hidden layer: 
        self.W2 = torch.nn.Parameter(torch.randn(4, 3) * 0.01)
        self.b2 = torch.nn.Parameter(torch.zeros(3))
        
        # output layer: 
        self.W3 = torch.nn.Parameter(torch.randn(3, output_dim) * 0.01)
        self.b3 = torch.nn.Parameter(torch.zeros(output_dim))

    def standard_scale(self, X, fit=True):
        """
        Scale input X using StandardScaler. If fit=True (default) a new scaler
        is fit and stored on the model; otherwise the stored scaler is used.
        X can be numpy array / list-like shaped (n_samples, n_features).
        Returns a torch.float32 tensor.
        """
        if fit or self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        return torch.tensor(X_scaled, dtype=torch.float32)

    def relu(self, x):
        # better to use torch.relu to keep gradients correct
        return torch.relu(x)
    
    def forward(self, x):
        """
        Forward pass through the network.
        """
        # First hidden layer
        z1 = torch.matmul(x, self.W1) + self.b1
        a1 = self.relu(z1)
        
        # Second hidden layer
        z2 = torch.matmul(a1, self.W2) + self.b2
        a2 = self.relu(z2)
        
        # Output layer
        z3 = torch.matmul(a2, self.W3) + self.b3
        return z3
    
    # a function to train the model. The training should be done using Adam optimizer
    # and a binary cross entropy loss function.
    def train_model(self, X, y, epochs=1000):
        """
        Train the neural network using Adam optimizer and binary cross entropy loss.
        Also saving the loss and making a plot against the number of epochs after the 
        training is done.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        criterion = torch.nn.BCEWithLogitsLoss()
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            outputs = self.forward(X)
            loss = criterion(outputs.squeeze(), y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())

            if (epoch+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

        # Plotting the loss curve
        plt.plot(range(epochs), losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.show()
    
    def update_weights(self):
        """
        Update weights using gradient descent.
        """
        with torch.no_grad():
            self.W1 -= self.lr * self.W1.grad
            self.b1 -= self.lr * self.b1.grad
            
            self.W2 -= self.lr * self.W2.grad
            self.b2 -= self.lr * self.b2.grad
            
            self.W3 -= self.lr * self.W3.grad
            self.b3 -= self.lr * self.b3.grad
            
            # Zero the gradients after updating
            self.W1.grad.zero_()
            self.b1.grad.zero_()
            self.W2.grad.zero_()
            self.b2.grad.zero_()
            self.W3.grad.zero_()
            self.b3.grad.zero_()

    def predict(self, sample, threshold=0.5):
        """
        Predict probability and binary label for a single sample or batch.
        - sample: 1D list/array (n_features,) or 2D (n_samples, n_features)
        Returns (probability, label) for single sample or list of tuples for batch.
        """
        self.eval()
        with torch.no_grad():
            import numpy as _np
            arr = _np.asarray(sample)
            # determine whether the input represents a single sample
            single_input = (arr.ndim == 1) or (arr.ndim == 2 and arr.shape[0] == 1)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)

            X_tensor = self.standard_scale(arr, fit=False)
            logits = self.forward(X_tensor)
            probs = torch.sigmoid(logits).squeeze(dim=-1)

            # If single input was provided, return a single (prob, label) tuple
            if single_input:
                # probs may be a scalar tensor or 1-d tensor with one element
                if isinstance(probs, torch.Tensor):
                    prob_val = probs.item() if probs.dim() == 0 else probs.squeeze().item()
                else:
                    prob_val = float(probs)
                label = 1 if prob_val >= threshold else 0
                return prob_val, label

            # Otherwise return list of (prob, label) tuples for the batch
            results = []
            for p in probs.tolist():
                label = 1 if p >= threshold else 0
                results.append((p, label))
            return results


if __name__ == "__main__":
    # get the diabetes data set:
    path = "/Users/syedraza/Desktop/UAH/Classes/Fall2025/CPE586-MachineLearning/HWs/hw5/diabetes.csv"
 
    diabetes = pl.read_csv(path)
    no_features = len(diabetes.columns) - 1 # because one is target

    # make the model
    model = TorchNet(input_dim=no_features, output_dim=1, lr=0.01)

    # prepare the data
    X = diabetes.select(pl.exclude("Outcome")).to_numpy()
    y = diabetes.select("Outcome").to_numpy()

    # scale the data using standard scaler 
    X_tensor = model.standard_scale(X)
    y_tensor = torch.tensor(y, dtype=torch.float32).squeeze()

    # train the model
    model.train_model(X_tensor, y_tensor, epochs=1000)

    # make a prediction for the sample from your image:
    sample = [7, 149, 73, 94, 94, 32, 0.672, 45]  # Pregnancies,Glucose,BP,Skin,Insulin,BMI,DPF,Age
    prob, label = model.predict(sample)
    print(f"Predicted probability: {prob:.4f}, label: {label}")
