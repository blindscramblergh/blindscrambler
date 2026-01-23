import torch 
import polars as pl
import sys
from math import *
from torch import optim

# helper functions

# 1. Forward pass fucntion
def forward_pass(X: torch.tensor, W1: torch.tensor, W2: torch.tensor, W3: torch.tensor, W4: torch.tensor):
    """
    Performs a full forward pass through the model
    Params: The feature tensor and all the four wight tensors 
    Returns: one value for the result
    """
    # First: Linear computation
    Z1 = X @ W1

    # Second: the first sigmoid
    A1 = torch.sigmoid(Z1 @ W2)

    # third: Linear computation
    Z2 = A1 @ W3

    # four: the second sigmoid
    A2 = torch.sigmoid(Z2 @ W4)

    # now simply return A2
    return A2

def binary_classification(d: int, n:int, epochs: int = 10000, lr: float = 0.001):
    """
    Performs binary classification using a two-layer neural network
    Params: d - no of features
            n - no of samples
            epochs - no of training epochs, with a default value of 10,000
            lr - learning rate, with a default value of 0.001
    Returns: list of weight matrices and the loss values over epochs -- after training
    """
    # generate a random matrix X of the above size as feature matrix. The type should be float32 
    # randn returns tensor filled with random numbers from a normal distribution with mean 0 and variance 1
    X = torch.randn((n, d), dtype=torch.float32)

    # make a lacel tensor y of size n x 1
    y = torch.zeros((n, 1), dtype=torch.float32)

    # for loop to populate the y label tensor
    for i in range(n):
        # if the sum of all the features of a sample is > 2, label it as 1 else 0
        if torch.sum(X[i, :]) > 2.0:
            y[i] = 1.0

    # make the weight matrices (enable autograd)
    W1 = torch.normal(mean=0.0, std=sqrt(2.0/d), size=(d, 48), dtype=torch.float32).requires_grad_()
    W2 = torch.normal(mean=0.0, std=sqrt(2.0/48.0), size=(48, 16), dtype=torch.float32).requires_grad_()
    W3 = torch.normal(mean=0.0, std=sqrt(2.0/16.0), size=(16, 32), dtype=torch.float32).requires_grad_()
    W4 = torch.normal(mean=0.0, std=sqrt(2.0/32.0), size=(32, 1), dtype=torch.float32).requires_grad_()

    # the loss fucntion 
    loss_fn = torch.nn.BCELoss()

    # ready to perform the training using SGD provided by torch
    losses = []

    for epoch in range(epochs):
        # do the forward pass and get the loss
        y_hat = forward_pass(X, W1, W2, W3, W4)
        loss = loss_fn(y_hat, y)

        # backpropagation
        loss.backward() 

        # update the weights (disable autograd for in-place ops)
        with torch.no_grad():
            W1 -= lr * W1.grad
            W2 -= lr * W2.grad
            W3 -= lr * W3.grad
            W4 -= lr * W4.grad
        
        # zero gradients for next iteration
        W1.grad.zero_()
        W2.grad.zero_()
        W3.grad.zero_()
        W4.grad.zero_()

        # store the loss
        losses.append(loss.item())

        # to print some results here and there
        if epoch % 200 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
     
    # return the weight matrices and the losses
    return [W1, W2, W3, W4, losses]

if __name__ == "__main__":

    # First check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # setting the size for the no of features and samples
    n = 500
    d = 20

    # call the function here
    binary_classification(d, n)