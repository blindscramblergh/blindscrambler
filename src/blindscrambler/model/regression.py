# Author metadata

__Name__ = "Syed Raza"
__email__ = "sar0033@uah.edu"

# import statements
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, Optional
import warnings
from torcheval.metrics import R2Score 
import polars
from sklearn.model_selection import train_test_split

# add a linear Regression class:
class LinearRegression:
    """
    A PyTorch-based Linear Regression implementation for one variable.
    
    Model: y = w_1 * x + w_0
    Loss: Mean Squared Error
    
    Features:
    - Gradient-based optimization using PyTorch
    - Confidence intervals for parameters w_1 and w_0
    """

    def __init__(self, learning_rate: float = 0.01, max_epochs: int = 1000,
                 tolerance: float = 1e-6):
        
        """
        The Constructor function for LinearRegression Class

        Params:
            - learning rate, for the gradient descent algorithm
            - maximum number of epochs 
            - tolerance, to know if things have converged
        """

        # make the arguments
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tolerance = tolerance

        self.nsamples = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        # to see if the instance is fitted or not
        self.fitted = False

        # the model parameters
        self.w_0 = nn.Parameter(torch.randn(1, requires_grad=True)) # intercept
        self.w_1 = nn.Parameter(torch.randn(1, requires_grad=True)) # slope

        # loss function and its optimizer
        self.lossfunction = nn.MSELoss()
        self.optimizer = optim.SGD([self.w_0, self.w_1], lr = self.learning_rate)

        # hold intermediate values of w_0 and w_1 and loss
        self.inter_w_0 = []
        self.inter_w_1 = []
        self.inter_loss = []

    
    def forward(self, X: torch.tensor) -> torch.tensor:
        """
        Forward function for to specify linear model and compute the response

        Params:
            - X: torch.tensor
            the input vector of size (n_samples, )
        Returns:
            - self.w_1 * X + self.w_0
    `       the output is linear model result
        
        """
        return self.w_1 * X + self.w_0

    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> 'LinearRegression':
        """
        The function where the training happens

        Params:
            - X, the training dataset of features 
            - y, the training dataset of target
        """

        # convert to Pytorch tensors:
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32)
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.float32)
        self.nsamples = len(X_train) # samples in the training set

        # the training loop:
        prev_loss = float('inf')

        # reset history
        self.inter_loss.clear()
        self.inter_w_0.clear()
        self.inter_w_1.clear()

        for epoch in range(self.max_epochs):
            # reset the gradients
            self.optimizer.zero_grad()

            # premature prediction
            y_train_pred = self.forward(self.X_train)

            # loss function
            loss = self.lossfunction(y_train_pred, self.y_train)

            # automatic gradient backward pass 
            loss.backward()

            # update model parameters
            self.optimizer.step()

            # get the current loss and save it 
            current_loss = float(loss.detach().item())

            # save intermediate loss and model parameters 
            self.inter_loss.append(current_loss)
            self.inter_w_0.append(float(self.w_0.detach().item()))
            self.inter_w_1.append(float(self.w_1.detach().item()))

            if abs(prev_loss - current_loss) < self.tolerance:
                print(f"Converged after {epoch + 1} epochs")
                break

            prev_loss = current_loss

        # make predictions on the test set
        y_test_pred = self.forward(self.X_test)

        # create an R^2 metric type 
        R2 = R2Score()
        R2.update(y_test_pred, self.y_test)
        print("The R2 score for the test set is :", R2.compute())

        self.fitted = True
        return self 

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on the new/unseen data

        Params:
            - feature vector X for the test set.
        Returns:
            - predictions in a numpy array
        """

        # making sure that the model is fitted lol
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # make it a tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            predictions = self.forward(X_tensor)

        return predictions.numpy()

    
    def analysis_plot(self, show: bool = True, save_path: Optional[str] = None):
        """
        Create a 2x2 figure showing:
        - Original data with fitted regression line
        - Training loss over epochs
        - w0 trajectory over epochs
        - w1 trajectory over epochs
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting.")
        if len(self.inter_loss) == 0:
            warnings.warn("No training history recorded; plots may be empty.")

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        # 1) Data + fitted line
        ax = axs[0, 0]

        # scatter only the test set
        if self.X_test is not None and self.y_test is not None:
            ax.scatter(
                self.X_test.detach().cpu().numpy(),
                self.y_test.detach().cpu().numpy(),
                s=12, alpha=0.7, label="Test"
            )
            # Line range from min/max of test X only
            xmin = float(torch.min(self.X_test).item())
            xmax = float(torch.max(self.X_test).item())
        else:
            xmin, xmax = -1.0, 1.0

        x_line = torch.linspace(xmin, xmax, 200)
        with torch.no_grad():
            y_line = self.forward(x_line).detach().cpu().numpy()
            w0 = float(self.w_0.detach().item())
            w1 = float(self.w_1.detach().item())
        ax.plot(
            x_line.detach().cpu().numpy(),
            y_line,
            color="crimson",
            label=f"Fit: y = {w1:.4f} x + {w0:.4f}"
        )

        ax.set_title("Test Data and Fitted Line")
        ax.set_xlabel("X")
        ax.set_ylabel("y")
        ax.legend()
        ax.grid(True, alpha=0.2)

        # 2) Loss
        ax = axs[0, 1]
        if self.inter_loss:
            ax.plot(range(1, len(self.inter_loss) + 1), self.inter_loss, color="steelblue")
        ax.set_title("Training Loss (MSE)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.2)

        # 3) w0 trajectory
        ax = axs[1, 0]
        if self.inter_w_0:
            ax.plot(range(1, len(self.inter_w_0) + 1), self.inter_w_0, color="darkgreen")
        ax.set_title("w0 trajectory")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("w0")
        ax.grid(True, alpha=0.2)

        # 4) w1 trajectory
        ax = axs[1, 1]
        if self.inter_w_1:
            ax.plot(range(1, len(self.inter_w_1) + 1), self.inter_w_1, color="darkorange")
        ax.set_title("w1 trajectory")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("w1")
        ax.grid(True, alpha=0.2)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig, axs
    
if __name__ == "__main__":

    # the path of the file
    csv_path = "/Users/syedraza/Desktop/UAH/Classes/Fall2025/CPE586-MachineLearning/HWs/hw3/Hydropower.csv"

    # read in the needed data
    data_frame = polars.read_csv(csv_path)["BCR", "AnnualProduction"]

    # separate out features and targets
    X = data_frame["BCR"]
    y = data_frame["AnnualProduction"]

    # train test split this
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # make a LinearRegression() instance
    model = LinearRegression()

    # .fit() takes test set as well because it has to calculate the R2 score
    model.fit(X_train, y_train, X_test, y_test)

    # make predictions
    predictions = model.predict(X_test)

    # make the required plots
    model.analysis_plot()
