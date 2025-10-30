# Author metadata

__Name__ = "Syed Raza"
__email__ = "sar0033@uah.edu"

import torch
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 

class LogisticRegression:
    """
    A simple PyTorch-based logistic regression (binary) implemented without autograd.
    Expects X as a torch.Tensor of shape (n_samples, n_features) and y as (n_samples,) with 0/1 labels.
    """

    def __init__(self, n_features: int, learning_rate: float = 0.0025,
        max_epochs: int = 50, split: float = 0.1):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.split = split

        self.w = torch.randn(n_features, dtype=torch.float32) * 1e-2
        self.b = torch.tensor(0.0, dtype=torch.float32)
        self.cross_ent = torch.zeros(max_epochs, dtype=torch.float32)

    def _sigmoid(self, X: torch.Tensor) -> torch.Tensor:
        # X may be (n_samples, n_features) or (n_features,)
        return torch.sigmoid(X @ self.w + self.b)

    def crossentropy(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Returns mean binary cross-entropy for (X, y).
        """
        eps = 1e-9
        p = self._sigmoid(X)
        y = y.to(dtype=torch.float32)
        ce = - (y * torch.log(p + eps) + (1.0 - y) * torch.log(1.0 - p + eps)).mean()
        return ce

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        Fit using batch gradient descent (vectorized) and record validation loss per epoch.
        """
        X = X.to(dtype=torch.float32)
        y = y.to(dtype=torch.float32)

        n = X.shape[0]
        # shuffle
        perm = torch.randperm(n)
        X = X[perm]
        y = y[perm]

        # compute train/validation split index (use Python arithmetic, not torch)
        M = int((1.0 - self.split) * n)
        M = max(1, min(n, M))  # ensure at least 1 and at most n

        Xtr, ytr = X[:M], y[:M]
        Xva, yva = X[M:], y[M:]

        for epoch in range(self.max_epochs):
            # forward
            p = self._sigmoid(Xtr)
            # gradients of mean BCE w.r.t w and b
            error = p - ytr  # shape (M,)
            grad_w = (Xtr.t() @ error) / max(1, Xtr.shape[0])
            grad_b = error.mean()

            # gradient descent step
            self.w -= self.learning_rate * grad_w
            self.b -= self.learning_rate * grad_b

            # record validation loss
            if Xva.shape[0] > 0:
                self.cross_ent[epoch] = self.crossentropy(Xva, yva)
            else:
                self.cross_ent[epoch] = self.crossentropy(Xtr, ytr)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(dtype=torch.float32)
        return self._sigmoid(X)

    def predict(self, X: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return (self.predict_proba(X) > threshold).to(dtype=torch.int64)

    def predict_log_proba(self, X: torch.Tensor) -> torch.Tensor:
        eps = 1e-9
        return torch.log(self.predict_proba(X) + eps)

if __name__ == "__main__":
    # read in the data that I downloaded from kaggle:
    file_path = "/Users/syedraza/Desktop/UAH/Classes/Fall2025/CPE586-MachineLearning/HWs/hw4/cleaned_star_data.csv"
    star_data = pl.read_csv(file_path)[1:]

    # remove missing data - empty strings
    str_cols = [c for c, dtype in zip(star_data.columns, star_data.dtypes) if dtype == pl.Utf8]
    if str_cols:
        star_data = star_data.with_columns([
            pl.when(pl.col(c) == " ").then(None).otherwise(pl.col(c)).alias(c) for c in str_cols
        ])
    # drop rows containing any nulls (if that's the desired behavior).
    star_data = star_data.drop_nulls()

    # Encode 'Star color' as integers 1, 2, 3, ...
    unique_colors = star_data["Star color"].unique().to_list()
    color_map = {color: i+1 for i, color in enumerate(unique_colors)}
    star_data = star_data.with_columns(
        pl.col("Star color").replace(color_map).cast(pl.Float64)
    )
    
    # Encode 'Spectral Class': M -> 1, all others -> 0. this is the target variable
    star_data = star_data.with_columns(
        (pl.col("Spectral Class") == "M").cast(pl.Float64)
    )

    # If Temperature and Star type are already numeric, ensure they’re floats too
    star_data = star_data.with_columns([
        pl.col("Temperature (K)").cast(pl.Float64),
        pl.col("Luminosity(L/Lo)").cast(pl.Float64),
        pl.col("Radius(R/Ro)").cast(pl.Float64),
        pl.col("Absolute magnitude(Mv)").cast(pl.Float64),
        pl.col("Star type").cast(pl.Float64)
    ])

    # features and targets
    X = star_data[["Temperature (K)", "Luminosity(L/Lo)", "Radius(R/Ro)", "Absolute magnitude(Mv)", "Star type", "Star color"]]
    y = star_data[["Spectral Class"]]

    # training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # TODO: -> training / evaluation protocol
    # convert polars frames to numpy -> torch tensors
    X_train_np = X_train.to_numpy().astype("float32")
    X_test_np = X_test.to_numpy().astype("float32")
    y_train_np = y_train.to_numpy().ravel().astype("float32")
    y_test_np = y_test.to_numpy().ravel().astype("float32")

    scale = StandardScaler()

    X_train_np = scale.fit_transform(X_train_np)
    X_test_np = scale.transform(X_test_np)

    X_train_t = torch.from_numpy(X_train_np)
    X_test_t = torch.from_numpy(X_test_np)
    y_train_t = torch.from_numpy(y_train_np)
    y_test_t = torch.from_numpy(y_test_np)

    # instantiate and fit model
    model = LogisticRegression(
        n_features=X_train_t.shape[1],
        learning_rate=0.01,
        max_epochs=100,
        split=0.1
    )
    model.fit(X_train_t, y_train_t)

    # evaluate
    probs = model.predict_proba(X_test_t).detach().cpu().numpy().ravel()
    predictions = (probs > 0.5).astype(int)

    # ...existing code...
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix)

    acc = accuracy_score(y_test_np, predictions)
    prec = precision_score(y_test_np, predictions, zero_division=0)
    rec = recall_score(y_test_np, predictions, zero_division=0)
    f1 = f1_score(y_test_np, predictions, zero_division=0)
    cm = confusion_matrix(y_test_np, predictions)

    print(f"accuracy: {acc:.4f}  precision: {prec:.4f}  recall: {rec:.4f}  f1: {f1:.4f}")

    print("confusion_matrix:\n", cm)