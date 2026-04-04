import pandas as pd
import polars as pl
from torch import nn 
import torch
import glob
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, ConfusionMatrixDisplay
from datasets import load_dataset
from datasets import load_from_disk
from torchvision import transforms
from torchvision.ops import sigmoid_focal_loss
import sys
from torch.utils.data import DataLoader
import subprocess
import glob
from sklearn.model_selection import train_test_split
import os

# Dice Loss implementation for binary classification
def dice_loss(y_pred, y_true, smooth=1.0):
    """
    Dice Loss for binary classification.
    Good for imbalanced datasets as it measures overlap directly.
    
    Args:
        y_pred: Predicted logits (raw model output)
        y_true: Ground truth binary labels (0 or 1)
        smooth: Smoothing constant to avoid division by zero
        
    Returns:
        Scalar dice loss value
    """
    # Apply sigmoid to convert logits to probabilities [0, 1]
    y_pred_prob = torch.sigmoid(y_pred)
    
    # Flatten predictions and targets
    y_pred_flat = y_pred_prob.view(-1)
    y_true_flat = y_true.view(-1)
    
    # Calculate intersection and union
    intersection = (y_pred_flat * y_true_flat).sum()
    
    # Dice coefficient: 2 * intersection / (sum of predictions + sum of targets)
    dice_coefficient = (2.0 * intersection + smooth) / (y_pred_flat.sum() + y_true_flat.sum() + smooth)
    
    # Dice loss is 1 - dice coefficient
    return 1.0 - dice_coefficient


class Dataset():
    def __init__(self, directory, datafile, labelfiles):

        # make the links using whcih we have to get data
        self.data_links = directory + "/" + "*" + datafile

        # make the data object.
        self.processed_data = pl.DataFrame()

    def time_enforcing(self, data, k):
        """
        Enforce time dependence on 1D velocity data by creating lagged features.
        
        Args:
            data: Polars DataFrame with velocity values (1D)
            k: Number of time lags to create
            
        Returns:
            DataFrame with original and lagged velocity columns (None rows removed)
        """
        # Convert to list for manipulation
        vel_list = data["v_t"].to_list()  # Get the first column as list
        N = len(vel_list)

        for i in range(k):
            new_list = [None] * N
            new_list[i+1:N] = vel_list[0:N-i-1]
            data = data.with_columns(pl.Series(f"v_t_minus_{i+1}", new_list))

        data = data.select([
            pl.all().exclude("acc_label"),
            "acc_label"
        ])

        # Remove rows with null values
        data = data.drop_nulls()
        
        return data

    def zero_order_holdout(self, data, labels):
        """
        Applies the zero order hold out technique and makes the number of samples equal for two data frames.

        Params:
            (1) data: polars.DataFrame
                containing the data
            (2) lables: polars.DataFrame
                containing the features 

        Returns:
            returns one data frame that is ready to be neural networked :)
        """
        # empty data frame with three columns: time (s), speed (m/s), label (0/1)
        combined_data = []

        # we have different number of samples here 
        label_samples = labels.height
        data_samples = data.height 

        count = 0
        for i in range(data_samples):
            if count < label_samples - 1:
                if (data["Time"][i] < labels["Time"][count+1]):
                    # make new sample based on zero order holdout
                    row = {"time": data["Time"][i], "v_t": data["Message"][i], "acc_label": labels["Message"][count]}

                    # extend the data frame
                    combined_data.append(row)

                else:
                    # increment counter
                    count += 1

                    # make new sample for this data
                    row = {"time": data["Time"][i], "v_t": data["Message"][i], "acc_label": labels["Message"][count]}

                    # extend the data frame
                    combined_data.append(row)

            else: 
                # make new sample by adding last one
                row = {"time": data["Time"][i], "v_t": data["Message"][i], "acc_label": labels["Message"][label_samples - 1]}

                # extend the data frame
                combined_data.append(row)

        return pl.DataFrame(combined_data)

    def read_data(self, preprocessed_data_saved, save_path):
        """
        This function will read data and prepare the features based on time_enforcing

        Args:
            preprocessed_data_saved: Boolean to tell the class to process data or not 

        Returns:
            A data frame, ready to do machine learning on.
        """
        if preprocessed_data_saved:
            print(f"loading the already saved data from {save_path}")
            return pl.read_csv(save_path)

        else:
            # glob them files
            data_files = glob.glob(self.data_links)

            # get the data frame to concat all the prepared data
            entire_data = pl.DataFrame()

            # for loop for each data file
            for i in range(len(data_files)):
                fl_data = pl.read_csv(data_files[i])

                # from km/h to m/s
                fl_data = fl_data.with_columns(
                    (pl.col("Message") * 1000 / 3600).alias("Message")
                )

                # read in the labels and make binary label (make 6 equal to 1, all else 0)
                fl_labels = pl.read_csv(
                    data_files[i].replace("wheel_speed_fl.csv", "acc_status.csv"), 
                )

                # make binary label
                fl_labels = fl_labels.with_columns(
                    pl.when(pl.col("Message") == 6).then(1).otherwise(0).alias("Message")
                )
                
                # only keep time and speed
                fl_data = fl_data.select(["Time", "Message"])

                # delete all the columns that have. Bus == 2 in labels
                fl_labels = fl_labels.filter(
                    pl.col("Bus") == 0
                )

                # only keep tiem and label
                fl_labels = fl_labels.select(["Time", "Message"])

                # create zero order holdout data 
                zoh_data = self.zero_order_holdout(fl_data, fl_labels)

                # some print statements
                print(f"I am at the {i}th iteration")
                print(f"The data file name is: ", data_files[i])
                print(f"The label file name is: ", data_files[i].replace("wheel_speed_fl.csv", "acc_status.csv"))
                print(f"The number of samples in this file is: ",  zoh_data.height)
                print("\n")

                # enforce the time dependence
                time_enforced_zoh_data = self.time_enforcing(zoh_data, 10)

                # concat it with the entire data dataframe
                entire_data = pl.concat([entire_data, time_enforced_zoh_data], how = "vertical")

            entire_data.write_csv(save_path)
            print(f"Data saved to {save_path}")
            return entire_data


# MAKE A RESNET ARCHITECTURE FOR THIS PROBLEM

# there will be 4 classes that I will make here:
# (1) acc_conv1d
# (2) ResidualBlock class 
# (3) ResNet class 
# (4) ACCTrainer class

# 1D convolution class 
class acc_conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()

        # store attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size 
        self.stride = stride
        self.padding = padding

        # initialize learnable parameters in the layer object
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # do the reset parameters:
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming initialization: a special ionitialization to avoid exploding and vanishing gradients
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # use pytorch's conv1d function here
        return nn.functional.conv1d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)

# Residual Net Class: This one is taken from figure 2 of the Kaiming He paper 
class Resblock1d(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None):
        super().__init__()

        # 2 conv layers
        self.conv1 = nn.Sequential(
            acc_conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            acc_conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels)
        )

        self.relu = nn.ReLU()
        self.out_channels = out_channels
        self.downsample = downsample

    # the forward function
    def forward(self, x):
        residual = x
        F_x = self.conv1(x)
        F_x = self.conv2(F_x)
        if self.downsample:
            residual = self.downsample(x)
        F_x = F_x + residual
        F_x = self.relu(F_x)
        return F_x

# A class to build the ResNet architecture
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # make relu just here:
        self.relu = nn.ReLU()

        # ---------- conv layer 1 ----------
        self.conv1 = acc_conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        
        # ---------- conv layer 1 ----------
        self.conv2 = acc_conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)

        # ---------- residual block 1 ----------
        self.residual1 = Resblock1d(32, 32)

        # ---------- residual block 2 ----------
        self.residual2 = Resblock1d(32, 32)

        # ---------- Average Pooling ----------
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

        # ---------- Fully connected ----------
        self.fc1 = nn.Linear(32, 1)
        # make a dropout just in case to avoid overfitting
        self.dropout = nn.Dropout(0.15)

        # ---------- sigmoid ----------
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # forward function to line up everything 
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.flatten(self.global_avg_pool(x))
        x = self.dropout(x)
        x = self.fc1(x)
        # x = self.sigmoid(x)

        return x
                
class ACCTrainer():
    # constructor
    def __init__(self, eta, epoch, train_dataloader, val_dataloader, loss_function, 
            optimizer, loss_vector, accuracy_vector, model, device, scheduler):
        super().__init__()

        # make the class attributes
        self.train_dataloader = train_dataloader    # the training data loader
        self.val_dataloader = val_dataloader        # the validation data loader
        self.eta = eta                              # learning rate 
        self.epoch = epoch                          # the epoch
        self.loss_function = loss_function          # loss function
        self.optimizer = optimizer                  # optimizer
        self.loss_vector = loss_vector              # to save loss
        self.accuracy_vector = accuracy_vector      # to save accuracy
        self.model = model                          
        self.device = device
        self.scheduler = scheduler

    def save_onnx(self, file_path="/home/sar0033/blindscrambler/scripts/models", epoch_num=None):
        """
        Saves the trained model in ONNX format.
        
        Args:
            file_path (str): Path where the ONNX model will be saved
            epoch_num (int): Epoch number to save in filename (defaults to total epochs)
            
        Raises:
            RuntimeError: If the model has not been trained yet
        """
        # Check if model has been trained
        if not self.loss_vector or len(self.loss_vector) == 0:
            raise RuntimeError("Model has not been trained yet. Please train the model before saving.")
        
        # Use provided epoch number or default to total epochs
        save_epoch = epoch_num if epoch_num is not None else self.epoch
        
        # Set model to evaluation mode
        self.model.eval()

        # Create a dummy input with the correct shape (batch_size=1, channels=1, sequence_length=11)
        dummy_input = torch.randn(1, 1, 11, device=self.device)

        # output file path 
        output_path = os.path.join(file_path, f"models_ACC_epoch_{save_epoch}.onnx") # saves the onnx model with epoch

        # Export the model to ONNX format
        torch.onnx.export(
            self.model,                          # Model to export
            dummy_input,                         # Model input
            output_path,                         # Output file path
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
        
        print(f'Model saved to {output_path}')

        self.model.train()

        return 0

    # the training function
    def train(self):
        """
        Train the ACC model and store epoch-level loss and accuracy.
        """

        # move model to device
        self.model = self.model.to(self.device)

        # training loop
        for epoch in range(self.epoch):
            self.model.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch_idx, (X_batch, y_batch) in enumerate(self.train_dataloader):
                # move batch to device
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).float().view(-1, 1)

                # zero gradients
                self.optimizer.zero_grad()

                # forward pass
                outputs = self.model(X_batch)

                # compute loss
                loss = self.loss_function(outputs, y_batch).mean()

                # backward pass + optimizer step
                loss.backward()
                self.optimizer.step()

                # binary predictions
                predicted = (outputs >= 0.5).float()

                # batch accuracy
                correct = (predicted == y_batch).sum().item()

                # accumulate statistics
                epoch_loss += loss.item()
                epoch_correct += correct
                epoch_total += y_batch.size(0)

                # print loss, train accuracy, and val accuracy every 2000 batches
                if (batch_idx + 1) % 1000 == 0:
                    batch_avg_loss = epoch_loss / (batch_idx + 1)
                    batch_avg_accuracy = epoch_correct / epoch_total
                    
                    # compute validation accuracy
                    self.model.eval()
                    val_correct = 0
                    val_total = 0
                    
                    with torch.no_grad():
                        for val_X, val_y in self.val_dataloader:
                            val_X = val_X.to(self.device)
                            val_y = val_y.to(self.device).float().view(-1, 1)
                            
                            val_outputs = self.model(val_X)
                            val_pred = (val_outputs >= 0.5).float()
                            
                            val_correct += (val_pred == val_y).sum().item()
                            val_total += val_y.size(0)
                    
                    val_accuracy = val_correct / val_total
                    
                    # switch back to train mode
                    self.model.train()
                    
                    print(
                        f"Epoch [{epoch+1}/{self.epoch}], Batch [{batch_idx+1}/{len(self.train_dataloader)}], "
                        f"Loss: {batch_avg_loss:.4f}, Train Acc: {batch_avg_accuracy:.4f}, Val Acc: {val_accuracy:.4f}"
                    )

            # epoch averages
            avg_loss = epoch_loss / len(self.train_dataloader)
            avg_accuracy = epoch_correct / epoch_total

            # store metrics
            self.loss_vector.append(avg_loss)
            self.accuracy_vector.append(avg_accuracy)

            # final validation pass for the epoch
            self.model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0.0

            with torch.no_grad():
                for val_X, val_y in self.val_dataloader:
                    val_X = val_X.to(self.device)
                    val_y = val_y.to(self.device).float().view(-1, 1)

                    val_outputs = self.model(val_X)
                    loss_val = self.loss_function(val_outputs, val_y).mean()

                    val_pred = (val_outputs >= 0.5).float()

                    val_correct += (val_pred == val_y).sum().item()
                    val_total += val_y.size(0)
                    val_loss += loss_val.item()

            avg_val_loss = val_loss / len(self.val_dataloader)
            avg_val_accuracy = val_correct / val_total

            # save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Saving checkpoint at epoch {epoch + 1}...")
                self.save_onnx(epoch_num=epoch + 1)
                self.model.train()  # switch back to train mode

            # scheduler step
            if self.scheduler is not None:
                self.scheduler.step()