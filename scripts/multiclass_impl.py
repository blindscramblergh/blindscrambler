import pandas as pd
from torch import nn 
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from blindscrambler.deepl import SimpleNN, ClassTrainer
import argparse
from datetime import datetime
import os
import sys


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train multiclass neural network classifier")
    parser.add_argument("keyword", type=str, help="Unique keyword to append to CSV file name for tracking runs")
    args = parser.parse_args()
    
    # get the data path
    data_path = "data/Android_Malware.csv"

    # read it using polars library:
    df = pd.read_csv(data_path, low_memory=False)

    # make all the columns have no empty spaces ahead or back
    df.columns = df.columns.str.strip()

    # delete the columns that are note important
    df = df.drop(columns=["Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port", "Protocol", "Timestamp"])

    # print statement
    print("Columns after claening the data: ", df.columns)

    # Map labels to numerical values
    label_mapping = {
        "Android_Adware": 0,
        "Android_Scareware": 1,
        "Android_SMS_Malware": 2,
        "Benign": 3
    }
    df["Label"] = df["Label"].map(label_mapping)

    # Convert all columns to numeric, coercing errors to NaN
    X = df.drop(columns=["Label"]).apply(pd.to_numeric, errors='coerce')
    y = df["Label"]
    
    # Drop rows with NaN values
    valid_indices = ~X.isnull().any(axis=1)
    X = X[valid_indices]
    y = y[valid_indices]

    # make a 80:20 random split of the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to torch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    # instantiate the SimpleNN class and the trainer class: calling them model and trainer
    epochs = 5000
    learning_rate = 0.0001

    # initialize the two classes 
    model = SimpleNN(in_features=X_train_tensor.shape[1], num_classes=4)
    trainer = ClassTrainer(X_train_tensor, y_train_tensor, eta=learning_rate, epoch=epochs, 
                        loss_function=nn.CrossEntropyLoss(), 
                        optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate), 
                        loss_vector=[], 
                        accuracy_vector=[], 
                        model=model, 
                        device='cuda' if torch.cuda.is_available() else 'cpu')

    # Train the model
    print("Starting training...")
    trainer.train()
    print("Training completed!")

    # Evaluate the model and get metrics
    print("Evaluating model...")
    # Get predictions for training set
    with torch.no_grad():
        train_outputs = model(X_train_tensor.to(trainer.device))
        _, train_predictions = torch.max(train_outputs.data, 1)
        train_predictions = train_predictions.cpu().numpy()
        y_train_np = y_train_tensor.cpu().numpy()
    
    # Get predictions for test set
    with torch.no_grad():
        test_outputs = model(X_test_tensor.to(trainer.device))
        _, test_predictions = torch.max(test_outputs.data, 1)
        test_predictions = test_predictions.cpu().numpy()
        y_test_np = y_test_tensor.cpu().numpy()
    
    # Calculate metrics for training set
    train_accuracy = accuracy_score(y_train_np, train_predictions)
    train_f1 = f1_score(y_train_np, train_predictions, average='weighted', zero_division=0)
    train_precision = precision_score(y_train_np, train_predictions, average='weighted', zero_division=0)
    train_recall = recall_score(y_train_np, train_predictions, average='weighted', zero_division=0)
    
    # Calculate metrics for test set
    test_accuracy = accuracy_score(y_test_np, test_predictions)
    test_f1 = f1_score(y_test_np, test_predictions, average='weighted', zero_division=0)
    test_precision = precision_score(y_test_np, test_predictions, average='weighted', zero_division=0)
    test_recall = recall_score(y_test_np, test_predictions, average='weighted', zero_division=0)
    
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
    
    # Create CSV with metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"metrics_{args.keyword}_{timestamp}.csv"
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    csv_filepath = os.path.join("logs", csv_filename)
    
    # Prepare metrics dataframe
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Training': [train_accuracy, train_precision, train_recall, train_f1],
        'Test': [test_accuracy, test_precision, test_recall, test_f1]
    })
    
    # Save to CSV
    metrics_df.to_csv(csv_filepath, index=False)
    print(f"Metrics saved to {csv_filepath}")
    
    # Call evaluation to generate plots
    trainer.evaluation(X_test_tensor, y_test_tensor)