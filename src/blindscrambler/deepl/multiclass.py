import pandas as pd
from torch import nn 

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
    def train(self, kwargs):
        # Calling train function will invokle teh training process 
        return 0
    
    def test(self, kwargs):
        # Calling this fucntion would perform test using the test data
        return 0
    
    def predict(self, kwargs):
        # Calling this function would help make a prediction using the set of features. 
        # It takes only one argument --> X 

        return 0

    def save(self, kwargs):
        # This will save the trained model in the onnx format for later use
        return 0
    
    def evaluation(self, kwargs):
        # This function takes in the loss vector and accuracy vector and plot to demonstrate 
        # the performance on the training set. In addition, it also creates a plot of confusion 
        # matrix showing teh correct label and predicted labels, as well as the F1 score, precision,
        # Recall, and Accuracy for both training and test dataset
        
        return 0

if __name__ == "__main__":
    # get the data path
    data_path = "/Users/syedraza/blindscrambler/scripts/data/Android_Malware.csv"

    # read it using polars library:
    df = pd.read_csv(data_path, low_memory=False)

    # make all the columns have no empty spaces ahead or back
    df.columns = df.columns.str.strip()


    # delete the columns that are note important
    df = df.drop(columns=["Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port", "Protocol", "Timestamp"])

    # print statement
    print("Columns afterclaening the data: ", df.columns)