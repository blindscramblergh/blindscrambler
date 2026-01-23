# script file for binary classification implementation

# import two_layer_binary_classification
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from blindscrambler.deepl.two_layer_binary_classification import binary_classification

# a helper function for the plots
def loss_plot(loss: list, show: bool = False, save: bool = True):
    """
    To plot loss history after the trainig is finished
    """

    # make the plot here 
    plt.plot(loss)
    plt.title('Loss history over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss value')
    plt.grid()

    if show: plt.show()

    # savfe the plot in the current directoy as a pdf
    if save:
        now = datetime.now()
        filename = f'crossentropyloss_{now.strftime("%Y%m%d%H%M%S")}.pdf'
        plt.savefig(filename)
        print(f'Loss plot saved as {filename}')

if __name__ == "__main__":
    # First check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # setting the size for the no of features and samples
    n = 500
    d = 20

    # call the binary classification
    # last index provides the loss vector - the others are weight matrices 
    loss_vector = binary_classification(d, n)[-1]

    # plot the loss history
    loss_plot(loss_vector)
