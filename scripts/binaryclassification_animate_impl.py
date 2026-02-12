# script file for binary classification implementation

# import two_layer_binary_classification
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from blindscrambler.deepl.two_layer_binary_classification import binary_classification
import blindscrambler.animation.largewt_animation as animation 

plot = False

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
    n = 200
    d = 40000

    # call the binary classification, and make weights animation
    # last index provides the loss vector - the others are weight matrices 
    result = binary_classification(d, n, epochs=5000, lr=0.01)
    weights = result[0:4]
    loss_vector = result[4]

    # plot the loss history
    if plot == True:
        loss_plot(loss_vector)


    print("Starting render for all the weights...")

    for i in range(4):
        animation.animate_large_heatmap(
            weights[i], 
            dt=0.04,
            file_name=f"weight_{i + 1}_evolution",
            title_str="Weight Evolution"
        )