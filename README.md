This is the version 0.1.15 of blindscrambler. This version is uploaded to PyPi now and can be downloaded using:

>>> pip install blindscrambler==0.1.15

The function binary clssification can be imported with the following command:

>>> from blindscrambler.deepl.two_layer_binary_classification import binary_classification

Once n and d are provided, the function will take care of everything else and return a list of weights vector and 
the loss vector over the epochs.