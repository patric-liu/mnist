# Neural Network Library with MNIST implementation

This is a library designed to use backpropogation to train an MLP network to
classify handwritten digits using the MNIST database. With this said, it can be
trained on any data whose inputs and outputs can be expressed as  vectors of 
values with some very minor modifications. 

This code is written for Python 3 and requires numpy, pickle, gzip, matplotlib, 
and scipy


# Files:

network.py - Main class which contains all functionality of the neural network,
			 and some helper classes/functions

trainer.py - Trains and saves a network under user-specified conditions

front_end.py - Allows user to feed their own MNIST 'style' images through 
			   previously trained networks to observe the output

mnist_loader.py - Library to load MNIST image data 
	(from Michael Nielsen's repository, 'neural-networks-and-deep-learning')

mnist.pkl.gz - Compressed MNIST image data

best_networks - folder containing the best performing parameters


# Details/Notes

This library can train any feedforward network shape and uses a cross-entropy
cost function and L2 regularization by default. Various options for 
monitoring performance during training are available, as well as early 
training termination and learning rate scheduling. 

When defining the network class, the input and output layer sizes should be 
included. 

When importing a user 'MNIST' image, the format must be the same as MNIST,
that is, 28x28 pixels with black text on a white background
