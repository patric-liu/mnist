import random
import numpy as np 

class Network(object):

	def __init__(self, sizes):			
		''' Initializes the Characteristics of the Network
		'''
		# number of network layers, including input and output layer
		self.num_layers = len(sizes)
		# list with the sizes of each layer in order
		self.sizes = sizes
		# random normal distrubtion of biases for all neurons in all layers but the input layer
		self.biases  = [np.random.randn(y, 1) for y in sizes[1:] ]
		# random normal distribution of weights for all connections, 
		# second dimension indexes the first layer and first dimension indexes the next layer
		self.weights = [np.random.randn(y, x) for x, y in zip( sizes[:-1],sizes[1:]) ]
		# performance of the network, initialized at 0. (successes/trials)
		self.performance = 0

	def feedforward(self, a): 			
		''' Takes input 'a' and feeds it through the current network
		'''
		for b, w in zip( self.biases,self.weights ):
			a = sigmoid( np.dot(w, a) + b )
		return a

	def SGD(self, training_data, epochs, mini_batch_size, eta, 
			test_data = None, show_progress = False ): 

		""" Trains network using Stochiastic Gradient Descent
		
		Incrementally improves the network by adjusting the weights and biases to better 
		fit 'minibatches' of multiple training examples. These minibatches are small random
		sets made from the whole training set. Once all minibatches in the training data set 
		are exhausted, the network can be re-trained over the same training data but with
		different minibatches. Can also show performance after each epoch

		Args:
			training_data: list of tuples "(x,y)" where x is the input 
				and y is the correct output
			epochs: number of epochs i.e. number of interations over all training data
			mini_batch_size: number of training examples to batch
			eta: learning rate
			test_data: same format as training_data, required for show_progress
			show_progress: (boolean) whether or not to display 
				network performance after every epoch

		Returns:
			None: update_mini_batch method updates the weights and biases
		"""
		training_data = list(training_data)
		n = len(training_data)		

		if show_progress:
			test_data = list(test_data)
			n_test = len(test_data)

		for j in range(epochs):
			random.shuffle(training_data)
			mini_batches = [
				training_data[ k:k+mini_batch_size ] 
				for k in range( 0, n, mini_batch_size) ]
			for mini_batch in mini_batches:
				self.update_mini_batch( mini_batch, eta )
			if show_progress:
				print( "Epoch {0}: {1}/{2} correct".format( j, self.evaluate(test_data), n_test ) )
			else:
				print( "Epoch {0} complete".format( j ) )
		self.performance = self.evaluate(test_data)/ n_test

	def update_mini_batch(self, mini_batch, eta): 
		''' Update networks weights and biases
		
		Applies Gradient Descent using backprop method for a single minibatch

		Args:
			mini_batch: list of tuples "(x,y)" where x is the input 
				and y is the correct output
			eta: learning rate

		Returns:
			None: updates weights and biases 
		'''

		# initializes nablas which sum the individual training examples' gradients
		nabla_b = [ np.zeros( b.shape ) for b in self.biases ]
		nabla_w = [ np.zeros( w.shape ) for w in self.weights ]
		
		# updates nablas for each training example in the minibatch
		for x, y in mini_batch:
		# find the change in gradients using backpropogation
			delta_nabla_b, delta_nabla_w = self.backprop( x, y )

			nabla_b = [ nb+dnb for nb, dnb in zip( nabla_b, delta_nabla_b ) ]
			nabla_w = [ nw+dnw for nw, dnw in zip( nabla_w, delta_nabla_w ) ]
		
		# updates weights and biases with nablas based on the fucntion v' = v - (eta/n) * nabla_v
		self.biases  = [ b - ( eta/len(mini_batch) ) * nb for b, nb in zip( self.biases,  nabla_b)  ]
		self.weights = [ w - ( eta/len(mini_batch) ) * nw for w, nw in zip( self.weights, nabla_w)  ]
		
	def backprop(self, x, y): 
		''' Finds the gradient of the function

			Calculates the gradient of the cost function with respect to the 
			current weights and biases for a single training example.
			Tracking the weighted inputs and their gradients are critical
			for simpler and faster calculations.

			Args:
				x: single training input
				y: desired output for input 'x'

			Returns:
				nabla_b: bias gradients
				nabla_w: weight gradients
		'''

		# initializes nablas as a list of numpy arrays of appropriate sizes
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		# variable containing the activation of the current layer
		activation = x
		# list to store the activations of each layer
		activations = [x]
		# list to store the weighted inputs [z] to each layer
		weighted_inputs = []

		# forward propogation with [z] tracking
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation)+b
			activation = sigmoid(z)
			# saves- 'z' and 'a'
			weighted_inputs.append(z)
			activations.append(activation)

		# derivative of **sigmoid** cost function with respect to weighted input to output layer
		# delta contains the gradient of the weighted inputs of the current layer
		delta = self.cost_derivative_sqr(activations[-1], y) * sigmoid_prime(weighted_inputs[-1])

		# set nabla_b,w for the last layer - not in loop because delta is defined differently to the other layers
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())

		# Loops through the rest of the layers backwards. l = 1 refers to last layer, l = 2 second last etc. 
		for l in range(2, self.num_layers):
			z = weighted_inputs[-l]
			# delta(l) = w(l+1)jk * delta(l+1)  *  sig'(z(l)) 
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_prime(z)
			# update nabla_b,w
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
		return (nabla_b, nabla_w)

	def evaluate(self, test_data): 
		# Returns the fraction of test examples the network correctly classifies
		test_results = [(np.argmax(self.feedforward(x)), y)
						for (x, y) in test_data]
		return sum(int(x == y) for (x, y) in test_results)

	def cost_derivative_sqr(self, output_activations, y): 
		# Returns vector of partial derivatibes dC/dA for a square cost function
		return (output_activations-y)

def sigmoid(z):						# implements element-wise sigmoid activation function
	return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):				# derivatibe of sigmoid function
	return sigmoid(z)*(1-sigmoid(z))