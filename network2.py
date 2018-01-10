import json
import random
import sys
import numpy as np 

#### Define Quadratic and Cross Entropy Cost Funcitons

class QuadraticCost(object):

	@staticmethod
	def fn(a, y):
		return 0.5*np.linalg.norm(a-y)**2

	@staticmethod
	def delta(z, a, y):
		return (a-y) * sigmoid_prime(z)
class CrossEntropyCost(object):

	@staticmethod
	def fn(a, y):
		return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

	@staticmethod
	def delta(z, a, y):
		return (a-y)
#### Main Network Class

class Network(object):

	def __init__(self, sizes, cost=CrossEntropyCost):			
		''' Initializes the Characteristics of the Network
		'''
		# number of network layers, including input and output layer
		self.num_layers = len(sizes)
		# list with the sizes of each layer in order
		self.sizes = sizes
		# 
		self.default_weight_initializer()
		# performance of the network, initialized at 0. (successes/trials)
		self.cost = cost
		self.performance = "str"

	def default_weight_initializer(self):
		self.biases  = [np.random.randn(y, 1) for y in self.sizes[1:]]
		self.weights = [np.random.randn(y, x)/np.sqrt(x) 
						for x, y in zip(self.sizes[:-1], self.sizes[1:])]

	def large_weight_initializer(self):

		self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
		self.weights = [np.random.randn(y, x) 
						for x, y in zip(self.sizes[:-1], self.sizes[1:])]

	def feedforward(self, a): 			
		"""Return the output of the network for input 'a'. """
		for b, w in zip( self.biases,self.weights ):
			a = sigmoid( np.dot(w, a) + b )
		return a

	def SGD(self, training_data, epochs, mini_batch_size, eta, 
			lmbda = 0.0,
			test_data = None,
			evaluation_data = None,
			monitor_evaluation_cost = False,
			monitor_evaluation_accuracy = False,
			monitor_training_cost = False,
			monitor_training_accuracy = False,
			keep_best = True,
			show_progress = False,
			early_stopping_n = 0): 

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

		if evaluation_data:
			evaluation_data = list(evaluation_data)
			n_eval = len(evaluation_data)

		if test_data:
			test_data = list(test_data)
			n_test = len(test_data)

		best_accuracy_stopping = 0
		best_accuracy_keeping = 0
		no_accuracy_change = 0
		self.best_biases = self.biases
		self.best_weights = self.weights

		evaluation_cost, evaluation_accuracy = [], []
		training_cost, training_accuracy = [], []

		for j in range(epochs):
			random.shuffle(training_data)
			mini_batches = [
				training_data[k:k+mini_batch_size]
				for k in range(0, n, mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_mini_batch(
					mini_batch, eta, lmbda, n)

			if show_progress and test_data:
				print( "Epoch {}: {}/{} correct".format(j, self.accuracy(test_data), n_test))
				self.performance = self.accuracy(test_data)/n_test
			else:
				print( "Epoch {} complete".format(j))

			if monitor_training_cost:
				cost = self.total_cost(training_data, lmbda)
				training_cost.append(cost)
				print("Cost on training data: {}".format(cost))
			if monitor_training_accuracy:
				accuracy = self.accuracy(training_data, convert=True)
				training_accuracy.append(accuracy)
				print("Accuracy on training data: {} / {}".format(accuracy, n))
			if monitor_evaluation_cost:
				cost = self.total_cost(evaluation_data, lmbda, convert=True)
				evaluation_cost.append(cost)
				print("Cost on evaluation data: {}".format(cost))
			if monitor_evaluation_accuracy:
				accuracy = self.accuracy(evaluation_data)
				evaluation_accuracy.append(accuracy)
				print("accuracy on evaluation data: {} / {}".format(accuracy, n_eval))

			if early_stopping_n > 0 and monitor_evaluation_accuracy:
				if accuracy > best_accuracy_stopping:
					best_accuracy_stopping = accuracy
					no_accuracy_change = 0
				else:
					no_accuracy_change += 1
				if (no_accuracy_change == early_stopping_n):
					return evaluation_cost, evaluation_accuracy, \
					 	   training_cost, training_accuracy

			if keep_best and monitor_evaluation_accuracy:
				if accuracy > best_accuracy_keeping:
					best_accuracy_keeping = accuracy
					self.best_biases = self.biases
					self.best_weights = self.weights

		if keep_best and monitor_evaluation_accuracy:
			self.biases = self.best_biases
			self.weights = self.best_weights
			performance = self.accuracy(test_data)
			self.performance = performance/n_test
			print("Best performance: {}/ {}".format(performance, n_test))			 	   

		return evaluation_cost, evaluation_accuracy, \
			   training_cost, training_accuracy



	def update_mini_batch(self, mini_batch, eta, lmbda, n): 
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

			nabla_b = [ nb + dnb for nb, dnb in zip( nabla_b, delta_nabla_b ) ]
			nabla_w = [ nw + dnw for nw, dnw in zip( nabla_w, delta_nabla_w ) ]
		
		# updates weights and biases with nablas based on the fucntion v' = v - (eta/n) * nabla_v
		self.biases  = [ b - ( eta/len(mini_batch) ) * nb 
						for b, nb in zip( self.biases,  nabla_b)  ]
		self.weights = [ (1-eta*(lmbda/n))*w - ( eta/len(mini_batch) ) * nw 
						for w, nw in zip( self.weights, nabla_w)  ]
		
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

		# derivative of cost function with respect to weighted input to output layer
		# delta stores the gradient of the weighted inputs of the current layer
		delta = (self.cost).delta(weighted_inputs[-1], activations[-1], y)

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

	def accuracy(self, data, convert=False):
		if convert:
			results = [(np.argmax(self.feedforward(x)), np.argmax(y))
						for (x, y) in data]
		else:
			results = [(np.argmax(self.feedforward(x)), y)
						for (x, y) in data]

		result_accuracy = sum(int(x == y) for (x, y) in results)
		return result_accuracy

	def total_cost(self, data, lmbda, convert=False):
		cost = 0.0
		for x, y in data:
			a = self.feedforward(x)
			if convert: y = vectorized_result(y)
			cost += self.cost.fn(a, y)/len(data)
			cost += 0.5 * (lmbda/len(data)) * sum(np.linalg.norm(w)**2 for w in self.weights)
		return cost

	def cost_derivative_sqr(self, output_activations, y): 
		# Returns vector of partial derivatibes dC/dA for a square cost function
		return (output_activations-y)

def vectorized_result(j):
	e = np.zeros((10, 1))
	e[j] = 1.0
	return e

def sigmoid(z):						# implements element-wise sigmoid activation function
	return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):				# derivatibe of sigmoid function
	return sigmoid(z)*(1-sigmoid(z))