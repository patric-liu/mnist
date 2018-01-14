import random
import numpy as np 

#### Define Quadratic and Cross Entropy Cost Funcitons

class QuadraticCost(object):
	# Cost is sum of squares of the error
	@staticmethod
	def fn(a, y):
		return 0.5*np.linalg.norm(a-y)**2

	@staticmethod
	def delta(z, a, y):
		return (a-y) * sigmoid_prime(z)

class CrossEntropyCost(object):
	# Cost is constructed such that the derivative of the cost with respect
	#to weighted inputs to the final layer is proportional to the error 
	@staticmethod
	def fn(a, y):
		return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

	@staticmethod
	def delta(z, a, y):
		return (a-y)

#### Main Network Class
class Network(object):

	def __init__(self, sizes, cost=CrossEntropyCost):			
		# Initializes the Characteristics of the Network

		# defines network shape: a list with the sizes of each layer in order
		self.sizes = sizes
		# number of network layers, including input and output layer
		self.num_layers = len(sizes)

		self.weight_initializer()
		# sets the cost function, either Cross Entropy (default) or Quadratic
		self.cost = cost
		# performance of the network, initialized at 0. (successes/trials) ################################################
		self.performance = "0"

	def weight_initializer(self):
		""" Initializes weights and biases randomly

		Initializes with a normal distrubtuion, scaled such that the weighted inputs 
		to the next layer has a standard deviation of 2 and therefore avoids the
		learning slowdown cause by the slope of the sigmoid function at abs(z) >> 1

		"""

		# std = 1, x = 0
		self.biases  = [np.random.randn(y, 1) for y in self.sizes[1:]]
		# std = 1/sqrt(len(next_layer)), x = 0
		self.weights = [np.random.randn(y, x)/np.sqrt(x) 
						for x, y in zip(self.sizes[:-1], self.sizes[1:])]

	def feedforward(self, a): 			
		# Return the output of the network for input 'a'

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
			keep_best = False,
			show_progress = False,
			early_stopping_n = 0,
			eta_schedule_change_n = 0,
			eta_decrease_factor = 2,
			eta_decrease_steps = 0): 

		""" Trains network using Stochiastic Gradient Descent
		
		Incrementally improves the network by adjusting the weights and biases to better 
		fit 'minibatches' of multiple training examples using gradient descent. 
		These minibatches are small data sets randomly selected from the full training set. 
		Once all data from the full training set has been selected and trained on,
		the network can be re-trained over the same training data but with differently
		selected minibatches. 

		There is the option to display the network's performance in various metrics after 
		each Epoch to keep track of progress. It's also possible to use this tracking to 
		apply early stopping and learning rate scheduling.

		Args:
			training_data: list of tuples "(x,y)" where x is the input 
				and y is the correct output
			epochs: maximum number of epochs i.e. number of interations over all training data
			mini_batch_size: number of training examples to batch
			eta: learning rate
			lmbda: regularization factor
			test_data: used to evaluate true network performance, similar format to training_data 
			evaluation_data: used to select hyperparameters, similar format to training_data 
			monitor_evaluation_cost: (boolean) display cost of network with evaluation_data 
			monitor_evaluation_accuracy: (boolean) display accuracy of network on evaluation_data
			monitor_training_cost: (boolean) display cost of network with training_data
			monitor_training_accuracy: (boolean) display accuracy of network on training_data
			keep_best: Whether or not to save the 'best' parameters as measured by 
				network accuracy on evaluation_data.
			show_progress: (boolean) display accuracy of network on test_data on test_data if given
			early_stopping_n: stops training after n epochs of no evaluation_data accuracy
				improvement. If == 0, stopping isn't implemented
			eta_schedule_change_n: reduces learning rate after n epochs of no evaluation_data accuracy
				improvement. If ==0, learning rate is never changed
			eta_decrease_factor: factor by which learning rate is decreased if decreased
			eta_decrease_steps: number of times learning rate decreases before training ends

		Returns:
			Lists containing performance of every tracked metric after each epoch of training
		"""


		# converts data from unusable tuple to usable list
		training_data = list(training_data)
		n = len(training_data)

		if evaluation_data:
			evaluation_data = list(evaluation_data)
			n_eval = len(evaluation_data)

		if test_data:
			test_data = list(test_data)
			n_test = len(test_data)

		# initializes state tracking variables
		best_accuracy= 0
		no_accuracy_change_stopping = 0
		no_accuracy_change_eta = 0
		eta_decrease_n = 0
		self.best_biases = self.biases
		self.best_weights = self.weights

		evaluation_cost, evaluation_accuracy = [], []
		training_cost, training_accuracy = [], []

		# main loop which trains the network for epochs
		for j in range(epochs):
			# randomly creating all mini_batches for this epoch
			random.shuffle(training_data)
			mini_batches = [
				training_data[k:k+mini_batch_size]
				for k in range(0, n, mini_batch_size)]

			# updates weights and biases incrementally for every mini_batch	
			for mini_batch in mini_batches:
				self.update_mini_batch(
					mini_batch, eta, lmbda, n, mini_batch_size)

			if show_progress and test_data:
				print( "Epoch {}: {}/{} accuracy on test data".format(j, self.accuracy(test_data), n_test))
				self.performance = self.accuracy(test_data)/n_test
			else:
				print( "Epoch {} complete".format(j))

			# optional evaluation metric display, causes slight speed decrease
			if monitor_training_cost:
				cost = self.total_cost(training_data, lmbda)
				training_cost.append(cost)
				print("Cost on training data: {}".format(cost))
			if monitor_training_accuracy:
				accuracy = self.accuracy(training_data, convert=True)
				training_accuracy.append(accuracy)
				print("Accuracy on training data: {}".format(accuracy/n))
			if monitor_evaluation_cost and evaluation_data:
				cost = self.total_cost(evaluation_data, lmbda, convert=True)
				evaluation_cost.append(cost)
				print("Cost on evaluation data: {}".format(cost))
			if monitor_evaluation_accuracy and evaluation_data:
				eval_accuracy = self.accuracy(evaluation_data)
				evaluation_accuracy.append(eval_accuracy)
				print("accuracy on evaluation data: {}".format(eval_accuracy/n_eval))

			# saves parameters to best_parameters if the network performs better on 
			# evaluation_data accuracy than the previous best
			try: 
				if keep_best and monitor_evaluation_accuracy and eval_accuracy > best_accuracy:
					self.best_biases = self.biases
					self.best_weights = self.weights
			except NameError:
				print("monitor_evaluation_accuracy needs to be True to keep best parameters")

			# implements early stopping
			# early stopping stops training if no improvment on evaluation accuracy is 
			# seen for n epochs
			if early_stopping_n > 0 and monitor_evaluation_accuracy:
				if eval_accuracy > best_accuracy:
					no_accuracy_change_stopping = 0
				else:
					no_accuracy_change_stopping += 1
				print("No improvements for {} epochs".format(no_accuracy_change_stopping))
				if (no_accuracy_change_stopping == early_stopping_n):
					print("Learning stopped at {} epochs due to no improvement".format(j), \
						  "for {} epochs".format(early_stopping_n))
					if keep_best and monitor_evaluation_accuracy: self.keep_best()
					self.save_performance(test_data, n_test)
					return evaluation_cost, evaluation_accuracy, \
					 	   training_cost, training_accuracy

			# implements learning schedule
			# This learning schedule decreases learning rate by some factor if no improvement
			# on evaluation accuracy is seen for n epochs. 
			if eta_schedule_change_n > 0 and monitor_evaluation_accuracy:
				if eval_accuracy > best_accuracy:
					no_accuracy_change_eta = 0
				else:
					no_accuracy_change_eta += 1
				print("Learning rate schedule: no improvement for {} epochs".\
					format(no_accuracy_change_eta))
				if (no_accuracy_change_eta == eta_schedule_change_n):
					eta = eta / eta_decrease_factor
					no_accuracy_change_eta = 0
					eta_decrease_n +=1
					print("Learning rate decreased from {:.6f} to {:.6f}"\
						.format(eta * eta_decrease_factor, eta))
				# optional termination of training after n decreases to eta
				if eta_decrease_n == eta_decrease_steps and eta_decrease_n > 0:
					if keep_best and monitor_evaluation_accuracy: self.keep_best() 	   
					self.save_performance(test_data, n_test)
					return evaluation_cost, evaluation_accuracy, \
					 	   training_cost, training_accuracy

			if keep_best and monitor_evaluation_accuracy:
				if eval_accuracy > best_accuracy:
					best_accuracy = eval_accuracy

			print("##################################################")

		if keep_best and monitor_evaluation_accuracy: self.keep_best() 	   
		self.save_performance(test_data, n_test)

		return evaluation_cost, evaluation_accuracy, \
			   training_cost, training_accuracy

	def update_mini_batch(self, mini_batch, eta, lmbda, n, mini_batch_size): 
		''' Update network parameters based on a single minibatch
		
		Uses gradient descent in the self.backprop method to find the gradient 
		of the cost function for a single minibatch, adds a regulatization factor,
		then updates the weights and biases.

		Args:
			mini_batch: list of tuples "(x,y)" where x is the input 
				and y is the correct output
			eta: learning rate
			lmbda: regularization factor

		Returns:
			None: updates weights and biases 
		'''

		# initializes array containing the sum of gradients of each individual training example
		nabla_b = [ np.zeros( b.shape ) for b in self.biases ]
		nabla_w = [ np.zeros( w.shape ) for w in self.weights ]
		
		# creates gradient array from sum of individual gradients
		for x, y in mini_batch:
		# find gradient of a single training example using backpropogation
			delta_nabla_b, delta_nabla_w = self.backprop( x, y )

			nabla_b = [ nb + dnb for nb, dnb in zip( nabla_b, delta_nabla_b ) ]
			nabla_w = [ nw + dnw for nw, dnw in zip( nabla_w, delta_nabla_w ) ]
		
		# updates parameters using the gradients based on the function v' = v - (eta/n)*nabla_v
		self.biases  = [ b - ( eta/mini_batch_size ) * nb 
						for b, nb in zip( self.biases,  nabla_b)  ]
		self.weights = [ (1-eta*(lmbda))*w - ( eta/mini_batch_size ) * nw 
						for w, nw in zip( self.weights, nabla_w)  ]
		
	def backprop(self, x, y): 
		''' Finds the gradient of the cost function

			Calculates the gradient of the cost function with respect to the 
			current weights and biases for a single training example.

			**LONG DESCRIPTION

			Args:
				x: single training input
				y: desired output for input 'x'

			Returns:
				nabla_b: bias gradients
				nabla_w: weight gradients
		'''

		# initializes containers
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]

		activation = x
		activations = [x]
		weighted_inputs = []

		# forward propogation with [z] tracking
		# Tracking the weighted inputs and their gradients are critical
		# for simpler and faster calculations.
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation)+b
			activation = sigmoid(z)
			# saves- 'z' and 'a'
			weighted_inputs.append(z)
			activations.append(activation)

		# derivative of cost function with respect to the weighted input of output layer
		# delta contains the gradient of the weighted inputs of the 'current' layer
		delta = (self.cost).delta(weighted_inputs[-1], activations[-1], y)
		# saves gradients of the parameters of the last layer - not in loop because 
		# delta is defined differently to the other layers
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())

		# Loops through the hidden layers in reverse order, starting from the last hidden layer
		for l in range(2, self.num_layers):
			z = weighted_inputs[-l]
			# delta(l) = w(l+1)jk * delta(l+1)  *  sig'(z(l)) 
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_prime(z)
			# save gradients for current layer
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
		return (nabla_b, nabla_w)

	def accuracy(self, data, convert=False):
		''' Calculates Accuracy

		Calculates number of correct classifications for the given set of data
		A classification is correct if the neuron with the highest activation
		corresponds with the correct label for a given input

		Convert is for MNIST data where the labels for evaluation and test data
		must first be converted from an int to a one-hot vector 
		'''
		if convert:
			results = [(np.argmax(self.feedforward(x)), np.argmax(y))
						for (x, y) in data]
		else:
			results = [(np.argmax(self.feedforward(x)), y)
						for (x, y) in data]
		result_accuracy = sum(int(x == y) for (x, y) in results)
		return result_accuracy

	def total_cost(self, data, lmbda, convert=False):
		''' Calculates Cost

		Calculates average cost per example for the given set of data
		Cost is the sum of cost from the cost function and L2 regularization cost

		L2 cost is defined such that the cost for a weight is proportional to
		the square of its size, where the proportion is constant across all weights

		'''
		cost = 0.0
		for x, y in data:
			a = self.feedforward(x)
			if convert: y = vectorized_result(y)
			cost += self.cost.fn(a, y)
			cost += 0.5 * lmbda * sum(np.linalg.norm(w)**2 for w in self.weights)
			cost = cost/len(data)
		return cost

	def feedforward_softmax_output(self, a, hardness): 			
		''' Return the softmax output of the network for input 'a'
		Softmax outputs are a probability distribution, where every output
		represents the probability of that neuron being the correct classification. 
		It does this by normalizing the exponential of each weighted input 
		so that they all add to 1. 

		hardness: parameter used to change the exponent's base
				  (e^z)^hardness
		'''
		n = 0
		for b, w in zip( self.biases,self.weights ):
			z = np.dot(w, a) + b
			n += 1
			if n == self.num_layers - 1:
				break
			a = sigmoid(z)
		z_exp = np.exp(z*hardness)
		normal = np.sum(z_exp)
		confidence = z_exp/normal
		return confidence

	def keep_best(self):
		# Saves best parameters for file saving once training is complete 
			self.biases = self.best_biases
			self.weights = self.best_weights

	def save_performance(self, test_data, n_test):
		# Saves performance on test_data  for file saving once training is complete
		performance = self.accuracy(test_data)
		self.performance = performance/n_test
		print("Test accuracy for best evaluation accuracy: {}/ {}".format(performance, n_test))



# data label helper
def vectorized_result(j):
	e = np.zeros((10, 1))
	e[j] = 1.0
	return e

# element-wise sigmoid activation function
def sigmoid(z):						
	return 1.0/(1.0+np.exp(-z))

# element-wise derivatibe of sigmoid function
def sigmoid_prime(z):				
	return sigmoid(z)*(1-sigmoid(z))