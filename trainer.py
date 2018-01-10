import mnist_loader
import network
import pickle


# Loads training data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Creates untrained network of desired shape
# Enter in the dimensions of the network in the form [a, b, c, d ... n], 
# where a is the input size and n is the output size
net = network.Network( [ 784, 30, 30, 10 ] )

# Does the actual training with the parameters: epochs, mini_batch_size, learning_rate
net.SGD(training_data, 10, 10, 3, test_data = test_data, show_progress = True )


file_name = str(net.sizes).replace("[","").replace("]","").replace(" ","").replace(",","_")+'.pkl'
new_net_properties = [net.performance, net.weights, net.biases]


# Saves the learned weights and biases. If the shape had been previously trained,
# it will override previous weights and biases if new performance is better.
# If it is a new shape, it will save to a new file
try:
	with open('best_networks/{0}'.format(file_name), 'rb') as f:
		old_net_properties = pickle.load(f)
	print(old_net_properties[0], "old")
	print(new_net_properties[0], "new")

	if new_net_properties[0] > old_net_properties[0]:
		with open('best_networks/{0}'.format(file_name), 'wb') as f:
			pickle.dump(new_net_properties, f)

		print('Found a better version of network {0}!'.format(file_name))
	else:
		print('New network not better than previous')

except FileNotFoundError:
	print('New Network Shape!')
	with open('best_networks/{0}'.format(file_name), 'wb') as f:
		pickle.dump(new_net_properties, f)
	print( new_net_properties[0] )