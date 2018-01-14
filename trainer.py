import mnist_loader
import pickle
import network

# Loads training data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Creates untrained network of desired shape
# Enter in the dimensions of the network in the form [a, b, c, d ... n], 
# where a is the input size and n is the output size
net = network.Network( [ 784, 20, 10] )

net.SGD(training_data, # Data to train on
	100, 			   # epochs
	10, 			   # mini_batch_size
	0.3, 			   # eta
	5e-4, 		   	   # lmbda
	test_data, 		   # test_data
	validation_data,   # evaluation_data
	False, 			   # monitor_evaluation_cost
	True, 			   # monitor_evaluation_accuracy
	False,			   # monitor_training_cost
	True, 			   # monitor_training_accuracy
	True, 			   # keep_best
	True, 			   # show_progress
	20, 			   # early_stopping_n
	1, 				   # eta_schedule_change_n
	2.0, 			   # eta_decrease_factor
	0)   			   # eta_decrease_steps
	


# Creates filename and prepares network properties(performance and parameters) for saving
# name follows naming scheme: inputsize_layer1size_layer2size_...outputsize 
file_name = str(net.sizes).replace("[","").replace("]","").replace(" ","").replace(",","_")+'.pkl'
new_net_properties = [net.performance, net.weights, net.biases, ]

''' Saves netwwork properties (learned parameters and performance). 
If the shape had been previously trained, it will override previous saved file if 
new performance is better. If it is a new shape, it will save to a new file
'''

try:
	with open('best_networks/{0}'.format(file_name), 'rb') as f:
		old_net_properties = pickle.load(f)
	if new_net_properties[0] > old_net_properties[0]:
		with open('best_networks/{0}'.format(file_name), 'wb') as f:
			pickle.dump(new_net_properties, f)
		print('Found a better version of network with shape {0}!'.format(file_name[:-4]))
	else:
		print('New network not better than previous')

	print(old_net_properties[0], "old")
	print(new_net_properties[0], "new")

except FileNotFoundError:
	print('New Network Shape!')
	with open('best_networks/{0}'.format(file_name), 'wb') as f:
		pickle.dump(new_net_properties, f)
	print("new performance", new_net_properties[0])

# if no test_data is given, performance cannot be compared
except TypeError:
	print('Must supply test_data to update file')