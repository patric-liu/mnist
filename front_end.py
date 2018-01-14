import numpy as np
from matplotlib import pyplot as plt
import scipy.ndimage
import network
import pickle

# import user image
user_picture = scipy.ndimage.imread('sample.jpg', flatten=True, mode=None)
user_picture = (-user_picture/255)+1
shape = np.shape(user_picture)

# plot imported image
imgplot = plt.imshow(user_picture, cmap = 'Greys')
plt.show()

# turns imported image (array) into a vector
vector = np.zeros( (784, 1) )
try:
	for i in range(shape[1]):
		for j in range(shape[0]):
			value = user_picture[ i, j ]	
			index = i*shape[1] + j
			vector[ index ] = value
except IndexError:
	print("\n"+"*** Error: image must be 28x28 ***"+"\n")
user_picture = vector

# chooses the network shape to use
net = network.Network( [ 784, 150, 50, 100, 10 ] )
	
# file name of the file containing best weights and biases for that shape
file_name = str(net.sizes).replace("[","").replace("]","").replace(" ","").replace(",","_")+'.pkl'

digit2word = {
	0 : 'ZERO',
	1 : 'ONE',
	2 : 'TWO',
	3 : 'THREE',
	4 : 'FOUR',
	5 : 'FIVE',
	6 : 'SIX',
	7 : 'SEVEN',
	8 : 'EIGHT',
	9 : 'NINE'
}
indices = [0,1,2,3,4,5,6,7,8,9]

# optional softmax output displays network output as a probability distribution
softmax = True
hardness = 1.0

# loads network and feeds vectorized image through the network
try:
	with open('best_networks/{0}'.format(file_name), 'rb') as f:
		net_properties = pickle.load(f)
	print("network performance:", net_properties[0],'\n')
	net.weights = net_properties[1]
	net.biases  = net_properties[2]
	
	output_vector = net.feedforward(user_picture)
	result = np.argmax(output_vector)
	print('My guess is {0}! '.format(digit2word[result]),'\n')
	print('Confidence:')

	if softmax:
		for output,indice in zip(net.feedforward_softmax_output(user_picture, hardness),indices):
			print ("{0}".format(indice)+":{:10.4f}".format(float(output)))
	else:
		for output,indice in zip(output_vector,indices):
			print ("{0}".format(indice)+":{:10.4f}".format(float(output)))

# brings up error if no weights/biases exist for the desired shape 
except FileNotFoundError:
	print('{0} shape has not been trained yet'.format(net.sizes))