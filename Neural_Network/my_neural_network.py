import numpy as np
import sklearn.datasets
import sklearn.linear_model
import matplotlib
import matplotlib.pyplot as plt

def sigmoid(input_):
	output = 1.0/(1+np.exp(-input_))
	return output

def forward_propagation(w, b, x, layers):
	input_prv = x
	caches = []

	for i in range(1, layers):
		#cache_i[input_prv, w[i], b[i], output, output_sig]
		cache = []	
		cache.append(input_prv)
		cache.append(w["w"+str(i)])
		cache.append(b["b"+str(i)])

		output = np.dot(w["w"+str(i)], input_prv) + b["b"+str(i)]
		cache.append(output)
		output_sig = sigmoid(output)
		cache.append(output_sig)

		#caches[cache_1, cache_2, ... , cache_i]
		caches.append(cache) 

		input_prv = output_sig

	caches = np.array(caches)

	return output_sig, caches

def back_propagation(y, output, caches, layers):
	gradients_w = {}
	gradients_b = {}
	length = layers - 1

	current = caches[-1]
	derror_out = current[4]-y

	for i in range(length):
		current = caches[length-i-1]
		input_prv = current[0]
		out_sig = current[4]

		derror_in = derror_out * out_sig*(1 - out_sig)
		derror_out = np.dot(current[1].T, derror_in)

		dE_dw = np.dot(derror_in, input_prv.T)
		dE_dw = dE_dw/dE_dw.shape[1]
		gradients_w["gw"+str(length-i)] = dE_dw
		
		dE_db = np.sum(derror_in, axis=1, keepdims=True)
		dE_db = dE_db/dE_dw.shape[1]
		gradients_b["gb"+str(length-i)] = dE_db
		
	return gradients_w, gradients_b

def update(w, b, gradients_w, gradients_b, learning_rate):
	length = len(gradients_w)
	for i in range(length):
		w["w"+str(i+1)] = w["w"+str(i+1)] - learning_rate * gradients_w["gw"+str(i+1)]
		b["b"+str(i+1)] = b["b"+str(i+1)] - learning_rate * gradients_b["gb"+str(i+1)]

	return w, b

def compute_errors(output, y):	
	m = y.shape[1]
	errors = -np.sum(np.multiply(np.log(output), y) + np.multiply( np.log(1-output), (1-y))  )/m
	errors = np.squeeze(errors)
	return errors

def plot_data(x, y, w, b, layers):
	plt.figure()#figsize=(16, 32))
	length = layers - 1
	
	horizontal_min, horizontal_max = x[0, :].min() - 0.5, x[0, :].max() + 0.5
	vertical_min, vertical_max = x[1, :].min() - 0.5, x[1, :].max() + 0.5
	h = .01
	xx, yy = np.meshgrid(np.arange(horizontal_min, horizontal_max, h), np.arange(vertical_min, vertical_max, h))

	int_prv = (np.c_[xx.ravel(), yy.ravel()]).T	
	for i in range(length):
		output = np.dot(w["w"+str(i+1)], int_prv) + b["b"+str(i+1)]
		out_sig = sigmoid(output)
		int_prv = out_sig
	
	z = np.int64(out_sig>0.5)
	z = z.reshape(xx.shape)

	plt.contourf(xx, yy, z, cmap=plt.cm.Spectral)	
	plt.scatter(x[0, :], x[1, :], c=np.squeeze(y), marker='o', cmap=plt.cm.Spectral, edgecolors='black')
	plt.show()

def training(x, y, w, b, iteration, learning_rate, layers):
	for i in range(iteration):
		output, caches = forward_propagation(w, b, x, layers)
		gradients_w, gradients_b = back_propagation(y, output, caches, layers)
		w, b = update(w, b, gradients_w, gradients_b, learning_rate)
		error = compute_errors(output, y)
		
		print(error)
	plot_data(x, y, w, b, layers)
	return w, b

#-----------------------------------------------------------------------------------
layers_structure = np.array([2, 30, 1]) #number of nodes in each layer
layers = len(layers_structure) #number of layers
learning_rate = 0.5
iteration = 50000

w = dict()
b = dict()
for l in range(1, layers):
	w["w"+str(l)] = np.random.randn(layers_structure[l], layers_structure[l-1])/np.sqrt(layers_structure[l-1])
	b["b"+str(l)] = np.zeros((layers_structure[l], 1))

#x, y = sklearn.datasets.make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=0, n_clusters_per_class=1)
#x, y = sklearn.datasets.make_circles(100, noise=0.2, factor=0.5, random_state=1)
x, y = sklearn.datasets.make_moons(100, noise=1.0)

x = x.T #coordinate
y = np.array(y).reshape(1,100) #color

w, b = training(x, y, w, b, iteration, learning_rate, layers)






