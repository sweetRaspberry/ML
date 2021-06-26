import numpy as np
from sklearn import preprocessing

 
class MySVM2:

	def __init__(self, dimension):
		self.w = np.ones((dimension, 1))
		self.b = 1


	def fit(self, X, y):
		#print(X.shape) # 404 x 13
		
		num = X.shape[0] # 404
		dimension = X.shape[1] # 13

		y = y.reshape(num,1) # 404 x 1

		learning_rate = 0.1
		max_itr = 100

		self.b = 1
		self.w = np.ones((dimension, 1)) # 13 x 1
		#print(self.w)
		for i in range(max_itr):
			error = 1 - y.T * (np.dot(X, self.w) + self.b ).T
			#print(error)
			zeroMask = np.int64(error > 0) # 1 x 404
			#print(zeroMask)

			YX_product = np.multiply(y.T, X.T) # 13 x 404
			YX_zero = np.multiply(zeroMask, YX_product) # 13 x 404

			#print(self.w.shape)
			#print(np.sum(YX_zero, axis=1).shape)
			d_w = - np.sum(YX_zero, axis=1).reshape(dimension,1) / num + 5 * self.w # 13
			d_b = - np.sum(zeroMask * y.T) / num # scalar

			self.w = self.w - learning_rate * d_w.reshape(dimension,1)
			self.b = self.b - learning_rate * d_b

			#cof = preprocessing.normalize(self.w, axis=0)[0,0] / self.w[0,0]
			#self.w = preprocessing.normalize(self.w, axis=0)
			#self.b = cof * self.b

		#print(self.w)
		#print(self.b)
		#print(np.sum(self.w))

	def predict(self, X):
		print(self.w)
		print(self.b)
		predict = (np.dot(X, self.w) + self.b).T
		predict = np.int64(predict > 0)
		predict = 2 * predict - 1
		#print(predict)
		return predict



