import numpy as np

# size of w : dimension * k
# size of X_mat : num * dimension
# size of y_mat : num * k
 
class MyLogisticReg2:

	def __init__(self, dimension):
		self.w = np.ones((dimension, 1))

	def fit(self, X, y):
		labels = np.unique(y)
		k = labels.shape[0] #number of class
		num = X.shape[0] # number of samples
		dimension = X.shape[1] # dimension of features

		X = np.append(X, np.ones((num, 1)), axis = 1 ) # expend for w0
		X_mat = np.mat(X)

		y2 = np.zeros((num, k))
		for i in range(num):
			y2[i,y[i]] = 1
		y_mat = np.mat(y2)

		learning_rate = 0.001
		max_itr = 500
		self.w = np.ones((dimension+1, k)) # expend dimension for w0
		for i in range(max_itr):
			h = 1.0 / (1 + np.exp(-(X_mat * self.w)))
			error = h - y_mat
			grad = X_mat.T * error
			self.w = self.w - learning_rate * grad


	def predict(self, X):
		X = np.append(X, np.ones((X.shape[0], 1)), axis = 1 )
		X_mat = np.mat(X)
		h = 1.0 / (1 + np.exp(-(X_mat * self.w)))
		predict = np.squeeze(np.asarray(np.argmax(h, axis=1)))
		#print(predict)
		return predict


