import numpy as np
import math
#from scipy.stats import multivariate_normal

def replaceZeroes(data):
	min_nonzero = np.min(data[np.nonzero(data)])
	data[data == 0] = min_nonzero
	return data

class MultiGaussClassify:
	# initial uniform prior, zero mean, identity covariance
	# k is number of classes
	def __init__(self, k):
		self.priors = np.full((k, 1), 1/k)
		self.means = np.zeros((k, 1))
		self.covariance = np.identity(k)

	def fit(self, X, y, diag):
		labels = np.unique(y)

		k = labels.shape[0]
		dimension = X.shape[1]

		# size of priors: k x 1 // 10 x 1
		self.priors = np.divide(np.bincount(y), k)

		# size of means: k x dimension // 10 x 64
		# size of covariance: k x dimension x dimension	// 10 x 64 x 64	
		self.means = np.zeros((k, dimension))
		self.covariance = np.zeros((k, dimension, dimension))
		for i in range(0, k):
			label = labels[i]
			indexs = np.where(y==label)
			single_class = X[indexs]
			sum_ = np.sum(single_class, axis = 0)
			mean = np.divide(sum_, single_class.shape[0])
			self.means[i,:] = mean[:];
			if diag == False:	
				self.covariance[i,:,:] = np.cov(single_class.T)
			else:
				self.covariance[i,:,:] = np.diag(np.diag(np.cov(single_class.T)))

	def predict(self, X):
		k = self.priors.shape[0]
		dimension = X.shape[1]

		prediction = np.zeros((k, X.shape[0]))
		for i in range(0, k):
			inv = np.linalg.pinv(self.covariance[i])
			det = np.linalg.det(self.covariance[i])
			
			discriminant = -0.5 * np.diag(np.dot((X-self.means[i]), np.dot(inv, (X-self.means[i]).T))) \
					- 0.5 * dimension * np.log(2*math.pi)
			if det != 0:
				discriminant = discriminant - 0.5 * np.log(det)
			if self.priors[i] != 0:
				discriminant = discriminant + np.log(self.priors[i])
			#mvn = multivariate_normal(self.means[i],self.covariance[i], allow_singular=True)
			#prediction[i,:] = np.log(replaceZeroes(mvn.pdf(X)))
			prediction[i,:] = discriminant			
			maximum = np.argmax(prediction, axis=0)
		return maximum


