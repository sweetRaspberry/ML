# has input "method, X, y, pi, k"
# give output : error rates for each k folders
# no data loading

import numpy as np
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from random import seed
from random import randrange
import math
from sklearn import preprocessing #??????????????

def SVC_(X_train, y_train, X_test):
	clc = SVC(gamma='auto', max_iter=500)
	clc.fit(X_train, y_train)
	return clc.predict(X_test)

def LinearSVC_(X_train, y_train, X_test):
	clf = LinearSVC(random_state=0, tol=1e-5, max_iter=500)
	clf.fit(X_train, y_train)
	return clf.predict(X_test)	

def LogisticRegression_(X_train, y_train, X_test):
	clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=500)
	clf.fit(X_train, y_train)
	return clf.predict(X_test);

def choose_method(method, X_train, y_train, X_test):
	if method == "SVC":
		#print("SVC")
		return SVC_(X_train, y_train, X_test)
	elif method == "LinearSVC":
		#print("LinearSVC")
		return LinearSVC_(X_train, y_train, X_test)
	else:
		#print("LogisticRegression")
		return LogisticRegression_(X_train, y_train, X_test)


def my_train_test(method, X, y, pi, k):

	X = preprocessing.normalize(X) # how to normalize data?
	result = []

	for itr in range(10):

		X_train = np.zeros(( round(pi*X.shape[0]), X.shape[1]))
		y_train = np.zeros(( round(pi*y.shape[0]) ))
		X_test = X
		y_test = y

		for i in range(X_train.shape[0]): 

			index = randrange(X_test.shape[0])
	
			X_train[i,:] = X_test[index,:]
			X_test = np.delete(X_test, index, axis=0)
	
			y_train[i] = y_test[index]
			y_test = np.delete(y_test, index)

		y_predict = choose_method(method, X_train, y_train, X_test)
		error = np.sum(y_test != y_predict)/y_test.shape[0]
		result.append(error)

	return result





