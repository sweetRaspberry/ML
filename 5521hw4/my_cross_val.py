# have input "method, X, y, num_folders"
# give output : error rates for each folder
# no data loading

import numpy as np
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
#from MultiGaussClassify import *
from MySVM2 import *

def MySVM2_(X_train, y_train, X_test):
	clf = MySVM2(2) # initiallize the default number of class
	clf.fit(X_train, y_train)
	return clf.predict(X_test)

def MyLogisticReg2_(X_train, y_train, X_test):
	clf = MyLogisticReg2(2) # initiallize the default number of class
	clf.fit(X_train, y_train)
	return clf.predict(X_test)

def MultiGaussClassify_(X_train, y_train, X_test, diag):
	clf = MultiGaussClassify(2) # initiallize the default number of class
	clf.fit(X_train, y_train, diag)
	return clf.predict(X_test)

def SVC_(X_train, y_train, X_test):
	clf = SVC(gamma='auto', max_iter=500)
	clf.fit(X_train, y_train)
	return clf.predict(X_test)

def LinearSVC_(X_train, y_train, X_test):
	clf = LinearSVC(random_state=0, tol=1e-5, max_iter=500)
	clf.fit(X_train, y_train)
	return clf.predict(X_test)	

def LogisticRegression_(X_train, y_train, X_test):
	clf = LogisticRegression(penalty='l2',solver='lbfgs',multi_class='multinomial',max_iter=500)
	clf.fit(X_train, y_train)
	return clf.predict(X_test);

def choose_method(method, X_train, y_train, X_test, diag):
	if method == "MultiGaussClassify":
		return MultiGaussClassify_(X_train, y_train, X_test, diag)
	elif method == "SVC":
		return SVC_(X_train, y_train, X_test)
	elif method == "LinearSVC":
		return LinearSVC_(X_train, y_train, X_test)
	elif method == "MyLogisticReg2":
		return MyLogisticReg2_(X_train, y_train, X_test)
	elif method == "MySVM2":
		return MySVM2_(X_train, y_train, X_test)
	else:
		return LogisticRegression_(X_train, y_train, X_test)


def my_cross_val(method, X, y, num_folders, diag=False):
	X_norm = preprocessing.normalize(X) # how to normalize data?
	#X_norm = X # not normalize?

	X_folds = np.array_split(X_norm, num_folders, axis=0)
	y_folds = np.array_split(y, num_folders, axis=0)
	result = []
	for i in range(num_folders):
		X_test = X_folds[i]
		y_test = y_folds[i]

		rest_X_folds = np.delete(X_folds, i)
		rest_y_folds = np.delete(y_folds, i)

		X_train = np.concatenate(rest_X_folds, axis=0)
		y_train = np.concatenate(rest_y_folds, axis=0)

		#print(i)
		print(y_test)
		#print(np.shape(y_test))
		
		y_predict = choose_method(method, X_train, y_train, X_test, diag)
		error = np.sum(y_test != y_predict)/y_test.shape[0]
		result.append(error)
	return result


