# have input "method, X, y, k"
# give output : error rates for each k folder
# no data loading

import numpy as np
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

def SVC_(X_train, y_train, X_test):
	clc = SVC(gamma='auto', max_iter=500)
	clc.fit(X_train, y_train)
	return clc.predict(X_test)

def LinearSVC_(X_train, y_train, X_test):
	clf = LinearSVC(random_state=0, tol=1e-5, max_iter=500)
	clf.fit(X_train, y_train)
	return clf.predict(X_test)	

def LogisticRegression_(X_train, y_train, X_test):
	clf = LogisticRegression(penalty='l2',solver='lbfgs',multi_class='multinomial',max_iter=500)
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


def my_cross_val(method, X, y, k):
	X_norm = preprocessing.normalize(X) # how to normalize data?
	X_folds = np.array_split(X_norm, 10, axis=0)
	y_folds = np.array_split(y, 10, axis=0)
	result = []
	for i in range(10):
		#print(i)
		X_test = X_folds[i]
		y_test = y_folds[i]

		nine_X_folds = np.delete(X_folds, i)
		nine_y_folds = np.delete(y_folds, i)

		X_train = np.concatenate(nine_X_folds, axis=0)
		y_train = np.concatenate(nine_y_folds, axis=0)
		
		y_predict = choose_method(method, X_train, y_train, X_test)
		error = np.sum(y_test != y_predict)/y_test.shape[0]
		result.append(error)
	return result


