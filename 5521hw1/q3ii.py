from my_train_test import *
import numpy as np
import sklearn.datasets
from sklearn.datasets import load_boston
from sklearn.datasets import load_digits

#load data
boston = load_boston()
digits = load_digits()

#calculate predicted y for boston data
mid = np.median(boston.target)
y_50 = np.int64(boston.target >= mid)
p_50 = np.sum(y_50) / boston.target.shape[0] 
print("Boston50:")
print("p(y=1) = " + str(p_50))
print("p(y=0) = " + str(1-p_50))
print()

sorted_target = np.sort(boston.target)
seventy_fifth = sorted_target[np.int64(0.75 * boston.target.shape[0])]
y_75 = np.int64(boston.target >= seventy_fifth)
p_75 = np.sum(y_75) / boston.target.shape[0]
print("Boston75")
print("p(y=1) = " + str(p_75))
print("p(y=0) = " + str(1-p_75))
print()

#print out error rate
folds_error_rate_1 = my_train_test("LinearSVC", boston.data, y_50, pi=0.75, k=10)
print("Error rates for LinearSVC with Boston50:")
print(folds_error_rate_1)
print("mean error = " + str(np.mean(folds_error_rate_1)))
print("standard deviation = " + str(np.std(folds_error_rate_1)))
print()

folds_error_rate_2 = my_train_test("LinearSVC", boston.data, y_75, pi=0.75, k=10)
print("Error rates for LinearSVC with Boston75:")
print(folds_error_rate_2)
print("mean error = " + str(np.mean(folds_error_rate_2)))
print("standard deviation = " + str(np.std(folds_error_rate_2)))
print()

folds_error_rate_3 = my_train_test("LinearSVC", digits.data, digits.target, pi=0.75, k=10)
print("Error rates for LinearSVC with digits:")
print(folds_error_rate_3)
print("mean error = " + str(np.mean(folds_error_rate_3)))
print("standard deviation = " + str(np.std(folds_error_rate_3)))
print()

folds_error_rate_4 = my_train_test("SVC", boston.data, y_50, pi=0.75, k=10)
print("Error rate for SVC with Boston50:")
print(folds_error_rate_4)
print("mean error = " + str(np.mean(folds_error_rate_4)))
print("standard deviation = " + str(np.std(folds_error_rate_4)))
print()

folds_error_rate_5 = my_train_test("SVC", boston.data, y_75, pi=0.75, k=10)
print("Error rate for SVC with Boston75:")
print(folds_error_rate_5)
print("mean error = " + str(np.mean(folds_error_rate_5)))
print("standard deviation = " + str(np.std(folds_error_rate_5)))
print()

folds_error_rate_6 = my_train_test("SVC", digits.data, digits.target, pi=0.75, k=10)
print("Error rates for SVC with digits:")
print(folds_error_rate_6)
print("mean error = " + str(np.mean(folds_error_rate_6)))
print("standard deviation = " + str(np.std(folds_error_rate_6)))
print()

folds_error_rate_7 = my_train_test("LogisticRegression", boston.data, y_50, pi=0.75, k=10)
print("Error rate for LogisticRegression with Boston50:")
print(folds_error_rate_7)
print("mean error = " + str(np.mean(folds_error_rate_7)))
print("standard deviation = " + str(np.std(folds_error_rate_7)))
print()

folds_error_rate_8 = my_train_test("LogisticRegression", boston.data, y_75, pi=0.75, k=10)
print("Error rate for LogisticRegression with Boston75:")
print(folds_error_rate_8)
print("mean error = " + str(np.mean(folds_error_rate_8)))
print("standard deviation = " + str(np.std(folds_error_rate_8)))
print()

folds_error_rate_9 = my_train_test("LogisticRegression", digits.data, digits.target, pi=0.75, k=10)
print("Error rates for LogisticRegression with digits:")
print(folds_error_rate_9)
print("mean error = " + str(np.mean(folds_error_rate_9)))
print("standard deviation = " + str(np.std(folds_error_rate_9)))
print()



