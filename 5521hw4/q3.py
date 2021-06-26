from MyLogisticReg2 import *
from my_cross_val import *
import numpy as np
import sklearn.datasets
from sklearn.datasets import load_boston
from sklearn.datasets import load_digits

# load data
boston = load_boston()
digits = load_digits()

# calculate predicted y for boston data
mid = np.median(boston.target)
y_50 = np.int64(boston.target >= mid)
p_50 = np.sum(y_50) / boston.target.shape[0] 
y_50 = 2 * y_50 - 1
print("Boston50:")
print("p(y=1) = " + str(p_50))
print("p(y=-1) = " + str(1-p_50))
print()

sorted_target = np.sort(boston.target)
seventy_fifth = sorted_target[np.int64(0.75 * boston.target.shape[0])]
y_75 = np.int64(boston.target >= seventy_fifth)
p_75 = np.sum(y_75) / boston.target.shape[0]
y_75 = 2 * y_75 - 1
print("Boston75")
print("p(y=1) = " + str(p_75))
print("p(y=-1) = " + str(1-p_75))
print()

folds_error_rate_1 = my_cross_val("MySVM2", boston.data, y_50, 5)
print("Error rates for MySVM2 with Boston50")
print(folds_error_rate_1)
print("mean error = " + str(np.mean(folds_error_rate_1)))
print("standard deviation = " + str(np.std(folds_error_rate_1)))
print()

folds_error_rate_2 = my_cross_val("MySVM2", boston.data, y_75, 5)
print("Error rates for MySVM2 with Boston75")
print(folds_error_rate_2)
print("mean error = " + str(np.mean(folds_error_rate_2)))
print("standard deviation = " + str(np.std(folds_error_rate_2)))
print()



'''
# fit, predict and print error rate
folds_error_rate_1 = my_cross_val("MyLogisticReg2", boston.data, y_50, 5)
print("Error rates for MyLogisticReg2 with Boston50")
print(folds_error_rate_1)
print("mean error = " + str(np.mean(folds_error_rate_1)))
print("standard deviation = " + str(np.std(folds_error_rate_1)))
print()

folds_error_rate_2 = my_cross_val("MyLogisticReg2", boston.data, y_75, 5)
print("Error rates for MyLogisticReg2 with Boston75")
print(folds_error_rate_2)
print("mean error = " + str(np.mean(folds_error_rate_2)))
print("standard deviation = " + str(np.std(folds_error_rate_2)))
print()

folds_error_rate_3 = my_cross_val("MyLogisticReg2", digits.data, digits.target, 5)
print("Error rates for MyLogisticReg2 with Digits")
print(folds_error_rate_3)
print("mean error = " + str(np.mean(folds_error_rate_3)))
print("standard deviation = " + str(np.std(folds_error_rate_3)))
print()

folds_error_rate_4 = my_cross_val("MyLogisticReg", boston.data, y_50, 5)
print("Error rates for MyLogisticReg with Boston50")
print(folds_error_rate_4)
print("mean error = " + str(np.mean(folds_error_rate_4)))
print("standard deviation = " + str(np.std(folds_error_rate_4)))
print()

folds_error_rate_5 = my_cross_val("MyLogisticReg", boston.data, y_75, 5)
print("Error rates for MyLogisticReg with Boston75")
print(folds_error_rate_5)
print("mean error = " + str(np.mean(folds_error_rate_5)))
print("standard deviation = " + str(np.std(folds_error_rate_5)))
print()

folds_error_rate_6 = my_cross_val("MyLogisticReg", digits.data, digits.target, 5)
print("Error rates for MyLogisticReg with Digits")
print(folds_error_rate_6)
print("mean error = " + str(np.mean(folds_error_rate_6)))
print("standard deviation = " + str(np.std(folds_error_rate_6)))
print()
'''







