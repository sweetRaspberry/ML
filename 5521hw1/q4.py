from rand_proj import *
from quad_proj import *
from my_cross_val import *
import numpy as np
import sklearn.datasets
from sklearn.datasets import load_digits


digits = load_digits()

X_proj = rand_proj(digits.data, 32)
X_quad = quad_proj(digits.data)
#print(X_proj.shape)
#print(X_quad.shape)

# LinearSVC
folds_error_rate_1 = my_cross_val("LinearSVC", X_proj, digits.target, k=10)
print("Error rates for LinearSVC with random projection:")
print(folds_error_rate_1)
print("mean error = " + str(np.mean(folds_error_rate_1)))
print("standard deviation = " + str(np.std(folds_error_rate_1)))
print()

folds_error_rate_2 = my_cross_val("LinearSVC", X_quad, digits.target, k=10)
print("Error rates for LinearSVC with quadratic projection:")
print(folds_error_rate_2)
print("mean error = " + str(np.mean(folds_error_rate_2)))
print("standard deviation = " + str(np.std(folds_error_rate_2)))
print()

# SVC
folds_error_rate_3 = my_cross_val("SVC", X_proj, digits.target, k=10)
print("Error rates for SVC with random projection:")
print(folds_error_rate_3)
print("mean error = " + str(np.mean(folds_error_rate_3)))
print("standard deviation = " + str(np.std(folds_error_rate_3)))
print()

X_quad = quad_proj(digits.data)
folds_error_rate_4 = my_cross_val("SVC", X_quad, digits.target, k=10)
print("Error rates for SVC with quadratic projection:")
print(folds_error_rate_4)
print("mean error = " + str(np.mean(folds_error_rate_4)))
print("standard deviation = " + str(np.std(folds_error_rate_4)))
print()

# LogisticRegression
folds_error_rate_5 = my_cross_val("LogisticRegression", X_proj, digits.target, k=10)
print("Error rates for LogisticRegression with random projection:")
print(folds_error_rate_5)
print("mean error = " + str(np.mean(folds_error_rate_5)))
print("standard deviation = " + str(np.std(folds_error_rate_5)))
print()

X_quad = quad_proj(digits.data)
folds_error_rate_6 = my_cross_val("LogisticRegression", X_quad, digits.target, k=10)
print("Error rates for LogisticRegression with quadratic projection:")
print(folds_error_rate_6)
print("mean error = " + str(np.mean(folds_error_rate_6)))
print("standard deviation = " + str(np.std(folds_error_rate_6)))
print()



