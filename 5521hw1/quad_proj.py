#input X  (1797 * 64)
#output X_proj (1797 * 2144)

import numpy as np
from random import seed

def quad_proj(X):
	m, n = X.shape
	result = X
	#print(result.shape)
	result = np.concatenate( (result, np.square(X)), axis=1)
	#print(result.shape)
	
	product = np.zeros((m, int(n*(n-1)/2)))
	#print(product.shape)
	count = 0
	for i in range(64):
		for j in range(i+1, 64):		
			product[:, count] = np.multiply(X[:, i], X[:, j])
			count = count + 1		
	#print(count)
	result = np.concatenate( (result, product), axis=1) 
	return result
