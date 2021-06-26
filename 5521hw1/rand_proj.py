#input X, d
# size of X = 1797 * 64
#output X_proj
#size of X_proj = 1797 * 32
#size of m = 64 * d

import numpy as np
from random import seed

def rand_proj(X, d):
	m = np.random.rand(64, d)
	return np.dot(X, m)
