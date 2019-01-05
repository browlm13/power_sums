import numpy as np
from numpy.linalg import matrix_power

#
# Actually a goood method for calculating sums of integers...
#

d = 3
n = 5

v = np.ones(shape=(n,))
L = np.tril(np.ones(shape=(n,n)))
R = matrix_power(L, 2)

# sum of integers
#print(R @ v)

P = np.repeat(R[np.newaxis, :, :], d, axis=0)
P = np.multiply.reduce(P, 0)

S = P @ v

# sum of first n integers to the dth power is at index n-1 of S
print(S)

